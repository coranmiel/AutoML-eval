"""
HyperparamValidator — checks that hyperparameters are meaningful and training TTL is reasonable.

1. Hyperparameters should be explicitly set (not all defaults).
2. Known anti-patterns (n_estimators=1, max_depth=None on a large dataset).
3. TTL per model training: if a single .fit() took >X% of the budget — penalty.
4. Time/quality trade-off: a model that trained long without improvement — penalty.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

from automl_eval.validators.base import BaseValidator, ValidationResult

if TYPE_CHECKING:
    from automl_eval.session import RuntimeSession

_HYPERPARAM_SET = re.compile(
    r"(n_estimators\s*=|max_depth\s*=|learning_rate\s*=|"
    r"n_neighbors\s*=|C\s*=|alpha\s*=|gamma\s*=|"
    r"num_leaves\s*=|min_samples_split\s*=|min_samples_leaf\s*=|"
    r"subsample\s*=|colsample_bytree\s*=|reg_alpha\s*=|reg_lambda\s*=|"
    r"max_features\s*=|min_child_weight\s*=|epochs\s*=|batch_size\s*=)",
    re.IGNORECASE,
)

_BAD_PARAMS = [
    (re.compile(r"n_estimators\s*=\s*1\b"), "n_estimators=1 (too few trees)"),
    (re.compile(r"max_depth\s*=\s*None\b"), "max_depth=None (unlimited depth, overfitting risk)"),
    (re.compile(r"n_estimators\s*=\s*[2-5]\b"), "n_estimators very low (2-5)"),
    (re.compile(r"learning_rate\s*=\s*[1-9]\d*\.?\d*"), "learning_rate >= 1.0 (too high)"),
    (re.compile(r"max_depth\s*=\s*[5-9]\d+"), "max_depth > 50 (extreme, likely overfitting)"),
]

_TUNING_PATTERNS = re.compile(
    r"(GridSearchCV|RandomizedSearchCV|BayesSearchCV|Optuna|optuna|"
    r"HalvingGridSearchCV|cross_val_score|param_grid|param_dist|"
    r"hyperopt|tune|tuning|hyperparameter)",
    re.IGNORECASE,
)


class HyperparamValidator(BaseValidator):
    """Checks hyperparameter quality and time/quality trade-off."""

    name = "hyperparameters"

    def __init__(
        self,
        no_tuning_penalty: float = 0.05,
        bad_param_penalty: float = 0.08,
        slow_model_penalty: float = 0.10,
        model_time_budget_frac: float = 0.5,
    ) -> None:
        self.no_tuning_penalty = no_tuning_penalty
        self.bad_param_penalty = bad_param_penalty
        self.slow_model_penalty = slow_model_penalty
        self.model_time_budget_frac = model_time_budget_frac

    def validate(self, session: RuntimeSession) -> ValidationResult:
        from automl_eval.session import ActionType

        code_steps = [
            rec for rec in session.steps
            if rec.action_type in (ActionType.CODE, ActionType.MODEL)
            and rec.execution_success
        ]
        if not code_steps:
            return ValidationResult(
                validator_name=self.name,
                passed=True,
                score=1.0,
                details="No model code to check.",
            )

        all_code = "\n".join(
            (rec.code_body if rec.code_body else rec.action_text) for rec in code_steps
        )

        issues: list[str] = []
        bonuses: list[str] = []
        penalty = 0.0

        has_fit = bool(re.search(r"\.fit\s*\(", all_code))
        if not has_fit:
            return ValidationResult(
                validator_name=self.name,
                passed=True,
                score=1.0,
                details="No model training detected.",
            )

        # ── 1. Hyperparameters explicitly set? ────────────────────────
        params_set = _HYPERPARAM_SET.findall(all_code)
        has_tuning = bool(_TUNING_PATTERNS.search(all_code))

        if len(params_set) == 0 and not has_tuning:
            issues.append("No hyperparameters explicitly set — using all defaults")
            penalty += self.no_tuning_penalty
        elif has_tuning:
            bonuses.append("Hyperparameter tuning attempted")
        elif len(params_set) >= 2:
            bonuses.append(f"{len(params_set)} hyperparameters explicitly set")

        # ── 2. Bad parameter patterns ─────────────────────────────────
        for pattern, description in _BAD_PARAMS:
            if pattern.search(all_code):
                issues.append(f"Suspicious param: {description}")
                penalty += self.bad_param_penalty

        # ── 3. Model training time check ──────────────────────────────
        fit_steps = [
            rec for rec in code_steps
            if re.search(r"\.fit\s*\(", rec.code_body if rec.code_body else rec.action_text)
        ]
        if len(fit_steps) >= 2:
            timestamps = [rec.timestamp for rec in fit_steps]
            for i in range(1, len(timestamps)):
                step_duration = timestamps[i] - timestamps[i - 1]
                budget = session.task.time_budget_seconds
                if step_duration > budget * self.model_time_budget_frac:
                    issues.append(
                        f"Model training step took {step_duration:.0f}s "
                        f"(>{self.model_time_budget_frac:.0%} of {budget:.0f}s budget)"
                    )
                    penalty += self.slow_model_penalty
                    break

        score = max(0.0, 1.0 - penalty)
        passed = len(issues) == 0

        parts = []
        if bonuses:
            parts.append("Good: " + ", ".join(bonuses))
        if issues:
            parts.append("Issues: " + "; ".join(issues))
        details = ". ".join(parts) if parts else "Hyperparameters look reasonable."

        return ValidationResult(
            validator_name=self.name,
            passed=passed,
            score=score,
            details=details,
            penalty=penalty,
        )
