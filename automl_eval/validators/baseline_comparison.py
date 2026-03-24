"""
BaselineComparisonValidator — compares the agent's model with a simple boosting baseline.

The environment automatically trains a baseline GBT on numeric features (no FE).
At FINAL_SUBMIT it compares:
  - If the agent beats the baseline -> bonus.
  - If worse -> penalty (why do FE if baseline is better?).
  - If approximately equal -> neutral (no harm, but no help either).

Also checks for diminishing returns:
  If the metric stopped growing (last N measurements are approximately stable),
  but the agent keeps running cycles -> time to stop.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np

from automl_eval.validators.base import BaseValidator, ValidationResult

if TYPE_CHECKING:
    from automl_eval.session import RuntimeSession

logger = logging.getLogger(__name__)

_PLATEAU_WINDOW = 3
_PLATEAU_THRESHOLD = 0.003


class BaselineComparisonValidator(BaseValidator):
    """Compares the agent's model with an automatic baseline."""

    name = "baseline_comparison"

    def __init__(
        self,
        worse_than_baseline_penalty: float = 0.15,
        better_bonus: float = 0.05,
        plateau_penalty: float = 0.05,
    ) -> None:
        self.worse_than_baseline_penalty = worse_than_baseline_penalty
        self.better_bonus = better_bonus
        self.plateau_penalty = plateau_penalty

    def validate(self, session: RuntimeSession) -> ValidationResult:
        if not session.done:
            return ValidationResult(
                validator_name=self.name,
                passed=True,
                score=1.0,
                details="Baseline comparison runs at FINAL_SUBMIT.",
            )

        issues: list[str] = []
        bonuses: list[str] = []
        penalty = 0.0
        bonus = 0.0

        # ── 1. Compute baseline ──────────────────────────────────────
        baseline_score = self._compute_baseline(session)
        agent_score = session.best_metric

        if baseline_score is not None and agent_score is not None:
            diff = agent_score - baseline_score

            if diff > _PLATEAU_THRESHOLD:
                bonuses.append(
                    f"Agent ({agent_score:.4f}) beats baseline ({baseline_score:.4f}) "
                    f"by +{diff:.4f}"
                )
                bonus += self.better_bonus
            elif diff < -_PLATEAU_THRESHOLD:
                issues.append(
                    f"Agent ({agent_score:.4f}) is worse than baseline "
                    f"({baseline_score:.4f}) by {diff:.4f}"
                )
                penalty += self.worse_than_baseline_penalty
            else:
                bonuses.append(
                    f"Agent ({agent_score:.4f}) ≈ baseline ({baseline_score:.4f})"
                )
        elif agent_score is None:
            issues.append("No agent metric available for comparison")
            penalty += self.worse_than_baseline_penalty

        # ── 2. Plateau detection ─────────────────────────────────────
        plateau_msg = self._check_plateau(session)
        if plateau_msg:
            issues.append(plateau_msg)
            penalty += self.plateau_penalty

        score = max(0.0, min(1.0, 1.0 - penalty + bonus))
        passed = len(issues) == 0

        parts = []
        if bonuses:
            parts.append("Good: " + ", ".join(bonuses))
        if issues:
            parts.append("Issues: " + "; ".join(issues))
        details = ". ".join(parts) if parts else "Baseline comparison not applicable."

        return ValidationResult(
            validator_name=self.name,
            passed=passed,
            score=score,
            details=details,
            penalty=penalty,
        )

    def _compute_baseline(self, session: RuntimeSession) -> float | None:
        """Train a simple GBT on the original numeric features."""
        try:
            from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
            from automl_eval.metrics import compute_metric
            from automl_eval.task import TaskType

            train_df = session.train_df
            test_df = session.test_df
            if train_df is None or test_df is None:
                return None

            target = session.task.target_column
            num_cols = train_df.drop(columns=[target]).select_dtypes(
                include="number"
            ).columns.tolist()

            if len(num_cols) < 1:
                return None

            X_tr = train_df[num_cols].fillna(0).values
            y_tr = train_df[target].values
            X_te = test_df[num_cols].fillna(0).values
            y_te = test_df[target].values

            is_clf = session.task.task_type != TaskType.REGRESSION
            if is_clf:
                model = GradientBoostingClassifier(
                    n_estimators=50, max_depth=3, random_state=42
                )
            else:
                model = GradientBoostingRegressor(
                    n_estimators=50, max_depth=3, random_state=42
                )

            model.fit(X_tr, y_tr)

            if is_clf and hasattr(model, "predict_proba"):
                preds = model.predict_proba(X_te)
                if preds.shape[1] == 2:
                    preds = preds[:, 1]
            else:
                preds = model.predict(X_te)

            return compute_metric(session.task.metric, y_te, preds)

        except Exception as e:
            logger.debug("Baseline computation failed: %s", e)
            return None

    def _check_plateau(self, session: RuntimeSession) -> str | None:
        """If the last N metrics are not improving and cycle > 2 — time to stop."""
        history = session.metric_history
        if len(history) < _PLATEAU_WINDOW:
            return None

        recent = [m for _, m in history[-_PLATEAU_WINDOW:]]
        spread = max(recent) - min(recent)

        if spread < _PLATEAU_THRESHOLD and session.cycle_count > 2:
            return (
                f"Metric plateau: last {_PLATEAU_WINDOW} values spread "
                f"= {spread:.4f} (<{_PLATEAU_THRESHOLD}), "
                f"but {session.cycle_count} cycles run — diminishing returns"
            )

        return None
