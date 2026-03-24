"""
EfficiencyValidator — penalizes inefficient use of time.

Three aspects:
1. E2E time limit: if total episode time exceeds hard_limit (default 3600s),
   the maximum penalty is applied.
2. Hyperparameter search: if GridSearchCV / exhaustive search is detected in code,
   a penalty is applied (slow, inefficient).
3. Smooth time penalty: the closer to the limit, the larger the penalty.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

from automl_eval.validators.base import BaseValidator, ValidationResult

if TYPE_CHECKING:
    from automl_eval.session import RuntimeSession

_GRIDSEARCH_PATTERNS = re.compile(
    r"(GridSearchCV|GridSearch|ParameterGrid|"
    r"itertools\.product.*param|"
    r"for\s+\w+\s+in\s+\[.*\]\s*:\s*\n.*\.fit\()",
    re.IGNORECASE | re.DOTALL,
)

_EFFICIENT_SEARCH = re.compile(
    r"(RandomizedSearchCV|BayesSearchCV|Optuna|optuna|HalvingGridSearchCV)",
    re.IGNORECASE,
)


class EfficiencyValidator(BaseValidator):
    """Penalizes excessive execution time and inefficient hyperparameter search."""

    name = "efficiency"

    def __init__(
        self,
        hard_time_limit: float = 3600.0,
        gridsearch_penalty: float = 0.15,
        time_penalty_max: float = 0.3,
    ) -> None:
        self.hard_time_limit = hard_time_limit
        self.gridsearch_penalty = gridsearch_penalty
        self.time_penalty_max = time_penalty_max

    def validate(self, session: RuntimeSession) -> ValidationResult:
        from automl_eval.session import ActionType

        elapsed = session.elapsed_seconds()
        budget = session.task.time_budget_seconds
        issues: list[str] = []
        penalty = 0.0

        if elapsed >= self.hard_time_limit:
            penalty += self.time_penalty_max
            issues.append(
                f"Hard time limit exceeded: {elapsed:.0f}s >= {self.hard_time_limit:.0f}s (max penalty)"
            )
        elif elapsed > budget:
            ratio = min((elapsed - budget) / budget, 1.0)
            time_pen = self.time_penalty_max * ratio
            penalty += time_pen
            issues.append(f"Over budget: {elapsed:.0f}s / {budget:.0f}s (penalty={time_pen:.3f})")

        code_steps = [
            rec for rec in session.steps
            if rec.action_type in (ActionType.CODE, ActionType.MODEL)
            and rec.execution_success
        ]
        all_code = "\n".join(
            (rec.code_body if rec.code_body else rec.action_text) for rec in code_steps
        )

        has_gridsearch = bool(_GRIDSEARCH_PATTERNS.search(all_code))
        has_efficient = bool(_EFFICIENT_SEARCH.search(all_code))

        if has_gridsearch and not has_efficient:
            penalty += self.gridsearch_penalty
            issues.append(
                "GridSearchCV / exhaustive param search detected — "
                "consider RandomizedSearchCV or Optuna"
            )

        score = max(0.0, 1.0 - penalty)
        passed = len(issues) == 0
        details = "; ".join(issues) if issues else f"Efficient: {elapsed:.0f}s / {budget:.0f}s budget."

        return ValidationResult(
            validator_name=self.name,
            passed=passed,
            score=score,
            details=details,
            penalty=penalty,
        )
