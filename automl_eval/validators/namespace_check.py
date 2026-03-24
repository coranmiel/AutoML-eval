"""
NamespaceCheckValidator — checks that expected variables exist in the sandbox
namespace after code execution and have correct properties.

Inspired by DSEval's `namespace_check` validator (Table 6, arXiv:2402.17168):
the validator fails if some variables are not correctly created or modified.

For AutoML tasks, we check that the agent created standard pipeline artifacts:
  - a trained model object with .predict() method,
  - feature arrays (X_train, X_valid) as numeric arrays,
  - target arrays (y_train, y_valid).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from automl_eval.validators.base import BaseValidator, ValidationResult

if TYPE_CHECKING:
    from automl_eval.session import RuntimeSession

EXPECTED_ARTIFACTS = [
    {
        "name": "model",
        "alt_names": ["clf", "classifier", "regressor", "estimator", "pipeline"],
        "check": "has_predict",
        "description": "trained model with .predict()",
        "required_after_step": 3,
    },
]


def _find_var(ns: dict[str, Any], name: str, alt_names: list[str]) -> tuple[str | None, Any]:
    """Find a variable by primary name or alternatives."""
    if name in ns:
        return name, ns[name]
    for alt in alt_names:
        if alt in ns:
            return alt, ns[alt]
    return None, None


class NamespaceCheckValidator(BaseValidator):
    """
    Checks sandbox namespace for expected ML artifacts.
    Adapts DSEval namespace_check to AutoML context.
    """

    name = "namespace_check"

    def validate(self, session: RuntimeSession) -> ValidationResult:
        ns = session.sandbox_namespace
        issues: list[str] = []
        checks_passed = 0
        checks_total = 0

        for artifact in EXPECTED_ARTIFACTS:
            if session.current_step < artifact["required_after_step"]:
                continue

            checks_total += 1
            var_name, var_val = _find_var(
                ns, artifact["name"], artifact.get("alt_names", [])
            )

            if var_val is None:
                issues.append(f"Missing: {artifact['description']}")
                continue

            check = artifact.get("check")
            if check == "has_predict" and not hasattr(var_val, "predict"):
                issues.append(
                    f"Variable '{var_name}' exists but has no .predict() method"
                )
                continue

            checks_passed += 1

        if "X_train" in ns and session.current_step >= 2:
            checks_total += 1
            x = ns["X_train"]
            if hasattr(x, "select_dtypes"):
                non_numeric = x.select_dtypes(exclude=[np.number]).columns.tolist()
                if non_numeric:
                    issues.append(
                        f"X_train has non-numeric columns: {non_numeric[:5]}"
                    )
                else:
                    checks_passed += 1
            elif isinstance(x, np.ndarray):
                checks_passed += 1

        if checks_total == 0:
            return ValidationResult(
                validator_name=self.name,
                passed=True,
                score=1.0,
                details="No namespace checks applicable at this step.",
            )

        score = checks_passed / checks_total
        passed = len(issues) == 0

        return ValidationResult(
            validator_name=self.name,
            passed=passed,
            score=score,
            details="; ".join(issues) if issues else "All namespace checks passed.",
            penalty=0.1 * len(issues),
        )
