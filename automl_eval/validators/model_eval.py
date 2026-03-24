"""
ModelEvalValidator — evaluates a trained model from the sandbox on held-out
validation data.

Inspired by DSEval's `model` validator (Table 6, arXiv:2402.17168):
"Fail if the defined model does not satisfy the criteria."

Unlike our CorrectnessValidator (which checks prediction arrays), this
validator finds the model object in the sandbox, calls .predict() on
valid data, and computes the metric — catching issues like:
  - model was trained on wrong data,
  - model overfits (train metric >> valid metric),
  - model produces wrong output shape.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from automl_eval.metrics import compute_metric
from automl_eval.validators.base import BaseValidator, ValidationResult

if TYPE_CHECKING:
    from automl_eval.session import RuntimeSession


_MODEL_NAMES = ["model", "clf", "classifier", "regressor", "estimator", "pipeline"]


def _find_model(ns: dict[str, Any]) -> Any | None:
    for name in _MODEL_NAMES:
        obj = ns.get(name)
        if obj is not None and hasattr(obj, "predict"):
            return obj
    return None


class ModelEvalValidator(BaseValidator):
    """
    If a model exists in sandbox, evaluate it on valid_df and check
    that performance is above a minimum threshold.
    """

    name = "model_eval"

    def __init__(self, min_score_above_baseline: float = 0.0) -> None:
        self.min_score_above_baseline = min_score_above_baseline

    def validate(self, session: RuntimeSession) -> ValidationResult:
        model = _find_model(session.sandbox_namespace)
        if model is None:
            return ValidationResult(
                validator_name=self.name,
                passed=True,
                score=0.5,
                details="No model found in sandbox (neutral).",
            )

        ns = session.sandbox_namespace
        valid_df = session.valid_df
        if valid_df is None:
            return ValidationResult(
                validator_name=self.name,
                passed=True,
                score=0.5,
                details="No validation data available.",
            )

        target = session.task.target_column
        feature_cols = [c for c in valid_df.columns if c != target]

        X_val = ns.get("X_valid")
        if X_val is None:
            try:
                X_val = valid_df[feature_cols]
            except Exception:
                return ValidationResult(
                    validator_name=self.name,
                    passed=False,
                    score=0.0,
                    details="Cannot construct X_valid from validation data.",
                    penalty=0.1,
                )

        y_val = valid_df[target].values

        try:
            if hasattr(model, "predict_proba"):
                preds = model.predict_proba(X_val)
                if preds.ndim == 2 and preds.shape[1] == 2:
                    preds = preds[:, 1]
            else:
                preds = model.predict(X_val)
        except Exception as exc:
            return ValidationResult(
                validator_name=self.name,
                passed=False,
                score=0.0,
                details=f"Model.predict failed on valid data: {exc}",
                penalty=0.15,
            )

        try:
            metric_val = compute_metric(session.task.metric, y_val, np.asarray(preds))
        except Exception as exc:
            return ValidationResult(
                validator_name=self.name,
                passed=False,
                score=0.0,
                details=f"Metric computation failed: {exc}",
                penalty=0.1,
            )

        baseline = session.task.baseline_score or 0.0
        above_baseline = metric_val - baseline

        if above_baseline < self.min_score_above_baseline:
            return ValidationResult(
                validator_name=self.name,
                passed=False,
                score=max(0.0, 0.5 + above_baseline),
                details=(
                    f"Model {session.task.metric.value}={metric_val:.4f} "
                    f"(baseline={baseline:.4f}, delta={above_baseline:+.4f}) — "
                    f"below minimum threshold."
                ),
                penalty=0.05,
            )

        score = min(1.0, 0.5 + above_baseline)
        return ValidationResult(
            validator_name=self.name,
            passed=True,
            score=score,
            details=(
                f"Model {session.task.metric.value}={metric_val:.4f} "
                f"(baseline={baseline:.4f}, delta={above_baseline:+.4f})."
            ),
        )
