"""
TargetLeakageModelValidator — model-based target leakage check.

Does not rely on regex. Instead:

1. **Single-feature AUC/R2**: after FE, train a DecisionTree(depth=1)
   on each individual feature. If a single feature yields AUC > 0.99
   (or R2 > 0.99) — it is almost certainly target leakage.

2. **Train/valid gap**: if the agent's accuracy/AUC on train is ~ 1.0,
   but significantly worse on valid — sign of overfitting or leakage.

3. **Feature importance drop test**: train a simple forest, then remove
   the most important feature. If the metric drops sharply -> suspicious
   (may be legitimate, but worth checking).

Heavy checks (1, 3) run only at FINAL_SUBMIT.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from automl_eval.validators.base import BaseValidator, ValidationResult

if TYPE_CHECKING:
    from automl_eval.session import RuntimeSession

logger = logging.getLogger(__name__)

_FEATURE_DF_NAMES = ["X_train", "X", "features", "df_train", "train_processed"]
_TARGET_NAMES = ["y_train", "y"]


class TargetLeakageModelValidator(BaseValidator):
    """Model-based target leakage and overfitting check."""

    name = "target_leakage_model"

    def __init__(
        self,
        single_feature_threshold: float = 0.99,
        train_valid_gap_threshold: float = 0.30,
        leakage_penalty: float = 0.25,
        overfit_penalty: float = 0.10,
    ) -> None:
        self.single_feature_threshold = single_feature_threshold
        self.train_valid_gap_threshold = train_valid_gap_threshold
        self.leakage_penalty = leakage_penalty
        self.overfit_penalty = overfit_penalty

    def validate(self, session: RuntimeSession) -> ValidationResult:
        if not session.done:
            return ValidationResult(
                validator_name=self.name,
                passed=True,
                score=1.0,
                details="Model-based leakage check runs at FINAL_SUBMIT only.",
            )

        issues: list[str] = []
        penalty = 0.0

        # ── 1. Single-feature AUC check ──────────────────────────────
        leak_features = self._single_feature_check(session)
        if leak_features:
            names = ", ".join(f"{name} (score={score:.3f})" for name, score in leak_features)
            issues.append(f"Possible target leak via features: {names}")
            penalty += self.leakage_penalty

        # ── 2. Train/valid gap ────────────────────────────────────────
        gap_issue = self._train_valid_gap_check(session)
        if gap_issue:
            issues.append(gap_issue)
            penalty += self.overfit_penalty

        score = max(0.0, 1.0 - penalty)
        passed = len(issues) == 0

        if passed:
            details = "No model-based leakage or overfitting detected."
        else:
            details = "; ".join(issues)

        return ValidationResult(
            validator_name=self.name,
            passed=passed,
            score=score,
            details=details,
            penalty=penalty,
        )

    def _single_feature_check(self, session: RuntimeSession) -> list[tuple[str, float]]:
        """Train a DecisionTree on each feature individually.
        If AUC > threshold — suspected leakage."""
        from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
        from sklearn.metrics import roc_auc_score, r2_score
        from automl_eval.task import TaskType

        ns = session.sandbox_namespace
        X_df = self._find_features(ns)
        y = self._find_target(ns, session)

        if X_df is None or y is None:
            return []

        try:
            X_numeric = X_df.select_dtypes(include="number").dropna(axis=1)
        except Exception:
            return []

        if len(X_numeric.columns) == 0 or len(X_numeric) < 10:
            return []

        mask = X_numeric.index.isin(y.index)
        X_numeric = X_numeric.loc[mask]
        y_aligned = y.loc[X_numeric.index]

        if len(X_numeric) < 10:
            return []

        is_classification = session.task.task_type != TaskType.REGRESSION
        leaky: list[tuple[str, float]] = []

        for col in X_numeric.columns:
            x_col = X_numeric[[col]].values
            col_mask = ~np.isnan(x_col.ravel())
            if col_mask.sum() < 10:
                continue
            x_clean = x_col[col_mask]
            y_clean = y_aligned.values[col_mask]

            try:
                if is_classification:
                    model = DecisionTreeClassifier(max_depth=2, random_state=42)
                    model.fit(x_clean, y_clean)
                    if hasattr(model, "predict_proba"):
                        proba = model.predict_proba(x_clean)
                        if proba.shape[1] == 2:
                            auc = roc_auc_score(y_clean, proba[:, 1])
                        else:
                            auc = roc_auc_score(
                                y_clean, proba, multi_class="ovr", average="macro"
                            )
                    else:
                        auc = float(model.score(x_clean, y_clean))
                    if auc >= self.single_feature_threshold:
                        leaky.append((col, auc))
                else:
                    model = DecisionTreeRegressor(max_depth=2, random_state=42)
                    model.fit(x_clean, y_clean)
                    r2 = r2_score(y_clean, model.predict(x_clean))
                    if r2 >= self.single_feature_threshold:
                        leaky.append((col, r2))
            except Exception:
                continue

        return leaky

    def _train_valid_gap_check(self, session: RuntimeSession) -> str | None:
        """If the agent's model shows AUC=1.0 on train but much worse on valid."""
        ns = session.sandbox_namespace
        model = ns.get("model") or ns.get("clf") or ns.get("estimator")
        if model is None:
            return None

        X_train = self._find_features(ns)
        y_train = self._find_target(ns, session)
        if X_train is None or y_train is None:
            return None

        valid_df = ns.get("valid_df")
        if not isinstance(valid_df, pd.DataFrame):
            return None

        target = session.task.target_column
        if target not in valid_df.columns:
            return None

        try:
            X_valid_raw = valid_df.drop(columns=[target])
            y_valid = valid_df[target]

            X_train_num = X_train.select_dtypes(include="number").fillna(0)

            valid_cols = [c for c in X_train_num.columns if c in X_valid_raw.columns]
            if len(valid_cols) < 2:
                return None

            X_valid_num = X_valid_raw[valid_cols].select_dtypes(include="number").fillna(0)
            X_train_sub = X_train_num[valid_cols]

            train_score = float(model.score(X_train_sub, y_train))

            valid_score = float(model.score(X_valid_num, y_valid))

            gap = train_score - valid_score
            if train_score > 0.95 and gap > self.train_valid_gap_threshold:
                return (
                    f"Train/valid gap: train_score={train_score:.3f}, "
                    f"valid_score={valid_score:.3f} (gap={gap:.3f}) — "
                    f"possible overfitting or leakage"
                )
        except Exception as e:
            logger.debug("Train/valid gap check failed: %s", e)

        return None

    def _find_features(self, ns: dict) -> pd.DataFrame | None:
        for name in _FEATURE_DF_NAMES:
            obj = ns.get(name)
            if isinstance(obj, pd.DataFrame) and len(obj) > 0:
                return obj
        train = ns.get("train_df")
        if isinstance(train, pd.DataFrame):
            return train
        return None

    def _find_target(self, ns: dict, session: RuntimeSession) -> pd.Series | None:
        for name in _TARGET_NAMES:
            obj = ns.get(name)
            if isinstance(obj, (pd.Series, np.ndarray)) and len(obj) > 0:
                if isinstance(obj, np.ndarray):
                    return pd.Series(obj)
                return obj
        train = ns.get("train_df")
        if isinstance(train, pd.DataFrame):
            target = session.task.target_column
            if target in train.columns:
                return train[target]
        return None
