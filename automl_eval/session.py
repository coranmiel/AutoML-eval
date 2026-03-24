"""
RuntimeSession — state of a single episode.

Stores:
- loaded data (train / valid / test),
- snapshot of the original data (for intactness checking),
- history of agent actions,
- current pipeline (applied transforms and models),
- per-step metrics,
- code execution results (stdout/stderr).
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from automl_eval.data_insights import DataInsights, analyze_dataset
from automl_eval.task import Task, TaskType


class ActionType(str, Enum):
    PLAN = "PLAN"
    FEATURE_ENGINEERING = "FEATURE_ENGINEERING"
    MODEL = "MODEL"
    CODE = "CODE"
    CODE_FIX = "CODE_FIX"
    FINAL_SUBMIT = "FINAL_SUBMIT"


@dataclass
class StepRecord:
    """Record of a single agent step."""

    step_idx: int
    action_type: ActionType
    action_text: str
    state_before: str
    state_after: str
    reward: float
    execution_success: bool
    error_message: str | None = None
    metric_value: float | None = None
    code_body: str = ""
    timestamp: float = field(default_factory=time.time)


class RuntimeSession:
    """State container for a single AutoML environment episode."""

    def __init__(self, task: Task, seed: int = 42) -> None:
        self.task = task
        self.seed = seed

        self.train_df: pd.DataFrame | None = None
        self.valid_df: pd.DataFrame | None = None
        self.test_df: pd.DataFrame | None = None

        self._train_snapshot_hash: str | None = None
        self._valid_snapshot_hash: str | None = None

        self.steps: list[StepRecord] = []
        self.current_step: int = 0

        self.plan_text: str | None = None
        self.applied_transforms: list[dict[str, Any]] = []
        self.trained_models: list[dict[str, Any]] = []
        self.best_metric: float | None = None
        self.predictions: np.ndarray | None = None

        self.sandbox_namespace: dict[str, Any] = {}

        # DSEval-inspired self-repair tracking (Section 6.5, arXiv:2402.17168)
        self.repair_attempts: int = 0
        self.repair_successes: int = 0
        self.consecutive_failures: int = 0

        self.data_insights: DataInsights | None = None

        # Cycle tracking: EDA/FE → Model → (back to EDA/FE → Model → ...)
        self.metric_history: list[tuple[int, float]] = []  # (step_idx, metric)
        self.cycle_count: int = 0
        self._last_phase: str = "init"  # "init", "data", "model"

        self.done: bool = False
        self.start_time: float = 0.0

    def initialize(self) -> None:
        """Load data and prepare the session."""
        df = pd.read_csv(self.task.dataset_path)

        if self.task.feature_columns:
            feature_cols = self.task.feature_columns
        else:
            feature_cols = [c for c in df.columns if c != self.task.target_column]

        X = df[feature_cols]
        y = df[self.task.target_column]

        stratify = y if self.task.task_type != TaskType.REGRESSION else None
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y, test_size=0.3, random_state=self.seed, stratify=stratify,
        )
        X_valid, X_test, y_valid, y_test = train_test_split(
            X_temp, y_temp, test_size=0.5, random_state=self.seed,
            stratify=y_temp if stratify is not None else None,
        )

        self.train_df = pd.concat([X_train, y_train], axis=1).reset_index(drop=True)
        self.valid_df = pd.concat([X_valid, y_valid], axis=1).reset_index(drop=True)
        self.test_df = pd.concat([X_test, y_test], axis=1).reset_index(drop=True)

        self._train_snapshot_hash = self._hash_df(self.train_df)
        self._valid_snapshot_hash = self._hash_df(self.valid_df)

        self.sandbox_namespace = {
            "train_df": self.train_df.copy(),
            "valid_df": self.valid_df.copy(),
            "pd": pd,
            "np": np,
        }

        self.data_insights = analyze_dataset(self.train_df, self.task.target_column)

        self.start_time = time.time()

    def elapsed_seconds(self) -> float:
        return time.time() - self.start_time

    def is_over_budget(self) -> bool:
        return self.elapsed_seconds() > self.task.time_budget_seconds

    def is_over_steps(self) -> bool:
        return self.current_step >= self.task.max_steps

    def record_step(self, record: StepRecord) -> None:
        self.steps.append(record)
        self.current_step += 1

        if record.action_type == ActionType.CODE_FIX:
            self.repair_attempts += 1
            if record.execution_success:
                self.repair_successes += 1
                self.consecutive_failures = 0
            else:
                self.consecutive_failures += 1
        elif not record.execution_success:
            self.consecutive_failures += 1
        else:
            self.consecutive_failures = 0

        # Cycle detection: data→model transition counts as a cycle
        if record.action_type in (ActionType.FEATURE_ENGINEERING, ActionType.CODE):
            code_text = record.code_body or record.action_text
            import re
            has_fit = bool(re.search(r"\.fit\s*\(", code_text))
            if has_fit:
                if self._last_phase == "data":
                    self.cycle_count += 1
                self._last_phase = "model"
            else:
                self._last_phase = "data"
        elif record.action_type == ActionType.MODEL:
            if self._last_phase == "data":
                self.cycle_count += 1
            self._last_phase = "model"

        if record.metric_value is not None:
            self.metric_history.append((record.step_idx, record.metric_value))

    def check_data_intact(self) -> bool:
        """Check whether the original data in the sandbox has been modified."""
        sandbox_train = self.sandbox_namespace.get("train_df")
        sandbox_valid = self.sandbox_namespace.get("valid_df")

        if sandbox_train is None or sandbox_valid is None:
            return False

        train_ok = self._hash_df(sandbox_train) == self._train_snapshot_hash
        valid_ok = self._hash_df(sandbox_valid) == self._valid_snapshot_hash
        return train_ok and valid_ok

    def state_summary(self) -> str:
        """Text summary of the current state for the agent."""
        lines = [f"Step: {self.current_step} / {self.task.max_steps}"]
        lines.append(f"Time elapsed: {self.elapsed_seconds():.1f}s / {self.task.time_budget_seconds}s")

        if self.plan_text:
            lines.append(f"Plan: submitted")
        else:
            lines.append("Plan: not yet submitted")

        lines.append(f"Transforms applied: {len(self.applied_transforms)}")
        lines.append(f"Models trained: {len(self.trained_models)}")

        if self.best_metric is not None:
            lines.append(f"Best {self.task.metric.value}: {self.best_metric:.4f}")
        else:
            lines.append(f"Best {self.task.metric.value}: N/A")

        if self.steps and not self.steps[-1].execution_success:
            lines.append(f"Last error: {self.steps[-1].error_message}")

        if self.repair_attempts > 0:
            lines.append(
                f"Self-repair: {self.repair_successes}/{self.repair_attempts} fixes succeeded"
            )

        return "\n".join(lines)

    @staticmethod
    def _hash_df(df: pd.DataFrame) -> str:
        return hashlib.sha256(pd.util.hash_pandas_object(df).values.tobytes()).hexdigest()
