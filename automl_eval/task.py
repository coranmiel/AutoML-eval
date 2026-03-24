"""
Task — description of a single task for the AutoML environment.

Each task contains:
- a dataset (path or loader),
- a text description for the agent,
- task type (classification / regression),
- target variable and metric,
- a checklist for plan evaluation,
- (optionally) a reference pipeline for comparison.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any


class TaskType(str, Enum):
    BINARY_CLASSIFICATION = "binary_classification"
    MULTICLASS_CLASSIFICATION = "multiclass_classification"
    REGRESSION = "regression"


class MetricName(str, Enum):
    ROC_AUC = "roc_auc"
    ACCURACY = "accuracy"
    F1 = "f1"
    LOG_LOSS = "log_loss"
    RMSE = "rmse"
    MAE = "mae"
    R2 = "r2"


@dataclass
class PlanChecklistItem:
    """A single checklist item for evaluating the agent's plan."""

    id: str
    description: str
    keywords: list[str] = field(default_factory=list)
    weight: float = 1.0
    required: bool = False

    def check(self, plan_text: str) -> bool:
        """Check whether this item is covered by the plan text (keyword match)."""
        text_lower = plan_text.lower()
        return any(kw.lower() in text_lower for kw in self.keywords)


@dataclass
class Task:
    """Full description of a single task for the RL environment."""

    task_id: str
    dataset_path: str
    target_column: str
    task_type: TaskType
    metric: MetricName
    description: str

    plan_checklist: list[PlanChecklistItem] = field(default_factory=list)

    # Columns the agent should work with (None means all)
    feature_columns: list[str] | None = None
    # Constraints
    time_budget_seconds: float = 300.0
    max_steps: int = 15
    # Optional: oracle metric and baseline
    oracle_score: float | None = None
    baseline_score: float | None = None
    # Additional metadata (column descriptions, hints, etc.)
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_json(cls, path: str | Path) -> Task:
        """Load a task from a JSON file."""
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        checklist = [
            PlanChecklistItem(**item) for item in data.pop("plan_checklist", [])
        ]
        data["task_type"] = TaskType(data["task_type"])
        data["metric"] = MetricName(data["metric"])
        return cls(**data, plan_checklist=checklist)

    def to_json(self, path: str | Path) -> None:
        """Save the task to a JSON file."""
        data = {
            "task_id": self.task_id,
            "dataset_path": self.dataset_path,
            "target_column": self.target_column,
            "task_type": self.task_type.value,
            "metric": self.metric.value,
            "description": self.description,
            "plan_checklist": [
                {
                    "id": item.id,
                    "description": item.description,
                    "keywords": item.keywords,
                    "weight": item.weight,
                    "required": item.required,
                }
                for item in self.plan_checklist
            ],
            "feature_columns": self.feature_columns,
            "time_budget_seconds": self.time_budget_seconds,
            "max_steps": self.max_steps,
            "oracle_score": self.oracle_score,
            "baseline_score": self.baseline_score,
            "metadata": self.metadata,
        }
        Path(path).write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")

    def observation_text(self) -> str:
        """Text description of the task visible to the agent during observe()."""
        lines = [
            f"Task ID: {self.task_id}",
            f"Task type: {self.task_type.value}",
            f"Target column: {self.target_column}",
            f"Metric: {self.metric.value}",
            f"Time budget: {self.time_budget_seconds}s",
            f"Max steps: {self.max_steps}",
            "",
            f"Description: {self.description}",
        ]
        if self.metadata.get("column_descriptions"):
            lines.append("")
            lines.append("Column descriptions:")
            for col, desc in self.metadata["column_descriptions"].items():
                lines.append(f"  - {col}: {desc}")
        return "\n".join(lines)
