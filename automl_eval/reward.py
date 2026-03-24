"""
RewardCalculator — assembles the final reward from validator results
and the model metric.

Formula:
  R = w_perf * r_perf
    + w_plan * r_plan
    + w_code * r_code
    - sum(penalties)

Where:
  r_perf  — normalized model metric (0..1),
  r_plan  — plan checklist coverage (0..1),
  r_code  — average of code validators (execution, correctness, intactness, leakage),
  penalties — penalties from validators for violations.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from automl_eval.validators.base import ValidationResult


@dataclass
class RewardWeights:
    performance: float = 0.5
    plan_coverage: float = 0.2
    code_quality: float = 0.3


@dataclass
class RewardBreakdown:
    """Detailed reward breakdown."""

    raw_performance: float = 0.0
    normalized_performance: float = 0.0
    plan_coverage_score: float = 0.0
    code_quality_score: float = 0.0
    total_penalty: float = 0.0
    final_reward: float = 0.0
    validator_details: dict[str, ValidationResult] = field(default_factory=dict)


class RewardCalculator:
    """Compute the total reward for a step or episode."""

    def __init__(self, weights: RewardWeights | None = None) -> None:
        self.weights = weights or RewardWeights()

    def compute(
        self,
        perf_score: float,
        validation_results: list[ValidationResult],
    ) -> RewardBreakdown:
        plan_score = 0.0
        code_scores: list[float] = []
        total_penalty = 0.0
        details: dict[str, ValidationResult] = {}

        for vr in validation_results:
            details[vr.validator_name] = vr
            total_penalty += vr.penalty

            if vr.validator_name == "plan_coverage":
                plan_score = vr.score
            else:
                code_scores.append(vr.score)

        code_quality = sum(code_scores) / len(code_scores) if code_scores else 1.0

        weighted = (
            self.weights.performance * perf_score
            + self.weights.plan_coverage * plan_score
            + self.weights.code_quality * code_quality
        )

        final = max(0.0, weighted - total_penalty)

        return RewardBreakdown(
            raw_performance=perf_score,
            normalized_performance=perf_score,
            plan_coverage_score=plan_score,
            code_quality_score=code_quality,
            total_penalty=total_penalty,
            final_reward=final,
            validator_details=details,
        )
