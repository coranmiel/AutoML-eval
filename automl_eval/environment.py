"""
AutoMLEnvironment — RL environment for evaluating AutoML agents.

Implements the AgentGym protocol:
  reset(task_id)  -> None
  observe()       -> str          (text state for the agent)
  step(content)   -> (str, float, bool)  (state, reward, done)
  close()         -> None

One episode = one task from the TaskRegistry.
Within an episode the agent takes multiple steps (multi-turn).
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from automl_eval.action_parser import ActionParser, ParsedAction
from automl_eval.metrics import compute_metric, normalize_score
from automl_eval.reward import RewardBreakdown, RewardCalculator, RewardWeights
from automl_eval.sandbox import Sandbox, ExecutionResult
from automl_eval.session import ActionType, RuntimeSession, StepRecord
from automl_eval.task import Task
from automl_eval.task_registry import TaskRegistry
from automl_eval.validators.base import BaseValidator, ValidationResult
from automl_eval.validators.correctness import CorrectnessValidator
from automl_eval.validators.execution import ExecutionValidator
from automl_eval.validators.intactness import IntactnessValidator
from automl_eval.validators.leakage import LeakageValidator
from automl_eval.validators.model_eval import ModelEvalValidator
from automl_eval.validators.namespace_check import NamespaceCheckValidator
from automl_eval.validators.plan_coverage import PlanCoverageValidator
from automl_eval.validators.backtracking import BacktrackingValidator
from automl_eval.validators.reproducibility import ReproducibilityValidator
from automl_eval.validators.efficiency import EfficiencyValidator
from automl_eval.validators.correlation import CorrelationValidator
from automl_eval.validators.missing_values import MissingValuesValidator
from automl_eval.validators.distribution import DistributionValidator
from automl_eval.validators.feature_pipeline import FeaturePipelineValidator
from automl_eval.validators.duplicate import DuplicateValidator
from automl_eval.validators.target_leakage_model import TargetLeakageModelValidator
from automl_eval.validators.feature_importance import FeatureImportanceValidator
from automl_eval.validators.hyperparam import HyperparamValidator
from automl_eval.validators.model_choice import ModelChoiceValidator
from automl_eval.validators.split_quality import SplitValidator
from automl_eval.validators.iterative_cycle import IterativeCycleValidator
from automl_eval.validators.baseline_comparison import BaselineComparisonValidator

logger = logging.getLogger(__name__)


class StepOutput:
    """Result of a single step() call — analogous to AgentGym StepOutput."""

    def __init__(self, state: str, reward: float, done: bool) -> None:
        self.state = state
        self.reward = reward
        self.done = done


class AutoMLEnvironment:
    """
    Main RL environment.

    Usage:
        env = AutoMLEnvironment(registry)
        env.reset("titanic_binary")
        obs = env.observe()
        # loop:
        output = env.step(agent_response_text)
        # output.state, output.reward, output.done
    """

    def __init__(
        self,
        registry: TaskRegistry,
        reward_weights: RewardWeights | None = None,
        sandbox_timeout: int = 60,
        seed: int = 42,
    ) -> None:
        self.registry = registry
        self.reward_calc = RewardCalculator(reward_weights)
        self.sandbox = Sandbox(timeout_seconds=sandbox_timeout)
        self.parser = ActionParser()
        self.seed = seed

        self.validators: list[BaseValidator] = [
            ExecutionValidator(),
            CorrectnessValidator(),
            IntactnessValidator(),
            LeakageValidator(),
            PlanCoverageValidator(),
            NamespaceCheckValidator(),
            ModelEvalValidator(),
            BacktrackingValidator(),
            ReproducibilityValidator(),
            EfficiencyValidator(),
            CorrelationValidator(),
            MissingValuesValidator(),
            DistributionValidator(),
            FeaturePipelineValidator(),
            DuplicateValidator(),
            TargetLeakageModelValidator(),
            FeatureImportanceValidator(),
            HyperparamValidator(),
            ModelChoiceValidator(),
            SplitValidator(),
            IterativeCycleValidator(),
            BaselineComparisonValidator(),
        ]

        self._session: RuntimeSession | None = None
        self._task: Task | None = None

    # ---------- AgentGym-compatible interface ----------

    def reset(self, task_id: str) -> None:
        """Start a new episode for the given task_id."""
        self._task = self.registry.get(task_id)
        self._session = RuntimeSession(self._task, seed=self.seed)
        self._session.initialize()
        logger.info("Environment reset for task '%s'", task_id)

    def observe(self) -> str:
        """Return a text description of the task and the current state."""
        self._check_active()
        task_text = self._task.observation_text()  # type: ignore[union-attr]

        session = self._session  # type: ignore[assignment]
        train_info = self._describe_dataframe(session.train_df, "train")
        valid_info = self._describe_dataframe(session.valid_df, "valid")

        parts = [
            task_text,
            "",
            "=== Data Overview ===",
            train_info,
            valid_info,
            "",
            "=== Session State ===",
            session.state_summary(),
            "",
            "Available actions: PLAN, FEATURE_ENGINEERING, MODEL, CODE, CODE_FIX, FINAL_SUBMIT",
            "Format your response as:",
            "ACTION: <type>",
            "<body>",
        ]
        return "\n".join(parts)

    def step(self, content: str) -> StepOutput:
        """Execute a single agent action.

        Order of operations (important for validator correctness):
          1. Execute the action (sandbox, best_metric on FINAL_SUBMIT).
          2. Determine done flag — validators like BaselineComparison
             check session.done to decide whether to run.
          3. Record the step — updates cycle_count, metric_history,
             current_step so validators see the freshest state.
          4. Run validators and compute reward.
          5. Back-patch reward into the already-recorded StepRecord.
        """
        self._check_active()
        session = self._session  # type: ignore[assignment]

        state_before = session.state_summary()
        parsed = self.parser.parse(content)

        exec_result = self._execute_action(parsed, session)

        done = self._check_done(session, parsed)
        session.done = done

        step_record = StepRecord(
            step_idx=session.current_step,
            action_type=parsed.action_type,
            action_text=parsed.raw_text,
            state_before=state_before,
            state_after=session.state_summary(),
            reward=0.0,
            execution_success=exec_result.success,
            error_message=exec_result.error,
            metric_value=session.best_metric,
            code_body=parsed.body,
        )
        session.record_step(step_record)

        validation_results = [v.validate(session) for v in self.validators]
        perf_score = self._current_performance(session)
        breakdown = self.reward_calc.compute(perf_score, validation_results)

        step_record.reward = breakdown.final_reward

        state_text = self._format_step_response(session, exec_result, breakdown, done)

        return StepOutput(state=state_text, reward=breakdown.final_reward, done=done)

    def close(self) -> None:
        """End the current episode."""
        self._session = None
        self._task = None

    # ---------- Internal logic ----------

    def _execute_action(
        self,
        parsed: ParsedAction,
        session: RuntimeSession,
    ) -> ExecutionResult:
        """Execute the action depending on its type."""

        if parsed.action_type == ActionType.PLAN:
            session.plan_text = parsed.body
            return ExecutionResult(success=True, stdout="Plan recorded.", stderr="")

        if parsed.action_type in (
            ActionType.CODE,
            ActionType.CODE_FIX,
            ActionType.FEATURE_ENGINEERING,
            ActionType.MODEL,
        ):
            result = self.sandbox.execute(parsed.body, session.sandbox_namespace)

            if result.success:
                self._sync_session_from_sandbox(session)

            if parsed.action_type == ActionType.FEATURE_ENGINEERING and result.success:
                session.applied_transforms.append({"code": parsed.body})
            elif parsed.action_type == ActionType.MODEL and result.success:
                session.trained_models.append({"code": parsed.body})

            return result

        if parsed.action_type == ActionType.FINAL_SUBMIT:
            return self._handle_final_submit(parsed, session)

        return ExecutionResult(
            success=False, stdout="", stderr="",
            error=f"Unknown action type: {parsed.action_type}",
        )

    def _handle_final_submit(
        self,
        parsed: ParsedAction,
        session: RuntimeSession,
    ) -> ExecutionResult:
        """Handle the final submission: extract predictions from the sandbox."""
        preds = session.sandbox_namespace.get("predictions")
        if preds is None:
            preds = session.sandbox_namespace.get("y_pred")

        if preds is None:
            return ExecutionResult(
                success=False, stdout="", stderr="",
                error="FINAL_SUBMIT: no 'predictions' or 'y_pred' found in sandbox. "
                      "Set predictions = model.predict(X_test) before submitting.",
            )

        session.predictions = np.asarray(preds)

        try:
            y_true = session.test_df[session.task.target_column].values  # type: ignore[index]
            score = compute_metric(session.task.metric, y_true, session.predictions)
            session.best_metric = score
        except Exception as exc:
            return ExecutionResult(
                success=False, stdout="", stderr="",
                error=f"Could not compute metric: {exc}",
            )

        return ExecutionResult(
            success=True,
            stdout=f"Final metric ({session.task.metric.value}): {score:.4f}",
            stderr="",
        )

    def _sync_session_from_sandbox(self, session: RuntimeSession) -> None:
        """Pull updates from the sandbox namespace into the session."""
        ns = session.sandbox_namespace

        if "best_metric" in ns:
            session.best_metric = float(ns["best_metric"])

        if "predictions" in ns:
            session.predictions = np.asarray(ns["predictions"])
        elif "y_pred" in ns:
            session.predictions = np.asarray(ns["y_pred"])

    def _current_performance(self, session: RuntimeSession) -> float:
        """Current normalized model metric."""
        if session.best_metric is None:
            return 0.0
        return normalize_score(
            session.best_metric,
            session.task.baseline_score,
            session.task.oracle_score,
        )

    def _check_done(self, session: RuntimeSession, parsed: ParsedAction) -> bool:
        if parsed.action_type == ActionType.FINAL_SUBMIT:
            return True
        if session.is_over_steps():
            return True
        if session.is_over_budget():
            return True
        return False

    def _format_step_response(
        self,
        session: RuntimeSession,
        exec_result: ExecutionResult,
        breakdown: RewardBreakdown,
        done: bool,
    ) -> str:
        """Format the environment's text response for the agent."""
        lines: list[str] = []

        if exec_result.success:
            lines.append("Execution: OK")
            if exec_result.stdout.strip():
                lines.append(f"Output: {exec_result.stdout.strip()}")
        else:
            lines.append(f"Execution: FAILED — {exec_result.error}")

        lines.append("")
        lines.append(session.state_summary())

        lines.append("")
        lines.append("--- Validator feedback ---")
        for name, vr in breakdown.validator_details.items():
            status = "PASS" if vr.passed else "FAIL"
            lines.append(f"  [{status}] {name}: {vr.details}")

        if done:
            lines.append("")
            lines.append(f"=== Episode finished. Final reward: {breakdown.final_reward:.4f} ===")

        return "\n".join(lines)

    def _check_active(self) -> None:
        if self._session is None:
            raise RuntimeError("No active session. Call reset(task_id) first.")

    @staticmethod
    def _describe_dataframe(df: Any, name: str) -> str:
        if df is None:
            return f"{name}: not loaded"
        lines = [
            f"{name}: {df.shape[0]} rows, {df.shape[1]} columns",
            f"  Columns: {', '.join(df.columns[:20])}{'...' if len(df.columns) > 20 else ''}",
            f"  Dtypes: {dict(df.dtypes.value_counts())}",
            f"  Missing: {int(df.isnull().sum().sum())} total nulls",
        ]
        return "\n".join(lines)
