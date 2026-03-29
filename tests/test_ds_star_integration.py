from __future__ import annotations

import sys
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

agentenv_module = types.ModuleType("agentenv")
agentenv_envs_module = types.ModuleType("agentenv.envs")
agentenv_envs_module.AutoMLTask = object
agentenv_module.envs = agentenv_envs_module
sys.modules.setdefault("agentenv", agentenv_module)
sys.modules.setdefault("agentenv.envs", agentenv_envs_module)

ds_star_package = types.ModuleType("ds_star")
ds_star_module = types.ModuleType("ds_star.ds_star")
ds_star_module.DSStar = object
ds_star_package.ds_star = ds_star_module
sys.modules.setdefault("ds_star", ds_star_package)
sys.modules.setdefault("ds_star.ds_star", ds_star_module)

utility_package = types.ModuleType("utility")
utility_chat_model_module = types.ModuleType("utility.chat_model")
utility_chat_model_module.GatewayChatModel = object
utility_package.chat_model = utility_chat_model_module
sys.modules.setdefault("utility", utility_package)
sys.modules.setdefault("utility.chat_model", utility_chat_model_module)

from agentgym_integration.ds_star_eval import (  # noqa: E402
    _build_episode_data_files,
    _build_query,
    _plan_text,
    _sync_state_output,
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def test_build_query_includes_stateful_env_contract() -> None:
    query = _build_query("Task ID: titanic_binary\nObservation body")

    assert "train_df" in query
    assert "valid_df" in query
    assert "Do NOT reload datasets" in query
    assert "Model steps must reuse those persisted artifacts" in query
    assert "predictions" in query


def test_build_episode_data_files_prefers_task_json_only() -> None:
    data_dir = PROJECT_ROOT / "automl_eval" / "tasks"
    observation = "Task ID: titanic_binary\nOther text"

    files = _build_episode_data_files(str(data_dir), observation)

    assert len(files) == 1
    assert files[0].endswith("titanic_binary.json")


def test_plan_text_contains_global_automl_strategy() -> None:
    plan = _plan_text(
        ["Create reusable encoded features.", "Train RandomForest with ROC AUC validation."],
        ["FEATURE_ENGINEERING", "MODEL"],
    )

    assert "missing values" in plan
    assert "cross-validation" in plan or "validation" in plan
    assert "LogisticRegression" in plan
    assert "RandomForest" in plan
    assert "Create reusable encoded features." in plan


def test_sync_state_output_overwrites_latest_output_only() -> None:
    state = {"outputs": ["old output", "older final"]}

    synced = _sync_state_output(state, "env feedback")

    assert synced["outputs"] == ["old output", "env feedback"]


def main() -> None:
    test_build_query_includes_stateful_env_contract()
    test_build_episode_data_files_prefers_task_json_only()
    test_plan_text_contains_global_automl_strategy()
    test_sync_state_output_overwrites_latest_output_only()
    print("DS-Star integration helper tests passed.")


if __name__ == "__main__":
    main()
