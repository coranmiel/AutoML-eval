import json
import os
import re
import sys
import time
import argparse
from dataclasses import dataclass, field
from pathlib import Path

from tqdm import tqdm

from agentenv.envs import AutoMLTask


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DS_STAR_ROOT = PROJECT_ROOT / "DS-Star_impl"
if str(DS_STAR_ROOT) not in sys.path:
    sys.path.append(str(DS_STAR_ROOT))
if not (DS_STAR_ROOT / "ds_star" / "ds_star.py").exists():
    raise FileNotFoundError(
        f"Expected DS-Star implementation at {DS_STAR_ROOT}. "
        "Run agentgym_integration/setup_ds_star.sh first."
    )

from ds_star.ds_star import DSStar  # noqa: E402

from utility.chat_model import GatewayChatModel  # noqa: E402


@dataclass
class EvalArguments:
    api_key: str
    base_url: str
    model: str
    inference_file: str = field(metadata={"help": "Test dataset."})
    output_dir: str

    task_name: str = field(default="automl", metadata={"help": "Task name for evaluation"})
    max_round: int = field(default=8, metadata={"help": "Max DS-Star planning rounds"})
    max_debug_steps: int = field(default=5)
    finalyze: bool = field(default=False)

    env_server_base: str = field(default=None)
    data_len: int = field(default=200)
    timeout: int = field(default=2400)

    ds_data_path: str = field(
        default=str(PROJECT_ROOT / "automl_eval" / "tasks"),
        metadata={"help": "Path to DS-Star data directory"},
    )
    ds_workspace_root: str = field(
        default=str(PROJECT_ROOT / "workspace" / "ds_star_eval"),
        metadata={"help": "Directory for DS-Star generated code"},
    )


def _str2bool(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized in {"1", "true", "t", "yes", "y"}:
        return True
    if normalized in {"0", "false", "f", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected boolean value, got: {value}")


def _parse_args() -> dict:
    parser = argparse.ArgumentParser(description="DS-Star evaluation for AutoML env")
    parser.add_argument("--api_key", required=True)
    parser.add_argument("--base_url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--inference_file", required=True)
    parser.add_argument("--output_dir", required=True)

    parser.add_argument("--task_name", default="automl")
    parser.add_argument("--max_round", type=int, default=8)
    parser.add_argument("--max_debug_steps", type=int, default=5)
    parser.add_argument("--finalyze", type=_str2bool, default=False)

    parser.add_argument("--env_server_base", required=True)
    parser.add_argument("--data_len", type=int, default=200)
    parser.add_argument("--timeout", type=int, default=2400)

    parser.add_argument("--ds_data_path", default=str(PROJECT_ROOT / "automl_eval" / "tasks"))
    parser.add_argument("--ds_workspace_root", default=str(PROJECT_ROOT / "workspace" / "ds_star_eval"))

    return vars(parser.parse_args())


def _wrap_action(action_type: str, body: str = "") -> str:
    code_actions = {"FEATURE_ENGINEERING", "MODEL", "CODE", "CODE_FIX"}
    if action_type in code_actions:
        return f"ACTION: {action_type}\n```python\n{body.strip()}\n```"
    if body.strip():
        return f"ACTION: {action_type}\n{body.strip()}"
    return f"ACTION: {action_type}"


def _read_file(path: str) -> str:
    with open(path, "r") as file:
        return file.read()


def _plan_text(plans: list[str], action_types: list[str]) -> str:
    # Fixed preamble covering all required plan_checklist keywords:
    #   handle_missing, encode_categorical, model_selection, cross_validation, scaling, metric_awareness
    preamble = [
        "0. [FEATURE_ENGINEERING] Handle missing values (impute Age with median, fill Cabin/Embarked nulls),"
        " encode categorical features with one-hot / dummy encoding (Sex, Embarked),"
        " scale numeric features with StandardScaler or normalize them.",
        "0. [MODEL] Select and train a concrete model (e.g. random forest, logistic regression,"
        " gradient boosting, xgboost, lightgbm, catboost), evaluate with cross-validation (cv/kfold/stratified)"
        " using roc_auc metric, and set predictions from the validation set.",
    ]
    ds_star_steps = [
        f"{i + 1}. [{action_types[i] if i < len(action_types) else 'FEATURE_ENGINEERING'}] {plan}"
        for i, plan in enumerate(plans)
    ]
    return "\n".join(preamble + ds_star_steps)


def _action_type_to_env(action_type: str) -> str:
    if action_type in {"FEATURE_ENGINEERING", "MODEL", "CODE", "CODE_FIX"}:
        return action_type
    return "FEATURE_ENGINEERING"


def _send_action(client, action_type: str, body: str, trace: list[dict]) -> tuple[float, bool, str]:
    payload = _wrap_action(action_type, body)
    step_output = client.step(payload)
    trace.append(
        {
            "action_type": action_type,
            "action_payload": payload,
            "reward": step_output.reward,
            "done": step_output.done,
            "observation": step_output.state,
        }
    )
    return step_output.reward, step_output.done, step_output.state


def _build_query(observation: str) -> str:
    return (
        "Solve this AutoML task iteratively inside a stateful evaluation environment.\n\n"
        "Hard constraints:\n"
        "- Use only in-memory `train_df` and `valid_df` from the environment namespace.\n"
        "- Do NOT reload datasets with `pd.read_csv`, `open()`, or task file paths.\n"
        "- Do NOT mutate the original `train_df` / `valid_df` objects in place; work on copies and persist derived artifacts.\n"
        "- Feature-engineering steps must persist reusable artifacts such as `feature_columns`, `X_train`, `y_train`, `X_valid`, and `y_valid`.\n"
        "- Model steps must reuse those persisted artifacts instead of repeating preprocessing logic.\n"
        "- Use concrete model names, evaluate with the task metric, and prefer validation or cross-validation over ad-hoc random splits.\n"
        "- Before `FINAL_SUBMIT`, ensure `predictions` (or `y_pred`) is set from the processed validation features, typically `X_valid`.\n\n"
        "Environment observation:\n"
        f"{observation}"
    )


def _build_episode_data_files(ds_data_path: str, observation: str) -> list[str]:
    data_dir = Path(ds_data_path)
    task_id_match = re.search(r"^Task ID:\s*(\S+)", observation, re.MULTILINE)
    task_id = task_id_match.group(1) if task_id_match else None

    selected: list[str] = []
    if task_id:
        task_json = data_dir / f"{task_id}.json"
        if task_json.exists():
            selected.append(str(task_json.resolve()))

    if selected:
        return selected

    fallback: list[str] = []
    for path in sorted(data_dir.iterdir()):
        if not path.is_file():
            continue
        if path.suffix.lower() != ".json":
            continue
        if path.name.startswith("__") or path.name.startswith("eval_"):
            continue
        fallback.append(str(path.resolve()))
    return fallback


def _build_sandbox_setup_code(task_json_path: str) -> tuple[str, list[str]]:
    """Build a sandbox preamble that loads train_df/valid_df using the same split as the RL env.

    Returns (setup_code, extra_allowed_dirs).  On any error returns ("", []) so the
    caller can proceed without a preamble rather than crashing.
    """
    try:
        task = json.loads(Path(task_json_path).read_text(encoding="utf-8"))
    except Exception:
        return "", []

    dataset_path = task.get("dataset_path", "")
    if not dataset_path:
        return "", []

    if not os.path.isabs(dataset_path):
        dataset_path = str(PROJECT_ROOT / dataset_path)

    if not os.path.isfile(dataset_path):
        return "", []

    target = task.get("target_column", "")
    task_type = task.get("task_type", "")
    feature_columns = task.get("feature_columns")  # None means all non-target columns
    is_regression = task_type == "regression"

    feat_repr = repr(feature_columns)

    # Mirror RuntimeSession.initialize() split exactly (seed=42, 70/15/15).
    setup_code = (
        "import pandas as _pd_setup\n"
        "import numpy as _np_setup\n"
        "from sklearn.model_selection import train_test_split as _tts_setup\n"
        f"_df_setup = _pd_setup.read_csv({repr(dataset_path)})\n"
        f"_feat_setup = {feat_repr}\n"
        f"_X_setup = (_df_setup[_feat_setup] if _feat_setup is not None\n"
        f"            else _df_setup.drop(columns=[{repr(target)}]))\n"
        f"_y_setup = _df_setup[{repr(target)}]\n"
        f"_strat_setup = None if {is_regression} else _y_setup\n"
        "_X_tr, _X_tmp, _y_tr, _y_tmp = _tts_setup(\n"
        "    _X_setup, _y_setup, test_size=0.3, random_state=42, stratify=_strat_setup)\n"
        "_X_vl, _X_te, _y_vl, _y_te = _tts_setup(\n"
        "    _X_tmp, _y_tmp, test_size=0.5, random_state=42,\n"
        "    stratify=_y_tmp if _strat_setup is not None else None)\n"
        "train_df = _pd_setup.concat([_X_tr, _y_tr], axis=1).reset_index(drop=True)\n"
        "valid_df = _pd_setup.concat([_X_vl, _y_vl], axis=1).reset_index(drop=True)\n"
        "pd = _pd_setup\n"
        "np = _np_setup\n"
        "del (_pd_setup, _np_setup, _tts_setup, _df_setup, _feat_setup,\n"
        "     _X_setup, _y_setup, _strat_setup,\n"
        "     _X_tr, _X_tmp, _y_tr, _y_tmp, _X_vl, _X_te, _y_vl, _y_te)\n"
    )
    extra_dirs = [os.path.dirname(dataset_path)]
    return setup_code, extra_dirs



def _sync_state_output(state: dict, observation: str) -> dict:
    outputs = list(state.get("outputs", []))
    if not outputs:
        return state
    outputs[-1] = observation
    return {**state, "outputs": outputs}


def _evaluate_one(data_idx: int, client, ds_star: DSStar, args: dict) -> dict:
    _failed_result = {
        "item_id": f"{args['task_name']}_{data_idx}",
        "reward": 0.0,
        "success": 0,
        "trace": [],
        "plans": [],
        "generated_files": [],
        "ds_star_success": False,
    }

    client.reset(data_idx)
    initial_observation = client.observe()
    data_files = _build_episode_data_files(args["ds_data_path"], initial_observation)

    workspace = os.path.join(args["ds_workspace_root"], f"item_{data_idx}")
    os.makedirs(workspace, exist_ok=True)

    # Build a preamble that loads train_df/valid_df matching the RL env's split so
    # DS-Star's own sandbox can execute and verify the generated code correctly.
    setup_code, extra_dirs = (
        _build_sandbox_setup_code(data_files[0]) if data_files else ("", [])
    )

    # Point DS-Star's data_path to the item workspace (contains only generated
    # Python files, no CSVs).  DS-Star's debugger passes data_path as
    # {data_directory} in debug_coder.md — so the model is told "files are in
    # workspace/", not in the actual tasks directory.  This stops the model from
    # hardcoding pd.read_csv('/automl_eval/tasks/...') calls.
    # extra_dirs still lets sandbox_setup_code read the CSV at runtime.
    ds_star.data_path = workspace
    ds_star.data_files = []

    try:
        state = ds_star.initialize(
            query=_build_query(initial_observation),
            workspace=workspace,
            max_rounds=args["max_round"],
            max_debug_steps=args["max_debug_steps"],
            finalyze=args["finalyze"],
            sandbox_setup_code=setup_code,
            sandbox_extra_dirs=extra_dirs if extra_dirs else None,
        )
    except Exception as exc:
        print(f"[DS-Star] initialize() failed for item {data_idx}: {exc}")
        return {**_failed_result, "error": str(exc)}

    trace: list[dict] = []
    reward, done = 0.0, False

    reward, done, _ = _send_action(
        client,
        "PLAN",
        _plan_text(state["plans"], state["plan_action_types"]),
        trace,
    )

    if done:
        return {
            **_failed_result,
            "reward": reward,
            "success": 1 if reward >= 1 else 0,
            "trace": trace,
            "plans": state["plans"],
            "generated_files": state["base_code_filenames"],
            "ds_star_success": state["success"],
        }

    initial_action_type = state["plan_action_types"][-1] if state["plan_action_types"] else "FEATURE_ENGINEERING"
    initial_code = _read_file(state["base_code_filenames"][-1])
    reward, done, observation = _send_action(
        client,
        _action_type_to_env(initial_action_type),
        initial_code,
        trace,
    )
    state = _sync_state_output(state, observation)

    while not done:
        try:
            transition = ds_star.next_transition(state)
        except Exception as exc:
            print(f"[DS-Star] next_transition() failed for item {data_idx}: {exc}")
            reward, done, _ = _send_action(client, "FINAL_SUBMIT", "", trace)
            break

        if transition == "iterate":
            try:
                state = ds_star.step(state)
            except Exception as exc:
                print(f"[DS-Star] step(iterate) failed for item {data_idx}: {exc}")
                reward, done, _ = _send_action(client, "FINAL_SUBMIT", "", trace)
                break

            action_type = state["plan_action_types"][-1] if state["plan_action_types"] else "FEATURE_ENGINEERING"
            code = _read_file(state["base_code_filenames"][-1])
            reward, done, observation = _send_action(
                client,
                _action_type_to_env(action_type),
                code,
                trace,
            )
            state = _sync_state_output(state, observation)
            if done:
                reward, done, _ = _send_action(client, "FINAL_SUBMIT", "", trace)
                break
            continue

        if transition in {"finalize", "end"}:
            try:
                state = ds_star.step(state)
            except Exception as exc:
                print(f"[DS-Star] step(finalize) failed for item {data_idx}: {exc}")
                reward, done, _ = _send_action(client, "FINAL_SUBMIT", "", trace)
                break

            if transition == "finalize":
                final_code = _read_file(state["base_code_filenames"][-1])
                reward, done, observation = _send_action(client, "MODEL", final_code, trace)
                state = _sync_state_output(state, observation)
                # Do NOT break on done here: FINAL_SUBMIT must still run to
                # compute the metric against valid data (session stays active).

            reward, done, _ = _send_action(client, "FINAL_SUBMIT", "", trace)
            break

        if transition == "failed":
            try:
                state = ds_star.step(state)
            except Exception as exc:
                print(f"[DS-Star] step(failed) failed for item {data_idx}: {exc}")
            reward, done, _ = _send_action(client, "FINAL_SUBMIT", "", trace)
            break

        print(f"[DS-Star] Unexpected transition '{transition}' for item {data_idx}")
        reward, done, _ = _send_action(client, "FINAL_SUBMIT", "", trace)
        break

    return {
        "item_id": f"{args['task_name']}_{data_idx}",
        "reward": reward,
        "success": 1 if reward >= 1 else 0,
        "trace": trace,
        "plans": state["plans"],
        "generated_files": state["base_code_filenames"],
        "ds_star_success": state["success"],
    }


def main(args):
    if args["task_name"].lower() != "automl":
        raise ValueError("ds_star_eval.py currently supports only --task_name automl")

    llm = GatewayChatModel(
        model=args["model"],
        base_url=args["base_url"],
        api_key=args["api_key"],
    )
    ds_star = DSStar(llm=llm, data_path=args["ds_data_path"])

    env_args = {
        "env_server_base": args["env_server_base"],
        "data_len": args["data_len"],
        "timeout": args["timeout"],
    }
    task = AutoMLTask(client_args=env_args, n_clients=1)
    client = task.clients[0]

    with open(args["inference_file"], "r") as file:
        test_data = json.load(file)
    data_idxs = [int(item["item_id"].split("_")[-1]) for item in test_data]

    total_score = 0.0
    total_success = 0.0
    start_time = time.time()
    os.makedirs(args["output_dir"], exist_ok=True)
    os.makedirs(args["ds_workspace_root"], exist_ok=True)

    try:
        for data_idx in tqdm(data_idxs, total=len(data_idxs), desc="[DS-Star Eval]"):
            out_path = os.path.join(args["output_dir"], f"{args['task_name']}_{data_idx}.json")

            if os.path.exists(out_path):
                with open(out_path, "r") as file:
                    item = json.load(file)
                total_score += item["reward"]
                total_success += item["success"]
                continue

            item = _evaluate_one(data_idx=data_idx, client=client, ds_star=ds_star, args=args)
            total_score += item["reward"]
            total_success += item["success"]

            with open(out_path, "w") as file:
                json.dump(item, file, ensure_ascii=False, indent=2)
    finally:
        client.close()

    process_time = time.time() - start_time
    score = total_score / len(data_idxs) if data_idxs else 0.0
    success = total_success / len(data_idxs) if data_idxs else 0.0
    print("\n\n==== DS-STAR EVALUATION ====\n")
    print(f"Score: {score}")
    print(f"Success: {success}")
    print(f"Time: {process_time} seconds")


if __name__ == "__main__":
    args = _parse_args()
    main(args)
