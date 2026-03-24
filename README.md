# AutoML-Eval: RL Environment for Evaluating AutoML Agents

An RL environment for evaluating LLM-based AutoML agents. The environment presents a dataset and ML task, then evaluates the agent's plan, code quality, feature engineering, model selection, and final predictions through a comprehensive suite of 21 validators.

Designed for integration with [AgentGym-RL](https://github.com/WooooDyy/AgentGym-RL) — a framework for training and evaluating LLM-based agents across diverse environments.

## Architecture

```
┌─────────────────┐     HTTP      ┌──────────────────────┐     HTTP      ┌─────────────────┐
│   AgentGym-RL   │ ──────────> │  agentenv_automl     │ ──────────> │  automl_eval    │
│   (LLM Agent)   │  port 8080   │  (proxy server)      │  port 8766   │  (RL env core)  │
│                 │ <────────── │                      │ <────────── │                 │
└─────────────────┘   actions    └──────────────────────┘   rewards    └─────────────────┘
                      + obs                                + state
```

**Three layers:**
1. **`automl_eval/`** — Core RL environment: tasks, sessions, sandbox, validators, reward calculation
2. **`agentenv_automl/`** — FastAPI proxy that translates AgentGym HTTP protocol to automl_eval calls
3. **`agentgym_integration/`** — Client code and setup script to register the environment in AgentGym-RL

> **Note:** AgentGym-RL itself is **not** included in this repository.
> It is cloned and configured automatically via the provided setup script.

## Quick Start

### 1. Install Dependencies

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Run Unit Tests (no server needed)

```bash
python tests/test_new_validators.py
python tests/test_data_quality_validators.py
python tests/test_pipeline_validators.py
python tests/test_model_validators.py
python tests/test_cycle_validators.py
```

All 58 unit tests should pass. See `tests/TESTING.md` for details.

### 3. Run the Integration Test (auto-starts servers)

```bash
python tests/test_integration.py
```

This automatically starts both the `automl_eval` server and the `agentenv_automl` proxy, runs 3 integration tests (direct server, full proxy chain, reset/new episode), and shuts everything down. No manual server setup required.

### 4. Run a Demo Episode

```bash
python demo_episode.py
```

## AgentGym-RL Integration

### How it Works

AgentGym-RL is an external framework ([GitHub](https://github.com/WooooDyy/AgentGym-RL)) that trains and evaluates LLM agents through multi-turn RL. To use our AutoML environment with AgentGym-RL, we need to:

1. Register our environment as a new "task" in AgentGym's environment client system
2. Run the `automl_eval` server (the actual RL environment)
3. Run the `agentenv_automl` proxy (translates AgentGym's HTTP protocol)
4. Point AgentGym-RL's evaluation/training scripts at the proxy

### Step 1: Automated Setup

The setup script clones AgentGym-RL, installs the `agentenv` package, and patches it to recognize the AutoML environment:

```bash
bash agentgym_integration/setup_agentgym.sh
```

This script:
- Clones [WooooDyy/AgentGym-RL](https://github.com/WooooDyy/AgentGym-RL) into `./AgentGym-RL/` (skips if already cloned)
- Runs `pip install -e .` for the `agentenv` package
- Copies `agentgym_integration/automl_client.py` into `agentenv/envs/automl.py`
- Patches `agentenv/envs/__init__.py` to import `AutoMLEnvClient`, `AutoMLTask`
- Patches `examples/basic/base_eval.py` to register the `"automl"` task

### Step 2: Start the Servers (2 terminals)

**Terminal 1** — Core environment:

```bash
source .venv/bin/activate
python run_server.py --tasks-dir automl_eval/tasks --port 8766
```

**Terminal 2** — AgentGym proxy:

```bash
source .venv/bin/activate
python -m uvicorn agentenv_automl.server:app --host 0.0.0.0 --port 8080
```

If the core environment runs on a non-default port, set the `AUTOML_SERVER_BASE` environment variable:

```bash
AUTOML_SERVER_BASE=http://localhost:9999 python -m uvicorn agentenv_automl.server:app --port 8080
```

### Step 3: Run Evaluation

**Terminal 3** — AgentGym evaluation:

```bash
cd AgentGym-RL/AgentGym/agentenv

python examples/basic/base_eval.py \
  --model_path /path/to/your/llm \
  --inference_file ../../../agentgym_integration/eval_titanic.json \
  --output_file results.jsonl \
  --task_name automl \
  --env_server_base http://localhost:8080 \
  --data_len 1 \
  --max_round 10
```

### Verifying the Connection

You can verify the full chain works **without** an LLM in two ways:

**Option A — Fully automatic (recommended):**

```bash
python tests/test_integration.py
```

This starts both servers, sends PLAN → CODE → FINAL_SUBMIT through the proxy, and validates the responses. No manual setup needed.

**Option B — Manual (with servers already running):**

```bash
# With servers from Step 2 running:
python tests/test_agentgym_chain.py
```

## Agent Action Protocol

The agent communicates via text actions in this format:

| Action | Purpose | Example |
|--------|---------|---------|
| `ACTION: PLAN` | Describe approach | `ACTION: PLAN\nI will handle missing values...` |
| `ACTION: CODE` | Execute code | `ACTION: CODE\n```python\ndf.fillna(0)\n``` ` |
| `ACTION: FEATURE_ENGINEERING` | Feature work | `ACTION: FEATURE_ENGINEERING\n```python\n...\n``` ` |
| `ACTION: MODEL` | Train model | `ACTION: MODEL\n```python\n...\n``` ` |
| `ACTION: CODE_FIX` | Fix errors | `ACTION: CODE_FIX\n```python\n...\n``` ` |
| `ACTION: FINAL_SUBMIT` | Submit predictions | `ACTION: FINAL_SUBMIT` |

The sandbox provides `train_df`, `valid_df`, `pd`, and `np` in the namespace. The agent's code must produce a `predictions` variable.

## Validators (21 total)

### Code Quality
| Validator | What it checks |
|-----------|---------------|
| `execution` | Code runs without errors |
| `correctness` | Predictions are valid (right length, no NaN, proper dtype) |
| `intactness` | Original train/valid data unchanged in sandbox |
| `leakage` | No data leakage (test data access, target in features) |
| `reproducibility` | Seeds fixed, deterministic predictions on re-run |
| `efficiency` | Within time budget, no exhaustive grid search |
| `namespace_check` | Expected variables exist in sandbox |

### Planning
| Validator | What it checks |
|-----------|---------------|
| `plan_coverage` | Plan covers checklist items (missing values, encoding, model, etc.) |
| `backtracking` | Penalizes returning to basic FE after model training |
| `iterative_cycles` | Escalating penalty for EDA→Model cycles with diminishing returns |

### Data Quality
| Validator | What it checks |
|-----------|---------------|
| `correlation` | Agent analyzes and handles correlated features |
| `missing_values` | Missing values properly handled (not too aggressive dropna) |
| `distribution` | EDA performed, outliers detected and handled |
| `feature_pipeline` | Outcome-based: no NaN left, categoricals encoded, scaling applied |
| `duplicates` | Duplicate rows detected and removed |

### Model Quality
| Validator | What it checks |
|-----------|---------------|
| `model_eval` | Trained model produces valid predictions on validation data |
| `target_leakage_model` | Model-based leakage detection (single-feature AUC, train/valid gap) |
| `feature_importance` | Agent inspects feature importance; new features are valuable |
| `hyperparameters` | Explicit params set, no anti-patterns (n_estimators=1) |
| `model_choice` | Penalizes ANN-first on tabular; rewards boosting/ensemble |
| `split_quality` | Proper split ratios, stratification, CV usage, time-series awareness |
| `baseline_comparison` | Agent must beat a simple GBT baseline |

## Reward Calculation

```
R = w_perf × performance + w_plan × plan_coverage + w_code × code_quality − penalties
```

Default weights: `performance=0.5`, `plan_coverage=0.2`, `code_quality=0.3`.

Each validator contributes a score (0-1) and optional penalty. The final reward is clamped to [0, 1].

## Adding a New Task

Create a JSON file in `automl_eval/tasks/`:

```json
{
  "task_id": "my_task",
  "dataset_path": "automl_eval/tasks/my_data.csv",
  "target_column": "target",
  "task_type": "binary_classification",
  "metric": "roc_auc",
  "description": "Predict target from features.",
  "plan_checklist": [
    {"id": "handle_missing", "description": "Handle missing values", "keywords": ["missing", "fillna"], "weight": 1.0, "required": true}
  ],
  "time_budget_seconds": 300,
  "max_steps": 15,
  "oracle_score": 0.90,
  "baseline_score": 0.50
}
```

Place the CSV file alongside it. The server auto-discovers all `.json` task files on startup.

## Project Structure

```
automl_eval/                    # Core RL environment
├── __init__.py
├── action_parser.py            # Parses agent text → structured actions
├── data_insights.py            # Pre-analyzes datasets (missing values, correlations, outliers)
├── environment.py              # Core RL environment (reset/step/observe/close)
├── metrics.py                  # Metric computation (ROC AUC, RMSE, accuracy, etc.)
├── reward.py                   # Composite reward from validator results
├── sandbox.py                  # Safe code execution with timeout
├── server.py                   # HTTP server exposing the environment
├── session.py                  # Episode state (data, steps, cycle tracking)
├── task.py                     # Task definition (dataset, metric, checklist)
├── task_registry.py            # Task collection with directory loading
├── tasks/
│   ├── titanic.csv             # Demo dataset
│   └── titanic_binary.json     # Demo task definition
└── validators/                 # 21 validators (see table above)
    ├── base.py
    ├── backtracking.py
    ├── ...
    └── target_leakage_model.py

agentenv_automl/                # AgentGym proxy server (FastAPI)
├── __init__.py
├── environment.py              # HTTP client to automl_eval server
├── env_wrapper.py              # Multi-instance manager
├── server.py                   # FastAPI endpoints (/create, /step, /reset, etc.)
├── model.py                    # Pydantic request models
└── launch.py                   # CLI entry point

agentgym_integration/           # AgentGym-RL registration
├── automl_client.py            # AutoMLEnvClient + AutoMLTask (installed into AgentGym)
├── setup_agentgym.sh           # One-command setup: clone, install, patch AgentGym-RL
└── eval_titanic.json           # Evaluation manifest for Titanic task

tests/                          # Test suite (61 tests total)
├── TESTING.md                  # Testing guide
├── test_new_validators.py      # 9 tests: backtracking, reproducibility, efficiency
├── test_data_quality_validators.py  # 12 tests: insights, correlation, missing, distribution
├── test_pipeline_validators.py # 10 tests: feature pipeline, duplicates, leakage model
├── test_model_validators.py    # 15 tests: importance, hyperparams, model choice, split
├── test_cycle_validators.py    # 12 tests: iterative cycles, baseline comparison
├── test_integration.py         # 3 integration tests: auto-starts servers, full chain
├── test_client.py              # Manual HTTP test (requires running servers)
└── test_agentgym_chain.py      # Manual chain test (requires running servers)

run_server.py                   # Entry point: starts the automl_eval HTTP server
demo_episode.py                 # Standalone demo of a full episode
requirements.txt                # Python dependencies
```

## License

MIT
