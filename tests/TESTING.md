# Testing Guide

This directory contains the test suite for the `automl_eval` environment. All tests run **without an HTTP server** — they instantiate sessions and validators directly in-process.

## Quick Start

From the **project root** (`automl_eval/` parent directory):

```bash
# Run all tests
python tests/test_new_validators.py
python tests/test_data_quality_validators.py
python tests/test_pipeline_validators.py
python tests/test_model_validators.py
python tests/test_cycle_validators.py

# Or run as modules
python -m tests.test_new_validators
python -m tests.test_data_quality_validators
python -m tests.test_pipeline_validators
python -m tests.test_model_validators
python -m tests.test_cycle_validators
```

## Test Suites

### `test_new_validators.py` — 9 tests

Tests for **BacktrackingValidator**, **ReproducibilityValidator**, and **EfficiencyValidator**.

| Test | What it checks |
|------|---------------|
| `test_backtracking_no_penalty` | Normal order (FE then model) → no penalty |
| `test_backtracking_with_penalty` | Basic operations after model training → penalty |
| `test_reproducibility_with_seed` | Code with `random_state` → deterministic → pass |
| `test_reproducibility_no_seed` | No seed + non-deterministic → penalty |
| `test_efficiency_within_budget` | Fast execution → pass |
| `test_efficiency_gridsearch` | `GridSearchCV` detected → penalty |
| `test_efficiency_randomizedsearch` | `RandomizedSearchCV` → no penalty |
| `test_efficiency_hard_limit` | Exceeds 3600s → max penalty |
| `test_full_pipeline` | End-to-end: all validators visible in feedback |

### `test_data_quality_validators.py` — 12 tests

Tests for **DataInsightsAnalyzer**, **CorrelationValidator**, **MissingValuesValidator**, and **DistributionValidator**.

| Test | What it checks |
|------|---------------|
| `test_data_insights_titanic` | Pre-analysis of Titanic: numeric/categorical columns, missing values, outliers |
| `test_corr_no_high` | No high correlations → pass |
| `test_corr_not_addressed` | High correlation pair present, not handled → penalty |
| `test_corr_addressed` | `.corr()` used + correlated feature dropped → pass |
| `test_missing_no_handling` | Missing values present, no fillna/dropna → penalty |
| `test_missing_proper_fill` | `fillna` + `drop(Cabin)` → pass |
| `test_missing_bad_fill` | Filling 85%-missing column instead of dropping → penalty |
| `test_dist_no_eda` | No `.describe()` or plots → penalty |
| `test_dist_with_clip` | `.describe()` + `.clip()` → pass |
| `test_dist_eda_in_plan` | EDA mentioned in plan text → pass |
| `test_dist_outliers_not_handled` | Outliers present, no handling → penalty |
| `test_full_pipeline_data_quality` | End-to-end pipeline with all data quality validators |

### `test_pipeline_validators.py` — 10 tests

Tests for **FeaturePipelineValidator**, **DuplicateValidator**, and **TargetLeakageModelValidator**.

| Test | What it checks |
|------|---------------|
| `test_fp_raw_data` | Unprocessed data → detects NaNs, object columns |
| `test_fp_clean_data` | Properly processed data → pass (soft scaling warning OK) |
| `test_fp_no_feature_eng` | Data cleaned but no new features → informational |
| `test_dup_no_duplicates` | Dataset has no duplicates → pass |
| `test_dup_not_handled` | Duplicates present, not removed → penalty |
| `test_dup_handled` | `drop_duplicates()` in code → pass |
| `test_leakage_no_leak` | Clean pipeline → no leakage detected |
| `test_leakage_detected` | Perfect-score feature → leakage flagged |
| `test_leakage_skip_before_submit` | Leakage check only runs at FINAL_SUBMIT |
| `test_full_pipeline_pipeline` | End-to-end: all pipeline validators visible |

### `test_model_validators.py` — 15 tests

Tests for **FeatureImportanceValidator**, **HyperparamValidator**, **ModelChoiceValidator**, and **SplitValidator**.

| Test | What it checks |
|------|---------------|
| `test_fi_no_inspection` | No `feature_importances_` check → penalty |
| `test_fi_with_inspection` | Inspects importances → pass |
| `test_fi_model_value` | Model value test: agent predictions improve baseline |
| `test_hp_all_defaults` | No explicit hyperparameters → penalty |
| `test_hp_explicit_params` | `n_estimators=100, max_depth=5` → pass |
| `test_hp_bad_params` | `n_estimators=1` → anti-pattern penalty |
| `test_mc_good_model` | RandomForest → pass + bonuses |
| `test_mc_ann_only` | Neural network only on tabular → penalty |
| `test_mc_both_nn_tree` | Both NN and tree tried → pass |
| `test_mc_no_model` | No model trained at FINAL_SUBMIT → penalty |
| `test_split_no_cv` | No cross-validation → penalty |
| `test_split_with_cv` | `StratifiedKFold` → pass |
| `test_split_ts_shuffle` | Time series with `shuffle=True` → penalty |
| `test_split_tiny_test` | `test_size=0.01` → penalty |
| `test_full_pipeline_model` | End-to-end: all model validators visible |

### `test_cycle_validators.py` — 12 tests

Tests for **IterativeCycleValidator** and **BaselineComparisonValidator**.

| Test | What it checks |
|------|---------------|
| `test_cycle_single_pass` | No iteration cycles → pass |
| `test_cycle_one_free` | 1 cycle within free allowance → pass |
| `test_cycle_escalating_penalty` | 3 cycles, no metric gain → escalating penalty |
| `test_cycle_with_improvement` | 3 cycles with metric gains → reduced penalty |
| `test_cycle_regression` | Metric drops after cycle → extra penalty |
| `test_cycle_error_multiplier` | Multiplier function: 1x→1.5x→2x→3x |
| `test_cycle_tracking_in_session` | `record_step()` correctly tracks cycle transitions |
| `test_baseline_better_than` | Agent beats GBT baseline → bonus |
| `test_baseline_worse_than` | Agent worse than baseline → penalty |
| `test_baseline_plateau` | Metric plateau with >2 cycles → diminishing returns warning |
| `test_baseline_not_done` | Before FINAL_SUBMIT → skip |
| `test_full_pipeline` | End-to-end: cycle validators visible in feedback |

### `test_integration.py` — 3 Automated Integration Tests

**Does not require manual server setup.** Automatically starts both `automl_eval` and `agentenv_automl` servers, runs tests, and shuts them down.

```bash
python tests/test_integration.py
```

| Test | What it checks |
|------|---------------|
| `test_direct_server` | automl_eval server starts, serves tasks, handles PLAN/close |
| `test_proxy_chain` | Full chain: create env → PLAN → CODE → FINAL_SUBMIT → reward, done=True |
| `test_reset_new_episode` | Reset via proxy starts a fresh episode, observation returned |

### `test_client.py` — Manual HTTP Test

Sends hardcoded PLAN → CODE → FINAL_SUBMIT actions to the `agentenv_automl` proxy server via HTTP. **Requires running servers** (see README.md for setup).

### `test_agentgym_chain.py` — Manual Chain Test

Tests the full communication chain: Python client → agentenv_automl (port 8080) → automl_eval server (port 8766). **Requires both servers running**.

## Adding New Tests

1. Create a new file `tests/test_<name>.py`
2. Add the standard path fix at the top:

```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
```

3. Import from `automl_eval.*` as needed
4. Follow the pattern: `make_task()` → `RuntimeSession` → validator → assert
5. Add a `main()` function that calls all tests and prints results
