#!/usr/bin/env bash
#
# Setup script: clone AgentGym-RL and register the AutoML environment.
#
# Usage:
#   cd /path/to/automl-eval
#   bash agentgym_integration/setup_agentgym.sh
#
# What this script does:
#   1. Clones AgentGym-RL (with the AgentGym submodule) into ./AgentGym-RL
#   2. Installs the agentenv package
#   3. Copies automl_client.py into agentenv/envs/automl.py
#   4. Patches agentenv/envs/__init__.py to import AutoMLEnvClient, AutoMLTask
#   5. Patches examples/basic/base_eval.py to register the "automl" task
#
# After running this script, you can evaluate your AutoML agent via:
#   python AgentGym-RL/AgentGym/agentenv/examples/basic/base_eval.py \
#     --task_name automl --env_server_base http://localhost:8080 ...
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

AGENTGYM_DIR="$PROJECT_ROOT/AgentGym-RL"
ENVS_DIR="$AGENTGYM_DIR/AgentGym/agentenv/agentenv/envs"
BASE_EVAL="$AGENTGYM_DIR/AgentGym/agentenv/examples/basic/base_eval.py"
DIST_EVAL="$AGENTGYM_DIR/AgentGym/agentenv/utils/distributed_eval_task.py"

echo "=== Step 1: Clone AgentGym-RL ==="
if [ -d "$AGENTGYM_DIR" ]; then
    echo "  AgentGym-RL already exists at $AGENTGYM_DIR, skipping clone."
else
    git clone --recurse-submodules https://github.com/WooooDyy/AgentGym-RL.git "$AGENTGYM_DIR"
fi

echo ""
echo "=== Step 2: Install agentenv package ==="
cd "$AGENTGYM_DIR/AgentGym/agentenv"
pip install -e . 2>&1 | tail -3
cd "$PROJECT_ROOT"

echo ""
echo "=== Step 3: Copy automl_client.py ==="
cp "$SCRIPT_DIR/automl_client.py" "$ENVS_DIR/automl.py"
echo "  Copied to $ENVS_DIR/automl.py"

echo ""
echo "=== Step 4: Patch envs/__init__.py ==="
INIT_FILE="$ENVS_DIR/__init__.py"
if grep -q "automl" "$INIT_FILE" 2>/dev/null; then
    echo "  Already patched."
else
    echo 'from .automl import AutoMLEnvClient, AutoMLTask' >> "$INIT_FILE"
    echo "  Added AutoML imports to $INIT_FILE"
fi

echo ""
echo "=== Step 5: Patch base_eval.py ==="
if [ -f "$BASE_EVAL" ]; then
    if grep -q "AutoMLTask" "$BASE_EVAL" 2>/dev/null; then
        echo "  base_eval.py already patched."
    else
        # Add import
        sed -i.bak 's/from agentenv.envs import (/from agentenv.envs import (\n    AutoMLTask,/' "$BASE_EVAL"
        # Add to task_classes dict
        sed -i.bak 's/"webshop": WebshopTask,/"webshop": WebshopTask,\n        "automl": AutoMLTask,/' "$BASE_EVAL"
        rm -f "${BASE_EVAL}.bak"
        echo "  Patched $BASE_EVAL"
    fi
else
    echo "  WARNING: base_eval.py not found at $BASE_EVAL"
fi

# Patch distributed_eval_task.py if it exists
if [ -f "$DIST_EVAL" ]; then
    if grep -q "AutoMLTask" "$DIST_EVAL" 2>/dev/null; then
        echo "  distributed_eval_task.py already patched."
    else
        sed -i.bak 's/from agentenv.envs import/from agentenv.envs import AutoMLTask,/' "$DIST_EVAL"
        sed -i.bak 's/"webshop": WebshopTask,/"webshop": WebshopTask,\n        "automl": AutoMLTask,/' "$DIST_EVAL"
        rm -f "${DIST_EVAL}.bak"
        echo "  Patched $DIST_EVAL"
    fi
fi

echo ""
echo "=== Setup complete! ==="
echo ""
echo "To run an evaluation:"
echo "  1. Start automl_eval server:       python run_server.py --tasks-dir automl_eval/tasks"
echo "  2. Start proxy server:             python -m uvicorn agentenv_automl.server:app --port 8080"
echo "  3. Run AgentGym evaluation:        python $BASE_EVAL \\"
echo "       --task_name automl --env_server_base http://localhost:8080 \\"
echo "       --model_path /path/to/llm --inference_file $SCRIPT_DIR/eval_titanic.json \\"
echo "       --output_file results.jsonl --data_len 1 --max_round 10"
