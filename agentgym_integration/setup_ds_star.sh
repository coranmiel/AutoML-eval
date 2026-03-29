#!/usr/bin/env bash
#
# Setup script: download/update DS-Star implementation.
#
# Usage:
#   cd /path/to/AutoML-eval
#   bash agentgym_integration/setup_ds_star.sh
#
# Optional env vars:
#   DS_STAR_REPO_URL   (default: https://github.com/rerum-nn/DS-Star_impl.git)
#   DS_STAR_REF        (branch/tag/commit to checkout, default: automl-eval-integration)
#   DS_STAR_DIR_NAME   (default: DS-Star_impl)
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

DS_STAR_REPO_URL="${DS_STAR_REPO_URL:-https://github.com/rerum-nn/DS-Star_impl.git}"
DS_STAR_REF="${DS_STAR_REF:-automl-eval-integration}"
DS_STAR_DIR_NAME="${DS_STAR_DIR_NAME:-DS-Star_impl}"
DS_STAR_DIR="$PROJECT_ROOT/$DS_STAR_DIR_NAME"

echo "=== DS-Star setup ==="
echo "Repository: $DS_STAR_REPO_URL"
echo "Ref:        $DS_STAR_REF"
echo "Target dir: $DS_STAR_DIR"
echo ""

if [ -d "$DS_STAR_DIR/.git" ]; then
  echo "Existing DS-Star repo found. Pulling updates..."
  git -C "$DS_STAR_DIR" fetch --all --tags
  git -C "$DS_STAR_DIR" checkout "$DS_STAR_REF"
  git -C "$DS_STAR_DIR" pull --ff-only origin "$DS_STAR_REF" || true
else
  if [ -e "$DS_STAR_DIR" ] && [ ! -d "$DS_STAR_DIR/.git" ]; then
    echo "ERROR: $DS_STAR_DIR exists but is not a git repository."
    echo "Delete/move it or set DS_STAR_DIR_NAME to another directory."
    exit 1
  fi
  echo "Cloning DS-Star..."
  git clone "$DS_STAR_REPO_URL" "$DS_STAR_DIR"
  git -C "$DS_STAR_DIR" checkout "$DS_STAR_REF"
fi

echo ""
echo "Installing DS-Star dependencies..."
if [ -f "$DS_STAR_DIR/requirements.txt" ]; then
  pip install -r "$DS_STAR_DIR/requirements.txt"
else
  echo "WARNING: no dependency manifest found (pyproject.toml/requirements*.txt)."
fi

echo ""
echo "DS-Star is ready at: $DS_STAR_DIR"
echo "Next: run eval with agentgym_integration/ds_star_eval.py"
