#!/usr/bin/env bash
# run.sh — Linux/macOS entry point for the forex ML pipeline (mirrors run.ps1).
#
# Usage:
#   ./run.sh download --start 2017-02-18
#   ./run.sh migrate
#   ./run.sh validate
#   ./run.sh data --start 2017-02-18
#   ./run.sh train
#   ./run.sh train --quick
#   ./run.sh backtest
#   ./run.sh all --start 2017-02-18

set -euo pipefail

_CMD="${1:-}"
if [[ -z "$_CMD" ]]; then
  echo "Usage: ./run.sh <download|migrate|validate|data|train|backtest|all> [args...]" >&2
  exit 2
fi
case "$_CMD" in
  download|migrate|validate|data|train|backtest|all) ;;
  *) echo "Unknown command '$_CMD'. Use: download|migrate|validate|data|train|backtest|all" >&2; exit 2 ;;
esac
shift

_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

_PY="$_ROOT/.venv/bin/python"
if [[ ! -x "$_PY" ]]; then
  _PY="$_ROOT/.venv-gpu/bin/python"
fi
if [[ ! -x "$_PY" ]]; then
  _PY="python"
fi

exec "$_PY" "$_ROOT/scripts/run_pipeline.py" "$_CMD" "$@"
