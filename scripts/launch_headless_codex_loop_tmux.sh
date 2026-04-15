#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
SUPERVISOR_SCRIPT="$REPO_ROOT/scripts/run_headless_codex_supervisor.sh"

SESSION_NAME="${1:-autoresearch_mangalam_codex}"

if ! command -v tmux >/dev/null 2>&1; then
  echo "tmux is required but not installed." >&2
  exit 1
fi

if [[ ! -x "$SUPERVISOR_SCRIPT" ]]; then
  echo "Supervisor script is missing or not executable: $SUPERVISOR_SCRIPT" >&2
  exit 1
fi

if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
  echo "tmux session already exists: $SESSION_NAME" >&2
  exit 1
fi

tmux new-session -d -s "$SESSION_NAME" "$SUPERVISOR_SCRIPT"

echo "Started tmux session: $SESSION_NAME"
echo "Attach: tmux attach -t $SESSION_NAME"
echo "Supervisor log: $REPO_ROOT/headless_runs/supervisor/supervisor.log"
