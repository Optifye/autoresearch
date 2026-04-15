#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

HEADLESS_ROOT="${HEADLESS_ROOT:-$REPO_ROOT/headless_runs}"
SUPERVISOR_LOG_DIR="$HEADLESS_ROOT/supervisor"
mkdir -p "$SUPERVISOR_LOG_DIR"
SUPERVISOR_LOG="${SUPERVISOR_LOG:-$SUPERVISOR_LOG_DIR/supervisor.log}"

exec > >(tee -a "$SUPERVISOR_LOG") 2>&1

log() {
  printf '[%s] %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*"
}

restore_generated_run_log() {
  if [[ ! -d "$REPO_ROOT/.git" ]]; then
    return 0
  fi
  if ! git -C "$REPO_ROOT" ls-files --error-unmatch run.log >/dev/null 2>&1; then
    return 0
  fi
  if git -C "$REPO_ROOT" diff --quiet -- run.log; then
    return 0
  fi
  if git -C "$REPO_ROOT" restore --source=HEAD -- run.log >/dev/null 2>&1; then
    log "Restored generated run.log back to HEAD"
  else
    log "Warning: failed to restore generated run.log"
  fi
}

CANONICAL_CACHE_PARENT="/tmp/autoresearch-mangalam-packing-cache"
CANONICAL_CACHE_DIR="$CANONICAL_CACHE_PARENT/mangalam_dense_v1"
SMOKE_CACHE_DIR="/tmp/autoresearch-mangalam-packing-cache-smoke/mangalam_dense_v1"

ensure_program_cache_path() {
  if [[ -f "$CANONICAL_CACHE_DIR/manifest.json" ]]; then
    return 0
  fi
  if [[ ! -f "$SMOKE_CACHE_DIR/manifest.json" ]]; then
    return 1
  fi
  mkdir -p "$CANONICAL_CACHE_PARENT"
  if [[ -e "$CANONICAL_CACHE_DIR" && ! -L "$CANONICAL_CACHE_DIR" ]]; then
    local backup_path="${CANONICAL_CACHE_DIR}.incomplete.$(date -u +%Y%m%dT%H%M%SZ)"
    mv "$CANONICAL_CACHE_DIR" "$backup_path"
    log "Moved incomplete canonical cache aside: $backup_path"
  fi
  ln -sfn "$SMOKE_CACHE_DIR" "$CANONICAL_CACHE_DIR"
  log "Linked canonical cache path to smoke cache: $CANONICAL_CACHE_DIR -> $SMOKE_CACHE_DIR"
}

resolve_cache_parent() {
  if [[ -n "${AUTORESEARCH_CACHE_DIR:-}" ]]; then
    printf '%s\n' "${AUTORESEARCH_CACHE_DIR}"
    return 0
  fi
  if [[ -f /tmp/autoresearch-mangalam-packing-cache/mangalam_dense_v1/manifest.json ]]; then
    printf '%s\n' "/tmp/autoresearch-mangalam-packing-cache"
    return 0
  fi
  if [[ -f /tmp/autoresearch-mangalam-packing-cache-smoke/mangalam_dense_v1/manifest.json ]]; then
    printf '%s\n' "/tmp/autoresearch-mangalam-packing-cache-smoke"
    return 0
  fi
  return 1
}

CODEX_MODEL="${CODEX_MODEL:-gpt-5.4}"
CODEX_REASONING_EFFORT="${CODEX_REASONING_EFFORT:-xhigh}"
CODEX_SESSION_TIMEOUT_SEC="${CODEX_SESSION_TIMEOUT_SEC:-7200}"
CODEX_COOLDOWN_SEC="${CODEX_COOLDOWN_SEC:-15}"
CODEX_MAX_SESSIONS="${CODEX_MAX_SESSIONS:-0}"
CODEX_PROMPT_FILE="${CODEX_PROMPT_FILE:-$REPO_ROOT/prompts/headless_codex_autoresearch_loop.md}"
OUTPUT_COLOR="${OUTPUT_COLOR:-never}"

ensure_program_cache_path || true

export AUTORESEARCH_CACHE_DIR="$(resolve_cache_parent)"
export AUTORESEARCH_FIXED_STAGE0_CHECKPOINT="${AUTORESEARCH_FIXED_STAGE0_CHECKPOINT:-/tmp/autoresearch-mangalam-packing-cache/fixed_stage0_prod_reduced_rf_halo16/boundary_model.pt}"
export AUTORESEARCH_WORKSPACE_ROOT="${AUTORESEARCH_WORKSPACE_ROOT:-/home/ubuntu/internvideo-attention}"
export AUTORESEARCH_ENV_PATH="${AUTORESEARCH_ENV_PATH:-/home/ubuntu/autoresearch/.env}"

if [[ ! -f "$CODEX_PROMPT_FILE" ]]; then
  log "Missing prompt file: $CODEX_PROMPT_FILE"
  exit 1
fi

if [[ ! -f "$AUTORESEARCH_FIXED_STAGE0_CHECKPOINT" ]]; then
  log "Missing fixed stage0 checkpoint: $AUTORESEARCH_FIXED_STAGE0_CHECKPOINT"
  exit 1
fi

if [[ ! -f "$AUTORESEARCH_CACHE_DIR/mangalam_dense_v1/manifest.json" ]]; then
  log "Missing autoresearch cache manifest under: $AUTORESEARCH_CACHE_DIR/mangalam_dense_v1"
  exit 1
fi

mkdir -p "$HEADLESS_ROOT"

RESULTS_TSV="$REPO_ROOT/results.tsv"
if [[ ! -f "$RESULTS_TSV" ]]; then
  printf 'commit\tprimary_metric\taux_metric\tmemory_gb\tstatus\tdescription\n' > "$RESULTS_TSV"
  log "Initialized results.tsv header"
fi

ITERATION=1
trap 'log "Supervisor exiting"' EXIT

log "Starting headless Codex supervisor"
log "repo_root=$REPO_ROOT"
log "cache_dir=$AUTORESEARCH_CACHE_DIR"
log "fixed_stage0=$AUTORESEARCH_FIXED_STAGE0_CHECKPOINT"
log "model=$CODEX_MODEL reasoning=$CODEX_REASONING_EFFORT timeout_sec=$CODEX_SESSION_TIMEOUT_SEC max_sessions=$CODEX_MAX_SESSIONS"

while true; do
  if [[ "$CODEX_MAX_SESSIONS" != "0" && "$ITERATION" -gt "$CODEX_MAX_SESSIONS" ]]; then
    log "Reached CODEX_MAX_SESSIONS=$CODEX_MAX_SESSIONS; exiting cleanly"
    exit 0
  fi

  STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
  RUN_DIR="$HEADLESS_ROOT/${STAMP}_iter$(printf '%04d' "$ITERATION")"
  mkdir -p "$RUN_DIR"
  cp "$CODEX_PROMPT_FILE" "$RUN_DIR/prompt.txt"

  {
    printf 'iteration=%s\n' "$ITERATION"
    printf 'timestamp_utc=%s\n' "$STAMP"
    printf 'repo_root=%s\n' "$REPO_ROOT"
    printf 'cache_dir=%s\n' "$AUTORESEARCH_CACHE_DIR"
    printf 'fixed_stage0_checkpoint=%s\n' "$AUTORESEARCH_FIXED_STAGE0_CHECKPOINT"
    printf 'model=%s\n' "$CODEX_MODEL"
    printf 'reasoning_effort=%s\n' "$CODEX_REASONING_EFFORT"
    printf 'timeout_sec=%s\n' "$CODEX_SESSION_TIMEOUT_SEC"
  } > "$RUN_DIR/launcher_env.txt"

  log "Launching Codex session iteration=$ITERATION run_dir=$RUN_DIR"

  set +e
  timeout --signal=TERM --kill-after=60s "${CODEX_SESSION_TIMEOUT_SEC}" \
    codex exec \
      --dangerously-bypass-approvals-and-sandbox \
      --json \
      --color "$OUTPUT_COLOR" \
      -C "$REPO_ROOT" \
      -m "$CODEX_MODEL" \
      -c "model_reasoning_effort=\"$CODEX_REASONING_EFFORT\"" \
      -o "$RUN_DIR/last_message.txt" \
      < "$CODEX_PROMPT_FILE" \
      > "$RUN_DIR/codex.jsonl" \
      2> "$RUN_DIR/codex.stderr"
  STATUS=$?
  set -e

  printf '%s\n' "$STATUS" > "$RUN_DIR/exit_status.txt"

  case "$STATUS" in
    0)
      log "Codex session iteration=$ITERATION exited cleanly"
      ;;
    124)
      log "Codex session iteration=$ITERATION hit timeout after ${CODEX_SESSION_TIMEOUT_SEC}s"
      ;;
    137)
      log "Codex session iteration=$ITERATION was killed after timeout escalation"
      ;;
    *)
      log "Codex session iteration=$ITERATION exited with status=$STATUS"
      ;;
  esac

  restore_generated_run_log

  ITERATION=$((ITERATION + 1))
  sleep "$CODEX_COOLDOWN_SEC"
done
