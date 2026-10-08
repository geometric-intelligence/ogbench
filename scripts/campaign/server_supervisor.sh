#!/usr/bin/env bash
# Keep one follow-mode launcher running on this server until its manifest is finished.
#
# Started detached by `remote_agent.py start` (its own session, so `remote_agent.py stop`
# can signal the launcher, its workers and their training jobs together). Settings come from
# the environment that command builds from scripts/campaign/servers.yaml.
set -uo pipefail

: "${REPO:?}" "${PYTHON:?}" "${RUN_ROOT:?}" "${DATA_ROOT:?}" "${CONFIGS:?}" "${GPUS:?}"
JOBS_PER_GPU=${JOBS_PER_GPU:-2}
WARMUP_JOBS=${WARMUP_JOBS:-2}
MIN_FREE_GPU_MIB=${MIN_FREE_GPU_MIB:-0}
OOM_RETRIES=${OOM_RETRIES:-1}
OOM_MIN_FREE_GPU_MIB=${OOM_MIN_FREE_GPU_MIB:-0}
POLL_SECONDS=${POLL_SECONDS:-30}
# Longer than the Optuna heartbeat grace period, so a restarted launcher can reclaim trials.
RELAUNCH_WAIT=${RELAUNCH_WAIT:-420}
GPU_LOG_EVERY=${GPU_LOG_EVERY:-300}

MANIFEST="$RUN_ROOT/manifest.txt"
STATE="$MANIFEST.state.json"
LOG_DIR="$RUN_ROOT/logs"
mkdir -p "$LOG_DIR"
cd "$REPO" || exit 1
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
read -r -a config_list <<< "$CONFIGS"
read -r -a gpu_ids <<< "$GPUS"

log() { printf '%s server_supervisor: %s\n' "$(date -u '+%F %T')" "$*"; }

finished() {
  "$PYTHON" -c 'import json, sys; sys.exit(0 if json.load(open(sys.argv[1]))["finished"] else 1)' \
    "$STATE" 2>/dev/null
}

gpu_log_loop() {
  while true; do
    {
      date -u '+%F %T'
      nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu \
        --format=csv,noheader
      nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv,noheader
    } >> "$LOG_DIR/gpu_usage.log" 2>&1 || true
    sleep "$GPU_LOG_EVERY"
  done
}

launcher_pid=''
cleanup() {
  [[ -n "$launcher_pid" ]] && kill "$launcher_pid" 2>/dev/null
  kill "$gpu_log_pid" 2>/dev/null
}
gpu_log_loop &
gpu_log_pid=$!
trap cleanup EXIT
trap 'exit 143' TERM INT HUP

attempt=0
while true; do
  attempt=$((attempt + 1))
  log "starting follow launcher (attempt $attempt) on GPUs ${gpu_ids[*]}"
  "$PYTHON" scripts/optuna_search.py \
    --config "${config_list[@]}" \
    --follow-manifest "$MANIFEST" \
    --candidates-file "$RUN_ROOT/candidates.json" \
    --output-dir "$RUN_ROOT" \
    --root-dir "$DATA_ROOT" \
    --gpus "${gpu_ids[@]}" \
    --jobs-per-gpu "$JOBS_PER_GPU" \
    --warmup-jobs "$WARMUP_JOBS" \
    --min-free-gpu-mib "$MIN_FREE_GPU_MIB" \
    --oom-retries "$OOM_RETRIES" \
    --oom-min-free-gpu-mib "$OOM_MIN_FREE_GPU_MIB" \
    --follow-poll-seconds "$POLL_SECONDS" \
    --retry-failed \
    >> "$LOG_DIR/launcher.log" 2>&1 &
  launcher_pid=$!
  wait "$launcher_pid"
  status=$?
  launcher_pid=''
  if finished; then
    log 'manifest finished; supervisor exiting'
    exit 0
  fi
  log "launcher exited with status $status; restarting in ${RELAUNCH_WAIT}s"
  sleep "$RELAUNCH_WAIT"
done
