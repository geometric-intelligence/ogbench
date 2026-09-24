#!/usr/bin/env bash
# Keep the ratio-0.3 transfer campaign running on Parka until the deadline.
#
# Start detached (survives logout):
#   cd /home/gbg141/bgbench && nohup setsid scripts/ratio03_supervisor.sh \
#     > "$ROOT/supervisor.log" 2>&1 < /dev/null & echo $! > "$ROOT/supervisor.pid"
# Stop everything it started (launcher, training jobs, side loops):
#   kill -TERM -- -"$(cat "$ROOT/supervisor.pid")"
set -euo pipefail

REPO=${REPO:-/home/gbg141/bgbench}
PYTHON=${PYTHON:-/home/gbg141/miniconda3/envs/bgbench/bin/python}
DATA_ROOT=${DATA_ROOT:-/scratch/lcornelis/ogbench}
ROOT=${ROOT:-$DATA_ROOT/search_results/sep24_ratio03_transfer}
PREP=${PREP:-$ROOT/prepare}
CONFIG=${CONFIG:-$REPO/configs/hparams_search/sep24_ratio03_transfer.yaml}
DEADLINE=${DEADLINE:-2026-09-25 07:00}
GPUS=${GPUS:-0 1 2 3 4 5 6 7}
JOBS_PER_GPU=${JOBS_PER_GPU:-2}
MIN_FREE_MIB=${MIN_FREE_MIB:-30000}
OOM_RETRIES=${OOM_RETRIES:-3}
OOM_MIN_FREE_MIB=${OOM_MIN_FREE_MIB:-60000}
RELAUNCH_WAIT=${RELAUNCH_WAIT:-420}
SNAPSHOT_EVERY=${SNAPSHOT_EVERY:-1800}
GPU_LOG_EVERY=${GPU_LOG_EVERY:-300}
# The seven-trial campaign's retry pass runs only after the ratio-0.3 queue drains.
OLD_LAUNCHER_PID=${OLD_LAUNCHER_PID:-720671}
OLD_ROOT=${OLD_ROOT:-$DATA_ROOT/search_results/sep24_factorial_tc10}
OLD_CONFIG=${OLD_CONFIG:-$REPO/configs/hparams_search/sep24_factorial_optuna.yaml}
OLD_MANIFEST=${OLD_MANIFEST:-$DATA_ROOT/search_results/sep24_factorial_tc10_rebalance_20260922/parka.txt}

cd "$REPO"
export PYTHONUNBUFFERED=1
deadline_epoch=$(date -d "$DEADLINE" +%s)
read -r -a gpu_ids <<< "$GPUS"

log() { printf '%s supervisor: %s\n' "$(date '+%F %T')" "$*"; }

for required in "$CONFIG" "$PREP/candidates.json" "$PREP/priority.txt" "$PREP/candidates.csv"; do
  [[ -f "$required" ]] || { log "missing $required"; exit 1; }
done
(cd "$PREP" && sha256sum --check --quiet SHA256SUMS.candidates) || {
  log 'candidate checksums do not match'
  exit 1
}

paths=(
  --config "$CONFIG"
  --root-dir "$DATA_ROOT"
  --output-dir "$ROOT"
  --storage "sqlite:///$ROOT/studies.db"
)

collect() {
  "$PYTHON" scripts/ratio03_collect.py \
    --config "$CONFIG" --output-dir "$ROOT" --storage "sqlite:///$ROOT/studies.db" \
    --prepare-dir "$PREP" --oom-retries "$OOM_RETRIES" "$@"
}

snapshot_loop() {
  while true; do
    sleep "$SNAPSHOT_EVERY"
    if collect > "$ROOT/collect_latest.log" 2>&1; then
      tail -n 1 "$ROOT/collect_latest.log" >> "$ROOT/status_history.jsonl"
    else
      log 'snapshot failed; see collect_latest.log'
    fi
  done
}

gpu_log_loop() {
  while true; do
    {
      date '+%F %T'
      nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu \
        --format=csv,noheader
      nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv,noheader
    } >> "$ROOT/gpu_usage.log" 2>&1 || true
    sleep "$GPU_LOG_EVERY"
  done
}

cleanup() {
  local children
  children=$(jobs -p)
  [[ -n "$children" ]] && kill $children 2> /dev/null || true
}
trap cleanup EXIT

snapshot_loop &
gpu_log_loop &

log "ratio-0.3 campaign until $DEADLINE on GPUs ${gpu_ids[*]} at $JOBS_PER_GPU jobs/GPU"
pass=0
while (( $(date +%s) < deadline_epoch )); do
  pass=$((pass + 1))
  log "launcher pass $pass starting"
  status=0
  "$PYTHON" scripts/optuna_search.py "${paths[@]}" \
    --studies-file "$PREP/priority.txt" \
    --candidates-file "$PREP/candidates.json" \
    --gpus "${gpu_ids[@]}" \
    --jobs-per-gpu "$JOBS_PER_GPU" \
    --min-free-gpu-mib "$MIN_FREE_MIB" \
    --oom-retries "$OOM_RETRIES" \
    --oom-min-free-gpu-mib "$OOM_MIN_FREE_MIB" \
    --retry-failed \
    --skip-warmup >> "$ROOT/launcher.log" 2>&1 || status=$?
  log "launcher pass $pass exited with status $status"

  remaining=0
  collect --check-remaining > "$ROOT/collect_latest.log" 2>&1 || remaining=$?
  tail -n 1 "$ROOT/collect_latest.log" >> "$ROOT/status_history.jsonl" || true
  if (( remaining == 3 )); then
    log 'every ratio-0.3 cell is complete or permanently failed'
    break
  fi
  if (( remaining != 0 )); then
    log "collector failed with status $remaining; relaunching anyway"
  fi
  # Wait out Optuna's heartbeat grace period so stale trials are failed and retried.
  log "waiting ${RELAUNCH_WAIT}s before relaunching"
  sleep "$RELAUNCH_WAIT"
done

if (( $(date +%s) < deadline_epoch )); then
  while kill -0 "$OLD_LAUNCHER_PID" 2> /dev/null; do
    log "waiting for the seven-trial launcher $OLD_LAUNCHER_PID to exit"
    sleep 600
  done
  log 'running the deferred seven-trial retry pass'
  "$PYTHON" scripts/optuna_search.py \
    --config "$OLD_CONFIG" \
    --root-dir "$DATA_ROOT" \
    --output-dir "$OLD_ROOT" \
    --storage "sqlite:///$OLD_ROOT/studies.db" \
    --studies-file "$OLD_MANIFEST" \
    --num-shards 7 --shard-indices 0 1 2 \
    --gpus "${gpu_ids[@]}" \
    --jobs-per-gpu "$JOBS_PER_GPU" \
    --min-free-gpu-mib "$MIN_FREE_MIB" \
    --retry-failed \
    --skip-warmup >> "$ROOT/old_campaign_retry.log" 2>&1 || log 'seven-trial retry pass failed'
fi

collect > "$ROOT/collect_latest.log" 2>&1 || true
log 'supervisor finished'
