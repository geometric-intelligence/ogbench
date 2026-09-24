#!/usr/bin/env bash
# Run an extra ratio-0.3 queue next to the supervisor until STOP_AT.
#
# Shares studies.db, the run ledger, and candidates.json with the supervisor but reads its
# own manifest (default: the balanced gin/gatv4/gps queue in $ROOT/extra_balanced). The
# supervisor's cheapest-first queue does not reach these models before the deadline, so the
# two launchers never run the same study.
#
# Start detached (survives logout):
#   cd /home/gbg141/bgbench && nohup setsid scripts/ratio03_extra.sh \
#     > "$ROOT/extra.log" 2>&1 < /dev/null & echo $! > "$ROOT/extra.pid"
# Stop it and its jobs early (the launcher runs in its own process group):
#   kill -TERM "$(cat "$ROOT/extra.pid")"
set -euo pipefail

REPO=${REPO:-/home/gbg141/bgbench}
PYTHON=${PYTHON:-/home/gbg141/miniconda3/envs/bgbench/bin/python}
DATA_ROOT=${DATA_ROOT:-/scratch/lcornelis/ogbench}
ROOT=${ROOT:-$DATA_ROOT/search_results/sep24_ratio03_transfer}
PREP=${PREP:-$ROOT/candidates_r05}
MANIFEST=${MANIFEST:-$ROOT/extra_balanced/priority.txt}
CONFIG=${CONFIG:-$REPO/configs/hparams_search/sep24_ratio03_transfer.yaml}
STOP_AT=${STOP_AT:-2026-09-25 06:59}
GPUS=${GPUS:-0 1 2 3 4 5 6 7}
JOBS_PER_GPU=${JOBS_PER_GPU:-1}
MIN_FREE_MIB=${MIN_FREE_MIB:-30000}
OOM_RETRIES=${OOM_RETRIES:-3}
OOM_MIN_FREE_MIB=${OOM_MIN_FREE_MIB:-60000}
TERM_GRACE=${TERM_GRACE:-120}

cd "$REPO"
export PYTHONUNBUFFERED=1
read -r -a gpu_ids <<< "$GPUS"
stop_epoch=$(date -d "$STOP_AT" +%s)

log() { printf '%s extra: %s\n' "$(date '+%F %T')" "$*"; }

log "queue $MANIFEST until $STOP_AT on GPUs ${gpu_ids[*]} at $JOBS_PER_GPU jobs/GPU"
setsid "$PYTHON" scripts/optuna_search.py \
  --config "$CONFIG" \
  --root-dir "$DATA_ROOT" \
  --output-dir "$ROOT" \
  --storage "sqlite:///$ROOT/studies.db" \
  --studies-file "$MANIFEST" \
  --candidates-file "$PREP/candidates.json" \
  --gpus "${gpu_ids[@]}" \
  --jobs-per-gpu "$JOBS_PER_GPU" \
  --min-free-gpu-mib "$MIN_FREE_MIB" \
  --oom-retries "$OOM_RETRIES" \
  --oom-min-free-gpu-mib "$OOM_MIN_FREE_MIB" \
  --skip-warmup >> "$ROOT/extra_launcher.log" 2>&1 &
launcher=$!
sleep 1
launcher_pgid=$(ps -o pgid= -p "$launcher" | tr -d ' ')
if [[ -z "$launcher_pgid" || "$launcher_pgid" == "$(ps -o pgid= -p $$ | tr -d ' ')" ]]; then
  log "launcher is not in its own process group (pgid '$launcher_pgid'); stopping"
  kill -TERM "$launcher" 2> /dev/null || true
  exit 1
fi

stop_launcher() {
  log "sending TERM to launcher process group $launcher_pgid"
  kill -TERM -- -"$launcher_pgid" 2> /dev/null || true
  for _ in $(seq "$TERM_GRACE"); do
    pgrep -g "$launcher_pgid" > /dev/null || return 0
    sleep 1
  done
  log "launcher process group still alive after ${TERM_GRACE}s; sending KILL"
  kill -KILL -- -"$launcher_pgid" 2> /dev/null || true
}
trap 'stop_launcher; log "stopped by signal"; exit 143' TERM INT

while kill -0 "$launcher" 2> /dev/null && (( $(date +%s) < stop_epoch )); do
  sleep 60 &
  wait $! || true
done

if kill -0 "$launcher" 2> /dev/null; then
  log 'deadline reached'
  stop_launcher
else
  wait "$launcher" && status=0 || status=$?
  log "launcher exited with status $status before the deadline"
fi
log 'done'
