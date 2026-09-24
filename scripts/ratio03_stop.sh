#!/usr/bin/env bash
# Stop the ratio-0.3 supervisor at STOP_AT and write the final deliverables.
#
# The supervisor only checks its deadline between launcher passes, so a pass that is
# still running at the deadline would otherwise continue until the queue drains.
#
# Start detached (survives logout):
#   cd /home/gbg141/bgbench && nohup setsid scripts/ratio03_stop.sh \
#     > "$ROOT/stop.log" 2>&1 < /dev/null & echo $! > "$ROOT/stop.pid"
set -euo pipefail

REPO=${REPO:-/home/gbg141/bgbench}
PYTHON=${PYTHON:-/home/gbg141/miniconda3/envs/bgbench/bin/python}
DATA_ROOT=${DATA_ROOT:-/scratch/lcornelis/ogbench}
ROOT=${ROOT:-$DATA_ROOT/search_results/sep24_ratio03_transfer}
PREP=${PREP:-$ROOT/candidates_r05}
CONFIG=${CONFIG:-$REPO/configs/hparams_search/sep24_ratio03_transfer.yaml}
STOP_AT=${STOP_AT:-2026-09-25 07:00}
OOM_RETRIES=${OOM_RETRIES:-3}
TERM_GRACE=${TERM_GRACE:-120}

cd "$REPO"
log() { printf '%s stop: %s\n' "$(date '+%F %T')" "$*"; }

stop_epoch=$(date -d "$STOP_AT" +%s)
log "waiting until $STOP_AT"
while (( $(date +%s) < stop_epoch )); do
  sleep $(( stop_epoch - $(date +%s) > 600 ? 600 : stop_epoch - $(date +%s) ))
done

pgid=$(ps -o pgid= -p "$(cat "$ROOT/supervisor.pid")" 2> /dev/null | tr -d ' ' || true)
if [[ -n "$pgid" ]]; then
  log "sending TERM to supervisor process group $pgid"
  kill -TERM -- -"$pgid" 2> /dev/null || true
  for _ in $(seq "$TERM_GRACE"); do
    pgrep -g "$pgid" > /dev/null || break
    sleep 1
  done
  if pgrep -g "$pgid" > /dev/null; then
    log "process group $pgid still alive after ${TERM_GRACE}s; sending KILL"
    kill -KILL -- -"$pgid" 2> /dev/null || true
  fi
else
  log 'supervisor already exited'
fi

log 'writing the final results_ratio03.csv and coverage.md'
"$PYTHON" scripts/ratio03_collect.py \
  --config "$CONFIG" --output-dir "$ROOT" --storage "sqlite:///$ROOT/studies.db" \
  --prepare-dir "$PREP" --oom-retries "$OOM_RETRIES" --final > "$ROOT/collect_final.log" 2>&1
tail -n 1 "$ROOT/collect_final.log" >> "$ROOT/status_history.jsonl"
log 'done'
