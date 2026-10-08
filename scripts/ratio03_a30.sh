#!/usr/bin/env bash
# Run this server's ratio-0.3 queue on an A30 server (Frank or Hall) until STOP_AT, then
# write the final results and upload them as a W&B artifact for the merge on Parka.
#
# Needs a worktree at the guille/ratio03_transfer commit and the downloaded
# ratio03-a30-handoff artifact in $HANDOFF; see A30_RATIO03_COMMANDS.md in the artifact.
#
# Start detached (survives logout):
#   nohup setsid env SERVER=frank PYTHON=... DATA_ROOT=... HANDOFF=... scripts/ratio03_a30.sh \
#     > "$ROOT/a30.log" 2>&1 < /dev/null & echo $! > "$ROOT/a30.pid"
# Stop early; the results written so far are still collected and uploaded:
#   kill -TERM "$(cat "$ROOT/a30.pid")"
set -euo pipefail

SERVER=${SERVER:?set SERVER=frank or SERVER=hall}
REPO=${REPO:-$(cd "$(dirname "$0")/.." && pwd)}
PYTHON=${PYTHON:?set PYTHON to the bgbench Python interpreter}
DATA_ROOT=${DATA_ROOT:?set DATA_ROOT to the local-scratch ogbench root}
HANDOFF=${HANDOFF:?set HANDOFF to the downloaded handoff artifact directory}
ROOT=${ROOT:-$DATA_ROOT/search_results/sep24_ratio03_transfer_$SERVER}
CONFIG=${CONFIG:-$REPO/configs/hparams_search/sep24_ratio03_transfer.yaml}
STOP_AT=${STOP_AT:-2026-09-25 06:59}
GPUS=${GPUS:-0 1 2 3 4 5 6 7}
JOBS_PER_GPU=${JOBS_PER_GPU:-2}
MIN_FREE_MIB=${MIN_FREE_MIB:-6000}
OOM_RETRIES=${OOM_RETRIES:-3}
OOM_MIN_FREE_MIB=${OOM_MIN_FREE_MIB:-12000}
ARTIFACT_PROJECT=${ARTIFACT_PROJECT:-bioshape-lab/ogbench_sep24_ratio03_transfer}
UPLOAD=${UPLOAD:-1}

cd "$REPO"
log() { printf '%s a30 %s: %s\n' "$(date '+%F %T')" "$SERVER" "$*"; }

(cd "$HANDOFF" && sha256sum -c --quiet SHA256SUMS)
mkdir -p "$ROOT"

log "queue $HANDOFF/$SERVER.txt until $STOP_AT at $JOBS_PER_GPU jobs/GPU on GPUs $GPUS"
env REPO="$REPO" PYTHON="$PYTHON" DATA_ROOT="$DATA_ROOT" ROOT="$ROOT" PREP="$HANDOFF" \
  MANIFEST="$HANDOFF/$SERVER.txt" CONFIG="$CONFIG" STOP_AT="$STOP_AT" GPUS="$GPUS" \
  JOBS_PER_GPU="$JOBS_PER_GPU" MIN_FREE_MIB="$MIN_FREE_MIB" OOM_RETRIES="$OOM_RETRIES" \
  OOM_MIN_FREE_MIB="$OOM_MIN_FREE_MIB" \
  bash scripts/ratio03_extra.sh >> "$ROOT/extra.log" 2>&1 &
queue=$!
trap 'log "stop requested"; kill -TERM "$queue" 2> /dev/null || true' TERM INT
while kill -0 "$queue" 2> /dev/null; do
  wait "$queue" || true
done
trap - TERM INT
log 'queue finished'

log 'writing the final results_ratio03.csv and coverage.md'
"$PYTHON" scripts/ratio03_collect.py \
  --config "$CONFIG" --output-dir "$ROOT" --storage "sqlite:///$ROOT/studies.db" \
  --prepare-dir "$HANDOFF" --oom-retries "$OOM_RETRIES" --final > "$ROOT/collect_final.log" 2>&1
tail -n 1 "$ROOT/collect_final.log"

upload_dir="$ROOT/upload"
rm -rf "$upload_dir"
mkdir -p "$upload_dir"
for name in results_ratio03.csv coverage.md status_latest.json trials.csv best_trials.csv \
  fold_attempts.csv failures.csv extra.log; do
  if [[ -f "$ROOT/$name" ]]; then
    cp "$ROOT/$name" "$upload_dir/"
  fi
done
"$PYTHON" - "$ROOT" "$upload_dir" << 'PY'
import sqlite3
import sys
from pathlib import Path

root, upload = Path(sys.argv[1]), Path(sys.argv[2])
for name in ('studies.db', 'run_ledger.sqlite3'):
    if (root / name).exists():
        with sqlite3.connect(root / name) as source, sqlite3.connect(upload / name) as target:
            source.backup(target)
PY
(cd "$upload_dir" && sha256sum -- * > SHA256SUMS)

if [[ "$UPLOAD" != 1 ]]; then
  log "UPLOAD=$UPLOAD; results are in $upload_dir"
  exit 0
fi
log "uploading $upload_dir to $ARTIFACT_PROJECT/ratio03-a30-results-$SERVER"
"$PYTHON" - "$ARTIFACT_PROJECT" "$SERVER" "$upload_dir" << 'PY'
import sys
import time

import wandb

project_path, server, upload = sys.argv[1:]
entity, project = project_path.split('/')
for attempt in range(1, 6):
    try:
        run = wandb.init(
            entity=entity,
            project=project,
            job_type='ratio03-a30-results',
            name=f'ratio03-a30-results-{server}',
        )
        artifact = wandb.Artifact(f'ratio03-a30-results-{server}', type='results')
        artifact.add_dir(upload)
        run.log_artifact(artifact)
        run.finish()
        break
    except Exception as error:  # noqa: BLE001
        print(f'upload attempt {attempt} failed: {error}', flush=True)
        time.sleep(60 * attempt)
else:
    sys.exit(f'upload failed; upload {upload} manually')
PY
log 'done'
