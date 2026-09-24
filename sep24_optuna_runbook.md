# September 24 Optuna campaign runbook

This campaign runs full-factorial structural ablations across six datasets with a
constrained seven-trial budget per cell. It contains 2,448 independent studies,
17,136 Optuna trials, and 85,680 five-fold training runs.
Every study belongs to exactly one of seven deterministic virtual shards.

For the experiment rationale and collaborator overview, see
[`sep24_optuna_collaborator_guide.md`](sep24_optuna_collaborator_guide.md).

> **September 22 operators:** Sections 2–4 describe the original shard launches and are
> historical. Use section 6 and the handoff's `A30_COMMANDS.md` for Frank/Hall remaining
> work. Starting an original shard command would duplicate completed studies.

## 1. Rules for every server

- Use the exact same git commit and Python environment on all three servers.
- Put datasets, caches, SQLite, logs, and checkpoints on that server's local scratch disk.
- Do not put SQLite on NFS and do not assign a shard to more than one server.
- Start with two jobs per GPU. Each training process is forced to one CPU thread.
- Use the same W&B project (`ogbench_sep24_factorial_tc10`) on every server.
- Do not merge this campaign with the archived static-threshold campaign. The cache,
  study, database, ledger, log, run, and W&B identities are intentionally separate.

On Parka, the superseded campaign is preserved at
`/scratch/lcornelis/ogbench/search_results/sep24_factorial_optuna_static_thresholds_archived_20260915`.
It used static dataset/ratio/method thresholds and is retained only for auditability.

Set these server-specific variables in each shell:

```bash
set -euo pipefail

export REPO=/path/to/bgbench
export PYTHON=/path/to/bgbench-python
export DATA_ROOT=/local/scratch/path/ogbench
export SWEEP_ROOT="$DATA_ROOT/search_results/sep24_factorial_tc10"
export CONFIG="$REPO/configs/hparams_search/sep24_factorial_optuna.yaml"
export STORAGE="sqlite:///$SWEEP_ROOT/studies.db"
export PATH="$(dirname "$PYTHON"):$PATH"
cd "$REPO"
```

Verify the host before starting:

```bash
git rev-parse HEAD
"$PYTHON" --version
"$PYTHON" -c "import optuna, torch; print('optuna', optuna.__version__, 'cuda', torch.cuda.device_count())"
"$PYTHON" -c "import wandb; assert wandb.api.api_key; print('W&B authentication available')"
df -h "$DATA_ROOT"
nvidia-smi
```

Check out the campaign branch after the factorial follow-up commit has been pushed:

```bash
cd "$REPO"
git fetch origin
git checkout guille/kfold_experiments
git pull --ff-only
test -f configs/hparams_search/sep24_factorial_optuna.yaml
git status --short
```

Do not launch until the fold-local WGCNA changes are committed and pushed. Record
`git rev-parse HEAD` on every server and require an exact match; the earlier
`3d94362` revision is insufficient.

## 2. Parka: shards 0, 1, and 2

Parka defaults:

```bash
export REPO=/home/gbg141/bgbench
export PYTHON=/home/gbg141/miniconda3/envs/bgbench/bin/python
export DATA_ROOT=/scratch/lcornelis/ogbench
export SWEEP_ROOT="$DATA_ROOT/search_results/sep24_factorial_tc10"
export CONFIG="$REPO/configs/hparams_search/sep24_factorial_optuna.yaml"
export STORAGE="sqlite:///$SWEEP_ROOT/studies.db"
export PATH="$(dirname "$PYTHON"):$PATH"
cd "$REPO"
```

Warm all Parka shard caches once:

```bash
PYTHONUNBUFFERED=1 "$PYTHON" scripts/optuna_search.py \
  --config "$CONFIG" \
  --root-dir "$DATA_ROOT" \
  --output-dir "$SWEEP_ROOT" \
  --storage "$STORAGE" \
  --num-shards 7 \
  --shard-indices 0 1 2 \
  --warmup-jobs 8 \
  --warmup-only
```

Launch on the four currently free GPUs:

```bash
nohup setsid env PYTHONUNBUFFERED=1 "$PYTHON" scripts/optuna_search.py \
  --config "$CONFIG" \
  --root-dir "$DATA_ROOT" \
  --output-dir "$SWEEP_ROOT" \
  --storage "$STORAGE" \
  --num-shards 7 \
  --shard-indices 0 1 2 \
  --gpus 0 1 2 7 \
  --jobs-per-gpu 2 \
  --skip-warmup \
  --retry-failed \
  >>"$SWEEP_ROOT/launcher.log" 2>&1 &
echo $! >"$SWEEP_ROOT/launcher.pid"
```

When GPUs 3, 4, 5, and 6 become free, stop the Parka launcher and its training children:

```bash
kill -TERM -- -"$(cat "$SWEEP_ROOT/launcher.pid")"
while kill -0 "$(cat "$SWEEP_ROOT/launcher.pid")" 2>/dev/null; do sleep 5; done
```

Wait at least six minutes for the five-minute Optuna heartbeat grace period, then resume
the same shards on all eight GPUs:

```bash
nohup setsid env PYTHONUNBUFFERED=1 "$PYTHON" scripts/optuna_search.py \
  --config "$CONFIG" \
  --root-dir "$DATA_ROOT" \
  --output-dir "$SWEEP_ROOT" \
  --storage "$STORAGE" \
  --num-shards 7 \
  --shard-indices 0 1 2 \
  --gpus 0 1 2 3 4 5 6 7 \
  --jobs-per-gpu 2 \
  --skip-warmup \
  --retry-failed \
  >>"$SWEEP_ROOT/launcher.log" 2>&1 &
echo $! >"$SWEEP_ROOT/launcher.pid"
```

Parka owns 1,082 studies and 37,870 expected fold runs.

## 3. A30 server A: shards 3 and 4

Set the four server-specific paths from section 1, then warm its local cache:

```bash
PYTHONUNBUFFERED=1 "$PYTHON" scripts/optuna_search.py \
  --config "$CONFIG" \
  --root-dir "$DATA_ROOT" \
  --output-dir "$SWEEP_ROOT" \
  --storage "$STORAGE" \
  --num-shards 7 \
  --shard-indices 3 4 \
  --warmup-jobs 16 \
  --warmup-only
```

Launch all eight A30s:

```bash
nohup setsid env PYTHONUNBUFFERED=1 "$PYTHON" scripts/optuna_search.py \
  --config "$CONFIG" \
  --root-dir "$DATA_ROOT" \
  --output-dir "$SWEEP_ROOT" \
  --storage "$STORAGE" \
  --num-shards 7 \
  --shard-indices 3 4 \
  --gpus 0 1 2 3 4 5 6 7 \
  --jobs-per-gpu 2 \
  --skip-warmup \
  --retry-failed \
  >>"$SWEEP_ROOT/launcher.log" 2>&1 &
echo $! >"$SWEEP_ROOT/launcher.pid"
```

A30 server A owns 655 studies and 22,925 expected fold runs.

## 4. A30 server B: shards 5 and 6

Set the four server-specific paths from section 1, then warm its local cache:

```bash
PYTHONUNBUFFERED=1 "$PYTHON" scripts/optuna_search.py \
  --config "$CONFIG" \
  --root-dir "$DATA_ROOT" \
  --output-dir "$SWEEP_ROOT" \
  --storage "$STORAGE" \
  --num-shards 7 \
  --shard-indices 5 6 \
  --warmup-jobs 16 \
  --warmup-only
```

Launch all eight A30s:

```bash
nohup setsid env PYTHONUNBUFFERED=1 "$PYTHON" scripts/optuna_search.py \
  --config "$CONFIG" \
  --root-dir "$DATA_ROOT" \
  --output-dir "$SWEEP_ROOT" \
  --storage "$STORAGE" \
  --num-shards 7 \
  --shard-indices 5 6 \
  --gpus 0 1 2 3 4 5 6 7 \
  --jobs-per-gpu 2 \
  --skip-warmup \
  --retry-failed \
  >>"$SWEEP_ROOT/launcher.log" 2>&1 &
echo $! >"$SWEEP_ROOT/launcher.pid"
```

A30 server B owns 711 studies and 24,885 expected fold runs.

## 5. Monitoring and partial exports

Run the matching status command on each server. Change only `--shard-indices`.

```bash
"$PYTHON" scripts/optuna_status.py \
  --config "$CONFIG" \
  --root-dir "$DATA_ROOT" \
  --output-dir "$SWEEP_ROOT" \
  --storage "$STORAGE" \
  --num-shards 7 \
  --shard-indices 0 1 2 \
  --window-hours 6 \
  --export
```

Operational checks:

```bash
tail -f "$SWEEP_ROOT/launcher.log"
nvidia-smi
ps -fp "$(cat "$SWEEP_ROOT/launcher.pid")"
```

After one hour on each A30 server, record `folds_per_hour` from the status command. Keep
two jobs/GPU unless a controlled comparison shows that one or three jobs/GPU produces
more completed folds per wall-clock hour.

Campaign checkpoints, measured by the sum of `completed_folds` from all three servers:

- Sep 17: at least 25,704 folds (30%).
- Sep 20: at least 59,976 folds (70%).
- Sep 22: at least 81,396 folds (95%).
- Sep 23: stop new work only after all study budgets are complete; rerun failures, export,
  collect, and validate final tables.

If a launcher exits, use the same launch command with `--retry-failed`. Completed folds
are read from the ledger and are not repeated.

## 6. September 22 emergency rebalance

Parka was frozen at `2026-09-22T17:33:03Z`. The immutable handoff is:

```text
/scratch/lcornelis/ogbench/search_results/sep24_factorial_tc10_rebalance_20260922
```

It excludes 492 completed Parka studies and assigns the 590 incomplete studies without
overlap:

- Parka: 295 studies / 9,459 estimated remaining folds.
- Frank: 148 studies / 4,709 estimated remaining folds.
- Hall: 147 studies / 4,712 estimated remaining folds.

Parka uses its original sweep root and `parka.txt`. Frank and Hall must use new local
rebalance roots; do not overwrite their completed shard databases.

Parka continuation command (four jobs per A100):

```bash
export PARKA_SWEEP=/scratch/lcornelis/ogbench/search_results/sep24_factorial_tc10
export PARKA_HANDOFF=/scratch/lcornelis/ogbench/search_results/sep24_factorial_tc10_rebalance_20260922
nohup setsid env PYTHONUNBUFFERED=1 \
  /home/gbg141/miniconda3/envs/bgbench/bin/python scripts/optuna_search.py \
  --config configs/hparams_search/sep24_factorial_optuna.yaml \
  --root-dir /scratch/lcornelis/ogbench \
  --output-dir "$PARKA_SWEEP" \
  --storage "sqlite:///$PARKA_SWEEP/studies.db" \
  --studies-file "$PARKA_HANDOFF/parka.txt" \
  --num-shards 7 \
  --shard-indices 0 1 2 \
  --gpus 0 1 2 3 4 5 6 7 \
  --jobs-per-gpu 4 \
  --skip-warmup \
  --retry-failed \
  >>"$PARKA_SWEEP/rebalance_parka.log" 2>&1 </dev/null &
echo $! >"$PARKA_SWEEP/launcher.pid"
```

On Frank or Hall, set server-local paths and select the matching manifest:

```bash
set -euo pipefail

export REPO=/path/to/bgbench
export PYTHON=/path/to/bgbench-python
export DATA_ROOT=/local/scratch/path/ogbench
export SERVER=frank                         # use hall on Hall
export ARTIFACT=bioshape-lab/ogbench_sep24_factorial_tc10/sep24-factorial-tc10-rebalance-20260922:v2
export BUNDLE="$DATA_ROOT/search_results/sep24_factorial_tc10_rebalance_20260922"
export SWEEP_ROOT="$DATA_ROOT/search_results/sep24_factorial_tc10_rebalance_20260922_$SERVER"
export CONFIG="$BUNDLE/sep24_factorial_optuna.yaml"
export MANIFEST="$BUNDLE/$SERVER.txt"
export STORAGE="sqlite:///$SWEEP_ROOT/studies.db"
export PATH="$(dirname "$PYTHON"):$PATH"

test "$(git -C "$REPO" rev-parse HEAD)" = 331f9b71442cb7ae022bb527a96c3360587d0ad2
"$PYTHON" -c "import optuna, torch, wandb; assert wandb.api.api_key; assert torch.cuda.device_count() == 8"
df -h "$DATA_ROOT"
if test -e "$SWEEP_ROOT/studies.db" || test -e "$SWEEP_ROOT/launcher.pid"; then
  echo "Refusing to overwrite an existing rebalance root: $SWEEP_ROOT" >&2
  exit 1
fi
```

Download the private W&B handoff artifact; Parka filesystem access is not required:

```bash
mkdir -p "$BUNDLE" "$SWEEP_ROOT"
ARTIFACT="$ARTIFACT" BUNDLE="$BUNDLE" "$PYTHON" - <<'PY'
import os
import wandb

wandb.Api().artifact(os.environ['ARTIFACT']).download(root=os.environ['BUNDLE'])
PY
(cd "$BUNDLE" && sha256sum -c SHA256SUMS)
cp "$BUNDLE/code/optuna_search.py" "$REPO/scripts/optuna_search.py"
cp "$BUNDLE/code/optuna_status.py" "$REPO/scripts/optuna_status.py"
echo 'fd8b4227a1ab9f9cc9fdb24cdfb1a04793f40720151cc7019aa6e82bdf7a35ea  scripts/optuna_search.py' | (cd "$REPO" && sha256sum -c -)
echo 'dffde2ac19c4dc18c2a345eb5b5b6bdf1e4c83d85d66399b9ad079c8d7f06c6f  scripts/optuna_status.py' | (cd "$REPO" && sha256sum -c -)
cp "$BUNDLE/studies.db" "$SWEEP_ROOT/studies.db"
cp "$BUNDLE/run_ledger.sqlite3" "$SWEEP_ROOT/run_ledger.sqlite3"
```

Warm only that server's assigned caches:

```bash
PYTHONUNBUFFERED=1 "$PYTHON" "$REPO/scripts/optuna_search.py" \
  --config "$CONFIG" \
  --root-dir "$DATA_ROOT" \
  --output-dir "$SWEEP_ROOT" \
  --storage "$STORAGE" \
  --studies-file "$MANIFEST" \
  --num-shards 7 \
  --shard-indices 0 1 2 \
  --warmup-jobs 16 \
  --warmup-only
```

Launch the assigned continuation:

```bash
nohup setsid env PYTHONUNBUFFERED=1 "$PYTHON" "$REPO/scripts/optuna_search.py" \
  --config "$CONFIG" \
  --root-dir "$DATA_ROOT" \
  --output-dir "$SWEEP_ROOT" \
  --storage "$STORAGE" \
  --studies-file "$MANIFEST" \
  --num-shards 7 \
  --shard-indices 0 1 2 \
  --gpus 0 1 2 3 4 5 6 7 \
  --jobs-per-gpu 2 \
  --skip-warmup \
  --retry-failed \
  >>"$SWEEP_ROOT/launcher.log" 2>&1 </dev/null &
echo $! >"$SWEEP_ROOT/launcher.pid"
```

Monitor only the assigned manifest:

```bash
"$PYTHON" "$REPO/scripts/optuna_status.py" \
  --config "$CONFIG" \
  --root-dir "$DATA_ROOT" \
  --output-dir "$SWEEP_ROOT" \
  --storage "$STORAGE" \
  --studies-file "$MANIFEST" \
  --num-shards 7 \
  --shard-indices 0 1 2 \
  --window-hours 2 \
  --export
```

The three SQLite files deliberately diverge after the handoff. Never copy them back over
one another and never run a study from more than one manifest.

## 7. Final collection

On each A30 server, run the status command with `--export`, then copy compact artifacts
to Parka. Replace hostnames and remote paths:

```bash
mkdir -p /scratch/lcornelis/ogbench/search_results/sep24_factorial_tc10_collected/a30-a
mkdir -p /scratch/lcornelis/ogbench/search_results/sep24_factorial_tc10_collected/a30-b

rsync -av \
  A30_SERVER_A:'/remote/sweep/root/*.csv' \
  /scratch/lcornelis/ogbench/search_results/sep24_factorial_tc10_collected/a30-a/

rsync -av \
  A30_SERVER_B:'/remote/sweep/root/*.csv' \
  /scratch/lcornelis/ogbench/search_results/sep24_factorial_tc10_collected/a30-b/
```

If a launcher's final CSVs are not present, use its `live_*.csv` exports. Keep the
server-local SQLite databases as the authoritative resumable records.

The original Frank/Hall shard exports remain disjoint from Parka. For the emergency
continuations, do not simply concatenate the frozen snapshot with continuation rows.
For each manifest, continuation rows replace snapshot rows having those study names;
the snapshot remains authoritative for the 492 excluded completed studies.

Merge disjoint final trial tables on Parka:

```bash
export COLLECTED=/scratch/lcornelis/ogbench/search_results/sep24_factorial_tc10_collected
"$PYTHON" - <<'PY'
from pathlib import Path
import pandas as pd

root = Path('/scratch/lcornelis/ogbench/search_results/sep24_factorial_tc10_collected')
parka = Path('/scratch/lcornelis/ogbench/search_results/sep24_factorial_tc10')
sources = [parka, root / 'a30-a', root / 'a30-b']
for name in ('trials.csv', 'best_trials.csv', 'fold_attempts.csv', 'failures.csv'):
    paths = [directory / name for directory in sources if (directory / name).is_file()]
    if paths:
        pd.concat([pd.read_csv(path) for path in paths], ignore_index=True).to_csv(
            root / name, index=False
        )
PY
```

Verify that the merged `trials.csv` contains 2,448 distinct study names and that every
study has 7 complete trials before using `best_trials.csv` for final analysis.

## 8. Ratio 0.3 transfer (September 23 to 25, Parka only)

The factorial campaign omitted `node_sample_ratio: 0.3`. There was no time for a seven-trial
search, so each of the 816 ratio-0.3 cells evaluates **one transferred configuration** with the
same 5-fold CV, fold-local WGCNA (target connectivity 0.10), STRING threshold 0.4, and
3,600 s timeout. Label these results as transferred, not tuned, in the paper.

The configuration for a cell is the best complete trial of, in order:

1. the same cell at ratio 0.5, then 0.8, then 1.0;
2. a sibling cell (same model, dataset, experiment, and adjacency; other selection method),
   at ratio 0.5, then 0.8, then 1.0;
3. a cell with another adjacency method, then another experiment.

`source_rule` records which rule applied. On September 23: 362 cells used their own ratio-0.5
trial, 156 their ratio-0.8 trial, 88 their ratio-1.0 trial, and 210 a sibling cell. Only
Parka-origin studies were on W&B or in Parka's ledger. The Frank/Hall shard studies were
not available.

Identities and paths:

- Config: `configs/hparams_search/sep24_ratio03_transfer.yaml`, study prefix
  `sep24tc10ratio03transfer`, W&B project `bioshape-lab/ogbench_sep24_ratio03_transfer`.
- Root: `/scratch/lcornelis/ogbench/search_results/sep24_ratio03_transfer` with its own
  `studies.db` and `run_ledger.sqlite3`.
- Candidates: `candidates_r05/` (`candidates.json`, `candidates.csv`, `priority.txt`,
  `SHA256SUMS.candidates`), built by `scripts/ratio03_prepare.py`. The cross-check against
  Parka's studies found 0 best-trial mismatches in 257 studies and a maximum fold-mean gap
  of 4e-11. Ignore the `prepare/` directory; it holds an abandoned first attempt.
- Priority: cheapest models first (mlp, sagn, gcn, chebnet, graph_sage, gatv2, gin, gatv4,
  gps), then cheapest cells, so partial results are complete per model.

Ratio 0.3 is the most expensive ratio. Graphs have `int(n_train / node_sample_ratio)` nodes,
so 1.67 times as many nodes as ratio 0.5 and 2.67 times as many as ratio 0.8. Measured
fold times are 1.1 to 1.3 times the source-run times with STRING graphs, and 1.5 to 3 times
with WGCNA graphs, where the edge count grows with the square of the node count. On the
morning of September 24 the projection for Friday 07:00 was: mlp, sagn, gcn, chebnet, and
graph_sage complete; about half of gatv2; nothing for gin, gatv4, or gps. The heavier models
on the large WGCNA graphs (BRCA, ADNI-neuromed) often hit the 3,600 s timeout. With the extra
queue below, the projection became: mlp through chebnet complete, about 60% of graph_sage,
little of gatv2, and about 45 cells each of gin, gatv4, and gps.

The supervisor runs detached and keeps relaunching the launcher with `--retry-failed`. It
does not start a new pass after Friday 07:00 PDT, but it does not interrupt a running pass.
It runs 2 jobs per GPU on all eight GPUs. A fold only starts on a GPU with
at least 30 GB free. Out-of-memory failures are retried up to 3 times per fold (on a GPU with
60 GB free when available) without using up the normal one-retry budget:

```bash
export ROOT=/scratch/lcornelis/ogbench/search_results/sep24_ratio03_transfer
cd /home/gbg141/bgbench
nohup setsid env PREP="$ROOT/candidates_r05" scripts/ratio03_supervisor.sh \
  > "$ROOT/supervisor.log" 2>&1 < /dev/null &
echo $! > "$ROOT/supervisor.pid"
```

Stop it (and every launcher and training job it started) with
`kill -TERM -- -"$(cat "$ROOT/supervisor.pid")"`, then wait at least 7 minutes before
relaunching. Override `JOBS_PER_GPU`, `GPUS`, `MIN_FREE_MIB`, or `DEADLINE` through the
environment. The retry pass for the seven-trial campaign is deferred until every ratio-0.3
cell is complete or permanently failed.

`scripts/ratio03_stop.sh` enforces the deadline. It waits until `STOP_AT` (default
Friday 07:00), stops the supervisor's process group, and writes the final
`results_ratio03.csv` and `coverage.md` with `ratio03_collect.py --final`, which labels
unfinished cells `incomplete`. It was started on September 24 with:

```bash
nohup setsid scripts/ratio03_stop.sh > "$ROOT/stop.log" 2>&1 < /dev/null &
echo $! > "$ROOT/stop.pid"
```

Cancel it with `kill "$(cat "$ROOT/stop.pid")"` to let the pass run until the queue drains.

So that gin, gatv4, and gps also get ratio-0.3 results, `scripts/ratio03_extra.sh` runs a
second launcher with one job per GPU (a third job on each GPU) next to the supervisor. It
shares `studies.db`, the ledger, and `candidates.json`, and reads the balanced manifest
`$ROOT/extra_balanced/priority.txt`. That manifest takes the cheapest remaining cell of each
model in turn, and puts the 59 cells whose projected fold time exceeds the 3,600 s timeout
last (`extra_queue.csv` has the projections). The supervisor's queue does not reach these
models before the deadline, so the two launchers never run the same study. Jobs on one GPU
share its time, so the extra queue takes roughly a third of the GPU time from the main
queue. It stops itself at Friday 06:59, before the final collect. It was started on
September 24 at 09:26 with:

```bash
nohup setsid scripts/ratio03_extra.sh > "$ROOT/extra.log" 2>&1 < /dev/null &
echo $! > "$ROOT/extra.pid"
```

Stop it early with `kill -TERM "$(cat "$ROOT/extra.pid")"`. It also stops its launcher's
process group. Launcher output is in `$ROOT/extra_launcher.log`.

Monitoring (the supervisor also does this every 30 minutes):

```bash
"$PYTHON" scripts/ratio03_collect.py --output-dir "$ROOT" \
  --storage "sqlite:///$ROOT/studies.db" --prepare-dir "$ROOT/candidates_r05"
cat "$ROOT/coverage.md"
tail -n 3 "$ROOT/status_history.jsonl" "$ROOT/supervisor.log"
```

Outputs in `$ROOT`: `results_ratio03.csv` (one row per cell in priority order, with status,
5-fold mean and std, fold scores, and source trial), `coverage.md` (completed cells by model
and dataset), `status_latest.json`, `trials.csv`, `best_trials.csv`, `fold_attempts.csv`,
`failures.csv`, `gpu_usage.log`, and `launcher.log`.

To merge with the other ratios, append the `complete` rows of `results_ratio03.csv` to the
factorial `best_trials.csv` by cell (model, dataset, experiment, adjacency method, selection
method). The ratio-0.3 study names are distinct from the factorial study names. Keep the
`protocol` column (`transferred_single_config`) so tables can mark the ratio-0.3 column.
