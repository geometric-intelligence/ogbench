# September 24 Optuna campaign runbook

This campaign runs full-factorial structural ablations across six datasets with a
constrained seven-trial budget per cell. It contains 2,448 independent studies,
17,136 Optuna trials, and 85,680 five-fold training runs.
Every study belongs to exactly one of seven deterministic virtual shards.

For the experiment rationale and collaborator overview, see
[`sep24_optuna_collaborator_guide.md`](sep24_optuna_collaborator_guide.md).

## 1. Rules for every server

- Use the exact same git commit and Python environment on all three servers.
- Put datasets, caches, SQLite, logs, and checkpoints on that server's local scratch disk.
- Do not put SQLite on NFS and do not assign a shard to more than one server.
- Start with two jobs per GPU. Each training process is forced to one CPU thread.
- Use the same W&B project (`ogbench_sep24_factorial_optuna`) on every server.

Set these server-specific variables in each shell:

```bash
export REPO=/path/to/bgbench
export PYTHON=/path/to/bgbench-python
export DATA_ROOT=/local/scratch/path/ogbench
export SWEEP_ROOT="$DATA_ROOT/search_results/sep24_factorial_optuna"
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
```

## 2. Parka: shards 0, 1, and 2

Parka defaults:

```bash
export REPO=/home/gbg141/bgbench
export PYTHON=/home/gbg141/miniconda3/envs/bgbench/bin/python
export DATA_ROOT=/scratch/lcornelis/ogbench
export SWEEP_ROOT="$DATA_ROOT/search_results/sep24_factorial_optuna"
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

Parka owns 1,052 studies and 36,820 expected fold runs.

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

A30 server A owns 676 studies and 23,660 expected fold runs.

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

A30 server B owns 720 studies and 25,200 expected fold runs.

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

## 6. Final collection

On each A30 server, run the status command with `--export`, then copy compact artifacts
to Parka. Replace hostnames and remote paths:

```bash
mkdir -p /scratch/lcornelis/ogbench/search_results/sep24_factorial_collected/a30-a
mkdir -p /scratch/lcornelis/ogbench/search_results/sep24_factorial_collected/a30-b

rsync -av \
  A30_SERVER_A:'/remote/sweep/root/*.csv' \
  /scratch/lcornelis/ogbench/search_results/sep24_factorial_collected/a30-a/

rsync -av \
  A30_SERVER_B:'/remote/sweep/root/*.csv' \
  /scratch/lcornelis/ogbench/search_results/sep24_factorial_collected/a30-b/
```

If a launcher's final CSVs are not present, use its `live_*.csv` exports. Keep the
server-local SQLite databases as the authoritative resumable records.

Merge disjoint final trial tables on Parka:

```bash
export COLLECTED=/scratch/lcornelis/ogbench/search_results/sep24_factorial_collected
"$PYTHON" - <<'PY'
from pathlib import Path
import pandas as pd

root = Path('/scratch/lcornelis/ogbench/search_results/sep24_factorial_collected')
parka = Path('/scratch/lcornelis/ogbench/search_results/sep24_factorial_optuna')
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
