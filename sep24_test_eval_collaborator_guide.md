# September 24 test evaluation: collaborator guide

## Executive summary

Validation search is finished per server when every assigned study has a COMPLETE
best trial (or only documented unresolved failures remain). The next step is to
score **that same trial** on the held-out test split, using the existing fold
checkpoints. Do not retrain and do not pick a global model×dataset winner first.

Each server tests only the studies it trained. Study ownership is still
shard-disjoint, so the three test CSVs can later be concatenated on one machine
and plotted with [`plotting/README.md`](plotting/README.md).

The launcher is [`scripts/optuna_test_eval.py`](scripts/optuna_test_eval.py).

## What this job does

For every COMPLETE row in that server's `live_best_trials.csv`:

- Find the five fold checkpoints under `$SEARCH_ROOT/runs/<study>/.../checkpoints/`.
- Run `python -m ogbench.run` with `train=false test=true ckpt_path=...`.
- Record `test/f1_macro` (and related metrics) in `$TEST_ROOT/test_ledger.sqlite3`.

Test jobs = (COMPLETE studies on this server) × 5 folds. Approximate maxima:

| Server        | Shards | Expected studies | Max test fold jobs |
| ------------- | ------ | ---------------- | ------------------ |
| Parka         | 0 1 2  | 1,052            | 5,260              |
| A30 A / Hall  | 3 4    | 676              | 3,380              |
| A30 B / Frank | 5 6    | 720              | 3,600              |

If some studies never completed, the launcher tests only the COMPLETE ones.

## Code version

Test eval and the Sep 24 plotting scripts live on **`louis/fold_test_eval`**,
branched from the search campaign. Search may still be on
`guille/kfold_experiments`; do not run test eval from that older tip.

Required files on this branch:

- `scripts/optuna_test_eval.py`
- `ogbench/run.py` (prints `metrics_payload`)
- `ogbench/utils/hparam_search.py` (`parse_metrics_payload`)
- `plotting/plot_sep24_test_best.py` and `plotting/plot_sep24_val_best.py`

```bash
git fetch origin
git checkout louis/fold_test_eval
git pull --ff-only
test -f scripts/optuna_test_eval.py
test -f plotting/plot_sep24_test_best.py
"$PYTHON" -c "from ogbench.utils.hparam_search import metrics_payload; print('ok')"
git rev-parse --abbrev-ref HEAD   # must print louis/fold_test_eval
```

If a server cannot pull yet, copy those files from a machine that has
`louis/fold_test_eval`, then `pip install -e .` in the same env that ran search.

## Conditions before launch (agents: check these)

Do not start test eval until every item below is true. Agents should run the
commands, not assume.

### 1. Search is no longer using the GPUs

```bash
export SEARCH_ROOT="$DATA_ROOT/search_results/sep24_factorial_optuna"
test -f "$SEARCH_ROOT/launcher.pid" && ps -fp "$(cat "$SEARCH_ROOT/launcher.pid")" || echo 'no search launcher pid'
pgrep -af 'optuna_search.py|ogbench.run' | grep -v grep || echo 'no search/train processes'
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv
```

The search PID file may still exist after the process exited. If `ps` shows no
`optuna_search.py` and no leftover `ogbench.run` from search, it is safe. On
Frank, GPU 0 may still hold deformetrica (~0.7 GiB); that is expected. Do not
use GPU 0 there.

### 2. This server's validation studies actually finished

Use the same shard indices the server searched with.

```bash
"$PYTHON" scripts/optuna_status.py \
  --config "$CONFIG" \
  --root-dir "$DATA_ROOT" \
  --output-dir "$SEARCH_ROOT" \
  --storage "$STORAGE" \
  --num-shards 7 \
  --shard-indices $SHARD_INDICES \
  --window-hours 6 \
  --export
```

Pass:

- Parka: `--shard-indices 0 1 2` and expect **1,052** studies / **36,820** folds.
- Hall: `--shard-indices 3 4` and expect **676** studies / **23,660** folds.
- Frank: `--shard-indices 5 6` and expect **720** studies / **25,200** folds.

Also inspect the search CSV:

```bash
"$PYTHON" - <<'PY'
import pandas as pd
from pathlib import Path
import os
p = Path(os.environ['SEARCH_ROOT']) / 'live_best_trials.csv'
assert p.is_file(), f'missing {p}'
df = pd.read_csv(p)
print(p)
print(df['state'].astype(str).str.upper().value_counts().to_string())
print('distinct studies', df['study_name'].nunique())
print('COMPLETE studies', (df['state'].astype(str).str.upper()=='COMPLETE').sum())
PY
```

Ready to test when:

- `completed_folds` is at (or extremely close to) `expected_folds`, **or**
- remaining gaps are only `unresolved_failures` that you have inspected, not
  still-running trials.
- `live_best_trials.csv` exists and COMPLETE count matches the studies you
  intend to test.
- The search launcher is gone.

Not ready if `optuna_search.py` is still running, COMPLETE count is far below
expected, or `live_best_trials.csv` is missing. In that case keep searching
(or `--retry-failed`); do not start test eval.

### 3. Checkpoints exist for the best-trial folds

```bash
"$PYTHON" scripts/optuna_test_eval.py \
  --config "$CONFIG" \
  --best-csv "$SEARCH_ROOT/live_best_trials.csv" \
  --search-output-dir "$SEARCH_ROOT" \
  --root-dir "$DATA_ROOT" \
  --output-dir "$TEST_ROOT" \
  --dry-run
```

The first printed line is `Studies: N | fold jobs: 5N`. Immediately after
launch (not dry-run) the script also prints `Checkpoints found for X/Y fold jobs`. `X` should equal `Y`. If many checkpoints are missing, stop and
investigate. Always pass `--require-ckpt` so missing files fail that fold
instead of silently retraining.

### 4. Environment and W&B

```bash
test -f configs/hparams_search/sep24_factorial_optuna.yaml
"$PYTHON" -c "import optuna, torch; print(optuna.__version__, torch.cuda.device_count())"
"$PYTHON" -c "import wandb; assert wandb.api.api_key; print('W&B ready')"
df -h "$DATA_ROOT"
```

Use the same env and W&B project as search (`ogbench_sep24_factorial_optuna`).

### 5. One launcher only

```bash
pgrep -af optuna_test_eval.py | grep -v grep || echo 'no test launcher'
```

Never start a second copy against the same `$TEST_ROOT`. Duplicate launchers
race the ledger.

## Server variables

```bash
export REPO=/path/to/ogbench
export PYTHON=/path/to/ogbench/env/bin/python
export DATA_ROOT=/local/scratch/path/ogbench
export SEARCH_ROOT="$DATA_ROOT/search_results/sep24_factorial_optuna"
export TEST_ROOT="$DATA_ROOT/search_results/sep24_factorial_optuna_test"
export CONFIG="$REPO/configs/hparams_search/sep24_factorial_optuna.yaml"
export STORAGE="sqlite:///$SEARCH_ROOT/studies.db"
export PATH="$(dirname "$PYTHON"):$PATH"
cd "$REPO"
mkdir -p "$TEST_ROOT"
```

Known defaults:

```text
Parka:  REPO=/home/gbg141/bgbench
        DATA_ROOT=/scratch/lcornelis/ogbench
        SHARD_INDICES="0 1 2"
        GPUS="0 1 2 3 4 5 6 7"   # or the subset Parka actually uses

Hall:   shards 3 4; set REPO / PYTHON / DATA_ROOT to that server's search paths
        GPUS="0 1 2 3 4 5 6 7" unless a GPU is already reserved

Frank:  REPO=/home/louisvl/ogbench
        PYTHON=/home/louisvl/miniconda/envs/ogbench/bin/python
        DATA_ROOT=/scratch/louisvl/ogbench
        SHARD_INDICES="5 6"
        GPUS="1 2 3 4 5 6 7"     # GPU 0 is forbidden
```

`$TEST_ROOT` must not be `$SEARCH_ROOT`. Checkpoints stay under search `runs/`;
test Hydra dirs and the test ledger go under `sep24_factorial_optuna_test`.

## Launch command

Paste as **one** command. Do not split the `nohup` line across prompts (a
trailing `&` on its own line does nothing useful).

Parka / Hall (all free GPUs; change `$GPUS` if needed):

```bash
nohup setsid env PYTHONUNBUFFERED=1 "$PYTHON" scripts/optuna_test_eval.py \
  --config "$CONFIG" \
  --best-csv "$SEARCH_ROOT/live_best_trials.csv" \
  --search-output-dir "$SEARCH_ROOT" \
  --root-dir "$DATA_ROOT" \
  --output-dir "$TEST_ROOT" \
  --gpus 0 1 2 3 4 5 6 7 \
  --jobs-per-gpu 2 \
  --require-ckpt \
  >>"$TEST_ROOT/launcher.log" 2>&1 </dev/null &
echo $! >"$TEST_ROOT/launcher.pid"
```

Frank (GPU 0 excluded):

```bash
nohup setsid env PYTHONUNBUFFERED=1 "$PYTHON" scripts/optuna_test_eval.py \
  --config "$CONFIG" \
  --best-csv "$SEARCH_ROOT/live_best_trials.csv" \
  --search-output-dir "$SEARCH_ROOT" \
  --root-dir "$DATA_ROOT" \
  --output-dir "$TEST_ROOT" \
  --gpus 1 2 3 4 5 6 7 \
  --jobs-per-gpu 2 \
  --require-ckpt \
  >>"$TEST_ROOT/launcher.log" 2>&1 </dev/null &
echo $! >"$TEST_ROOT/launcher.pid"
```

Confirm it detached:

```bash
ps -o pid,ppid,etime,cmd -p "$(cat "$TEST_ROOT/launcher.pid")"
```

`PPID` should become `1` after the shell exits. First log lines should show
`Studies: …`, `Checkpoints found for …/…`, `GPUs: […]`, and
`Pending fold jobs after resume: …`.

## Tracking progress

```bash
tail -f "$TEST_ROOT/launcher.log"
ps -fp "$(cat "$TEST_ROOT/launcher.pid")"
nvidia-smi
```

Joblib lines like `[Parallel(n_jobs=14)]: Done  44 tasks` count finished fold
jobs toward the pending total printed at start.

Live counts from the ledger (this is the source of truth while the job runs):

```bash
"$PYTHON" - <<'PY'
import sqlite3, os
from pathlib import Path
p = Path(os.environ['TEST_ROOT']) / 'test_ledger.sqlite3'
con = sqlite3.connect(p)
print('status', list(con.execute('SELECT status, COUNT(*) FROM test_folds GROUP BY status')))
print('unique success folds', con.execute(
    "SELECT COUNT(*) FROM (SELECT 1 FROM test_folds WHERE status='success' GROUP BY study_name, trial_number, fold)"
).fetchone()[0])
print('studies with 5 success folds', con.execute("""
SELECT COUNT(*) FROM (
  SELECT 1 FROM test_folds WHERE status='success'
  GROUP BY study_name, trial_number HAVING COUNT(DISTINCT fold) >= 5
)
""").fetchone()[0])
PY
```

`live_test_best_trials.csv` and `live_test_folds.csv` appear only after the
launcher finishes, or if you snapshot with `--export-only` (safe; does not
stop the running job):

```bash
"$PYTHON" scripts/optuna_test_eval.py \
  --config "$CONFIG" \
  --best-csv "$SEARCH_ROOT/live_best_trials.csv" \
  --search-output-dir "$SEARCH_ROOT" \
  --root-dir "$DATA_ROOT" \
  --output-dir "$TEST_ROOT" \
  --export-only
```

If the launcher dies, confirm no leftover `ogbench.run` / `optuna_test_eval.py`
processes, then rerun the **same** detached command. Successful folds are
skipped from the ledger.

Do not delete `test_ledger.sqlite3`, its `-wal`/`-shm` files, or `$TEST_ROOT/runs`.

## Intermediate plots on this server

See [`plotting/README.md`](plotting/README.md). Short form:

```bash
python plotting/plot_sep24_test_best.py \
  --best-csv "$SEARCH_ROOT/live_best_trials.csv" \
  --ledger "$TEST_ROOT/test_ledger.sqlite3" \
  --min-folds 5 \
  --out-dir plotting/plots_sep24_test
```

Those figures cover only this server's finished studies.

## Final goal: one server, then plot

When a server's test launcher exits and the ledger shows every intended study
with 5 successful folds:

1. Run `--export-only` once more so the CSVs are current.
2. Copy these files (not the checkpoints) to the plotting machine:

```text
$TEST_ROOT/live_test_best_trials.csv
$TEST_ROOT/live_test_folds.csv
```

Keep a server subfolder so they do not overwrite each other. Example on Parka:

```bash
export COLLECTED=/scratch/lcornelis/ogbench/search_results/sep24_factorial_collected
mkdir -p "$COLLECTED/parka" "$COLLECTED/hall" "$COLLECTED/frank"

# On Parka, local copy:
cp /scratch/lcornelis/ogbench/search_results/sep24_factorial_optuna_test/live_test_*.csv \
  "$COLLECTED/parka/"

# From Hall / Frank (replace hosts and remote TEST_ROOT):
rsync -av hall:'/remote/sep24_factorial_optuna_test/live_test_*.csv' "$COLLECTED/hall/"
rsync -av frank:'/scratch/louisvl/ogbench/search_results/sep24_factorial_optuna_test/live_test_*.csv' \
  "$COLLECTED/frank/"
```

3. On that one machine, plot with all three CSVs as documented in
   [`plotting/README.md`](plotting/README.md):

```bash
python plotting/plot_sep24_test_best.py \
  --csv "$COLLECTED/frank/live_test_best_trials.csv" \
  --csv "$COLLECTED/hall/live_test_best_trials.csv" \
  --csv "$COLLECTED/parka/live_test_best_trials.csv" \
  --min-folds 5 \
  --out-dir plotting/plots_sep24_test
```

Before treating those figures as final, check:

```bash
"$PYTHON" - <<'PY'
from pathlib import Path
import pandas as pd
import os
root = Path(os.environ['COLLECTED'])
frames = [pd.read_csv(p) for p in [
    root/'frank'/'live_test_best_trials.csv',
    root/'hall'/'live_test_best_trials.csv',
    root/'parka'/'live_test_best_trials.csv',
] if p.is_file()]
df = pd.concat(frames, ignore_index=True)
print('files', len(frames), 'rows', len(df), 'studies', df['study_name'].nunique())
print(df.groupby(df['n_folds'].fillna(0).astype(int)).size())
PY
```

Target: **2,448** distinct studies, all with `n_folds == 5`. If a server is
still short, plot only what exists (`--min-folds 5` already drops incomplete
studies) and wait for that server to finish.

## Who should run what

- Parka operator: test shards `0 1 2` after Parka search is done.
- Hall operator: test shards `3 4` after Hall search is done.
- Frank operator: test shards `5 6` after Frank search is done; GPUs `1–7` only.
- Campaign coordinator: collect the three `live_test_best_trials.csv` files onto
  one server and run the commands in [`plotting/README.md`](plotting/README.md).
