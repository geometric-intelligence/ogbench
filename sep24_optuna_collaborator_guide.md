# September 24 Optuna campaign: collaborator guide

## Executive summary

We need final hyperparameter and ablation results by September 24. We retained the
full-factorial structural design so that interactions between ablation factors remain
measurable, but reduced the Optuna budget from 20 to 7 trials per cell. The campaign
contains:

- 6 datasets.
- 9 models.
- 2,448 independent Optuna studies.
- 7 Optuna trials per study.
- 5 cross-validation folds per trial.
- 17,136 total Optuna trials.
- 85,680 total fold-level training runs.

Parka is rebuilding its fold-local caches and relaunching its assigned studies. Two
additional servers, each with eight A30 GPUs, must run the remaining disjoint shards.

The detailed operational commands are in
[`sep24_optuna_runbook.md`](sep24_optuna_runbook.md).

## Why we changed the trial budget

The structural choices below are scientific ablation variables, not hyperparameters that
Optuna should compare within one study:

- Dataset.
- Model.
- Readout experiment.
- Adjacency construction method.
- Node sampling ratio.
- Node selection method.

The original plan used the complete Cartesian product and 20 Optuna trials inside every
cell. That required 244,800 fold-level runs and could not finish before the deadline.
One-factor-at-a-time was considered, but rejected because it only estimates effects at
one baseline and cannot identify interactions. We therefore kept every valid structural
combination and reduced only the within-cell optimization budget.

Seven trials is a constrained hyperparameter search, not an exhaustive optimization.
Three trials initialize TPE and the remaining four are TPE-guided. Results should be
described as performance under a fixed seven-trial search budget.

## Scientific design

For every model and dataset, the campaign evaluates the Cartesian product of:

- Experiments: `omics_readout` and `no_readout`.
- Adjacency methods: `wgcna` and `string`.
- Node sample ratios: `1.0`, `0.8`, and `0.5`.
- Node selection methods: `variance`, `random`, `correlation`, and
  `distance_correlation`.

This gives 48 cells per non-MLP model/dataset. MLP cannot use `omics_readout`, so its
invalid combinations are omitted, leaving 24 cells per MLP/dataset. Each dataset has 408
studies and the six-dataset campaign has 2,448.

Within each fixed cell, Optuna tunes the optimizer and model-specific architecture
parameters. Every trial is evaluated on all five folds, and its objective is the mean
`best_val/f1_macro` across those folds.

Fixed reproducibility settings:

- Training seed: `42`.
- Folds: `0, 1, 2, 3, 4`.
- Trials per study: `7`.
- TPE sampler seed: `1234`.
- TPE startup trials: `3`.
- One CPU thread per training job.
- Zero dataloader workers in search subprocesses.

Corrected campaign identities:

- W&B project: `ogbench_sep24_factorial_tc10`.
- Study prefix: `relaunchsep24tc10factorialwgcna`.
- Per-server sweep directory: `search_results/sep24_factorial_tc10`.

WGCNA adjacency is calibrated independently inside every fold. After train-only
correction, imputation, and node selection, the strongest undirected edges are retained
to obtain the nearest possible graph connectivity to 10%. Equal-weight edges are
resolved deterministically. The requested connectivity, achieved connectivity, and
effective cutoff are saved in that fold cache's `split_info.json`. STRING continues to
use its fixed confidence threshold of `0.4`.

The campaign configuration is
[`configs/hparams_search/sep24_factorial_optuna.yaml`](configs/hparams_search/sep24_factorial_optuna.yaml).

## Distributed execution design

Study names are deterministically hashed into seven virtual shards. A study belongs to
exactly one shard, so separate servers never optimize or write the same study.

Shard ownership is fixed:

- Parka owns shards `0 1 2`: 1,082 studies and 37,870 fold runs.
- A30 server A owns shards `3 4`: 655 studies and 22,925 fold runs.
- A30 server B owns shards `5 6`: 711 studies and 24,885 fold runs.

This was the original ownership. On September 22, Frank and Hall had completed their
original shards while Parka still had 590 incomplete studies. Parka was frozen and those
remaining studies were reassigned by estimated runtime:

- Parka: 295 studies / 9,459 estimated remaining folds.
- Frank: 148 studies / 4,709 estimated remaining folds.
- Hall: 147 studies / 4,712 estimated remaining folds.

The handoff manifests are disjoint and their union is exactly the incomplete set. Parka
keeps 50% of estimated work because its A100 GPUs are faster; Frank and Hall each receive
25%. Parka uses four jobs per A100 (32 workers); Frank and Hall use two jobs per A30
(16 workers each). All three continuations use `--retry-failed`.

Each server must use its own local scratch directory and SQLite database. SQLite must not
be placed on NFS. W&B is the shared cross-server view, while each server's SQLite database
and run ledger are its authoritative resumable state.

The common frozen handoff is
`/scratch/lcornelis/ogbench/search_results/sep24_factorial_tc10_rebalance_20260922`.
Frank and Hall copy it into new server-local rebalance roots; their completed original
shard outputs must not be overwritten. Parka filesystem access is not required: the
private W&B artifact is
`bioshape-lab/ogbench_sep24_factorial_tc10/sep24-factorial-tc10-rebalance-20260922:v2`.
Detailed commands are in the runbook.

## Code version

All servers must run the same revision of `guille/kfold_experiments`. Commit `3d94362`
contains the distributed launcher but predates the fold-local WGCNA correction. Do not
launch from it. The correction, factorial configuration, and this documentation must be
committed and pushed together before collaborators launch.

On each A30 server:

```bash
git fetch origin
git checkout guille/kfold_experiments
git pull --ff-only
git status --short
test -f configs/hparams_search/sep24_factorial_optuna.yaml
```

`git status --short` should print nothing. Every operator should record
`git rev-parse HEAD` and confirm that the hashes match. Do not run from different
revisions: shard assignment, study fingerprints, cache behavior, or objective parsing
could differ.

## What the implementation adds

The campaign launcher,
[`scripts/optuna_search.py`](scripts/optuna_search.py), now provides:

- Explicit full-factorial or one-factor-at-a-time ablation construction.
- Deterministic `--num-shards` and `--shard-indices`.
- Multiple shard indices per server for weighted allocation.
- Newline-delimited `--studies-file` manifests for exact continuation ownership.
- Server-local `--root-dir`, `--output-dir`, and `--storage`.
- `--warmup-only` and `--skip-warmup`.
- Resumable Optuna studies and durable fold-level attempt history.
- Failed/interrupted trial retries that reuse successful folds.
- Strict one-thread subprocess isolation.

Training is invoked through the launcher's Python interpreter with
`python -m ogbench.run`; an activated shell or globally installed `ogbench-train`
executable is not required.

The status utility,
[`scripts/optuna_status.py`](scripts/optuna_status.py), reports:

- Expected and completed studies, trials, and folds.
- Progress by virtual shard.
- Recent fold throughput.
- ETA based on recent throughput.
- Trial state counts and unresolved failures.
- Status and exports filtered to one handoff manifest.
- Live CSV exports while a launcher is still running.

The STRING data code uses file locks, validated gzip downloads, and atomic cache writes so
parallel cache warmup cannot corrupt shared local downloads.

## Current Parka state

Parka's full-factorial campaign is running independently of Cursor under `nohup` and `setsid`.
Closing Cursor or disconnecting SSH will not stop it.

Parka paths:

```text
Data/cache root: /scratch/lcornelis/ogbench
Sweep root:      /scratch/lcornelis/ogbench/search_results/sep24_factorial_tc10
Optuna database: /scratch/lcornelis/ogbench/search_results/sep24_factorial_tc10/studies.db
Run ledger:      /scratch/lcornelis/ogbench/search_results/sep24_factorial_tc10/run_ledger.sqlite3
Launcher log:    /scratch/lcornelis/ogbench/search_results/sep24_factorial_tc10/launcher.log
Launcher PID:    /scratch/lcornelis/ogbench/search_results/sep24_factorial_tc10/launcher.pid
Rebalance log:   /scratch/lcornelis/ogbench/search_results/sep24_factorial_tc10/rebalance_parka.log
Parka manifest:  /scratch/lcornelis/ogbench/search_results/sep24_factorial_tc10_rebalance_20260922/parka.txt
```

The superseded static-threshold factorial output is preserved at
`/scratch/lcornelis/ogbench/search_results/sep24_factorial_optuna_static_thresholds_archived_20260915`.
It used static dataset/ratio/method thresholds. Its SQLite database, ledger, logs, and
W&B runs are audit artifacts only and must never be merged with the fold-local campaign.
The earlier OFAT attempt is also separate and archived.

## A30 collaborator quick start

The original commands below document the initial shard launch and must not be rerun after
the September 22 handoff. For remaining Parka work, use section 6 of
[`sep24_optuna_runbook.md`](sep24_optuna_runbook.md) or the handoff's
`A30_COMMANDS.md`.

The A30 servers use different filesystem paths, so set these variables separately on each
server:

```bash
export REPO=/path/to/bgbench
export PYTHON=/path/to/the/bgbench/environment/bin/python
export DATA_ROOT=/local/scratch/path/ogbench
export SWEEP_ROOT="$DATA_ROOT/search_results/sep24_factorial_tc10"
export CONFIG="$REPO/configs/hparams_search/sep24_factorial_optuna.yaml"
export STORAGE="sqlite:///$SWEEP_ROOT/studies.db"
export PATH="$(dirname "$PYTHON"):$PATH"
cd "$REPO"
```

Before launching, verify:

```bash
test -f configs/hparams_search/sep24_factorial_optuna.yaml
"$PYTHON" -c "import optuna, torch; print(optuna.__version__, torch.cuda.device_count())"
"$PYTHON" -c "import wandb; assert wandb.api.api_key; print('W&B ready')"
test "$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)" -eq 8
df -h "$DATA_ROOT"
```

### A30 server A

Warm only the caches required by shards 3 and 4:

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

Then launch two jobs on each A30:

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
  >>"$SWEEP_ROOT/launcher.log" 2>&1 </dev/null &
echo $! >"$SWEEP_ROOT/launcher.pid"
```

### A30 server B

Warm only the caches required by shards 5 and 6:

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

Then launch two jobs on each A30:

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
  >>"$SWEEP_ROOT/launcher.log" 2>&1 </dev/null &
echo $! >"$SWEEP_ROOT/launcher.pid"
```

Do not exchange the shard indices between servers after either server has started.

## Monitoring

On server A, use `--shard-indices 3 4`; on server B, use `--shard-indices 5 6`:

```bash
"$PYTHON" scripts/optuna_status.py \
  --config "$CONFIG" \
  --root-dir "$DATA_ROOT" \
  --output-dir "$SWEEP_ROOT" \
  --storage "$STORAGE" \
  --num-shards 7 \
  --shard-indices 3 4 \
  --window-hours 6 \
  --export
```

Also check:

```bash
ps -fp "$(cat "$SWEEP_ROOT/launcher.pid")"
nvidia-smi
tail -f "$SWEEP_ROOT/launcher.log"
```

After the first hour, send the `completed_folds`, `folds_per_hour`,
`unresolved_failures`, and `eta_hours` values to the campaign coordinator. We will use
the observed A30 throughput to confirm whether two jobs/GPU is optimal.

## Failure and resume behavior

If a launcher exits:

1. Confirm that no child `ogbench.run` processes remain.
2. Wait at least six minutes for the five-minute Optuna heartbeat grace period.
3. Run the exact same detached launch command with `--retry-failed`.

The run ledger is append-only. Successful folds are reused, failed folds retain their
attempt history, and unresolved failures remain visible in the status export.

Do not delete `studies.db`, `run_ledger.sqlite3`, their WAL files, or the run directories.
Deleting them removes resumability.

## Deadline checkpoints

Add the `completed_folds` values from Parka and both A30 servers:

- September 17: at least 25,704 folds (30%).
- September 20: at least 59,976 folds (70%).
- September 22: at least 81,396 folds (95%).
- September 23: retries, final exports, collection, and validation.

If the campaign is behind a checkpoint, do not independently reduce trials or change
models. Coordinate any scope adjustment so every affected study remains comparable.

## Outputs and final collection

Each server writes:

- `studies.db`: Optuna study/trial state.
- `run_ledger.sqlite3`: fold attempt state and metrics.
- `runs/`: Hydra, checkpoint, and W&B run artifacts.
- `logs/`: captured training logs.
- `live_status.csv`, `live_trials.csv`, `live_best_trials.csv`,
  `live_fold_attempts.csv`, and `live_failures.csv`: status exports.
- `trials.csv`, `best_trials.csv`, `fold_attempts.csv`, and `failures.csv`: final exports
  after that server's launcher finishes.

At completion, follow the collection and merge commands in
[`sep24_optuna_runbook.md`](sep24_optuna_runbook.md). Because study ownership is
disjoint, the three trial tables can be concatenated. Before analysis, verify:

- 2,448 distinct study names.
- 7 complete trials per study.
- 5 successful fold scores per complete trial.
- No unresolved failures.

## Ratio 0.3 transfer results

The factorial design omitted `node_sample_ratio: 0.3`. It was added late, on Parka only,
without an Optuna search. Each of the 816 ratio-0.3 cells runs **one transferred
configuration** with the same 5-fold CV and data construction as the other ratios. The
configuration is the best trial of the same cell at ratio 0.5, else 0.8, else 1.0, else of the
closest sibling cell (another selection method). The `source_rule` column records which.
Report these cells as "transferred configuration" rather than "tuned".

- Results: `/scratch/lcornelis/ogbench/search_results/sep24_ratio03_transfer/results_ratio03.csv`
  and `coverage.md`, refreshed every 30 minutes until Friday 07:00 PDT.
- Live runs: W&B project `bioshape-lab/ogbench_sep24_ratio03_transfer`.
- Cells run cheapest models first (mlp, sagn, gcn, chebnet, graph_sage, gatv2, gin, gatv4,
  gps), plus an extra queue that gives gin, gatv4, and gps their cheapest cells. Ratio 0.3
  has the largest graphs, so by Friday 07:00 expect mlp through chebnet to be complete,
  about 60% of graph_sage, little of gatv2, and about 45 cells each of gin, gatv4, and gps.
- Operational details: section 8 of [`sep24_optuna_runbook.md`](sep24_optuna_runbook.md).

## Who should run what

- Parka operator: keep shards `0 1 2` alive and expand to eight A100s when available.
- A30 server A operator: run only shards `3 4`.
- A30 server B operator: run only shards `5 6`.
- Campaign coordinator: collect status numbers, compare them with deadline checkpoints,
  and merge final exports on Parka.
