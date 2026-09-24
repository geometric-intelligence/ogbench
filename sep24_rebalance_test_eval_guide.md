# Sep 24 rebalance — test evaluation guide

This guide covers test evaluation for the **rebalanced studies**
(`relaunchsep24tc10factorialwgcna_*`) together with any original
(`sep24f_*`) studies that have not yet been evaluated.  
The new pipeline uses W&B as the single source of truth instead of
per-server `live_best_trials.csv` files.

---

## Step 0 — Get `wandb_ckpt_manifest.csv`

The manifest is a small CSV that records, for every study, which trial had
the best mean val F1 and exactly which checkpoint directory it used.
It contains rows for all servers (Frank, Hall, Parka), so you only need to
generate it **once** and copy it to each machine.

### Option A — Ask a collaborator

If someone already ran the manifest builder and has a fresh copy, just `scp`
it over:

```bash
# On the machine that already has it (e.g. frank):
scp /path/to/wandb_ckpt_manifest.csv hall:~/ogbench_optuna/tmp/
scp /path/to/wandb_ckpt_manifest.csv parka:~/ogbench_optuna/tmp/
```

The file lives at `tmp/wandb_ckpt_manifest.csv` in this repo on all servers.

### Option B — Generate it yourself (takes a while)

You need W&B credentials (`wandb login`) on whichever machine you run this on.
It queries both projects (original + rebalanced) and may take several minutes.

```bash
python scripts/optuna_wandb_ckpt_map.py \
    --project bioshape-lab/ogbench_sep24_factorial_optuna \
    --project bioshape-lab/ogbench_sep24_factorial_tc10 \
    --output tmp/wandb_ckpt_manifest.csv
```

Then copy `tmp/wandb_ckpt_manifest.csv` to every server as shown in Option A.

---

## Step 1 — Run test evaluation on each server

Each server reads the **same manifest** but filters to the checkpoint
directories that physically live on that machine via `--server-root`.

### Hall  (`louisvl` — `/scratch/louisvl/ogbench`)

```bash
python scripts/optuna_ckpt_test_eval.py \
    --config configs/hparams_search/sep24_factorial_optuna.yaml \
    --ckpt-map tmp/wandb_ckpt_manifest.csv \
    --server-root /scratch/louisvl/ogbench \
    --output-dir /scratch/louisvl/ogbench/search_results/sep24_rebalance_test \
    --root-dir /scratch/louisvl/ogbench \
    --gpus 0 1 2 3 4 5 6 7 --jobs-per-gpu 2
```

### Frank  (`louisvl` — `/scratch/louisvl/ogbench`)

```bash
python scripts/optuna_ckpt_test_eval.py \
    --config configs/hparams_search/sep24_factorial_optuna.yaml \
    --ckpt-map tmp/wandb_ckpt_manifest.csv \
    --server-root /scratch/louisvl/ogbench \
    --output-dir /scratch/louisvl/ogbench/search_results/sep24_rebalance_test \
    --root-dir /scratch/louisvl/ogbench \
    --gpus 1 2 3 4 5 6 7 --jobs-per-gpu 2
```

> **Note:** Hall and Frank share the same username and root path.  
> The manifest's `ckpt_dir` values tell them apart automatically — each server
> only resolves directories that exist locally (non-existent paths are silently
> skipped).

### Parka  (`lcornelis` — `/scratch/lcornelis/ogbench`)

This server picks up:
- original `sep24f_*` studies that trained on Parka, **and**
- all rebalanced studies whose **best trial** ran on Parka before
  redistribution (trials 0–6 of each rebalanced study).

```bash
python scripts/optuna_ckpt_test_eval.py \
    --config configs/hparams_search/sep24_factorial_optuna.yaml \
    --ckpt-map tmp/wandb_ckpt_manifest.csv \
    --server-root /scratch/lcornelis/ogbench \
    --output-dir /scratch/lcornelis/ogbench/search_results/sep24_rebalance_test \
    --root-dir /scratch/lcornelis/ogbench \
    --gpus 0 1 2 3 4 5 6 7 --jobs-per-gpu 2
```

### Outputs

Each server writes two files to its `--output-dir`:

| File | Contents |
|------|----------|
| `live_test_folds.csv` | One row per (study × fold) with `test_f1` |
| `live_test_best_trials.csv` | One row per study (mean ± std across folds) |

The job is idempotent — interrupted runs can be re-run with the same command
and will skip already-completed folds.

---

## Step 2 — Gather the CSVs

Once all three servers are done, collect the **`live_test_best_trials.csv`**
from each into one folder (e.g. on Frank or Hall):

```bash
mkdir -p /scratch/louisvl/ogbench/search_results/sep24_rebalance_test_all

# from Hall (run locally on Hall):
cp /scratch/louisvl/ogbench/search_results/sep24_rebalance_test/live_test_best_trials.csv \
   /scratch/louisvl/ogbench/search_results/sep24_rebalance_test_all/live_test_best_trials_hall.csv

# from Frank (scp to Hall or the plotting machine):
scp frank:/scratch/louisvl/ogbench/search_results/sep24_rebalance_test/live_test_best_trials.csv \
    /scratch/louisvl/ogbench/search_results/sep24_rebalance_test_all/live_test_best_trials_frank.csv

# from Parka:
scp parka:/scratch/lcornelis/ogbench/search_results/sep24_rebalance_test/live_test_best_trials.csv \
    /scratch/louisvl/ogbench/search_results/sep24_rebalance_test_all/live_test_best_trials_parka.csv
```

---

## Step 3 — Plot

```bash
python plotting/plot_sep24_test_best.py \
    --csv /scratch/louisvl/ogbench/search_results/sep24_rebalance_test_all/live_test_best_trials_hall.csv \
    --csv /scratch/louisvl/ogbench/search_results/sep24_rebalance_test_all/live_test_best_trials_frank.csv \
    --csv /scratch/louisvl/ogbench/search_results/sep24_rebalance_test_all/live_test_best_trials_parka.csv \
    --out-dir plotting/plots_sep24_rebalance_test
```

Figures are written to `plotting/plots_sep24_rebalance_test/`.

---

## Quick reference

| Server | `--server-root` | Who runs the rebalanced best trials? |
|--------|----------------|--------------------------------------|
| Hall   | `/scratch/louisvl/ogbench`  | Trials 7–10 of Hall-assigned rebalanced studies (if those are best) |
| Frank  | `/scratch/louisvl/ogbench`  | Trials 7–10 of Frank-assigned rebalanced studies (if those are best) |
| Parka  | `/scratch/lcornelis/ogbench` | **Trials 0–6 of ALL rebalanced studies** (ran before redistribution) + original Parka studies |

In practice, the best trial of nearly all rebalanced studies is expected to
be on **Parka** (trials 0–6 ran there), so Parka carries the largest load.
