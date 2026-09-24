# Parka test eval (send results to Louis)

Hall and Frank test CSVs are already in `plotting/test_results/{hall,frank}/`.
Parka only needs to evaluate the checkpoints that live on Parka.

`optuna_wandb_ckpt_map.py` already accepts extra `--project` flags. No code
change is required for `ogbench_sep24_ratio03_transfer`.

______________________________________________________________________

## 1. When training is done — scrape W&B (once)

On any machine with `wandb login` (Hall is fine):

```bash
python scripts/optuna_wandb_ckpt_map.py \
    --project bioshape-lab/ogbench_sep24_factorial_optuna \
    --project bioshape-lab/ogbench_sep24_factorial_tc10 \
    --project bioshape-lab/ogbench_sep24_ratio03_transfer \
    --output plotting/wandb_ckpt_manifest.csv
```

Copy `plotting/wandb_ckpt_manifest.csv` to Parka.

______________________________________________________________________

## 2. On Parka — test-eval local checkpoints

```bash
python scripts/optuna_ckpt_test_eval.py \
    --config configs/hparams_search/sep24_factorial_optuna.yaml \
    --ckpt-map plotting/wandb_ckpt_manifest.csv \
    --server-root /scratch/lcornelis/ogbench \
    --output-dir /scratch/lcornelis/ogbench/test_eval_rebalance \
    --root-dir /scratch/lcornelis/ogbench \
    --gpus 0 1 2 3 4 5 6 7 --jobs-per-gpu 2
```

`--server-root` keeps only rows whose `ckpt_dir` is on Parka.

When it finishes, these two files are written:

- `/scratch/lcornelis/ogbench/test_eval_rebalance/live_test_folds.csv`
- `/scratch/lcornelis/ogbench/test_eval_rebalance/live_test_best_trials.csv`

______________________________________________________________________

## 3. Send Louis those two CSVs

Same pair as Hall:

- `live_test_folds.csv`
- `live_test_best_trials.csv`

Louis will drop them in `plotting/test_results/parka/`.
