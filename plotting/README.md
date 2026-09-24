# Sep 24 Optuna plots

These scripts live on **`louis/fold_test_eval`**. They plot the best trial in
each structural cell. Validation search chooses that trial; test evaluation
scores the same trial on the held-out split.

```text
plotting/plot_sep24_val_best.py    # validation F1 (search CSVs)
plotting/plot_sep24_test_best.py   # test F1 (test CSVs or live ledger)
```

Run from the repo root with the `ogbench` conda env activated.

## Intermediate plots (this server, test eval still running)

`live_test_best_trials.csv` is written only when `optuna_test_eval.py` finishes.
While it is still running, plot from the SQLite ledger plus this server's search
CSV. A study is kept only if it already has `--min-folds` successful test folds
(default 5).

On Frank:

```bash
conda activate ogbench
cd /home/louisvl/ogbench

python plotting/plot_sep24_test_best.py \
  --best-csv /scratch/louisvl/ogbench/search_results/sep24_factorial_optuna/live_best_trials.csv \
  --ledger /scratch/louisvl/ogbench/search_results/sep24_factorial_optuna_test/test_ledger.sqlite3 \
  --min-folds 5 \
  --out-dir plotting/plots_sep24_test
```

On Hall or Parka, keep the same flags and change only the two paths (that
server's `live_best_trials.csv` and `test_ledger.sqlite3`).

If you pass `--csv .../live_test_best_trials.csv` before the job finishes, the
script now falls back to `test_ledger.sqlite3` in the same directory.

Optional: dump a snapshot CSV without stopping the launcher:

```bash
"$PYTHON" scripts/optuna_test_eval.py \
  --config "$CONFIG" \
  --best-csv "$SEARCH_ROOT/live_best_trials.csv" \
  --search-output-dir "$SEARCH_ROOT" \
  --root-dir "$DATA_ROOT" \
  --output-dir "$TEST_ROOT" \
  --export-only
```

That writes `live_test_folds.csv` and `live_test_best_trials.csv` under
`$TEST_ROOT`. They will be incomplete until every fold succeeds.

To peek at partial studies (not for final figures):

```bash
python plotting/plot_sep24_test_best.py \
  --best-csv "$SEARCH_ROOT/live_best_trials.csv" \
  --ledger "$TEST_ROOT/test_ledger.sqlite3" \
  --min-folds 1 \
  --out-dir plotting/plots_sep24_test_peek
```

## Final plots (all shards on one server)

Copy each server's finished test export to one machine, then concatenate them.
Do not re-pick winners on test: each CSV row is already the validation-best
trial with its test-fold mean.

```bash
python plotting/plot_sep24_test_best.py \
  --csv /path/to/frank/live_test_best_trials.csv \
  --csv /path/to/hall/live_test_best_trials.csv \
  --csv /path/to/parka/live_test_best_trials.csv \
  --min-folds 5 \
  --out-dir plotting/plots_sep24_test
```

Validation-only figures (search CSVs, no test eval required):

```bash
python plotting/plot_sep24_val_best.py \
  --csv /path/to/frank/live_best_trials.csv \
  --csv /path/to/hall/live_best_trials.csv \
  --csv /path/to/parka/live_best_trials.csv \
  --out-dir plotting/plots_sep24_val
```

Expected after a complete collection: 2,448 distinct `study_name` values across
the three test CSVs, each with `n_folds == 5`.

Figures written to `--out-dir`:

- `best_overall_{val,test}_f1.{png,pdf}`
- `readout_effect_{val,test}_f1.{png,pdf}`
- `adjacency_effect_{val,test}_f1.{png,pdf}`
- `best_{val,test}_f1_by_ratio_and_method.{png,pdf}`
- `best_per_model_dataset.csv`, `best_per_dataset.csv`

How to produce the test CSVs, and how to copy them onto one server, is in
[`../sep24_test_eval_collaborator_guide.md`](../sep24_test_eval_collaborator_guide.md).
