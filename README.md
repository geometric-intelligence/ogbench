[![Code Quality](https://github.com/geometric-intelligence/ogbench/actions/workflows/code-quality-main.yaml/badge.svg)](https://github.com/geometric-intelligence/ogbench/actions/workflows/code-quality-main.yaml)
[![Dependencies](https://github.com/geometric-intelligence/ogbench/actions/workflows/dependabot/dependabot-updates/badge.svg)](https://github.com/geometric-intelligence/ogbench/actions/workflows/dependabot/dependabot-updates)
[![Tests](https://github.com/geometric-intelligence/ogbench/actions/workflows/test.yml/badge.svg)](https://github.com/geometric-intelligence/ogbench/actions/workflows/test.yml)

# OGBench — Omics Graph Benchmarking

A benchmarking framework for Graph Neural Networks on omics datasets. OGBench provides standardized datasets, graph construction pipelines, GNN architectures, and sklearn baselines to enable fair comparison of models on biological classification tasks (genomics, transcriptomics, proteomics).

## Overview

- **6 curated omics datasets** on Hugging Face Hub with automatic download
- **9 GNN architectures** — GCN, GATv2, GATv4, GIN, GraphSAGE, ChebNet, SAGN, GPS, MLP
- **2 graph construction methods** — WGCNA co-expression and STRING protein-protein interaction
- **Five-fold evaluation** with every preprocessing step fit on the training split only
- **Optuna hyperparameter search** scored on the mean validation macro F1 across folds
- **Sklearn baselines** — SVM and Elastic Net on the exact features and folds the GNN pipeline builds
- **Hydra configs** for reproducible, composable experiments
- **PyTorch Lightning** training with WandB logging and multi-GPU support
- **Interactive leaderboard** webapp with dataset explorer

## Installation

```bash
git clone git@github.com:geometric-intelligence/ogbench.git
cd ogbench

conda create -n ogbench python=3.12
curl -LsSf https://astral.sh/uv/install.sh | sh
conda activate ogbench

uv venv
uv pip install -e '.[all]'

pre-commit install
```

A CLI entry point is also installed: `ogbench-train` (equivalent to `python ogbench/run.py`).

## Datasets

OGBench includes six curated omics datasets for graph-based classification. All are stored on Hugging Face Hub at [`geometric-intelligence/ogbench`](https://huggingface.co/datasets/geometric-intelligence/ogbench) in Parquet format and downloaded automatically on first use.

| Dataset          | Domain                                  | Samples | Features                           | Classes | Task                          |
| ---------------- | --------------------------------------- | ------- | ---------------------------------- | ------- | ----------------------------- |
| **MotrPac**      | Proteomics (exercise response)          | 654     | ~4,976 proteins                    | 2       | Responder vs non-responder    |
| **Parkinson's**  | Gene expression (PD study)              | 535     | ~21,755 genes                      | 2       | Dementia vs MCI/normal        |
| **AddNeuroMed**  | Gene expression (AD study)              | 711     | ~17,197 genes                      | 3       | AD vs MCI vs Control          |
| **BRCA**         | Gene expression (breast cancer)         | 640     | ~19,049 genes                      | 4       | Cancer subtype classification |
| **Tuberculosis** | Protein microarray (GSE19433, sera)     | 561     | ~3,814 proteins                    | 2       | Culture negative vs positive  |
| **Smoking**      | DNA methylation (GSE50660, blood, 450k) | 464     | 139,125 TSS probes → ~20,763 genes | 2       | Never- vs ever-smoker         |

### Downloading and Processing Datasets

```bash
python scripts/download_datasets.py motrpac
python scripts/download_datasets.py parkinsons
python scripts/download_datasets.py addneuromed
python scripts/download_datasets.py tuberculosis
python scripts/download_datasets.py smoking
python scripts/download_datasets.py all
```

### Train / validation / test splits

Omics datasets use `dataset.split_params.split_type` (default **`k-fold`**):

- **`k-fold`** (default) — stratified 3/1/1 rotation over `k` folds (`k` defaults to 5 → about 60 / 20 / 20). `data_seed` is the **test fold**; validation is the next fold. Each sample is test once and validation once across folds `0 .. k-1`, and results are averaged over the folds.
- **`fixed`** — shuffle with seed 42, then cut 70 / 15 / 15.

Imputation, gene selection, adjacency, and feature normalization are always fit on **training samples only**, then applied to val/test. Each run trains, validates, and tests on the fold stored in the dataset cache; a cache without `split_info.json`, or built for a different split, raises an error instead of being re-split.

`class_weights: balanced` computes balanced loss weights from each run's training labels (`null` leaves the loss unweighted). `method=random` draws genes with a seed derived from the dataset, fold, and `node_sample_ratio`, so reruns and the sklearn baselines select the same genes.

MoTrPAC covariate adjustment, AddNeuroMed ComBat, and smoking promoter-probe pick plus median-centering are also train-only. Hub matrices are uncorrected; sidecars are `motrpac_covariates.parquet`, `addneuromed_batches.parquet`, `tuberculosis_batches.parquet`, and `smoking_probe_map.parquet`. Defaults are `corrections=[covariate_adjust]` (MoTrPAC), `corrections=[combat]` (AddNeuroMed), and `corrections=[promoter_min_beta, median_center]` (smoking). Smoking Hub columns are TSS1500/TSS200 probes; after the split, each gene keeps the candidate probe with the lowest mean beta on **training** never-smokers, remaining NaNs are imputed with training column means, and each gene is median-centered on train. Parkinson GEO characteristics are in `parkinsons_sample_meta.parquet`, including the hybridization-date `batch` field. Parkinson defaults to `dataset.split_params.grouping=batch`, so `StratifiedGroupKFold` keeps every batch inside a single fold and no batch is split across train / val / test. Batch sizes are very uneven (70 down to 1), so grouped folds are not equal sized: for `k=5` the test fold ranges from 95 to 120 samples. Tuberculosis also defaults to `grouping=batch`, where `tuberculosis_batches.parquet` holds the country of serum collection: culture positivity differs sharply by country (Brazil and Peru are all positive, Canada and Tanzania all negative), so a random split would let models learn country. Its 11 countries give test folds of 88 to 139 samples. `grouping` applies to `k-fold` only; a `fixed` split with `grouping=batch` raises an error. Grouped caches are stored separately (`..._group_batch`).

```bash
# One fold: test fold 0, validation fold 1
python -m ogbench dataset=brca model=gcn dataset.split_params.data_seed=0

# Full 5-fold rotation
python -m ogbench dataset=brca model=gcn dataset.split_params.data_seed=0,1,2,3,4 -m
```

#### Fixed split

The single 70 / 15 / 15 split is still available. Datasets that group folds by batch need `grouping=null` for it:

```bash
python -m ogbench dataset=motrpac model=gcn dataset.split_params.split_type=fixed
python -m ogbench dataset=parkinsons model=gcn dataset.split_params.split_type=fixed dataset.split_params.grouping=null
python -m ogbench dataset=tuberculosis model=gcn dataset.split_params.split_type=fixed dataset.split_params.grouping=null
```

## Graph Construction

Graphs are constructed from omics feature matrices. Two adjacency methods are supported:

- **WGCNA** (default) — weighted gene co-expression network analysis with soft thresholding. The graph keeps the strongest training-split edges until `adjacency_target_connectivity` (0.10) of possible gene pairs are connected.
- **STRING PPI** — protein-protein interaction edges from the STRING database, kept when the normalized combined score exceeds `adjacency_threshold` (0.4, shared by all datasets)

Node (feature) selection methods: `variance`, `correlation`, `distance_correlation`, `random`. The `node_sample_ratio` parameter controls the fraction of features retained.

```bash
# Switch adjacency method
python ogbench/run.py dataset=motrpac dataset.loader.parameters.adjacency_method=string dataset.split_params.data_seed=0,1,2,3,4 -m

# Change node selection
python ogbench/run.py dataset=motrpac dataset.loader.parameters.method=distance_correlation dataset.split_params.data_seed=0,1,2,3,4 -m

# Adjust sampling ratio
python ogbench/run.py dataset=motrpac dataset.loader.parameters.node_sample_ratio=0.3 dataset.split_params.data_seed=0,1,2,3,4 -m
```

## Usage

### Training a Model

Each command trains on one fold (`data_seed`, default 0). The examples sweep all five folds as a Hydra multirun; average the test metrics across them.

```bash
# Train GATv2 on MotrPac (default: WGCNA, variance selection, GPU)
python ogbench/run.py dataset=motrpac model=gatv2 dataset.split_params.data_seed=0,1,2,3,4 -m

# Train GCN on Parkinson's with specific selection method
python ogbench/run.py dataset=parkinsons model=gcn dataset.loader.parameters.method=correlation dataset.split_params.data_seed=0,1,2,3,4 -m

# Train GPS on BRCA with STRING adjacency
python ogbench/run.py dataset=brca model=gps dataset.loader.parameters.adjacency_method=string dataset.split_params.data_seed=0,1,2,3,4 -m

# Distributed training on one fold
python ogbench/run.py dataset=addneuromed model=graph_sage trainer=ddp dataset.split_params.data_seed=0
```

### Available Models

| Model     | Config name  | Description                                           |
| --------- | ------------ | ----------------------------------------------------- |
| GCN       | `gcn`        | Graph Convolutional Network                           |
| GATv2     | `gatv2`      | Graph Attention Network v2                            |
| GATv4     | `gatv4`      | Graph Attention Network v4 (per-layer heads/channels) |
| GIN       | `gin`        | Graph Isomorphism Network                             |
| GraphSAGE | `graph_sage` | Graph Sample and Aggregate                            |
| ChebNet   | `chebnet`    | Chebyshev Spectral Graph Convolution                  |
| SAGN      | `sagn`       | Structure-Aware Graph Network                         |
| GPS       | `gps`        | General, Powerful, Scalable Graph Transformer         |
| MLP       | `mlp`        | Multi-layer Perceptron (non-graph baseline)           |

### Configuration

OGBench uses [Hydra](https://hydra.cc/) for configuration management. Key config groups:

- `configs/dataset/` — dataset-specific settings (features, classes, splits, baselines)
- `configs/gene_identity/` — optional learnable node identity before message passing
- `configs/model/` — model architectures and hyperparameters
- `configs/trainer/` — training backend (`cpu`, `gpu`, `mps`, `ddp`, `ddp_sim`)
- `configs/logger/` — logging backends (WandB, TensorBoard, CSV, MLflow, etc.)
- `configs/experiment/` — experiment presets (e.g. `omics_readout`, `no_readout`)
- `configs/transforms/` — data manipulations and topological liftings
- `configs/hparams_search/` — Optuna search spaces and ablation grids

Override any parameter from the command line:

```bash
python ogbench/run.py dataset=brca model=gin \
    optimizer.parameters.lr=0.001 \
    trainer.max_epochs=200 \
    seed=123 \
    dataset.split_params.data_seed=0,1,2,3,4 -m
```

`callbacks.time_budget.budget_seconds=3600` stops training after one hour of wall-clock time and still tests the best checkpoint.

### Hyperparameter search

[`scripts/optuna_search.py`](scripts/optuna_search.py) runs resumable Optuna studies. Each outer
ablation cell (dataset, model, experiment, graph method, node ratio, and selector) is a separate
study. A trial samples one model configuration, evaluates that same configuration on all five
validation folds, and optimizes the arithmetic mean of `best_val/f1_macro`. Test folds never take
part in hyperparameter selection; each fold's test metrics are recorded alongside it.

Start with the smoke config. `--dry-run` composes Hydra configs and instantiates models without
training or creating a persistent study:

```bash
python scripts/optuna_search.py \
    --config configs/hparams_search/optuna_smoke_test.yaml \
    --dry-run
```

Studies and the fold-attempt ledger are written under `search_results/` by default.
Machine-specific locations belong on the command line:

```bash
python scripts/optuna_search.py \
    --config configs/hparams_search/kfold_factorial.yaml \
    --models gcn gin \
    --datasets motrpac brca \
    --gpus 0 1 \
    --jobs-per-gpu 1 \
    --root-dir /path/to/ogbench-project \
    --output-dir /path/to/search-results \
    --storage sqlite:////path/to/search-results/studies.db
```

The launcher warms every selected fold-aware dataset cache before training. It constrains each
subprocess to one CPU thread and sets each dataloader's worker count to zero so parallel jobs do
not oversubscribe the host. STRING cache downloads and JSON artifacts are locked and installed
atomically. `training.time_budget` stops each fold gracefully at the budget and tests its best
checkpoint; the fold process is killed only after `time_budget + timeout_grace` seconds.

Interrupted searches resume from the configured storage. Use `--retry-failed` to re-enqueue
failed trials while reusing successful fold records. Large searches can be partitioned
deterministically with `--num-shards` and `--shard-indices`, or filtered by exact study names with
`--studies`. Use `--warmup-only` to build caches without opening studies. Each run exports
`trials.csv`, `best_trials.csv` (the best trial per study, with fold-mean test metrics),
`fold_attempts.csv`, and `failures.csv`.

### Learnable node identity

Graphs share a fixed node order, but expression alone does not identify which
gene or marker each row represents. Append a trainable embedding for every node
before the feature encoder and message-passing layers:

```bash
python -m ogbench dataset=brca model=gcn gene_identity=learnable dataset.split_params.data_seed=0,1,2,3,4 -m
```

The default `combine=concat` appends 32 identity channels and updates the
configured encoder dimensions. Set `gene_identity.embed_dim=64` to change the
embedding size, or use `gene_identity.combine=add` to project identity into the
existing feature dimension. SAGN is not supported because it trains on node
features aggregated before training, so the identity embedding would never reach it.

## Sklearn Baselines

Sklearn classifiers are trained on the same input the GNNs see, without a graph, so comparing them with the GNNs isolates the value of the graph structure and message passing. Each dataset's YAML configures them under the `baselines` key (e.g. `configs/dataset/motrpac.yaml`):

- **`svm`** and **`elastic_net`** (standard) — LinearSVC with calibration and elastic-net logistic regression. They rebuild the GNN input from the Hub data: the same split and fold, train-only corrections and imputation, and the same node selection (including the seeded `random` method).
- **`svm_gnn_features`** and **`elastic_net_gnn_features`** — the same classifiers on the exact matrix the GNN pipeline caches as `selected_data.parquet`, with the split from `split_info.json`. With `split_type=k-fold` they need the cache of every fold, so run the GNN pipeline on all folds first.

With `split_type=k-fold`, every grid configuration is scored on the validation fold of all `k` folds, and the one with the best mean validation score is kept (`baseline_hparam_selection: global_kfold_mean_validation`). Each fold is then refit on its training split with that configuration and evaluated on its test fold. The selection is cached under `baseline_hparam_cache_dir`, so the other folds reuse it. Results are logged to WandB.

```bash
# Standard baselines on one fold of one dataset
python ogbench/baseline.py dataset=motrpac baseline_filter=standard dataset.split_params.data_seed=0

# All datasets, folds, node ratios and selection methods
bash run_baselines.sh
```

`run_baselines.sh` reads `DATASETS`, `FOLDS`, `NODE_RATIOS`, `METHODS`, and `PYTHON` from the environment, and passes extra arguments to Hydra as overrides:

```bash
DATASETS=motrpac NODE_RATIOS=0.5 bash run_baselines.sh logger.wandb.project=my_baselines
```

`scripts/verify_baseline_features.py --root-dir <root>` checks that the standard baselines select exactly the features stored in every cached GNN fold.

## Reproducing the paper

The GNN results come from one Optuna study per ablation cell: 6 datasets, 9 models, two experiments
(MLP runs `no_readout` only), two graph methods, four node ratios, and four selectors, for 3,264
studies of seven trials each. Every fold trains for at most one hour and is tested on its best
checkpoint:

```bash
python scripts/optuna_search.py --config configs/hparams_search/kfold_factorial.yaml
```

The sklearn baselines use the same datasets, folds, node ratios, and selectors:

```bash
bash run_baselines.sh
```

The learnable node identity ablation reruns the best configuration of every non-SAGN study with
the identity embedding. Each fold's exact Hydra overrides are saved in
`runs/<study>/<run>/.hydra/overrides.yaml` under the search output directory; rerun all five folds
of a study's best trial with them plus `gene_identity=learnable callbacks.time_budget.budget_seconds=3600`.

The fold, gene selection, and graph of every run are fixed by the dataset cache, and training is
seeded (`seed=42`). GPU message-passing kernels are not deterministic (`trainer.deterministic` is
off), so reruns match closely but not bit for bit. Folds that reach the one-hour budget (counted
in `best_trials.csv` as `time_budget_hit_folds`) also stop at an epoch that depends on GPU speed.

## Leaderboard & Dataset Explorer

An interactive webapp provides a leaderboard comparing all models and a dataset explorer for visualizing graph statistics across parameter combinations. See [webapp/README.md](webapp/README.md) for setup and deployment details.

## Development

### Code Quality

```bash
pre-commit install
pre-commit run -a
# or
make format
```

Pre-commit hooks: Ruff formatting/linting, import sorting, docstring formatting, Bandit security checks, YAML/shell validation, CodeSpell.

### Testing

```bash
make test          # fast tests (excludes slow)
make test-full     # all tests
pytest tests/nn/ -v  # specific module
```

### Project Structure

```
ogbench/
├── ogbench/                    # Main Python package
│   ├── run.py                  # Training entry point
│   ├── baseline.py             # Sklearn baseline experiments
│   ├── callbacks/              # Lightning callbacks (time budget)
│   ├── data/
│   │   ├── loaders/            # Dataset loaders (omics, TU, Planetoid)
│   │   ├── adjacency/          # Graph construction (WGCNA, STRING PPI)
│   │   ├── selectors/          # Node selection methods
│   │   ├── datasets/           # HF dataset integration
│   │   └── preprocessor/       # Preprocessing pipeline
│   ├── nn/
│   │   ├── backbones/          # GNN architectures (GATv4, GPS, ChebNet, etc.)
│   │   ├── wrappers/           # Domain wrappers (graph, cell, hypergraph)
│   │   ├── encoders/           # Feature encoders (flat, DGM)
│   │   └── readouts/           # Readout layers (OmicsReadOut, etc.)
│   ├── transforms/             # Data manipulations and liftings
│   ├── model/                  # Lightning module
│   ├── evaluator/              # Metrics and evaluation
│   ├── loss/                   # Loss functions
│   └── optimizer/              # Optimizer construction
├── configs/                    # Hydra YAML configs
├── scripts/                    # Utilities (download, processors, export, Optuna search)
├── tests/                      # Pytest suite
├── webapp/                     # Astro/React leaderboard & explorer
├── tutorials/                  # Notebooks and analysis scripts
└── notebooks/                  # Dataset exploration notebooks
```

## License

MIT — see [LICENSE](LICENSE).

## Acknowledgments

- [PyTorch Geometric](https://pyg.org/) and [Lightning](https://lightning.ai/) for the training stack
- [TopoModelX](https://github.com/pyt-team/TopoModelX) / [TopoNetX](https://github.com/pyt-team/TopoNetX) for topological operations
- Datasets sourced from public repositories (GEO, MoTrPAC, TCGA)
- [Hugging Face Hub](https://huggingface.co/) for dataset storage and distribution
