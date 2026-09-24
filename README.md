[![Code Quality](https://github.com/geometric-intelligence/ogbench/actions/workflows/code-quality-main.yaml/badge.svg)](https://github.com/geometric-intelligence/ogbench/actions/workflows/code-quality-main.yaml)
[![Dependencies](https://github.com/geometric-intelligence/ogbench/actions/workflows/dependabot/dependabot-updates/badge.svg)](https://github.com/geometric-intelligence/ogbench/actions/workflows/dependabot/dependabot-updates)
[![Tests](https://github.com/geometric-intelligence/ogbench/actions/workflows/test.yml/badge.svg)](https://github.com/geometric-intelligence/ogbench/actions/workflows/test.yml)

# OGBench — Omics Graph Benchmarking

A benchmarking framework for Graph Neural Networks on omics datasets. OGBench provides standardized datasets, graph construction pipelines, GNN architectures, and sklearn baselines to enable fair comparison of models on biological classification tasks (genomics, transcriptomics, proteomics).

## Overview

- **6 curated omics datasets** on Hugging Face Hub with automatic download
- **9 GNN architectures** — GCN, GATv2, GATv4, GIN, GraphSAGE, ChebNet, SAGN, GPS, MLP
- **2 graph construction methods** — WGCNA co-expression and STRING protein-protein interaction
- **GNN-features baselines** — sklearn classifiers (SVM, Elastic Net) on learned GNN embeddings
- **Hydra configs** for reproducible, composable experiments
- **Resumable Optuna search** with one configuration evaluated across every fold
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
python scripts/download_datasets.py brca
python scripts/download_datasets.py tuberculosis
python scripts/download_datasets.py smoking
python scripts/download_datasets.py all
```

### Train / validation / test splits

Omics datasets use `dataset.split_params.split_type` (default **`k-fold`**):

- **`k-fold`** — stratified 3/1/1 rotation over `k` folds (`k` defaults to 5 → about 60 / 20 / 20). `data_seed` is the **test fold**; validation is the next fold. Each sample is test once and validation once across folds `0 .. k-1`.
- **`fixed`** — legacy 70 / 15 / 15 split after shuffling with seed 42. Fixed-split graph caches retain their historical path.

A normal training command runs one fold; the dataset configs default to fold 0. Run folds
`0,1,2,3,4` to evaluate the full rotation.

```bash
python ogbench/run.py --multirun dataset=brca model=gcn \
    dataset.split_params.data_seed=0,1,2,3,4
```

Imputation, gene selection, adjacency, and feature normalization are always fit on **training samples only**, then applied to val/test.

MoTrPAC covariate adjustment, AddNeuroMed ComBat, and smoking promoter-probe pick plus median-centering are also train-only. Hub matrices are uncorrected; sidecars are `motrpac_covariates.parquet`, `addneuromed_batches.parquet`, and `smoking_probe_map.parquet`. Defaults are `corrections=[covariate_adjust]` (MoTrPAC), `corrections=[combat]` (AddNeuroMed), and `corrections=[promoter_min_beta, median_center]` (smoking). Smoking Hub columns are TSS1500/TSS200 probes; after the split, each gene keeps the candidate probe with the lowest mean beta on **training** never-smokers, remaining NaNs are imputed with training column means, and each gene is median-centered on train. Parkinson GEO characteristics are in `parkinsons_sample_meta.parquet`, including the hybridization-date `batch` field. Parkinson defaults to `dataset.split_params.grouping=batch`, so `StratifiedGroupKFold` keeps every batch inside a single fold and no batch is split across train / val / test. Batch sizes are very uneven (70 down to 1), so grouped folds are not equal sized: for `k=5` the test fold ranges from 95 to 120 samples. `grouping` applies to `k-fold` only and is ignored for `fixed`. Grouped caches are stored separately (`..._group_batch`).

```bash
python -m ogbench dataset=brca model=gcn
python -m ogbench dataset=brca model=gcn dataset.split_params.split_type=k-fold dataset.split_params.data_seed=0
```

## Graph Construction

Graphs are constructed from omics feature matrices. Two adjacency methods are supported:

- **WGCNA** (default) — weighted gene co-expression network analysis with soft thresholding
- **STRING PPI** — protein-protein interaction edges from the STRING database

WGCNA is computed from each fold's training samples and keeps the strongest edges nearest to
`adjacency_target_connectivity` (0.10 in the dataset configs). STRING uses
`adjacency_threshold` as a fixed confidence cutoff. Neither method silently falls back to the
other method's parameter.

Node (feature) selection methods: `variance`, `correlation`, `distance_correlation`, `random`. The `node_sample_ratio` parameter controls the fraction of features retained.

```bash
# Switch adjacency method
python ogbench/run.py dataset=motrpac dataset.loader.parameters.adjacency_method=string

# Change node selection
python ogbench/run.py dataset=motrpac dataset.loader.parameters.method=distance_correlation

# Adjust sampling ratio
python ogbench/run.py dataset=motrpac dataset.loader.parameters.node_sample_ratio=0.3
```

## Usage

### Training a Model

```bash
# Train GATv2 on MotrPac (default: WGCNA, variance selection, GPU)
python ogbench/run.py dataset=motrpac model=gatv2

# Run another fold (the default is fold 0)
python ogbench/run.py dataset=motrpac model=gatv2 dataset.split_params.data_seed=1

# Train GCN on Parkinson's with specific selection method
python ogbench/run.py dataset=parkinsons model=gcn dataset.loader.parameters.method=correlation

# Train GPS on BRCA with STRING adjacency
python ogbench/run.py dataset=brca model=gps dataset.loader.parameters.adjacency_method=string

# Distributed training
python ogbench/run.py dataset=addneuromed model=graph_sage trainer=ddp
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
- `configs/hparams_search/` — standalone Optuna search definitions
- `configs/transforms/` — data manipulations and topological liftings

Override any parameter from the command line:

```bash
python ogbench/run.py dataset=brca model=gin \
    optimizer.parameters.lr=0.001 \
    trainer.max_epochs=200 \
    seed=123
```

### Hyperparameter search

[`scripts/optuna_search.py`](scripts/optuna_search.py) runs resumable Optuna studies. Each outer
ablation cell (dataset, model, experiment, graph method, node ratio, and selector) is a separate
study. A trial samples one model configuration, evaluates that same configuration on all five
validation folds, and optimizes the arithmetic mean of `best_val/f1_macro`. Test folds never take
part in hyperparameter selection.

Start with the smoke config. `--dry-run` composes Hydra configs and instantiates models without
training or creating a persistent study:

```bash
python scripts/optuna_search.py \
    --config configs/hparams_search/optuna_smoke_test.yaml \
    --dry-run
```

The reusable multi-dataset config writes its SQLite study and fold-attempt ledger under
`search_results/`. Machine-specific locations belong on the command line:

```bash
python scripts/optuna_search.py \
    --config configs/hparams_search/multi_dataset_optuna_search.yaml \
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
atomically.

Interrupted searches resume from the configured storage. Use `--retry-failed` to re-enqueue
failed trials while reusing successful fold records. Large campaigns can be partitioned
deterministically with `--num-shards` and `--shard-indices`, or filtered by exact study names with
`--studies` / `--studies-file`. Use `--warmup-only` to build caches without opening studies.

### Learnable node identity

Graphs share a fixed node order, but expression alone does not identify which
gene or marker each row represents. Append a trainable embedding for every node
before the feature encoder and message-passing layers:

```bash
python -m ogbench dataset=brca model=gcn gene_identity=learnable
```

The default `combine=concat` appends 32 identity channels and updates the
configured encoder dimensions. Set `gene_identity.embed_dim=64` to change the
embedding size, or use `gene_identity.combine=add` to project identity into the
existing feature dimension.

## Baselines

OGBench supports a hybrid baseline approach: train a GNN to learn node embeddings, then use those embeddings as features for sklearn classifiers. This isolates the value of the graph structure from the classifier head.

Two GNN-features baselines are configured per dataset:

- **`svm_gnn_features`** — LinearSVC with calibration on GNN-learned embeddings
- **`elastic_net_gnn_features`** — Logistic regression with elastic net penalty on GNN-learned embeddings

The standard SVM and elastic-net baselines use the same train-only node selection and node budget
as the GNNs; they do not apply a second `SelectKBest` pass. For k-fold runs,
`baseline_hparam_selection: global_kfold_mean_validation` scores every candidate on every
validation fold, chooses one configuration by mean score, and refits every fold with that shared
configuration. GNN-features baselines also skip manual feature selection because the GNN has
already produced the representation.

```bash
# Run baselines on a specific dataset
python ogbench/baseline.py dataset=motrpac

# Run all baselines across datasets
bash run_baselines.sh
```

Baselines are configured in each dataset's YAML under the `baselines` key (e.g. `configs/dataset/motrpac.yaml`). Results are logged to WandB.

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
├── configs/                    # Hydra configs and standalone search definitions
│   └── hparams_search/         # Optuna and search smoke-test YAMLs
├── scripts/
│   ├── optuna_search.py        # Resumable fold-mean Optuna launcher
│   └── ...                     # Download, processing, and export utilities
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
