# OGBench — Omics Graph Benchmarking

A benchmarking framework for Graph Neural Networks on omics datasets. OGBench provides standardized datasets, graph construction pipelines, GNN architectures, and sklearn baselines to enable fair comparison of models on biological classification tasks (genomics, transcriptomics, proteomics).

## Overview

- **4 curated omics datasets** on Hugging Face Hub with automatic download
- **9 GNN architectures** — GCN, GATv2, GATv4, GIN, GraphSAGE, ChebNet, SAGN, GPS, MLP
- **2 graph construction methods** — WGCNA co-expression and STRING protein-protein interaction
- **GNN-features baselines** — sklearn classifiers (SVM, Elastic Net) on learned GNN embeddings
- **Hydra configs** for reproducible, composable experiments
- **PyTorch Lightning** training with WandB logging and multi-GPU support
- **Interactive leaderboard** webapp with dataset explorer

## Installation

```bash
git clone <anonymous-repo>
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

OGBench includes four curated omics datasets for graph-based classification. All are stored on Hugging Face Hub at [`ogbench-neurips/ogbench-neurips`](https://huggingface.co/datasets/ogbench-neurips/ogbench-neurips) in Parquet format and downloaded automatically on first use.

| Dataset         | Domain                          | Samples | Features        | Classes | Task                          |
| --------------- | ------------------------------- | ------- | --------------- | ------- | ----------------------------- |
| **MotrPac**     | Proteomics (exercise response)  | 654     | ~4,976 proteins | 2       | Responder vs non-responder    |
| **Parkinson's** | Gene expression (PD study)      | 535     | ~21,755 genes   | 2       | Dementia vs MCI/normal        |
| **AddNeuroMed** | Gene expression (AD study)      | 711     | ~17,198 genes   | 3       | AD vs MCI vs Control          |
| **BRCA**        | Gene expression (breast cancer) | 640     | ~19,049 genes   | 4       | Cancer subtype classification |

### Downloading and Processing Datasets

```bash
python scripts/download_datasets.py motrpac
python scripts/download_datasets.py parkinsons
python scripts/download_datasets.py addneuromed
python scripts/download_datasets.py all
```

## Graph Construction

Graphs are constructed from omics feature matrices. Two adjacency methods are supported:

- **WGCNA** (default) — weighted gene co-expression network analysis with soft thresholding
- **STRING PPI** — protein-protein interaction edges from the STRING database

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
- `configs/model/` — model architectures and hyperparameters
- `configs/trainer/` — training backend (`cpu`, `gpu`, `mps`, `ddp`, `ddp_sim`)
- `configs/logger/` — logging backends (WandB, TensorBoard, CSV, MLflow, etc.)
- `configs/experiment/` — experiment presets (e.g. `omics_readout`, `no_readout`)
- `configs/transforms/` — data manipulations and topological liftings

Override any parameter from the command line:

```bash
python ogbench/run.py dataset=brca model=gin \
    optimizer.parameters.lr=0.001 \
    trainer.max_epochs=200 \
    seed=123
```

## Baselines — GNN-Features Pipeline

OGBench supports a hybrid baseline approach: train a GNN to learn node embeddings, then use those embeddings as features for sklearn classifiers. This isolates the value of the graph structure from the classifier head.

Two GNN-features baselines are configured per dataset:

- **`svm_gnn_features`** — LinearSVC with calibration on GNN-learned embeddings
- **`elastic_net_gnn_features`** — Logistic regression with elastic net penalty on GNN-learned embeddings

Both skip the manual feature selection step (no `SelectKBest`) since the GNN already performs representation learning.

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
├── configs/                    # Hydra YAML configs
├── scripts/                    # Utilities (download, processors, export)
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
