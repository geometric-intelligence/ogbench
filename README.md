[![Code Quality](https://github.com/geometric-intelligence/bgbench/actions/workflows/code-quality-main.yaml/badge.svg)](https://github.com/geometric-intelligence/bgbench/actions/workflows/code-quality-main.yaml)
[![Dependencies](https://github.com/geometric-intelligence/bgbench/actions/workflows/dependabot/dependabot-updates/badge.svg)](https://github.com/geometric-intelligence/bgbench/actions/workflows/dependabot/dependabot-updates)
[![Tests](https://github.com/geometric-intelligence/bgbench/actions/workflows/test.yml/badge.svg)](https://github.com/geometric-intelligence/bgbench/actions/workflows/test.yml)

# Big Graph Bench (BGBench)

A comprehensive benchmarking framework for Graph Neural Networks (GNNs) on omics datasets for classification tasks. This repository provides standardized datasets, preprocessing pipelines, and evaluation metrics to enable fair comparison of different GNN architectures on biological data.

## Overview

BGBench focuses on benchmarking GNNs for classification tasks using omics datasets (genomics, transcriptomics, proteomics). The framework includes:

- **Standardized Datasets**: Preprocessed omics datasets stored on Hugging Face Hub
- **Multiple GNN Architectures**: GCN, GAT, GraphSAGE, GIN, and more
- **Comprehensive Evaluation**: Multiple metrics and baseline comparisons
- **Reproducible Experiments**: Hydra-based configuration management
- **Scalable Training**: Support for distributed training and hyperparameter optimization

## Installation

### Quick Setup with Environment Script

The easiest way to set up BGBench is using the provided environment setup script:

```bash
# Clone the repository
git clone git@github.com:geometric-intelligence/bgbench.git
cd bgbench

# Run the automated setup script
bash env_setup.sh
```

This script will:

- Install Miniconda if not present
- Create a `bgbench` conda environment with Python 3.12
- Install all dependencies including development tools
- Set up pre-commit hooks

### Manual Installation

If you prefer manual setup:

```bash
# Clone the repository
git clone git@github.com:geometric-intelligence/bgbench.git
cd bgbench

# Create conda environment
conda create -n bgbench python=3.12
conda activate bgbench

# Install dependencies
pip install -e '.[all]'

# Install pre-commit hooks
pre-commit install
```

## Datasets

BGBench includes three curated omics datasets for graph-based classification:

### 1. MotrPac Dataset

- **Type**: Proteomics data from exercise response study
- **Samples**: 654 samples
- **Features**: ~4,977 proteins (sampled to ~4,698 nodes)
- **Task**: Binary classification (exercise responders vs non-responders)
- **Classes**: 2 (responder/non-responder based on 15% VO₂max improvement)
- **Preprocessing**: Log-transformed, covariate-adjusted (age, sex, BMI, race, baseline VO₂)

### 2. Parkinson's Dataset

- **Type**: Gene expression data from Parkinson's disease study
- **Samples**: 535 samples
- **Features**: ~21,755 genes (sampled to ~10,878 nodes)
- **Task**: Binary classification (dementia vs MCI/normal)
- **Classes**: 2 (dementia: MoCA \<21, MCI/Normal: MoCA ≥21)
- **Preprocessing**: Probe-to-gene mapping, variance filtering

### 3. AddNeuroMed Dataset

- **Type**: Gene expression data from Alzheimer's disease study
- **Samples**: 711 samples
- **Features**: ~17,112 genes (full dataset)
- **Task**: Multi-class classification (AD vs MCI vs Control)
- **Classes**: 3 (AD, MCI, Control)
- **Preprocessing**: Combat batch correction, class balancing

### Dataset Storage and Access

All datasets are stored on Hugging Face Hub at `geometric-intelligence/bgbench` and automatically downloaded when needed. The datasets are preprocessed and stored in Parquet format for efficient loading.

## Dataset Preprocessing

### Downloading and Processing Datasets

To download and process datasets for Hugging Face storage:

```bash
# Process individual datasets
python scripts/download_datasets.py motrpac
python scripts/download_datasets.py parkinsons
python scripts/download_datasets.py addneuromed

# Process all datasets at once
python scripts/download_datasets.py all
```

### Preprocessing Pipeline

Each dataset undergoes standardized preprocessing:

1. **Data Download**: Raw data downloaded from public repositories
2. **Quality Control**: Missing value filtering, outlier detection
3. **Feature Selection**: Node sampling based on variance/correlation/random methods
4. **Graph Construction**: WGCNA-based adjacency matrix with soft thresholding
5. **Normalization**: Mean-std normalization fitted on training data only
6. **Splitting**: Fixed train/validation/test splits (70/15/15 or 70/20/10)

### Graph Construction

Graphs are constructed using:

- **Node Selection**: Multiple methods (variance, correlation, random)
- **Adjacency Matrix**: WGCNA with optimal power selection and soft thresholding
- **Binarization**: Threshold-based edge creation (default: 0.85)
- **Node Sampling**: Configurable ratio for computational efficiency

## Usage

### Running Single Experiments

Train a model with default configuration:

```bash
# Train GATv2 on MotrPac dataset
python ogbench/run.py dataset=motrpac model=gatv2

# Train GCN on Parkinson's dataset with specific sampling method
python ogbench/run.py dataset=parkinsons model=gcn dataset.loader.parameters.method=variance

# Train on GPU with distributed training
python ogbench/run.py dataset=addneuromed model=graph_sage trainer=ddp
```

### Running All Experiments

Run comprehensive benchmarking across all model-dataset combinations:

```bash
# Sequential execution
bash run_all_experiments.sh

# Parallel execution (requires multiple GPUs)
bash run_all_experiments.sh --parallel
```

The script tests:

- **Models**: ChebNet, GATv4, GAT, GATv2, GCN, MLP, GraphSAGE
- **Datasets**: AddNeuroMed, MotrPac, Parkinson's
- **Sampling Methods**: variance, random, correlation

### Available Models

- **GCN**: Graph Convolutional Network
- **GAT/GATv2/GATv4**: Graph Attention Networks (various versions)
- **GraphSAGE**: Graph Sample and Aggregate
- **GIN**: Graph Isomorphism Network
- **ChebNet**: Chebyshev Spectral Graph Convolution
- **SAGN**: Structure-Aware Graph Network
- **MLP**: Multi-layer Perceptron (baseline)

### Configuration Management

BGBench uses Hydra for configuration management. Key configuration files:

- `configs/dataset/`: Dataset-specific configurations
- `configs/model/`: Model architectures
- `configs/trainer/`: Training configurations (CPU, GPU, DDP)
- `configs/logger/`: Logging backends (WandB, TensorBoard, etc.)

## Baseline Experiments

BGBench includes comprehensive baseline experiments using traditional machine learning methods to establish performance benchmarks for comparison with GNN models.

### Available Baselines

Each dataset includes multiple baseline models:

- **SVM**: Support Vector Machine with feature selection and calibration
- **Elastic Net**: Logistic regression with elastic net regularization
- **Feature Selection**: SelectKBest with F-test for feature importance
- **Preprocessing**: StandardScaler normalization and SimpleImputer for missing values

### Running Baseline Experiments

#### Quick Start

Run all baseline experiments across the three main datasets:

```bash
# Run all baseline experiments
bash run_baselines.sh
```

#### Individual Baseline Runs

Run baseline experiments for specific datasets:

```bash
# Run baseline on MotrPac dataset
python ogbench/baseline.py dataset=motrpac

# Run baseline on Parkinson's dataset
python ogbench/baseline.py dataset=parkinsons

# Run baseline on AddNeuroMed dataset
python ogbench/baseline.py dataset=addneuromed
```

### What the Baseline Script Does

The baseline experiment script (`run_baselines.sh`) performs the following:

1. **Data Loading**: Loads the specified dataset using the same preprocessing pipeline as GNN experiments
2. **Feature Selection**: Applies SelectKBest with F-test to select top features (k=50, 100, 500)
3. **Hyperparameter Search**: Uses GridSearchCV to find optimal parameters for each baseline model
4. **Model Training**: Trains SVM and Elastic Net models with cross-validation
5. **Evaluation**: Computes comprehensive metrics including accuracy, precision, recall, F1-score, AUROC
6. **Visualization**: Generates confusion matrices, ROC curves, precision-recall curves, and feature importance plots
7. **Logging**: Logs all results to WandB with detailed tracking and comparison

### Baseline Configuration

Baseline models are configured in each dataset's YAML file under the `baselines` section:

```yaml
baselines:
  svm:
    pipeline:
      - name: feature_selection
        _target_: sklearn.feature_selection.SelectKBest
        score_func:
          _target_: sklearn.feature_selection.f_classif
        k: 100
      - name: scaler
        _target_: sklearn.preprocessing.StandardScaler
      - name: calibrated_svm
        _target_: sklearn.calibration.CalibratedClassifierCV
        estimator:
          _target_: sklearn.svm.LinearSVC
          class_weight: balanced
          max_iter: 5000
        method: sigmoid
    param_grid:
      feature_selection__k: [50, 100, 500]
      calibrated_svm__estimator__C: [1e-3, 1e-2, 1e-1]
    scoring: f1_macro
    n_jobs: -1
```

### Viewing Baseline Results

#### WandB Dashboard

1. **Access**: Navigate to your WandB dashboard
2. **Project**: Look for project named `bgbench-baselines` (or as configured)
3. **Tags**: Filter by tags `['baseline', 'sklearn', dataset_name]`
4. **Metrics**: Compare performance across different baseline models
5. **Plots**: View automatically generated confusion matrices, ROC curves, and feature importance plots

#### Local Results

- **Logs**: Check `logs/` directory for detailed experiment logs
- **Plots**: Generated plots are saved in experiment output directories
- **Metrics**: Performance metrics are logged and can be exported from WandB

#### Key Metrics to Compare

- **Primary Metric**: F1-macro (or F1-weighted for multi-class)
- **Accuracy**: Overall classification accuracy
- **AUROC**: Area under ROC curve
- **Precision/Recall**: Per-class performance metrics
- **Feature Importance**: Top selected features for each model

### Baseline vs GNN Comparison

After running both baseline and GNN experiments:

1. **WandB Comparison**: Use WandB's compare feature to analyze performance differences
2. **Statistical Significance**: Check if GNN improvements are statistically significant
3. **Feature Analysis**: Compare which features are important for different model types
4. **Computational Cost**: Analyze training time and resource usage differences

### Example Results Interpretation

Typical baseline performance ranges:

- **MotrPac**: F1-macro ~0.65-0.75 (exercise response prediction)
- **Parkinson's**: F1-weighted ~0.70-0.80 (dementia classification)
- **AddNeuroMed**: F1-weighted ~0.60-0.70 (3-class AD classification)

GNN models should ideally outperform these baselines, especially on datasets where graph structure provides meaningful signal.

## Development

### Code Quality and Pre-commit

BGBench enforces high code quality through pre-commit hooks. **It's essential to run pre-commit before committing**:

```bash
# Install pre-commit hooks (done automatically by env_setup.sh)
pre-commit install

# Run pre-commit on all files
pre-commit run -a

# Or use the Makefile
make format
```

Pre-commit hooks include:

- **Code Formatting**: Ruff formatter and linter
- **Import Sorting**: Automatic import organization
- **Docstring Formatting**: Consistent docstring style
- **Security Checks**: Bandit security linter
- **YAML/Shell Validation**: Configuration file validation
- **Spelling Checks**: CodeSpell for typo detection

### Testing

```bash
# Run fast tests (excludes slow integration tests)
make test

# Run all tests including slow ones
make test-full

# Run specific test categories
pytest tests/data/ -v
pytest tests/nn/ -v
```

### Development Workflow

1. **Fork and Clone**: Fork the repository and clone your fork
2. **Create Branch**: Create a feature branch for your changes
3. **Install Dev Dependencies**: `pip install -e '.[dev]'`
4. **Make Changes**: Implement your feature/fix
5. **Run Tests**: Ensure all tests pass
6. **Pre-commit**: Run `pre-commit run -a` to fix formatting issues
7. **Commit**: Commit your changes with descriptive messages
8. **Push and PR**: Push to your fork and create a pull request

### Project Structure

```
bgbench/
├── ogbench/                 # Main package
│   ├── data/               # Data loading and preprocessing
│   ├── nn/                 # Neural network architectures
│   ├── model/              # Model definitions
│   ├── evaluator/          # Evaluation metrics
│   └── run.py              # Main training script
├── configs/                 # Hydra configuration files
├── scripts/                 # Utility scripts
│   ├── processors/         # Dataset processing scripts
│   └── download_datasets.py # Dataset download CLI
├── tests/                   # Test suite
├── tutorials/              # Jupyter notebooks and tutorials
└── notebooks/              # Analysis notebooks
```

## Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details on:

- Code style and standards
- Testing requirements
- Documentation expectations
- Pull request process

## Citation

If you use BGBench in your research, please cite:

```bibtex
@software{bgbench2024,
  title={Big Graph Bench: A Benchmarking Framework for Graph Neural Networks on Omics Data},
  author={Geometric Intelligence Team},
  year={2024},
  url={https://github.com/geometric-intelligence/bgbench}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Built on top of PyTorch Geometric and Lightning
- Datasets sourced from public repositories (GEO, MotrPac)
- Hugging Face Hub for dataset storage and distribution
