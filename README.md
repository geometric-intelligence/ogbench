<div align="center">

# Big Graph Bench

.. list-table::
:header-rows: 0

- - **Code**
  - |PyPI version|\\ |Downloads|\\ |Zenodo|\\
- - **Continuous Integration**
  - |Build Status|\\ |python|\\
- - **Code coverage (np, autograd, torch)**
  - |Coverage Status np|\\ |Coverage Status autograd|\\ |Coverage Status torch|

</div>

## Description

## Installation

#### Pip

```bash
# clone project
git clone https://github.com/geometric-intelligence/bgbench
cd bgbench

conda create -n bgbench python=3.12
conda activate bgbench

pip install -r requirements.txt
```

## How to run one train

Example: Train model with default configuration for gatv2 as model and pancancer as dataset:

```bash
# train on GPU with DDP
python src/train.py trainer=ddp experiment=gatv2 data=pancancer
```

## Run a grid search over parameters:

```bash
python src/train.py trainer=ddp hparams_search=gcn_basic experiment=gatv2 data=pancancer
```
