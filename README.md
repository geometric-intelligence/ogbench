# Big Graph Bench

[![Code Quality](https://github.com/geometric-intelligence/bgbench/actions/workflows/code-quality-main.yaml/badge.svg)](https://github.com/geometric-intelligence/bgbench/actions/workflows/code-quality-main.yaml)
[![Dependencies](https://github.com/geometric-intelligence/bgbench/actions/workflows/dependabot/dependabot-updates/badge.svg)](https://github.com/geometric-intelligence/bgbench/actions/workflows/dependabot/dependabot-updates)
[![Tests](https://github.com/geometric-intelligence/bgbench/actions/workflows/test.yml/badge.svg)](https://github.com/geometric-intelligence/bgbench/actions/workflows/test.yml)

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
