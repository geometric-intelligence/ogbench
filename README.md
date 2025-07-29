<div align="center">

# Big Graph Bench

<a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white"></a>
<a href="https://pytorchlightning.ai/"><img alt="Lightning" src="https://img.shields.io/badge/-Lightning-792ee5?logo=pytorchlightning&logoColor=white"></a>
<a href="https://hydra.cc/"><img alt="Config: Hydra" src="https://img.shields.io/badge/Config-Hydra-89b8cd"></a>
<a href="https://github.com/ashleve/lightning-hydra-template"><img alt="Template" src="https://img.shields.io/badge/-Lightning--Hydra--Template-017F2F?style=flat&logo=github&labelColor=gray"></a><br>
[![Paper](http://img.shields.io/badge/paper-arxiv.1001.2234-B31B1B.svg)](todo)
[![Conference](http://img.shields.io/badge/AnyConference-year-4b44ce.svg)](todo)

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
