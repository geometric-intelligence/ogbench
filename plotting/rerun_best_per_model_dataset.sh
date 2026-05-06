#!/bin/bash
# Re-run training: all seeds of the bucket with highest mean best_val_f1_macro per (model_name, data_name) (same fingerprint rules as aggregated export).
# Overrides from W&B (summary or metadata ``args``).
# format=cluster  n_gpus=8  launcher='python -m ogbench'  background=True
# Drops: hydra.run.dir, paths.root_dir. Adds W&B tag: 'best_rerun'

python -m ogbench \
    model=chebnet \
    dataset=addneuromed \
    seed=42 \
    logger.wandb.tags=[chebnet,addneuromed,hpsearch,best_rerun] \
    logger.wandb.entity=bioshape-lab \
    experiment=omics_readout \
    dataset.loader.parameters.adjacency_method=wgcna \
    dataset.loader.parameters.node_sample_ratio=0.5 \
    dataset.loader.parameters.method=variance \
    optimizer.parameters.lr=0.0001 \
    optimizer.parameters.weight_decay=0.0 \
    model.backbone.dropout=0.1 \
    model.feature_encoder.out_channels=64 \
    model.backbone.num_layers=2 \
    dataset.loader.parameters.adjacency_threshold=0.1786 \
    model.readout.fc_dropout=0.1 \
    logger.wandb.project=best_model_reruns \
    trainer.devices=\[0\] \
    &

python -m ogbench \
    model=chebnet \
    dataset=addneuromed \
    seed=123 \
    logger.wandb.tags=[chebnet,addneuromed,hpsearch,best_rerun] \
    logger.wandb.entity=bioshape-lab \
    experiment=omics_readout \
    dataset.loader.parameters.adjacency_method=wgcna \
    dataset.loader.parameters.node_sample_ratio=0.5 \
    dataset.loader.parameters.method=variance \
    optimizer.parameters.lr=0.0001 \
    optimizer.parameters.weight_decay=0.0 \
    model.backbone.dropout=0.1 \
    model.feature_encoder.out_channels=64 \
    model.backbone.num_layers=2 \
    dataset.loader.parameters.adjacency_threshold=0.1786 \
    model.readout.fc_dropout=0.1 \
    logger.wandb.project=best_model_reruns \
    trainer.devices=\[1\] \
    &

python -m ogbench \
    model=chebnet \
    dataset=addneuromed \
    seed=456 \
    logger.wandb.tags=[chebnet,addneuromed,hpsearch,best_rerun] \
    logger.wandb.entity=bioshape-lab \
    experiment=omics_readout \
    dataset.loader.parameters.adjacency_method=wgcna \
    dataset.loader.parameters.node_sample_ratio=0.5 \
    dataset.loader.parameters.method=variance \
    optimizer.parameters.lr=0.0001 \
    optimizer.parameters.weight_decay=0.0 \
    model.backbone.dropout=0.1 \
    model.feature_encoder.out_channels=64 \
    model.backbone.num_layers=2 \
    dataset.loader.parameters.adjacency_threshold=0.1786 \
    model.readout.fc_dropout=0.1 \
    logger.wandb.project=best_model_reruns \
    trainer.devices=\[2\] \
    &

python -m ogbench \
    model=chebnet \
    dataset=brca \
    seed=42 \
    logger.wandb.tags=[chebnet,brca,hpsearch,best_rerun] \
    logger.wandb.entity=bioshape-lab \
    experiment=omics_readout \
    dataset.loader.parameters.adjacency_method=string \
    dataset.loader.parameters.node_sample_ratio=1.0 \
    dataset.loader.parameters.method=distance_correlation \
    optimizer.parameters.lr=0.001 \
    optimizer.parameters.weight_decay=0.0 \
    model.backbone.dropout=0.1 \
    model.feature_encoder.out_channels=16 \
    model.backbone.num_layers=2 \
    dataset.loader.parameters.adjacency_threshold=0.4 \
    model.readout.fc_dropout=0.1 \
    logger.wandb.project=best_model_reruns \
    trainer.devices=\[3\] \
    &

python -m ogbench \
    model=chebnet \
    dataset=brca \
    seed=123 \
    logger.wandb.tags=[chebnet,brca,hpsearch,best_rerun] \
    logger.wandb.entity=bioshape-lab \
    experiment=omics_readout \
    dataset.loader.parameters.adjacency_method=string \
    dataset.loader.parameters.node_sample_ratio=1.0 \
    dataset.loader.parameters.method=distance_correlation \
    optimizer.parameters.lr=0.001 \
    optimizer.parameters.weight_decay=0.0 \
    model.backbone.dropout=0.1 \
    model.feature_encoder.out_channels=16 \
    model.backbone.num_layers=2 \
    dataset.loader.parameters.adjacency_threshold=0.4 \
    model.readout.fc_dropout=0.1 \
    logger.wandb.project=best_model_reruns \
    trainer.devices=\[4\] \
    &

python -m ogbench \
    model=chebnet \
    dataset=brca \
    seed=456 \
    logger.wandb.tags=[chebnet,brca,hpsearch,best_rerun] \
    logger.wandb.entity=bioshape-lab \
    experiment=omics_readout \
    dataset.loader.parameters.adjacency_method=string \
    dataset.loader.parameters.node_sample_ratio=1.0 \
    dataset.loader.parameters.method=distance_correlation \
    optimizer.parameters.lr=0.001 \
    optimizer.parameters.weight_decay=0.0 \
    model.backbone.dropout=0.1 \
    model.feature_encoder.out_channels=16 \
    model.backbone.num_layers=2 \
    dataset.loader.parameters.adjacency_threshold=0.4 \
    model.readout.fc_dropout=0.1 \
    logger.wandb.project=best_model_reruns \
    trainer.devices=\[5\] \
    &

python -m ogbench \
    model=chebnet \
    dataset=motrpac \
    seed=42 \
    logger.wandb.tags=[chebnet,motrpac,hpsearch,best_rerun] \
    logger.wandb.entity=bioshape-lab \
    experiment=omics_readout \
    dataset.loader.parameters.adjacency_method=wgcna \
    dataset.loader.parameters.node_sample_ratio=0.5 \
    dataset.loader.parameters.method=random \
    optimizer.parameters.lr=0.0001 \
    optimizer.parameters.weight_decay=0.0 \
    model.backbone.dropout=0.1 \
    model.feature_encoder.out_channels=64 \
    model.backbone.num_layers=4 \
    dataset.loader.parameters.adjacency_threshold=0.0693 \
    model.readout.fc_dropout=0.1 \
    logger.wandb.project=best_model_reruns \
    trainer.devices=\[6\] \
    &

python -m ogbench \
    model=chebnet \
    dataset=motrpac \
    seed=123 \
    logger.wandb.tags=[chebnet,motrpac,hpsearch,best_rerun] \
    logger.wandb.entity=bioshape-lab \
    experiment=omics_readout \
    dataset.loader.parameters.adjacency_method=wgcna \
    dataset.loader.parameters.node_sample_ratio=0.5 \
    dataset.loader.parameters.method=random \
    optimizer.parameters.lr=0.0001 \
    optimizer.parameters.weight_decay=0.0 \
    model.backbone.dropout=0.1 \
    model.feature_encoder.out_channels=64 \
    model.backbone.num_layers=4 \
    dataset.loader.parameters.adjacency_threshold=0.0693 \
    model.readout.fc_dropout=0.1 \
    logger.wandb.project=best_model_reruns \
    trainer.devices=\[7\] \
    &

python -m ogbench \
    model=chebnet \
    dataset=motrpac \
    seed=456 \
    logger.wandb.tags=[chebnet,motrpac,hpsearch,best_rerun] \
    logger.wandb.entity=bioshape-lab \
    experiment=omics_readout \
    dataset.loader.parameters.adjacency_method=wgcna \
    dataset.loader.parameters.node_sample_ratio=0.5 \
    dataset.loader.parameters.method=random \
    optimizer.parameters.lr=0.0001 \
    optimizer.parameters.weight_decay=0.0 \
    model.backbone.dropout=0.1 \
    model.feature_encoder.out_channels=64 \
    model.backbone.num_layers=4 \
    dataset.loader.parameters.adjacency_threshold=0.0693 \
    model.readout.fc_dropout=0.1 \
    logger.wandb.project=best_model_reruns \
    trainer.devices=\[0\] \
    &

python -m ogbench \
    model=chebnet \
    dataset=parkinsons \
    seed=42 \
    logger.wandb.tags=[chebnet,parkinsons,hpsearch,best_rerun] \
    logger.wandb.entity=bioshape-lab \
    experiment=omics_readout \
    dataset.loader.parameters.adjacency_method=wgcna \
    dataset.loader.parameters.node_sample_ratio=0.3 \
    dataset.loader.parameters.method=random \
    optimizer.parameters.lr=0.001 \
    optimizer.parameters.weight_decay=0.0 \
    model.backbone.dropout=0.1 \
    model.feature_encoder.out_channels=64 \
    model.backbone.num_layers=2 \
    dataset.loader.parameters.adjacency_threshold=0.0043 \
    model.readout.fc_dropout=0.1 \
    logger.wandb.project=best_model_reruns \
    trainer.devices=\[1\] \
    &

python -m ogbench \
    model=chebnet \
    dataset=parkinsons \
    seed=123 \
    logger.wandb.tags=[chebnet,parkinsons,hpsearch,best_rerun] \
    logger.wandb.entity=bioshape-lab \
    experiment=omics_readout \
    dataset.loader.parameters.adjacency_method=wgcna \
    dataset.loader.parameters.node_sample_ratio=0.3 \
    dataset.loader.parameters.method=random \
    optimizer.parameters.lr=0.001 \
    optimizer.parameters.weight_decay=0.0 \
    model.backbone.dropout=0.1 \
    model.feature_encoder.out_channels=64 \
    model.backbone.num_layers=2 \
    dataset.loader.parameters.adjacency_threshold=0.0043 \
    model.readout.fc_dropout=0.1 \
    logger.wandb.project=best_model_reruns \
    trainer.devices=\[2\] \
    &

python -m ogbench \
    model=chebnet \
    dataset=parkinsons \
    seed=456 \
    logger.wandb.tags=[chebnet,parkinsons,hpsearch,best_rerun] \
    logger.wandb.entity=bioshape-lab \
    experiment=omics_readout \
    dataset.loader.parameters.adjacency_method=wgcna \
    dataset.loader.parameters.node_sample_ratio=0.3 \
    dataset.loader.parameters.method=random \
    optimizer.parameters.lr=0.001 \
    optimizer.parameters.weight_decay=0.0 \
    model.backbone.dropout=0.1 \
    model.feature_encoder.out_channels=64 \
    model.backbone.num_layers=2 \
    dataset.loader.parameters.adjacency_threshold=0.0043 \
    model.readout.fc_dropout=0.1 \
    logger.wandb.project=best_model_reruns \
    trainer.devices=\[3\] \
    &

python -m ogbench \
    model=gatv2 \
    dataset=addneuromed \
    seed=42 \
    logger.wandb.tags=[gatv2,addneuromed,hpsearch,best_rerun] \
    logger.wandb.entity=bioshape-lab \
    experiment=omics_readout \
    dataset.loader.parameters.adjacency_method=string \
    dataset.loader.parameters.node_sample_ratio=0.3 \
    dataset.loader.parameters.method=variance \
    optimizer.parameters.lr=0.001 \
    optimizer.parameters.weight_decay=0.0 \
    model.backbone.dropout=0.1 \
    model.feature_encoder.out_channels=16 \
    model.backbone.num_layers=2 \
    model.backbone.heads=4 \
    dataset.loader.parameters.adjacency_threshold=0.4 \
    model.readout.fc_dropout=0.1 \
    logger.wandb.project=best_model_reruns \
    trainer.devices=\[4\] \
    &

python -m ogbench \
    model=gatv2 \
    dataset=addneuromed \
    seed=123 \
    logger.wandb.tags=[gatv2,addneuromed,hpsearch,best_rerun] \
    logger.wandb.entity=bioshape-lab \
    experiment=omics_readout \
    dataset.loader.parameters.adjacency_method=string \
    dataset.loader.parameters.node_sample_ratio=0.3 \
    dataset.loader.parameters.method=variance \
    optimizer.parameters.lr=0.001 \
    optimizer.parameters.weight_decay=0.0 \
    model.backbone.dropout=0.1 \
    model.feature_encoder.out_channels=16 \
    model.backbone.num_layers=2 \
    model.backbone.heads=4 \
    dataset.loader.parameters.adjacency_threshold=0.4 \
    model.readout.fc_dropout=0.1 \
    logger.wandb.project=best_model_reruns \
    trainer.devices=\[5\] \
    &

python -m ogbench \
    model=gatv2 \
    dataset=addneuromed \
    seed=456 \
    logger.wandb.tags=[gatv2,addneuromed,hpsearch,best_rerun] \
    logger.wandb.entity=bioshape-lab \
    experiment=omics_readout \
    dataset.loader.parameters.adjacency_method=string \
    dataset.loader.parameters.node_sample_ratio=0.3 \
    dataset.loader.parameters.method=variance \
    optimizer.parameters.lr=0.001 \
    optimizer.parameters.weight_decay=0.0 \
    model.backbone.dropout=0.1 \
    model.feature_encoder.out_channels=16 \
    model.backbone.num_layers=2 \
    model.backbone.heads=4 \
    dataset.loader.parameters.adjacency_threshold=0.4 \
    model.readout.fc_dropout=0.1 \
    logger.wandb.project=best_model_reruns \
    trainer.devices=\[6\] \
    &

python -m ogbench \
    model=gatv2 \
    dataset=brca \
    seed=42 \
    logger.wandb.tags=[gatv2,brca,hpsearch,best_rerun] \
    logger.wandb.entity=bioshape-lab \
    experiment=omics_readout \
    dataset.loader.parameters.adjacency_method=string \
    dataset.loader.parameters.node_sample_ratio=1.0 \
    dataset.loader.parameters.method=distance_correlation \
    optimizer.parameters.lr=0.001 \
    optimizer.parameters.weight_decay=0.0 \
    model.backbone.dropout=0.1 \
    model.feature_encoder.out_channels=64 \
    model.backbone.num_layers=2 \
    model.backbone.heads=4 \
    dataset.loader.parameters.adjacency_threshold=0.4 \
    model.readout.fc_dropout=0.1 \
    logger.wandb.project=best_model_reruns \
    trainer.devices=\[7\] \
    &

python -m ogbench \
    model=gatv2 \
    dataset=brca \
    seed=123 \
    logger.wandb.tags=[gatv2,brca,hpsearch,best_rerun] \
    logger.wandb.entity=bioshape-lab \
    experiment=omics_readout \
    dataset.loader.parameters.adjacency_method=string \
    dataset.loader.parameters.node_sample_ratio=1.0 \
    dataset.loader.parameters.method=distance_correlation \
    optimizer.parameters.lr=0.001 \
    optimizer.parameters.weight_decay=0.0 \
    model.backbone.dropout=0.1 \
    model.feature_encoder.out_channels=64 \
    model.backbone.num_layers=2 \
    model.backbone.heads=4 \
    dataset.loader.parameters.adjacency_threshold=0.4 \
    model.readout.fc_dropout=0.1 \
    logger.wandb.project=best_model_reruns \
    trainer.devices=\[0\] \
    &

python -m ogbench \
    model=gatv2 \
    dataset=brca \
    seed=456 \
    logger.wandb.tags=[gatv2,brca,hpsearch,best_rerun] \
    logger.wandb.entity=bioshape-lab \
    experiment=omics_readout \
    dataset.loader.parameters.adjacency_method=string \
    dataset.loader.parameters.node_sample_ratio=1.0 \
    dataset.loader.parameters.method=distance_correlation \
    optimizer.parameters.lr=0.001 \
    optimizer.parameters.weight_decay=0.0 \
    model.backbone.dropout=0.1 \
    model.feature_encoder.out_channels=64 \
    model.backbone.num_layers=2 \
    model.backbone.heads=4 \
    dataset.loader.parameters.adjacency_threshold=0.4 \
    model.readout.fc_dropout=0.1 \
    logger.wandb.project=best_model_reruns \
    trainer.devices=\[1\] \
    &

python -m ogbench \
    model=gatv2 \
    dataset=motrpac \
    seed=42 \
    logger.wandb.tags=[gatv2,motrpac,hpsearch,best_rerun] \
    logger.wandb.entity=bioshape-lab \
    experiment=omics_readout \
    dataset.loader.parameters.adjacency_method=wgcna \
    dataset.loader.parameters.node_sample_ratio=0.5 \
    dataset.loader.parameters.method=random \
    optimizer.parameters.lr=0.001 \
    optimizer.parameters.weight_decay=0.0 \
    model.backbone.dropout=0.1 \
    model.feature_encoder.out_channels=16 \
    model.backbone.num_layers=4 \
    model.backbone.heads=2 \
    dataset.loader.parameters.adjacency_threshold=0.0693 \
    model.readout.fc_dropout=0.1 \
    logger.wandb.project=best_model_reruns \
    trainer.devices=\[2\] \
    &

python -m ogbench \
    model=gatv2 \
    dataset=motrpac \
    seed=123 \
    logger.wandb.tags=[gatv2,motrpac,hpsearch,best_rerun] \
    logger.wandb.entity=bioshape-lab \
    experiment=omics_readout \
    dataset.loader.parameters.adjacency_method=wgcna \
    dataset.loader.parameters.node_sample_ratio=0.5 \
    dataset.loader.parameters.method=random \
    optimizer.parameters.lr=0.001 \
    optimizer.parameters.weight_decay=0.0 \
    model.backbone.dropout=0.1 \
    model.feature_encoder.out_channels=16 \
    model.backbone.num_layers=4 \
    model.backbone.heads=2 \
    dataset.loader.parameters.adjacency_threshold=0.0693 \
    model.readout.fc_dropout=0.1 \
    logger.wandb.project=best_model_reruns \
    trainer.devices=\[3\] \
    &

python -m ogbench \
    model=gatv2 \
    dataset=motrpac \
    seed=456 \
    logger.wandb.tags=[gatv2,motrpac,hpsearch,best_rerun] \
    logger.wandb.entity=bioshape-lab \
    experiment=omics_readout \
    dataset.loader.parameters.adjacency_method=wgcna \
    dataset.loader.parameters.node_sample_ratio=0.5 \
    dataset.loader.parameters.method=random \
    optimizer.parameters.lr=0.001 \
    optimizer.parameters.weight_decay=0.0 \
    model.backbone.dropout=0.1 \
    model.feature_encoder.out_channels=16 \
    model.backbone.num_layers=4 \
    model.backbone.heads=2 \
    dataset.loader.parameters.adjacency_threshold=0.0693 \
    model.readout.fc_dropout=0.1 \
    logger.wandb.project=best_model_reruns \
    trainer.devices=\[4\] \
    &

python -m ogbench \
    model=gatv2 \
    dataset=parkinsons \
    seed=42 \
    logger.wandb.tags=[gatv2,parkinsons,hpsearch,best_rerun] \
    logger.wandb.entity=bioshape-lab \
    experiment=omics_readout \
    dataset.loader.parameters.adjacency_method=wgcna \
    dataset.loader.parameters.node_sample_ratio=0.3 \
    dataset.loader.parameters.method=distance_correlation \
    optimizer.parameters.lr=0.001 \
    optimizer.parameters.weight_decay=0.0 \
    model.backbone.dropout=0.1 \
    model.feature_encoder.out_channels=64 \
    model.backbone.num_layers=4 \
    model.backbone.heads=2 \
    dataset.loader.parameters.adjacency_threshold=0.0005 \
    model.readout.fc_dropout=0.1 \
    logger.wandb.project=best_model_reruns \
    trainer.devices=\[5\] \
    &

python -m ogbench \
    model=gatv2 \
    dataset=parkinsons \
    seed=123 \
    logger.wandb.tags=[gatv2,parkinsons,hpsearch,best_rerun] \
    logger.wandb.entity=bioshape-lab \
    experiment=omics_readout \
    dataset.loader.parameters.adjacency_method=wgcna \
    dataset.loader.parameters.node_sample_ratio=0.3 \
    dataset.loader.parameters.method=distance_correlation \
    optimizer.parameters.lr=0.001 \
    optimizer.parameters.weight_decay=0.0 \
    model.backbone.dropout=0.1 \
    model.feature_encoder.out_channels=64 \
    model.backbone.num_layers=4 \
    model.backbone.heads=2 \
    dataset.loader.parameters.adjacency_threshold=0.0005 \
    model.readout.fc_dropout=0.1 \
    logger.wandb.project=best_model_reruns \
    trainer.devices=\[6\] \
    &

python -m ogbench \
    model=gatv2 \
    dataset=parkinsons \
    seed=456 \
    logger.wandb.tags=[gatv2,parkinsons,hpsearch,best_rerun] \
    logger.wandb.entity=bioshape-lab \
    experiment=omics_readout \
    dataset.loader.parameters.adjacency_method=wgcna \
    dataset.loader.parameters.node_sample_ratio=0.3 \
    dataset.loader.parameters.method=distance_correlation \
    optimizer.parameters.lr=0.001 \
    optimizer.parameters.weight_decay=0.0 \
    model.backbone.dropout=0.1 \
    model.feature_encoder.out_channels=64 \
    model.backbone.num_layers=4 \
    model.backbone.heads=2 \
    dataset.loader.parameters.adjacency_threshold=0.0005 \
    model.readout.fc_dropout=0.1 \
    logger.wandb.project=best_model_reruns \
    trainer.devices=\[7\] \
    &

python -m ogbench \
    model=gatv4 \
    dataset=addneuromed \
    seed=42 \
    logger.wandb.tags=[gatv4,addneuromed,hpsearch,best_rerun] \
    logger.wandb.entity=bioshape-lab \
    experiment=omics_readout \
    dataset.loader.parameters.adjacency_method=string \
    dataset.loader.parameters.node_sample_ratio=0.5 \
    dataset.loader.parameters.method=variance \
    optimizer.parameters.lr=0.001 \
    optimizer.parameters.weight_decay=0.0 \
    model.backbone.dropout=0.1 \
    model.backbone.hidden_channels=[32,64] \
    model.backbone.heads=[4,4] \
    model.backbone.use_layer_norm=false \
    dataset.loader.parameters.adjacency_threshold=0.4 \
    model.readout.fc_dropout=0.1 \
    logger.wandb.project=best_model_reruns \
    trainer.devices=\[0\] \
    &

python -m ogbench \
    model=gatv4 \
    dataset=addneuromed \
    seed=123 \
    logger.wandb.tags=[gatv4,addneuromed,hpsearch,best_rerun] \
    logger.wandb.entity=bioshape-lab \
    experiment=omics_readout \
    dataset.loader.parameters.adjacency_method=string \
    dataset.loader.parameters.node_sample_ratio=0.5 \
    dataset.loader.parameters.method=variance \
    optimizer.parameters.lr=0.001 \
    optimizer.parameters.weight_decay=0.0 \
    model.backbone.dropout=0.1 \
    model.backbone.hidden_channels=[32,64] \
    model.backbone.heads=[4,4] \
    model.backbone.use_layer_norm=true \
    dataset.loader.parameters.adjacency_threshold=0.4 \
    model.readout.fc_dropout=0.1 \
    logger.wandb.project=best_model_reruns \
    trainer.devices=\[1\] \
    &

python -m ogbench \
    model=gatv4 \
    dataset=addneuromed \
    seed=456 \
    logger.wandb.tags=[gatv4,addneuromed,hpsearch,best_rerun] \
    logger.wandb.entity=bioshape-lab \
    experiment=omics_readout \
    dataset.loader.parameters.adjacency_method=string \
    dataset.loader.parameters.node_sample_ratio=0.5 \
    dataset.loader.parameters.method=variance \
    optimizer.parameters.lr=0.001 \
    optimizer.parameters.weight_decay=0.0 \
    model.backbone.dropout=0.1 \
    model.backbone.hidden_channels=[32,64] \
    model.backbone.heads=[4,4] \
    model.backbone.use_layer_norm=false \
    dataset.loader.parameters.adjacency_threshold=0.4 \
    model.readout.fc_dropout=0.1 \
    logger.wandb.project=best_model_reruns \
    trainer.devices=\[2\] \
    &

python -m ogbench \
    model=gatv4 \
    dataset=brca \
    seed=42 \
    logger.wandb.tags=[gatv4,brca,hpsearch,best_rerun] \
    logger.wandb.entity=bioshape-lab \
    experiment=omics_readout \
    dataset.loader.parameters.adjacency_method=string \
    dataset.loader.parameters.node_sample_ratio=0.8 \
    dataset.loader.parameters.method=distance_correlation \
    optimizer.parameters.lr=0.001 \
    optimizer.parameters.weight_decay=0.0 \
    model.backbone.dropout=0.1 \
    model.backbone.hidden_channels=[32,64] \
    model.backbone.heads=[4,4] \
    model.backbone.use_layer_norm=false \
    dataset.loader.parameters.adjacency_threshold=0.4 \
    model.readout.fc_dropout=0.1 \
    logger.wandb.project=best_model_reruns \
    trainer.devices=\[3\] \
    &

python -m ogbench \
    model=gatv4 \
    dataset=brca \
    seed=123 \
    logger.wandb.tags=[gatv4,brca,hpsearch,best_rerun] \
    logger.wandb.entity=bioshape-lab \
    experiment=omics_readout \
    dataset.loader.parameters.adjacency_method=string \
    dataset.loader.parameters.node_sample_ratio=0.8 \
    dataset.loader.parameters.method=distance_correlation \
    optimizer.parameters.lr=0.001 \
    optimizer.parameters.weight_decay=0.0 \
    model.backbone.dropout=0.1 \
    model.backbone.hidden_channels=[32,64] \
    model.backbone.heads=[4,4] \
    model.backbone.use_layer_norm=false \
    dataset.loader.parameters.adjacency_threshold=0.4 \
    model.readout.fc_dropout=0.1 \
    logger.wandb.project=best_model_reruns \
    trainer.devices=\[4\] \
    &

python -m ogbench \
    model=gatv4 \
    dataset=brca \
    seed=456 \
    logger.wandb.tags=[gatv4,brca,hpsearch,best_rerun] \
    logger.wandb.entity=bioshape-lab \
    experiment=omics_readout \
    dataset.loader.parameters.adjacency_method=string \
    dataset.loader.parameters.node_sample_ratio=0.8 \
    dataset.loader.parameters.method=distance_correlation \
    optimizer.parameters.lr=0.001 \
    optimizer.parameters.weight_decay=0.0 \
    model.backbone.dropout=0.1 \
    model.backbone.hidden_channels=[32,64] \
    model.backbone.heads=[4,4] \
    model.backbone.use_layer_norm=true \
    dataset.loader.parameters.adjacency_threshold=0.4 \
    model.readout.fc_dropout=0.1 \
    logger.wandb.project=best_model_reruns \
    trainer.devices=\[5\] \
    &

python -m ogbench \
    model=gatv4 \
    dataset=motrpac \
    seed=42 \
    logger.wandb.tags=[gatv4,motrpac,hpsearch,best_rerun] \
    logger.wandb.entity=bioshape-lab \
    experiment=omics_readout \
    dataset.loader.parameters.adjacency_method=string \
    dataset.loader.parameters.node_sample_ratio=0.3 \
    dataset.loader.parameters.method=correlation \
    optimizer.parameters.lr=0.001 \
    optimizer.parameters.weight_decay=0.0 \
    model.backbone.dropout=0.1 \
    model.backbone.hidden_channels=[32,64] \
    model.backbone.heads=[4,4] \
    model.backbone.use_layer_norm=true \
    dataset.loader.parameters.adjacency_threshold=0.4 \
    model.readout.fc_dropout=0.1 \
    logger.wandb.project=best_model_reruns \
    trainer.devices=\[6\] \
    &

python -m ogbench \
    model=gatv4 \
    dataset=motrpac \
    seed=123 \
    logger.wandb.tags=[gatv4,motrpac,hpsearch,best_rerun] \
    logger.wandb.entity=bioshape-lab \
    experiment=omics_readout \
    dataset.loader.parameters.adjacency_method=string \
    dataset.loader.parameters.node_sample_ratio=0.3 \
    dataset.loader.parameters.method=correlation \
    optimizer.parameters.lr=0.001 \
    optimizer.parameters.weight_decay=0.0 \
    model.backbone.dropout=0.1 \
    model.backbone.hidden_channels=[32,64] \
    model.backbone.heads=[4,4] \
    model.backbone.use_layer_norm=true \
    dataset.loader.parameters.adjacency_threshold=0.4 \
    model.readout.fc_dropout=0.1 \
    logger.wandb.project=best_model_reruns \
    trainer.devices=\[7\] \
    &

python -m ogbench \
    model=gatv4 \
    dataset=motrpac \
    seed=456 \
    logger.wandb.tags=[gatv4,motrpac,hpsearch,best_rerun] \
    logger.wandb.entity=bioshape-lab \
    experiment=omics_readout \
    dataset.loader.parameters.adjacency_method=string \
    dataset.loader.parameters.node_sample_ratio=0.3 \
    dataset.loader.parameters.method=correlation \
    optimizer.parameters.lr=0.001 \
    optimizer.parameters.weight_decay=0.0 \
    model.backbone.dropout=0.1 \
    model.backbone.hidden_channels=[32,64] \
    model.backbone.heads=[4,4] \
    model.backbone.use_layer_norm=true \
    dataset.loader.parameters.adjacency_threshold=0.4 \
    model.readout.fc_dropout=0.1 \
    logger.wandb.project=best_model_reruns \
    trainer.devices=\[0\] \
    &

python -m ogbench \
    model=gatv4 \
    dataset=parkinsons \
    seed=42 \
    logger.wandb.tags=[gatv4,parkinsons,hpsearch,best_rerun] \
    logger.wandb.entity=bioshape-lab \
    experiment=omics_readout \
    dataset.loader.parameters.adjacency_method=wgcna \
    dataset.loader.parameters.node_sample_ratio=0.8 \
    dataset.loader.parameters.method=random \
    optimizer.parameters.lr=0.0001 \
    optimizer.parameters.weight_decay=0.0 \
    model.backbone.dropout=0.1 \
    model.backbone.hidden_channels=[32,64] \
    model.backbone.heads=[4,4] \
    model.backbone.use_layer_norm=true \
    dataset.loader.parameters.adjacency_threshold=0.0216 \
    model.readout.fc_dropout=0.1 \
    logger.wandb.project=best_model_reruns \
    trainer.devices=\[1\] \
    &

python -m ogbench \
    model=gatv4 \
    dataset=parkinsons \
    seed=123 \
    logger.wandb.tags=[gatv4,parkinsons,hpsearch,best_rerun] \
    logger.wandb.entity=bioshape-lab \
    experiment=omics_readout \
    dataset.loader.parameters.adjacency_method=wgcna \
    dataset.loader.parameters.node_sample_ratio=0.8 \
    dataset.loader.parameters.method=random \
    optimizer.parameters.lr=0.0001 \
    optimizer.parameters.weight_decay=0.0 \
    model.backbone.dropout=0.1 \
    model.backbone.hidden_channels=[32,64] \
    model.backbone.heads=[4,4] \
    model.backbone.use_layer_norm=false \
    dataset.loader.parameters.adjacency_threshold=0.0216 \
    model.readout.fc_dropout=0.1 \
    logger.wandb.project=best_model_reruns \
    trainer.devices=\[2\] \
    &

python -m ogbench \
    model=gatv4 \
    dataset=parkinsons \
    seed=456 \
    logger.wandb.tags=[gatv4,parkinsons,hpsearch,best_rerun] \
    logger.wandb.entity=bioshape-lab \
    experiment=omics_readout \
    dataset.loader.parameters.adjacency_method=wgcna \
    dataset.loader.parameters.node_sample_ratio=0.8 \
    dataset.loader.parameters.method=random \
    optimizer.parameters.lr=0.0001 \
    optimizer.parameters.weight_decay=0.0 \
    model.backbone.dropout=0.1 \
    model.backbone.hidden_channels=[32,64] \
    model.backbone.heads=[4,4] \
    model.backbone.use_layer_norm=true \
    dataset.loader.parameters.adjacency_threshold=0.0216 \
    model.readout.fc_dropout=0.1 \
    logger.wandb.project=best_model_reruns \
    trainer.devices=\[3\] \
    &

python -m ogbench \
    model=gcn \
    dataset=addneuromed \
    seed=42 \
    logger.wandb.tags=[gcn,addneuromed,hpsearch,best_rerun] \
    logger.wandb.entity=bioshape-lab \
    experiment=omics_readout \
    dataset.loader.parameters.adjacency_method=string \
    dataset.loader.parameters.node_sample_ratio=0.3 \
    dataset.loader.parameters.method=variance \
    optimizer.parameters.lr=0.001 \
    optimizer.parameters.weight_decay=0.0 \
    model.backbone.dropout=0.1 \
    model.feature_encoder.out_channels=64 \
    model.backbone.num_layers=2 \
    dataset.loader.parameters.adjacency_threshold=0.4 \
    model.readout.fc_dropout=0.1 \
    logger.wandb.project=best_model_reruns \
    trainer.devices=\[4\] \
    &

python -m ogbench \
    model=gcn \
    dataset=addneuromed \
    seed=123 \
    logger.wandb.tags=[gcn,addneuromed,hpsearch,best_rerun] \
    logger.wandb.entity=bioshape-lab \
    experiment=omics_readout \
    dataset.loader.parameters.adjacency_method=string \
    dataset.loader.parameters.node_sample_ratio=0.3 \
    dataset.loader.parameters.method=variance \
    optimizer.parameters.lr=0.001 \
    optimizer.parameters.weight_decay=0.0 \
    model.backbone.dropout=0.1 \
    model.feature_encoder.out_channels=64 \
    model.backbone.num_layers=2 \
    dataset.loader.parameters.adjacency_threshold=0.4 \
    model.readout.fc_dropout=0.1 \
    logger.wandb.project=best_model_reruns \
    trainer.devices=\[5\] \
    &

python -m ogbench \
    model=gcn \
    dataset=addneuromed \
    seed=456 \
    logger.wandb.tags=[gcn,addneuromed,hpsearch,best_rerun] \
    logger.wandb.entity=bioshape-lab \
    experiment=omics_readout \
    dataset.loader.parameters.adjacency_method=string \
    dataset.loader.parameters.node_sample_ratio=0.3 \
    dataset.loader.parameters.method=variance \
    optimizer.parameters.lr=0.001 \
    optimizer.parameters.weight_decay=0.0 \
    model.backbone.dropout=0.1 \
    model.feature_encoder.out_channels=64 \
    model.backbone.num_layers=2 \
    dataset.loader.parameters.adjacency_threshold=0.4 \
    model.readout.fc_dropout=0.1 \
    logger.wandb.project=best_model_reruns \
    trainer.devices=\[6\] \
    &

python -m ogbench \
    model=gcn \
    dataset=brca \
    seed=42 \
    logger.wandb.tags=[gcn,brca,hpsearch,best_rerun] \
    logger.wandb.entity=bioshape-lab \
    experiment=omics_readout \
    dataset.loader.parameters.adjacency_method=wgcna \
    dataset.loader.parameters.node_sample_ratio=0.3 \
    dataset.loader.parameters.method=distance_correlation \
    optimizer.parameters.lr=0.001 \
    optimizer.parameters.weight_decay=0.0 \
    model.backbone.dropout=0.1 \
    model.feature_encoder.out_channels=16 \
    model.backbone.num_layers=2 \
    dataset.loader.parameters.adjacency_threshold=0.0005 \
    model.readout.fc_dropout=0.1 \
    logger.wandb.project=best_model_reruns \
    trainer.devices=\[7\] \
    &

python -m ogbench \
    model=gcn \
    dataset=brca \
    seed=123 \
    logger.wandb.tags=[gcn,brca,hpsearch,best_rerun] \
    logger.wandb.entity=bioshape-lab \
    experiment=omics_readout \
    dataset.loader.parameters.adjacency_method=wgcna \
    dataset.loader.parameters.node_sample_ratio=0.3 \
    dataset.loader.parameters.method=distance_correlation \
    optimizer.parameters.lr=0.001 \
    optimizer.parameters.weight_decay=0.0 \
    model.backbone.dropout=0.1 \
    model.feature_encoder.out_channels=16 \
    model.backbone.num_layers=2 \
    dataset.loader.parameters.adjacency_threshold=0.0005 \
    model.readout.fc_dropout=0.1 \
    logger.wandb.project=best_model_reruns \
    trainer.devices=\[0\] \
    &

python -m ogbench \
    model=gcn \
    dataset=brca \
    seed=456 \
    logger.wandb.tags=[gcn,brca,hpsearch,best_rerun] \
    logger.wandb.entity=bioshape-lab \
    experiment=omics_readout \
    dataset.loader.parameters.adjacency_method=wgcna \
    dataset.loader.parameters.node_sample_ratio=0.3 \
    dataset.loader.parameters.method=distance_correlation \
    optimizer.parameters.lr=0.001 \
    optimizer.parameters.weight_decay=0.0 \
    model.backbone.dropout=0.1 \
    model.feature_encoder.out_channels=16 \
    model.backbone.num_layers=2 \
    dataset.loader.parameters.adjacency_threshold=0.0005 \
    model.readout.fc_dropout=0.1 \
    logger.wandb.project=best_model_reruns \
    trainer.devices=\[1\] \
    &

python -m ogbench \
    model=gcn \
    dataset=motrpac \
    seed=42 \
    logger.wandb.tags=[gcn,motrpac,hpsearch,best_rerun] \
    logger.wandb.entity=bioshape-lab \
    experiment=omics_readout \
    dataset.loader.parameters.adjacency_method=wgcna \
    dataset.loader.parameters.node_sample_ratio=0.3 \
    dataset.loader.parameters.method=correlation \
    optimizer.parameters.lr=0.001 \
    optimizer.parameters.weight_decay=0.0 \
    model.backbone.dropout=0.1 \
    model.feature_encoder.out_channels=16 \
    model.backbone.num_layers=4 \
    dataset.loader.parameters.adjacency_threshold=0.0005 \
    model.readout.fc_dropout=0.1 \
    logger.wandb.project=best_model_reruns \
    trainer.devices=\[2\] \
    &

python -m ogbench \
    model=gcn \
    dataset=motrpac \
    seed=123 \
    logger.wandb.tags=[gcn,motrpac,hpsearch,best_rerun] \
    logger.wandb.entity=bioshape-lab \
    experiment=omics_readout \
    dataset.loader.parameters.adjacency_method=wgcna \
    dataset.loader.parameters.node_sample_ratio=0.3 \
    dataset.loader.parameters.method=correlation \
    optimizer.parameters.lr=0.001 \
    optimizer.parameters.weight_decay=0.0 \
    model.backbone.dropout=0.1 \
    model.feature_encoder.out_channels=16 \
    model.backbone.num_layers=4 \
    dataset.loader.parameters.adjacency_threshold=0.0005 \
    model.readout.fc_dropout=0.1 \
    logger.wandb.project=best_model_reruns \
    trainer.devices=\[3\] \
    &

python -m ogbench \
    model=gcn \
    dataset=motrpac \
    seed=456 \
    logger.wandb.tags=[gcn,motrpac,hpsearch,best_rerun] \
    logger.wandb.entity=bioshape-lab \
    experiment=omics_readout \
    dataset.loader.parameters.adjacency_method=wgcna \
    dataset.loader.parameters.node_sample_ratio=0.3 \
    dataset.loader.parameters.method=correlation \
    optimizer.parameters.lr=0.001 \
    optimizer.parameters.weight_decay=0.0 \
    model.backbone.dropout=0.1 \
    model.feature_encoder.out_channels=16 \
    model.backbone.num_layers=4 \
    dataset.loader.parameters.adjacency_threshold=0.0005 \
    model.readout.fc_dropout=0.1 \
    logger.wandb.project=best_model_reruns \
    trainer.devices=\[4\] \
    &

python -m ogbench \
    model=gcn \
    dataset=parkinsons \
    seed=42 \
    logger.wandb.tags=[gcn,parkinsons,hpsearch,best_rerun] \
    logger.wandb.entity=bioshape-lab \
    experiment=omics_readout \
    dataset.loader.parameters.adjacency_method=string \
    dataset.loader.parameters.node_sample_ratio=0.8 \
    dataset.loader.parameters.method=random \
    optimizer.parameters.lr=0.0001 \
    optimizer.parameters.weight_decay=0.0 \
    model.backbone.dropout=0.1 \
    model.feature_encoder.out_channels=64 \
    model.backbone.num_layers=4 \
    dataset.loader.parameters.adjacency_threshold=0.4 \
    model.readout.fc_dropout=0.1 \
    logger.wandb.project=best_model_reruns \
    trainer.devices=\[5\] \
    &

python -m ogbench \
    model=gcn \
    dataset=parkinsons \
    seed=123 \
    logger.wandb.tags=[gcn,parkinsons,hpsearch,best_rerun] \
    logger.wandb.entity=bioshape-lab \
    experiment=omics_readout \
    dataset.loader.parameters.adjacency_method=string \
    dataset.loader.parameters.node_sample_ratio=0.8 \
    dataset.loader.parameters.method=random \
    optimizer.parameters.lr=0.0001 \
    optimizer.parameters.weight_decay=0.0 \
    model.backbone.dropout=0.1 \
    model.feature_encoder.out_channels=64 \
    model.backbone.num_layers=4 \
    dataset.loader.parameters.adjacency_threshold=0.4 \
    model.readout.fc_dropout=0.1 \
    logger.wandb.project=best_model_reruns \
    trainer.devices=\[6\] \
    &

python -m ogbench \
    model=gcn \
    dataset=parkinsons \
    seed=456 \
    logger.wandb.tags=[gcn,parkinsons,hpsearch,best_rerun] \
    logger.wandb.entity=bioshape-lab \
    experiment=omics_readout \
    dataset.loader.parameters.adjacency_method=string \
    dataset.loader.parameters.node_sample_ratio=0.8 \
    dataset.loader.parameters.method=random \
    optimizer.parameters.lr=0.0001 \
    optimizer.parameters.weight_decay=0.0 \
    model.backbone.dropout=0.1 \
    model.feature_encoder.out_channels=64 \
    model.backbone.num_layers=4 \
    dataset.loader.parameters.adjacency_threshold=0.4 \
    model.readout.fc_dropout=0.1 \
    logger.wandb.project=best_model_reruns \
    trainer.devices=\[7\] \
    &

python -m ogbench \
    model=gin \
    dataset=addneuromed \
    seed=42 \
    logger.wandb.tags=[gin,addneuromed,hpsearch,best_rerun] \
    logger.wandb.entity=bioshape-lab \
    experiment=no_readout \
    dataset.loader.parameters.adjacency_method=string \
    dataset.loader.parameters.node_sample_ratio=0.8 \
    dataset.loader.parameters.method=distance_correlation \
    optimizer.parameters.lr=0.001 \
    optimizer.parameters.weight_decay=0.0 \
    model.backbone.dropout=0.1 \
    model.feature_encoder.out_channels=64 \
    model.backbone.num_layers=2 \
    dataset.loader.parameters.adjacency_threshold=0.4 \
    logger.wandb.project=best_model_reruns \
    trainer.devices=\[0\] \
    &

python -m ogbench \
    model=gin \
    dataset=addneuromed \
    seed=123 \
    logger.wandb.tags=[gin,addneuromed,hpsearch,best_rerun] \
    logger.wandb.entity=bioshape-lab \
    experiment=no_readout \
    dataset.loader.parameters.adjacency_method=string \
    dataset.loader.parameters.node_sample_ratio=0.8 \
    dataset.loader.parameters.method=distance_correlation \
    optimizer.parameters.lr=0.001 \
    optimizer.parameters.weight_decay=0.0 \
    model.backbone.dropout=0.1 \
    model.feature_encoder.out_channels=64 \
    model.backbone.num_layers=2 \
    dataset.loader.parameters.adjacency_threshold=0.4 \
    logger.wandb.project=best_model_reruns \
    trainer.devices=\[1\] \
    &

python -m ogbench \
    model=gin \
    dataset=addneuromed \
    seed=456 \
    logger.wandb.tags=[gin,addneuromed,hpsearch,best_rerun] \
    logger.wandb.entity=bioshape-lab \
    experiment=no_readout \
    dataset.loader.parameters.adjacency_method=string \
    dataset.loader.parameters.node_sample_ratio=0.8 \
    dataset.loader.parameters.method=distance_correlation \
    optimizer.parameters.lr=0.001 \
    optimizer.parameters.weight_decay=0.0 \
    model.backbone.dropout=0.1 \
    model.feature_encoder.out_channels=64 \
    model.backbone.num_layers=2 \
    dataset.loader.parameters.adjacency_threshold=0.4 \
    logger.wandb.project=best_model_reruns \
    trainer.devices=\[2\] \
    &

python -m ogbench \
    model=gin \
    dataset=brca \
    seed=42 \
    logger.wandb.tags=[gin,brca,hpsearch,best_rerun] \
    logger.wandb.entity=bioshape-lab \
    experiment=omics_readout \
    dataset.loader.parameters.adjacency_method=string \
    dataset.loader.parameters.node_sample_ratio=1.0 \
    dataset.loader.parameters.method=correlation \
    optimizer.parameters.lr=0.001 \
    optimizer.parameters.weight_decay=0.0 \
    model.backbone.dropout=0.1 \
    model.feature_encoder.out_channels=16 \
    model.backbone.num_layers=2 \
    dataset.loader.parameters.adjacency_threshold=0.4 \
    model.readout.fc_dropout=0.1 \
    logger.wandb.project=best_model_reruns \
    trainer.devices=\[3\] \
    &

python -m ogbench \
    model=gin \
    dataset=brca \
    seed=123 \
    logger.wandb.tags=[gin,brca,hpsearch,best_rerun] \
    logger.wandb.entity=bioshape-lab \
    experiment=omics_readout \
    dataset.loader.parameters.adjacency_method=string \
    dataset.loader.parameters.node_sample_ratio=1.0 \
    dataset.loader.parameters.method=correlation \
    optimizer.parameters.lr=0.001 \
    optimizer.parameters.weight_decay=0.0 \
    model.backbone.dropout=0.1 \
    model.feature_encoder.out_channels=16 \
    model.backbone.num_layers=2 \
    dataset.loader.parameters.adjacency_threshold=0.4 \
    model.readout.fc_dropout=0.1 \
    logger.wandb.project=best_model_reruns \
    trainer.devices=\[4\] \
    &

python -m ogbench \
    model=gin \
    dataset=brca \
    seed=456 \
    logger.wandb.tags=[gin,brca,hpsearch,best_rerun] \
    logger.wandb.entity=bioshape-lab \
    experiment=omics_readout \
    dataset.loader.parameters.adjacency_method=string \
    dataset.loader.parameters.node_sample_ratio=1.0 \
    dataset.loader.parameters.method=correlation \
    optimizer.parameters.lr=0.001 \
    optimizer.parameters.weight_decay=0.0 \
    model.backbone.dropout=0.1 \
    model.feature_encoder.out_channels=16 \
    model.backbone.num_layers=2 \
    dataset.loader.parameters.adjacency_threshold=0.4 \
    model.readout.fc_dropout=0.1 \
    logger.wandb.project=best_model_reruns \
    trainer.devices=\[5\] \
    &

python -m ogbench \
    model=gin \
    dataset=motrpac \
    seed=42 \
    logger.wandb.tags=[gin,motrpac,hpsearch,best_rerun] \
    logger.wandb.entity=bioshape-lab \
    experiment=omics_readout \
    dataset.loader.parameters.adjacency_method=string \
    dataset.loader.parameters.node_sample_ratio=1.0 \
    dataset.loader.parameters.method=random \
    optimizer.parameters.lr=0.001 \
    optimizer.parameters.weight_decay=0.0 \
    model.backbone.dropout=0.1 \
    model.feature_encoder.out_channels=16 \
    model.backbone.num_layers=2 \
    dataset.loader.parameters.adjacency_threshold=0.4 \
    model.readout.fc_dropout=0.1 \
    logger.wandb.project=best_model_reruns \
    trainer.devices=\[6\] \
    &

python -m ogbench \
    model=gin \
    dataset=motrpac \
    seed=123 \
    logger.wandb.tags=[gin,motrpac,hpsearch,best_rerun] \
    logger.wandb.entity=bioshape-lab \
    experiment=omics_readout \
    dataset.loader.parameters.adjacency_method=string \
    dataset.loader.parameters.node_sample_ratio=1.0 \
    dataset.loader.parameters.method=random \
    optimizer.parameters.lr=0.001 \
    optimizer.parameters.weight_decay=0.0 \
    model.backbone.dropout=0.1 \
    model.feature_encoder.out_channels=16 \
    model.backbone.num_layers=2 \
    dataset.loader.parameters.adjacency_threshold=0.4 \
    model.readout.fc_dropout=0.1 \
    logger.wandb.project=best_model_reruns \
    trainer.devices=\[7\] \
    &

python -m ogbench \
    model=gin \
    dataset=motrpac \
    seed=456 \
    logger.wandb.tags=[gin,motrpac,hpsearch,best_rerun] \
    logger.wandb.entity=bioshape-lab \
    experiment=omics_readout \
    dataset.loader.parameters.adjacency_method=string \
    dataset.loader.parameters.node_sample_ratio=1.0 \
    dataset.loader.parameters.method=random \
    optimizer.parameters.lr=0.001 \
    optimizer.parameters.weight_decay=0.0 \
    model.backbone.dropout=0.1 \
    model.feature_encoder.out_channels=16 \
    model.backbone.num_layers=2 \
    dataset.loader.parameters.adjacency_threshold=0.4 \
    model.readout.fc_dropout=0.1 \
    logger.wandb.project=best_model_reruns \
    trainer.devices=\[0\] \
    &

python -m ogbench \
    model=gin \
    dataset=parkinsons \
    seed=42 \
    logger.wandb.tags=[gin,parkinsons,hpsearch,best_rerun] \
    logger.wandb.entity=bioshape-lab \
    experiment=omics_readout \
    dataset.loader.parameters.adjacency_method=string \
    dataset.loader.parameters.node_sample_ratio=0.3 \
    dataset.loader.parameters.method=correlation \
    optimizer.parameters.lr=0.0001 \
    optimizer.parameters.weight_decay=0.0 \
    model.backbone.dropout=0.1 \
    model.feature_encoder.out_channels=64 \
    model.backbone.num_layers=2 \
    dataset.loader.parameters.adjacency_threshold=0.4 \
    model.readout.fc_dropout=0.1 \
    logger.wandb.project=best_model_reruns \
    trainer.devices=\[1\] \
    &

python -m ogbench \
    model=gin \
    dataset=parkinsons \
    seed=123 \
    logger.wandb.tags=[gin,parkinsons,hpsearch,best_rerun] \
    logger.wandb.entity=bioshape-lab \
    experiment=omics_readout \
    dataset.loader.parameters.adjacency_method=string \
    dataset.loader.parameters.node_sample_ratio=0.3 \
    dataset.loader.parameters.method=correlation \
    optimizer.parameters.lr=0.0001 \
    optimizer.parameters.weight_decay=0.0 \
    model.backbone.dropout=0.1 \
    model.feature_encoder.out_channels=64 \
    model.backbone.num_layers=2 \
    dataset.loader.parameters.adjacency_threshold=0.4 \
    model.readout.fc_dropout=0.1 \
    logger.wandb.project=best_model_reruns \
    trainer.devices=\[2\] \
    &

python -m ogbench \
    model=gin \
    dataset=parkinsons \
    seed=456 \
    logger.wandb.tags=[gin,parkinsons,hpsearch,best_rerun] \
    logger.wandb.entity=bioshape-lab \
    experiment=omics_readout \
    dataset.loader.parameters.adjacency_method=string \
    dataset.loader.parameters.node_sample_ratio=0.3 \
    dataset.loader.parameters.method=correlation \
    optimizer.parameters.lr=0.0001 \
    optimizer.parameters.weight_decay=0.0 \
    model.backbone.dropout=0.1 \
    model.feature_encoder.out_channels=64 \
    model.backbone.num_layers=2 \
    dataset.loader.parameters.adjacency_threshold=0.4 \
    model.readout.fc_dropout=0.1 \
    logger.wandb.project=best_model_reruns \
    trainer.devices=\[3\] \
    &

python -m ogbench \
    model=gps \
    dataset=addneuromed \
    seed=42 \
    logger.wandb.tags=[gps,addneuromed,hpsearch,best_rerun] \
    logger.wandb.entity=bioshape-lab \
    experiment=omics_readout \
    dataset.loader.parameters.adjacency_method=wgcna \
    dataset.loader.parameters.node_sample_ratio=0.5 \
    dataset.loader.parameters.method=variance \
    optimizer.parameters.lr=0.001 \
    optimizer.parameters.weight_decay=0.0 \
    model.backbone.dropout=0.1 \
    model.feature_encoder.out_channels=64 \
    model.backbone.num_layers=2 \
    model.encodings=[RWSE] \
    dataset.loader.parameters.adjacency_threshold=0.1786 \
    model.readout.fc_dropout=0.1 \
    logger.wandb.project=best_model_reruns \
    trainer.devices=\[4\] \
    &

python -m ogbench \
    model=gps \
    dataset=addneuromed \
    seed=123 \
    logger.wandb.tags=[gps,addneuromed,hpsearch,best_rerun] \
    logger.wandb.entity=bioshape-lab \
    experiment=omics_readout \
    dataset.loader.parameters.adjacency_method=wgcna \
    dataset.loader.parameters.node_sample_ratio=0.5 \
    dataset.loader.parameters.method=variance \
    optimizer.parameters.lr=0.001 \
    optimizer.parameters.weight_decay=0.0 \
    model.backbone.dropout=0.1 \
    model.feature_encoder.out_channels=64 \
    model.backbone.num_layers=2 \
    model.encodings=[RWSE] \
    dataset.loader.parameters.adjacency_threshold=0.1786 \
    model.readout.fc_dropout=0.1 \
    logger.wandb.project=best_model_reruns \
    trainer.devices=\[5\] \
    &

python -m ogbench \
    model=gps \
    dataset=addneuromed \
    seed=456 \
    logger.wandb.tags=[gps,addneuromed,hpsearch,best_rerun] \
    logger.wandb.entity=bioshape-lab \
    experiment=omics_readout \
    dataset.loader.parameters.adjacency_method=wgcna \
    dataset.loader.parameters.node_sample_ratio=0.5 \
    dataset.loader.parameters.method=variance \
    optimizer.parameters.lr=0.001 \
    optimizer.parameters.weight_decay=0.0 \
    model.backbone.dropout=0.1 \
    model.feature_encoder.out_channels=64 \
    model.backbone.num_layers=2 \
    model.encodings=[RWSE] \
    dataset.loader.parameters.adjacency_threshold=0.1786 \
    model.readout.fc_dropout=0.1 \
    logger.wandb.project=best_model_reruns \
    trainer.devices=\[6\] \
    &

python -m ogbench \
    model=gps \
    dataset=brca \
    seed=42 \
    logger.wandb.tags=[gps,brca,hpsearch,best_rerun] \
    logger.wandb.entity=bioshape-lab \
    experiment=omics_readout \
    dataset.loader.parameters.adjacency_method=wgcna \
    dataset.loader.parameters.node_sample_ratio=0.8 \
    dataset.loader.parameters.method=distance_correlation \
    optimizer.parameters.lr=0.001 \
    optimizer.parameters.weight_decay=0.0 \
    model.backbone.dropout=0.1 \
    model.feature_encoder.out_channels=64 \
    model.backbone.num_layers=2 \
    model.encodings=[LapPE] \
    dataset.loader.parameters.adjacency_threshold=0.0006 \
    model.readout.fc_dropout=0.1 \
    logger.wandb.project=best_model_reruns \
    trainer.devices=\[7\] \
    &

python -m ogbench \
    model=gps \
    dataset=brca \
    seed=123 \
    logger.wandb.tags=[gps,brca,hpsearch,best_rerun] \
    logger.wandb.entity=bioshape-lab \
    experiment=omics_readout \
    dataset.loader.parameters.adjacency_method=wgcna \
    dataset.loader.parameters.node_sample_ratio=0.8 \
    dataset.loader.parameters.method=distance_correlation \
    optimizer.parameters.lr=0.001 \
    optimizer.parameters.weight_decay=0.0 \
    model.backbone.dropout=0.1 \
    model.feature_encoder.out_channels=64 \
    model.backbone.num_layers=2 \
    model.encodings=[LapPE] \
    dataset.loader.parameters.adjacency_threshold=0.0006 \
    model.readout.fc_dropout=0.1 \
    logger.wandb.project=best_model_reruns \
    trainer.devices=\[0\] \
    &

python -m ogbench \
    model=gps \
    dataset=brca \
    seed=456 \
    logger.wandb.tags=[gps,brca,hpsearch,best_rerun] \
    logger.wandb.entity=bioshape-lab \
    experiment=omics_readout \
    dataset.loader.parameters.adjacency_method=wgcna \
    dataset.loader.parameters.node_sample_ratio=0.8 \
    dataset.loader.parameters.method=distance_correlation \
    optimizer.parameters.lr=0.001 \
    optimizer.parameters.weight_decay=0.0 \
    model.backbone.dropout=0.1 \
    model.feature_encoder.out_channels=64 \
    model.backbone.num_layers=2 \
    model.encodings=[LapPE] \
    dataset.loader.parameters.adjacency_threshold=0.0006 \
    model.readout.fc_dropout=0.1 \
    logger.wandb.project=best_model_reruns \
    trainer.devices=\[1\] \
    &

python -m ogbench \
    model=gps \
    dataset=motrpac \
    seed=42 \
    logger.wandb.tags=[gps,motrpac,hpsearch,best_rerun] \
    logger.wandb.entity=bioshape-lab \
    experiment=omics_readout \
    dataset.loader.parameters.adjacency_method=string \
    dataset.loader.parameters.node_sample_ratio=1.0 \
    dataset.loader.parameters.method=correlation \
    optimizer.parameters.lr=0.001 \
    optimizer.parameters.weight_decay=0.0 \
    model.backbone.dropout=0.1 \
    model.feature_encoder.out_channels=64 \
    model.backbone.num_layers=2 \
    model.encodings=[LapPE] \
    dataset.loader.parameters.adjacency_threshold=0.4 \
    model.readout.fc_dropout=0.1 \
    logger.wandb.project=best_model_reruns \
    trainer.devices=\[2\] \
    &

python -m ogbench \
    model=gps \
    dataset=motrpac \
    seed=123 \
    logger.wandb.tags=[gps,motrpac,hpsearch,best_rerun] \
    logger.wandb.entity=bioshape-lab \
    experiment=omics_readout \
    dataset.loader.parameters.adjacency_method=string \
    dataset.loader.parameters.node_sample_ratio=1.0 \
    dataset.loader.parameters.method=correlation \
    optimizer.parameters.lr=0.001 \
    optimizer.parameters.weight_decay=0.0 \
    model.backbone.dropout=0.1 \
    model.feature_encoder.out_channels=64 \
    model.backbone.num_layers=2 \
    model.encodings=[LapPE] \
    dataset.loader.parameters.adjacency_threshold=0.4 \
    model.readout.fc_dropout=0.1 \
    logger.wandb.project=best_model_reruns \
    trainer.devices=\[3\] \
    &

python -m ogbench \
    model=gps \
    dataset=motrpac \
    seed=456 \
    logger.wandb.tags=[gps,motrpac,hpsearch,best_rerun] \
    logger.wandb.entity=bioshape-lab \
    experiment=omics_readout \
    dataset.loader.parameters.adjacency_method=string \
    dataset.loader.parameters.node_sample_ratio=1.0 \
    dataset.loader.parameters.method=correlation \
    optimizer.parameters.lr=0.001 \
    optimizer.parameters.weight_decay=0.0 \
    model.backbone.dropout=0.1 \
    model.feature_encoder.out_channels=64 \
    model.backbone.num_layers=2 \
    model.encodings=[LapPE] \
    dataset.loader.parameters.adjacency_threshold=0.4 \
    model.readout.fc_dropout=0.1 \
    logger.wandb.project=best_model_reruns \
    trainer.devices=\[4\] \
    &

python -m ogbench \
    model=gps \
    dataset=parkinsons \
    seed=42 \
    logger.wandb.tags=[gps,parkinsons,hpsearch,best_rerun] \
    logger.wandb.entity=bioshape-lab \
    experiment=omics_readout \
    dataset.loader.parameters.adjacency_method=wgcna \
    dataset.loader.parameters.node_sample_ratio=0.8 \
    dataset.loader.parameters.method=random \
    optimizer.parameters.lr=0.001 \
    optimizer.parameters.weight_decay=0.0 \
    model.backbone.dropout=0.1 \
    model.feature_encoder.out_channels=64 \
    model.backbone.num_layers=2 \
    model.encodings=[RWSE] \
    dataset.loader.parameters.adjacency_threshold=0.0216 \
    model.readout.fc_dropout=0.1 \
    logger.wandb.project=best_model_reruns \
    trainer.devices=\[5\] \
    &

python -m ogbench \
    model=gps \
    dataset=parkinsons \
    seed=123 \
    logger.wandb.tags=[gps,parkinsons,hpsearch,best_rerun] \
    logger.wandb.entity=bioshape-lab \
    experiment=omics_readout \
    dataset.loader.parameters.adjacency_method=wgcna \
    dataset.loader.parameters.node_sample_ratio=0.8 \
    dataset.loader.parameters.method=random \
    optimizer.parameters.lr=0.001 \
    optimizer.parameters.weight_decay=0.0 \
    model.backbone.dropout=0.1 \
    model.feature_encoder.out_channels=64 \
    model.backbone.num_layers=2 \
    model.encodings=[RWSE] \
    dataset.loader.parameters.adjacency_threshold=0.0216 \
    model.readout.fc_dropout=0.1 \
    logger.wandb.project=best_model_reruns \
    trainer.devices=\[6\] \
    &

python -m ogbench \
    model=gps \
    dataset=parkinsons \
    seed=456 \
    logger.wandb.tags=[gps,parkinsons,hpsearch,best_rerun] \
    logger.wandb.entity=bioshape-lab \
    experiment=omics_readout \
    dataset.loader.parameters.adjacency_method=wgcna \
    dataset.loader.parameters.node_sample_ratio=0.8 \
    dataset.loader.parameters.method=random \
    optimizer.parameters.lr=0.001 \
    optimizer.parameters.weight_decay=0.0 \
    model.backbone.dropout=0.1 \
    model.feature_encoder.out_channels=64 \
    model.backbone.num_layers=2 \
    model.encodings=[RWSE] \
    dataset.loader.parameters.adjacency_threshold=0.0216 \
    model.readout.fc_dropout=0.1 \
    logger.wandb.project=best_model_reruns \
    trainer.devices=\[7\] \
    &

python -m ogbench \
    model=mlp \
    dataset=addneuromed \
    seed=42 \
    logger.wandb.tags=[mlp,addneuromed,hpsearch,best_rerun] \
    logger.wandb.entity=bioshape-lab \
    experiment=no_readout \
    dataset.loader.parameters.adjacency_method=wgcna \
    dataset.loader.parameters.node_sample_ratio=0.3 \
    dataset.loader.parameters.method=variance \
    optimizer.parameters.lr=0.001 \
    optimizer.parameters.weight_decay=0.0 \
    model.backbone.dropout=0.1 \
    model.backbone.hidden_channels=[32,64,16] \
    model.backbone.norm=batch \
    dataset.loader.parameters.adjacency_threshold=0.2378 \
    logger.wandb.project=best_model_reruns \
    trainer.devices=\[0\] \
    &

python -m ogbench \
    model=mlp \
    dataset=addneuromed \
    seed=123 \
    logger.wandb.tags=[mlp,addneuromed,hpsearch,best_rerun] \
    logger.wandb.entity=bioshape-lab \
    experiment=no_readout \
    dataset.loader.parameters.adjacency_method=wgcna \
    dataset.loader.parameters.node_sample_ratio=0.3 \
    dataset.loader.parameters.method=variance \
    optimizer.parameters.lr=0.001 \
    optimizer.parameters.weight_decay=0.0 \
    model.backbone.dropout=0.1 \
    model.backbone.hidden_channels=[32,64,16] \
    model.backbone.norm=null \
    dataset.loader.parameters.adjacency_threshold=0.2378 \
    logger.wandb.project=best_model_reruns \
    trainer.devices=\[1\] \
    &

python -m ogbench \
    model=mlp \
    dataset=addneuromed \
    seed=456 \
    logger.wandb.tags=[mlp,addneuromed,hpsearch,best_rerun] \
    logger.wandb.entity=bioshape-lab \
    experiment=no_readout \
    dataset.loader.parameters.adjacency_method=wgcna \
    dataset.loader.parameters.node_sample_ratio=0.3 \
    dataset.loader.parameters.method=variance \
    optimizer.parameters.lr=0.001 \
    optimizer.parameters.weight_decay=0.0 \
    model.backbone.dropout=0.1 \
    model.backbone.hidden_channels=[32,64,16] \
    model.backbone.norm=null \
    dataset.loader.parameters.adjacency_threshold=0.2378 \
    logger.wandb.project=best_model_reruns \
    trainer.devices=\[2\] \
    &

python -m ogbench \
    model=mlp \
    dataset=brca \
    seed=42 \
    logger.wandb.tags=[mlp,brca,hpsearch,best_rerun] \
    logger.wandb.entity=bioshape-lab \
    experiment=no_readout \
    dataset.loader.parameters.adjacency_method=string \
    dataset.loader.parameters.node_sample_ratio=0.3 \
    dataset.loader.parameters.method=distance_correlation \
    optimizer.parameters.lr=0.0001 \
    optimizer.parameters.weight_decay=0.0 \
    model.backbone.dropout=0.1 \
    model.backbone.hidden_channels=[64,128,32] \
    model.backbone.norm=batch \
    dataset.loader.parameters.adjacency_threshold=0.4 \
    logger.wandb.project=best_model_reruns \
    trainer.devices=\[3\] \
    &

python -m ogbench \
    model=mlp \
    dataset=brca \
    seed=123 \
    logger.wandb.tags=[mlp,brca,hpsearch,best_rerun] \
    logger.wandb.entity=bioshape-lab \
    experiment=no_readout \
    dataset.loader.parameters.adjacency_method=string \
    dataset.loader.parameters.node_sample_ratio=0.3 \
    dataset.loader.parameters.method=distance_correlation \
    optimizer.parameters.lr=0.0001 \
    optimizer.parameters.weight_decay=0.0 \
    model.backbone.dropout=0.1 \
    model.backbone.hidden_channels=[64,128,32] \
    model.backbone.norm=batch \
    dataset.loader.parameters.adjacency_threshold=0.4 \
    logger.wandb.project=best_model_reruns \
    trainer.devices=\[4\] \
    &

python -m ogbench \
    model=mlp \
    dataset=brca \
    seed=456 \
    logger.wandb.tags=[mlp,brca,hpsearch,best_rerun] \
    logger.wandb.entity=bioshape-lab \
    experiment=no_readout \
    dataset.loader.parameters.adjacency_method=string \
    dataset.loader.parameters.node_sample_ratio=0.3 \
    dataset.loader.parameters.method=distance_correlation \
    optimizer.parameters.lr=0.0001 \
    optimizer.parameters.weight_decay=0.0 \
    model.backbone.dropout=0.1 \
    model.backbone.hidden_channels=[64,128,32] \
    model.backbone.norm=null \
    dataset.loader.parameters.adjacency_threshold=0.4 \
    logger.wandb.project=best_model_reruns \
    trainer.devices=\[5\] \
    &

python -m ogbench \
    model=mlp \
    dataset=motrpac \
    seed=42 \
    logger.wandb.tags=[mlp,motrpac,hpsearch,best_rerun] \
    logger.wandb.entity=bioshape-lab \
    experiment=no_readout \
    dataset.loader.parameters.adjacency_method=string \
    dataset.loader.parameters.node_sample_ratio=0.5 \
    dataset.loader.parameters.method=distance_correlation \
    optimizer.parameters.lr=0.001 \
    optimizer.parameters.weight_decay=0.0 \
    model.backbone.dropout=0.1 \
    model.backbone.hidden_channels=[32,64,16] \
    model.backbone.norm=batch \
    dataset.loader.parameters.adjacency_threshold=0.4 \
    logger.wandb.project=best_model_reruns \
    trainer.devices=\[6\] \
    &

python -m ogbench \
    model=mlp \
    dataset=motrpac \
    seed=123 \
    logger.wandb.tags=[mlp,motrpac,hpsearch,best_rerun] \
    logger.wandb.entity=bioshape-lab \
    experiment=no_readout \
    dataset.loader.parameters.adjacency_method=string \
    dataset.loader.parameters.node_sample_ratio=0.5 \
    dataset.loader.parameters.method=distance_correlation \
    optimizer.parameters.lr=0.001 \
    optimizer.parameters.weight_decay=0.0 \
    model.backbone.dropout=0.1 \
    model.backbone.hidden_channels=[32,64,16] \
    model.backbone.norm=null \
    dataset.loader.parameters.adjacency_threshold=0.4 \
    logger.wandb.project=best_model_reruns \
    trainer.devices=\[7\] \
    &

python -m ogbench \
    model=mlp \
    dataset=motrpac \
    seed=456 \
    logger.wandb.tags=[mlp,motrpac,hpsearch,best_rerun] \
    logger.wandb.entity=bioshape-lab \
    experiment=no_readout \
    dataset.loader.parameters.adjacency_method=string \
    dataset.loader.parameters.node_sample_ratio=0.5 \
    dataset.loader.parameters.method=distance_correlation \
    optimizer.parameters.lr=0.001 \
    optimizer.parameters.weight_decay=0.0 \
    model.backbone.dropout=0.1 \
    model.backbone.hidden_channels=[32,64,16] \
    model.backbone.norm=null \
    dataset.loader.parameters.adjacency_threshold=0.4 \
    logger.wandb.project=best_model_reruns \
    trainer.devices=\[0\] \
    &

python -m ogbench \
    model=mlp \
    dataset=parkinsons \
    seed=42 \
    logger.wandb.tags=[mlp,parkinsons,hpsearch,best_rerun] \
    logger.wandb.entity=bioshape-lab \
    experiment=no_readout \
    dataset.loader.parameters.adjacency_method=wgcna \
    dataset.loader.parameters.node_sample_ratio=0.5 \
    dataset.loader.parameters.method=random \
    optimizer.parameters.lr=0.001 \
    optimizer.parameters.weight_decay=0.0 \
    model.backbone.dropout=0.1 \
    model.backbone.hidden_channels=[32,64,16] \
    model.backbone.norm=null \
    dataset.loader.parameters.adjacency_threshold=0.0005 \
    logger.wandb.project=best_model_reruns \
    trainer.devices=\[1\] \
    &

python -m ogbench \
    model=mlp \
    dataset=parkinsons \
    seed=123 \
    logger.wandb.tags=[mlp,parkinsons,hpsearch,best_rerun] \
    logger.wandb.entity=bioshape-lab \
    experiment=no_readout \
    dataset.loader.parameters.adjacency_method=wgcna \
    dataset.loader.parameters.node_sample_ratio=0.5 \
    dataset.loader.parameters.method=random \
    optimizer.parameters.lr=0.001 \
    optimizer.parameters.weight_decay=0.0 \
    model.backbone.dropout=0.1 \
    model.backbone.hidden_channels=[32,64,16] \
    model.backbone.norm=batch \
    dataset.loader.parameters.adjacency_threshold=0.0005 \
    logger.wandb.project=best_model_reruns \
    trainer.devices=\[2\] \
    &

python -m ogbench \
    model=mlp \
    dataset=parkinsons \
    seed=456 \
    logger.wandb.tags=[mlp,parkinsons,hpsearch,best_rerun] \
    logger.wandb.entity=bioshape-lab \
    experiment=no_readout \
    dataset.loader.parameters.adjacency_method=wgcna \
    dataset.loader.parameters.node_sample_ratio=0.5 \
    dataset.loader.parameters.method=random \
    optimizer.parameters.lr=0.001 \
    optimizer.parameters.weight_decay=0.0 \
    model.backbone.dropout=0.1 \
    model.backbone.hidden_channels=[32,64,16] \
    model.backbone.norm=batch \
    dataset.loader.parameters.adjacency_threshold=0.0005 \
    logger.wandb.project=best_model_reruns \
    trainer.devices=\[3\] \
    &

python -m ogbench \
    model=graph_sage \
    dataset=addneuromed \
    seed=42 \
    logger.wandb.tags=[graph_sage,addneuromed,hpsearch,best_rerun] \
    logger.wandb.entity=bioshape-lab \
    experiment=omics_readout \
    dataset.loader.parameters.adjacency_method=wgcna \
    dataset.loader.parameters.node_sample_ratio=0.5 \
    dataset.loader.parameters.method=variance \
    optimizer.parameters.lr=0.001 \
    optimizer.parameters.weight_decay=0.0 \
    model.backbone.dropout=0.1 \
    model.feature_encoder.out_channels=16 \
    model.backbone.num_layers=2 \
    dataset.loader.parameters.adjacency_threshold=0.1786 \
    model.readout.fc_dropout=0.1 \
    logger.wandb.project=best_model_reruns \
    trainer.devices=\[4\] \
    &

python -m ogbench \
    model=graph_sage \
    dataset=addneuromed \
    seed=123 \
    logger.wandb.tags=[graph_sage,addneuromed,hpsearch,best_rerun] \
    logger.wandb.entity=bioshape-lab \
    experiment=omics_readout \
    dataset.loader.parameters.adjacency_method=wgcna \
    dataset.loader.parameters.node_sample_ratio=0.5 \
    dataset.loader.parameters.method=variance \
    optimizer.parameters.lr=0.001 \
    optimizer.parameters.weight_decay=0.0 \
    model.backbone.dropout=0.1 \
    model.feature_encoder.out_channels=16 \
    model.backbone.num_layers=2 \
    dataset.loader.parameters.adjacency_threshold=0.1786 \
    model.readout.fc_dropout=0.1 \
    logger.wandb.project=best_model_reruns \
    trainer.devices=\[5\] \
    &

python -m ogbench \
    model=graph_sage \
    dataset=addneuromed \
    seed=456 \
    logger.wandb.tags=[graph_sage,addneuromed,hpsearch,best_rerun] \
    logger.wandb.entity=bioshape-lab \
    experiment=omics_readout \
    dataset.loader.parameters.adjacency_method=wgcna \
    dataset.loader.parameters.node_sample_ratio=0.5 \
    dataset.loader.parameters.method=variance \
    optimizer.parameters.lr=0.001 \
    optimizer.parameters.weight_decay=0.0 \
    model.backbone.dropout=0.1 \
    model.feature_encoder.out_channels=16 \
    model.backbone.num_layers=2 \
    dataset.loader.parameters.adjacency_threshold=0.1786 \
    model.readout.fc_dropout=0.1 \
    logger.wandb.project=best_model_reruns \
    trainer.devices=\[6\] \
    &

python -m ogbench \
    model=graph_sage \
    dataset=brca \
    seed=42 \
    logger.wandb.tags=[graph_sage,brca,hpsearch,best_rerun] \
    logger.wandb.entity=bioshape-lab \
    experiment=omics_readout \
    dataset.loader.parameters.adjacency_method=wgcna \
    dataset.loader.parameters.node_sample_ratio=0.3 \
    dataset.loader.parameters.method=distance_correlation \
    optimizer.parameters.lr=0.001 \
    optimizer.parameters.weight_decay=0.0 \
    model.backbone.dropout=0.1 \
    model.feature_encoder.out_channels=16 \
    model.backbone.num_layers=2 \
    dataset.loader.parameters.adjacency_threshold=0.0005 \
    model.readout.fc_dropout=0.1 \
    logger.wandb.project=best_model_reruns \
    trainer.devices=\[7\] \
    &

python -m ogbench \
    model=graph_sage \
    dataset=brca \
    seed=123 \
    logger.wandb.tags=[graph_sage,brca,hpsearch,best_rerun] \
    logger.wandb.entity=bioshape-lab \
    experiment=omics_readout \
    dataset.loader.parameters.adjacency_method=wgcna \
    dataset.loader.parameters.node_sample_ratio=0.3 \
    dataset.loader.parameters.method=distance_correlation \
    optimizer.parameters.lr=0.001 \
    optimizer.parameters.weight_decay=0.0 \
    model.backbone.dropout=0.1 \
    model.feature_encoder.out_channels=16 \
    model.backbone.num_layers=2 \
    dataset.loader.parameters.adjacency_threshold=0.0005 \
    model.readout.fc_dropout=0.1 \
    logger.wandb.project=best_model_reruns \
    trainer.devices=\[0\] \
    &

python -m ogbench \
    model=graph_sage \
    dataset=brca \
    seed=456 \
    logger.wandb.tags=[graph_sage,brca,hpsearch,best_rerun] \
    logger.wandb.entity=bioshape-lab \
    experiment=omics_readout \
    dataset.loader.parameters.adjacency_method=wgcna \
    dataset.loader.parameters.node_sample_ratio=0.3 \
    dataset.loader.parameters.method=distance_correlation \
    optimizer.parameters.lr=0.001 \
    optimizer.parameters.weight_decay=0.0 \
    model.backbone.dropout=0.1 \
    model.feature_encoder.out_channels=16 \
    model.backbone.num_layers=2 \
    dataset.loader.parameters.adjacency_threshold=0.0005 \
    model.readout.fc_dropout=0.1 \
    logger.wandb.project=best_model_reruns \
    trainer.devices=\[1\] \
    &

python -m ogbench \
    model=graph_sage \
    dataset=motrpac \
    seed=42 \
    logger.wandb.tags=[graph_sage,motrpac,hpsearch,best_rerun] \
    logger.wandb.entity=bioshape-lab \
    experiment=omics_readout \
    dataset.loader.parameters.adjacency_method=wgcna \
    dataset.loader.parameters.node_sample_ratio=0.3 \
    dataset.loader.parameters.method=correlation \
    optimizer.parameters.lr=0.001 \
    optimizer.parameters.weight_decay=0.0 \
    model.backbone.dropout=0.1 \
    model.feature_encoder.out_channels=16 \
    model.backbone.num_layers=4 \
    dataset.loader.parameters.adjacency_threshold=0.0005 \
    model.readout.fc_dropout=0.1 \
    logger.wandb.project=best_model_reruns \
    trainer.devices=\[2\] \
    &

python -m ogbench \
    model=graph_sage \
    dataset=motrpac \
    seed=123 \
    logger.wandb.tags=[graph_sage,motrpac,hpsearch,best_rerun] \
    logger.wandb.entity=bioshape-lab \
    experiment=omics_readout \
    dataset.loader.parameters.adjacency_method=wgcna \
    dataset.loader.parameters.node_sample_ratio=0.3 \
    dataset.loader.parameters.method=correlation \
    optimizer.parameters.lr=0.001 \
    optimizer.parameters.weight_decay=0.0 \
    model.backbone.dropout=0.1 \
    model.feature_encoder.out_channels=16 \
    model.backbone.num_layers=4 \
    dataset.loader.parameters.adjacency_threshold=0.0005 \
    model.readout.fc_dropout=0.1 \
    logger.wandb.project=best_model_reruns \
    trainer.devices=\[3\] \
    &

python -m ogbench \
    model=graph_sage \
    dataset=motrpac \
    seed=456 \
    logger.wandb.tags=[graph_sage,motrpac,hpsearch,best_rerun] \
    logger.wandb.entity=bioshape-lab \
    experiment=omics_readout \
    dataset.loader.parameters.adjacency_method=wgcna \
    dataset.loader.parameters.node_sample_ratio=0.3 \
    dataset.loader.parameters.method=correlation \
    optimizer.parameters.lr=0.001 \
    optimizer.parameters.weight_decay=0.0 \
    model.backbone.dropout=0.1 \
    model.feature_encoder.out_channels=16 \
    model.backbone.num_layers=4 \
    dataset.loader.parameters.adjacency_threshold=0.0005 \
    model.readout.fc_dropout=0.1 \
    logger.wandb.project=best_model_reruns \
    trainer.devices=\[4\] \
    &

python -m ogbench \
    model=graph_sage \
    dataset=parkinsons \
    seed=42 \
    logger.wandb.tags=[graph_sage,parkinsons,hpsearch,best_rerun] \
    logger.wandb.entity=bioshape-lab \
    experiment=omics_readout \
    dataset.loader.parameters.adjacency_method=wgcna \
    dataset.loader.parameters.node_sample_ratio=0.3 \
    dataset.loader.parameters.method=distance_correlation \
    optimizer.parameters.lr=0.001 \
    optimizer.parameters.weight_decay=0.0 \
    model.backbone.dropout=0.1 \
    model.feature_encoder.out_channels=64 \
    model.backbone.num_layers=2 \
    dataset.loader.parameters.adjacency_threshold=0.0005 \
    model.readout.fc_dropout=0.1 \
    logger.wandb.project=best_model_reruns \
    trainer.devices=\[5\] \
    &

python -m ogbench \
    model=graph_sage \
    dataset=parkinsons \
    seed=123 \
    logger.wandb.tags=[graph_sage,parkinsons,hpsearch,best_rerun] \
    logger.wandb.entity=bioshape-lab \
    experiment=omics_readout \
    dataset.loader.parameters.adjacency_method=wgcna \
    dataset.loader.parameters.node_sample_ratio=0.3 \
    dataset.loader.parameters.method=distance_correlation \
    optimizer.parameters.lr=0.001 \
    optimizer.parameters.weight_decay=0.0 \
    model.backbone.dropout=0.1 \
    model.feature_encoder.out_channels=64 \
    model.backbone.num_layers=2 \
    dataset.loader.parameters.adjacency_threshold=0.0005 \
    model.readout.fc_dropout=0.1 \
    logger.wandb.project=best_model_reruns \
    trainer.devices=\[6\] \
    &

python -m ogbench \
    model=graph_sage \
    dataset=parkinsons \
    seed=456 \
    logger.wandb.tags=[graph_sage,parkinsons,hpsearch,best_rerun] \
    logger.wandb.entity=bioshape-lab \
    experiment=omics_readout \
    dataset.loader.parameters.adjacency_method=wgcna \
    dataset.loader.parameters.node_sample_ratio=0.3 \
    dataset.loader.parameters.method=distance_correlation \
    optimizer.parameters.lr=0.001 \
    optimizer.parameters.weight_decay=0.0 \
    model.backbone.dropout=0.1 \
    model.feature_encoder.out_channels=64 \
    model.backbone.num_layers=2 \
    dataset.loader.parameters.adjacency_threshold=0.0005 \
    model.readout.fc_dropout=0.1 \
    logger.wandb.project=best_model_reruns \
    trainer.devices=\[7\] \
    &

python -m ogbench \
    model=sagn \
    dataset=addneuromed \
    seed=42 \
    logger.wandb.tags=[sagn,addneuromed,hpsearch,best_rerun] \
    logger.wandb.entity=bioshape-lab \
    experiment=omics_readout \
    dataset.loader.parameters.adjacency_method=wgcna \
    dataset.loader.parameters.node_sample_ratio=1.0 \
    dataset.loader.parameters.method=variance \
    optimizer.parameters.lr=0.001 \
    optimizer.parameters.weight_decay=0.0 \
    model.backbone.dropout=0.1 \
    model.backbone.hidden_channels=64 \
    model.backbone.num_layers=4 \
    model.backbone.num_heads=4 \
    dataset.loader.parameters.adjacency_threshold=0.2996 \
    model.readout.fc_dropout=0.1 \
    logger.wandb.project=best_model_reruns \
    trainer.devices=\[0\] \
    &

python -m ogbench \
    model=sagn \
    dataset=addneuromed \
    seed=123 \
    logger.wandb.tags=[sagn,addneuromed,hpsearch,best_rerun] \
    logger.wandb.entity=bioshape-lab \
    experiment=omics_readout \
    dataset.loader.parameters.adjacency_method=wgcna \
    dataset.loader.parameters.node_sample_ratio=1.0 \
    dataset.loader.parameters.method=variance \
    optimizer.parameters.lr=0.001 \
    optimizer.parameters.weight_decay=0.0 \
    model.backbone.dropout=0.1 \
    model.backbone.hidden_channels=64 \
    model.backbone.num_layers=4 \
    model.backbone.num_heads=4 \
    dataset.loader.parameters.adjacency_threshold=0.2996 \
    model.readout.fc_dropout=0.1 \
    logger.wandb.project=best_model_reruns \
    trainer.devices=\[1\] \
    &

python -m ogbench \
    model=sagn \
    dataset=addneuromed \
    seed=456 \
    logger.wandb.tags=[sagn,addneuromed,hpsearch,best_rerun] \
    logger.wandb.entity=bioshape-lab \
    experiment=omics_readout \
    dataset.loader.parameters.adjacency_method=wgcna \
    dataset.loader.parameters.node_sample_ratio=1.0 \
    dataset.loader.parameters.method=variance \
    optimizer.parameters.lr=0.001 \
    optimizer.parameters.weight_decay=0.0 \
    model.backbone.dropout=0.1 \
    model.backbone.hidden_channels=64 \
    model.backbone.num_layers=4 \
    model.backbone.num_heads=4 \
    dataset.loader.parameters.adjacency_threshold=0.2996 \
    model.readout.fc_dropout=0.1 \
    logger.wandb.project=best_model_reruns \
    trainer.devices=\[2\] \
    &

python -m ogbench \
    model=sagn \
    dataset=brca \
    seed=42 \
    logger.wandb.tags=[sagn,brca,hpsearch,best_rerun] \
    logger.wandb.entity=bioshape-lab \
    experiment=omics_readout \
    dataset.loader.parameters.adjacency_method=string \
    dataset.loader.parameters.node_sample_ratio=0.8 \
    dataset.loader.parameters.method=distance_correlation \
    optimizer.parameters.lr=0.001 \
    optimizer.parameters.weight_decay=0.0 \
    model.backbone.dropout=0.1 \
    model.backbone.hidden_channels=16 \
    model.backbone.num_layers=2 \
    model.backbone.num_heads=4 \
    dataset.loader.parameters.adjacency_threshold=0.4 \
    model.readout.fc_dropout=0.1 \
    logger.wandb.project=best_model_reruns \
    trainer.devices=\[3\] \
    &

python -m ogbench \
    model=sagn \
    dataset=brca \
    seed=123 \
    logger.wandb.tags=[sagn,brca,hpsearch,best_rerun] \
    logger.wandb.entity=bioshape-lab \
    experiment=omics_readout \
    dataset.loader.parameters.adjacency_method=string \
    dataset.loader.parameters.node_sample_ratio=0.8 \
    dataset.loader.parameters.method=distance_correlation \
    optimizer.parameters.lr=0.001 \
    optimizer.parameters.weight_decay=0.0 \
    model.backbone.dropout=0.1 \
    model.backbone.hidden_channels=16 \
    model.backbone.num_layers=2 \
    model.backbone.num_heads=4 \
    dataset.loader.parameters.adjacency_threshold=0.4 \
    model.readout.fc_dropout=0.1 \
    logger.wandb.project=best_model_reruns \
    trainer.devices=\[4\] \
    &

python -m ogbench \
    model=sagn \
    dataset=brca \
    seed=456 \
    logger.wandb.tags=[sagn,brca,hpsearch,best_rerun] \
    logger.wandb.entity=bioshape-lab \
    experiment=omics_readout \
    dataset.loader.parameters.adjacency_method=string \
    dataset.loader.parameters.node_sample_ratio=0.8 \
    dataset.loader.parameters.method=distance_correlation \
    optimizer.parameters.lr=0.001 \
    optimizer.parameters.weight_decay=0.0 \
    model.backbone.dropout=0.1 \
    model.backbone.hidden_channels=16 \
    model.backbone.num_layers=2 \
    model.backbone.num_heads=4 \
    dataset.loader.parameters.adjacency_threshold=0.4 \
    model.readout.fc_dropout=0.1 \
    logger.wandb.project=best_model_reruns \
    trainer.devices=\[5\] \
    &

python -m ogbench \
    model=sagn \
    dataset=motrpac \
    seed=42 \
    logger.wandb.tags=[sagn,motrpac,hpsearch,best_rerun] \
    logger.wandb.entity=bioshape-lab \
    experiment=omics_readout \
    dataset.loader.parameters.adjacency_method=wgcna \
    dataset.loader.parameters.node_sample_ratio=0.5 \
    dataset.loader.parameters.method=random \
    optimizer.parameters.lr=0.0001 \
    optimizer.parameters.weight_decay=0.0 \
    model.backbone.dropout=0.1 \
    model.backbone.hidden_channels=64 \
    model.backbone.num_layers=4 \
    model.backbone.num_heads=2 \
    dataset.loader.parameters.adjacency_threshold=0.0693 \
    model.readout.fc_dropout=0.1 \
    logger.wandb.project=best_model_reruns \
    trainer.devices=\[6\] \
    &

python -m ogbench \
    model=sagn \
    dataset=motrpac \
    seed=123 \
    logger.wandb.tags=[sagn,motrpac,hpsearch,best_rerun] \
    logger.wandb.entity=bioshape-lab \
    experiment=omics_readout \
    dataset.loader.parameters.adjacency_method=wgcna \
    dataset.loader.parameters.node_sample_ratio=0.5 \
    dataset.loader.parameters.method=random \
    optimizer.parameters.lr=0.0001 \
    optimizer.parameters.weight_decay=0.0 \
    model.backbone.dropout=0.1 \
    model.backbone.hidden_channels=64 \
    model.backbone.num_layers=4 \
    model.backbone.num_heads=2 \
    dataset.loader.parameters.adjacency_threshold=0.0693 \
    model.readout.fc_dropout=0.1 \
    logger.wandb.project=best_model_reruns \
    trainer.devices=\[7\] \
    &

python -m ogbench \
    model=sagn \
    dataset=motrpac \
    seed=456 \
    logger.wandb.tags=[sagn,motrpac,hpsearch,best_rerun] \
    logger.wandb.entity=bioshape-lab \
    experiment=omics_readout \
    dataset.loader.parameters.adjacency_method=wgcna \
    dataset.loader.parameters.node_sample_ratio=0.5 \
    dataset.loader.parameters.method=random \
    optimizer.parameters.lr=0.0001 \
    optimizer.parameters.weight_decay=0.0 \
    model.backbone.dropout=0.1 \
    model.backbone.hidden_channels=64 \
    model.backbone.num_layers=4 \
    model.backbone.num_heads=2 \
    dataset.loader.parameters.adjacency_threshold=0.0693 \
    model.readout.fc_dropout=0.1 \
    logger.wandb.project=best_model_reruns \
    trainer.devices=\[0\] \
    &

python -m ogbench \
    model=sagn \
    dataset=parkinsons \
    seed=42 \
    logger.wandb.tags=[sagn,parkinsons,hpsearch,best_rerun] \
    logger.wandb.entity=bioshape-lab \
    experiment=omics_readout \
    dataset.loader.parameters.adjacency_method=wgcna \
    dataset.loader.parameters.node_sample_ratio=0.3 \
    dataset.loader.parameters.method=random \
    optimizer.parameters.lr=0.001 \
    optimizer.parameters.weight_decay=0.0 \
    model.backbone.dropout=0.1 \
    model.backbone.hidden_channels=64 \
    model.backbone.num_layers=2 \
    model.backbone.num_heads=2 \
    dataset.loader.parameters.adjacency_threshold=0.0043 \
    model.readout.fc_dropout=0.1 \
    logger.wandb.project=best_model_reruns \
    trainer.devices=\[1\] \
    &

python -m ogbench \
    model=sagn \
    dataset=parkinsons \
    seed=123 \
    logger.wandb.tags=[sagn,parkinsons,hpsearch,best_rerun] \
    logger.wandb.entity=bioshape-lab \
    experiment=omics_readout \
    dataset.loader.parameters.adjacency_method=wgcna \
    dataset.loader.parameters.node_sample_ratio=0.3 \
    dataset.loader.parameters.method=random \
    optimizer.parameters.lr=0.001 \
    optimizer.parameters.weight_decay=0.0 \
    model.backbone.dropout=0.1 \
    model.backbone.hidden_channels=64 \
    model.backbone.num_layers=2 \
    model.backbone.num_heads=2 \
    dataset.loader.parameters.adjacency_threshold=0.0043 \
    model.readout.fc_dropout=0.1 \
    logger.wandb.project=best_model_reruns \
    trainer.devices=\[2\] \
    &

python -m ogbench \
    model=sagn \
    dataset=parkinsons \
    seed=456 \
    logger.wandb.tags=[sagn,parkinsons,hpsearch,best_rerun] \
    logger.wandb.entity=bioshape-lab \
    experiment=omics_readout \
    dataset.loader.parameters.adjacency_method=wgcna \
    dataset.loader.parameters.node_sample_ratio=0.3 \
    dataset.loader.parameters.method=random \
    optimizer.parameters.lr=0.001 \
    optimizer.parameters.weight_decay=0.0 \
    model.backbone.dropout=0.1 \
    model.backbone.hidden_channels=64 \
    model.backbone.num_layers=2 \
    model.backbone.num_heads=2 \
    dataset.loader.parameters.adjacency_threshold=0.0043 \
    model.readout.fc_dropout=0.1 \
    logger.wandb.project=best_model_reruns \
    trainer.devices=\[3\] \
    &

wait

