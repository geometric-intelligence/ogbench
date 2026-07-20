#!/bin/bash
# Re-run training: rows missing best_test_f1_macro in the export CSV.
# Overrides from W&B (summary or metadata ``args``).
# format=cluster  n_gpus=8  launcher='python -m ogbench'  background=True
# Drops: hydra.run.dir, paths.root_dir. Adds W&B tag: 'rerun'

python -m ogbench \
    model=gcn \
    dataset=parkinsons \
    seed=42 \
    logger.wandb.tags=[gcn,parkinsons,hpsearch,rerun] \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    experiment=no_readout \
    dataset.loader.parameters.adjacency_method=wgcna \
    dataset.loader.parameters.node_sample_ratio=1.0 \
    dataset.loader.parameters.method=random \
    optimizer.parameters.lr=0.0001 \
    optimizer.parameters.weight_decay=0.0 \
    model.backbone.dropout=0.1 \
    model.feature_encoder.out_channels=64 \
    model.backbone.num_layers=2 \
    dataset.loader.parameters.adjacency_threshold=0.0013 \
    trainer.devices=\[0\] \
    &

python -m ogbench \
    model=gcn \
    dataset=parkinsons \
    seed=42 \
    logger.wandb.tags=[gcn,parkinsons,hpsearch,rerun] \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    experiment=no_readout \
    dataset.loader.parameters.adjacency_method=wgcna \
    dataset.loader.parameters.node_sample_ratio=0.8 \
    dataset.loader.parameters.method=random \
    optimizer.parameters.lr=0.001 \
    optimizer.parameters.weight_decay=0.0 \
    model.backbone.dropout=0.1 \
    model.feature_encoder.out_channels=16 \
    model.backbone.num_layers=2 \
    dataset.loader.parameters.adjacency_threshold=0.0216 \
    trainer.devices=\[1\] \
    &

python -m ogbench \
    model=gcn \
    dataset=parkinsons \
    seed=123 \
    logger.wandb.tags=[gcn,parkinsons,hpsearch,rerun] \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    experiment=no_readout \
    dataset.loader.parameters.adjacency_method=wgcna \
    dataset.loader.parameters.node_sample_ratio=0.8 \
    dataset.loader.parameters.method=correlation \
    optimizer.parameters.lr=0.0001 \
    optimizer.parameters.weight_decay=0.0 \
    model.backbone.dropout=0.1 \
    model.feature_encoder.out_channels=16 \
    model.backbone.num_layers=4 \
    dataset.loader.parameters.adjacency_threshold=0.0019 \
    trainer.devices=\[2\] \
    &

python -m ogbench \
    model=gcn \
    dataset=parkinsons \
    seed=456 \
    logger.wandb.tags=[gcn,parkinsons,hpsearch,rerun] \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    experiment=no_readout \
    dataset.loader.parameters.adjacency_method=wgcna \
    dataset.loader.parameters.node_sample_ratio=0.8 \
    dataset.loader.parameters.method=distance_correlation \
    optimizer.parameters.lr=0.0001 \
    optimizer.parameters.weight_decay=0.0 \
    model.backbone.dropout=0.1 \
    model.feature_encoder.out_channels=16 \
    model.backbone.num_layers=4 \
    dataset.loader.parameters.adjacency_threshold=0.0054 \
    trainer.devices=\[3\] \
    &

python -m ogbench \
    model=gcn \
    dataset=parkinsons \
    seed=42 \
    logger.wandb.tags=[gcn,parkinsons,hpsearch,rerun] \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    experiment=no_readout \
    dataset.loader.parameters.adjacency_method=string \
    dataset.loader.parameters.node_sample_ratio=1.0 \
    dataset.loader.parameters.method=variance \
    optimizer.parameters.lr=0.0001 \
    optimizer.parameters.weight_decay=0.0 \
    model.backbone.dropout=0.1 \
    model.feature_encoder.out_channels=64 \
    model.backbone.num_layers=4 \
    dataset.loader.parameters.adjacency_threshold=0.4 \
    trainer.devices=\[4\] \
    &

python -m ogbench \
    model=gcn \
    dataset=parkinsons \
    seed=123 \
    logger.wandb.tags=[gcn,parkinsons,hpsearch,rerun] \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    experiment=no_readout \
    dataset.loader.parameters.adjacency_method=string \
    dataset.loader.parameters.node_sample_ratio=1.0 \
    dataset.loader.parameters.method=correlation \
    optimizer.parameters.lr=0.0001 \
    optimizer.parameters.weight_decay=0.0 \
    model.backbone.dropout=0.1 \
    model.feature_encoder.out_channels=64 \
    model.backbone.num_layers=2 \
    dataset.loader.parameters.adjacency_threshold=0.4 \
    trainer.devices=\[5\] \
    &

python -m ogbench \
    model=gcn \
    dataset=parkinsons \
    seed=456 \
    logger.wandb.tags=[gcn,parkinsons,hpsearch,rerun] \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    experiment=no_readout \
    dataset.loader.parameters.adjacency_method=string \
    dataset.loader.parameters.node_sample_ratio=1.0 \
    dataset.loader.parameters.method=correlation \
    optimizer.parameters.lr=0.001 \
    optimizer.parameters.weight_decay=0.0 \
    model.backbone.dropout=0.1 \
    model.feature_encoder.out_channels=16 \
    model.backbone.num_layers=2 \
    dataset.loader.parameters.adjacency_threshold=0.4 \
    trainer.devices=\[6\] \
    &

python -m ogbench \
    model=gcn \
    dataset=parkinsons \
    seed=123 \
    logger.wandb.tags=[gcn,parkinsons,hpsearch,rerun] \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    experiment=no_readout \
    dataset.loader.parameters.adjacency_method=string \
    dataset.loader.parameters.node_sample_ratio=1.0 \
    dataset.loader.parameters.method=distance_correlation \
    optimizer.parameters.lr=0.0001 \
    optimizer.parameters.weight_decay=0.0 \
    model.backbone.dropout=0.1 \
    model.feature_encoder.out_channels=16 \
    model.backbone.num_layers=4 \
    dataset.loader.parameters.adjacency_threshold=0.4 \
    trainer.devices=\[7\] \
    &

python -m ogbench \
    model=gcn \
    dataset=parkinsons \
    seed=456 \
    logger.wandb.tags=[gcn,parkinsons,hpsearch,rerun] \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    experiment=no_readout \
    dataset.loader.parameters.adjacency_method=string \
    dataset.loader.parameters.node_sample_ratio=1.0 \
    dataset.loader.parameters.method=distance_correlation \
    optimizer.parameters.lr=0.001 \
    optimizer.parameters.weight_decay=0.0 \
    model.backbone.dropout=0.1 \
    model.feature_encoder.out_channels=16 \
    model.backbone.num_layers=2 \
    dataset.loader.parameters.adjacency_threshold=0.4 \
    trainer.devices=\[0\] \
    &

python -m ogbench \
    model=gcn \
    dataset=parkinsons \
    seed=42 \
    logger.wandb.tags=[gcn,parkinsons,hpsearch,rerun] \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    experiment=no_readout \
    dataset.loader.parameters.adjacency_method=string \
    dataset.loader.parameters.node_sample_ratio=0.8 \
    dataset.loader.parameters.method=variance \
    optimizer.parameters.lr=0.0001 \
    optimizer.parameters.weight_decay=0.0 \
    model.backbone.dropout=0.1 \
    model.feature_encoder.out_channels=16 \
    model.backbone.num_layers=2 \
    dataset.loader.parameters.adjacency_threshold=0.4 \
    trainer.devices=\[1\] \
    &

python -m ogbench \
    model=gcn \
    dataset=parkinsons \
    seed=42 \
    logger.wandb.tags=[gcn,parkinsons,hpsearch,rerun] \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    experiment=omics_readout \
    dataset.loader.parameters.adjacency_method=wgcna \
    dataset.loader.parameters.node_sample_ratio=1.0 \
    dataset.loader.parameters.method=correlation \
    optimizer.parameters.lr=0.0001 \
    optimizer.parameters.weight_decay=0.0 \
    model.backbone.dropout=0.1 \
    model.feature_encoder.out_channels=64 \
    model.backbone.num_layers=2 \
    dataset.loader.parameters.adjacency_threshold=0.0005 \
    model.readout.fc_dropout=0.1 \
    trainer.devices=\[2\] \
    &

python -m ogbench \
    model=gcn \
    dataset=parkinsons \
    seed=42 \
    logger.wandb.tags=[gcn,parkinsons,hpsearch,rerun] \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    experiment=omics_readout \
    dataset.loader.parameters.adjacency_method=string \
    dataset.loader.parameters.node_sample_ratio=1.0 \
    dataset.loader.parameters.method=random \
    optimizer.parameters.lr=0.001 \
    optimizer.parameters.weight_decay=0.0 \
    model.backbone.dropout=0.1 \
    model.feature_encoder.out_channels=16 \
    model.backbone.num_layers=4 \
    dataset.loader.parameters.adjacency_threshold=0.4 \
    model.readout.fc_dropout=0.1 \
    trainer.devices=\[3\] \
    &

python -m ogbench \
    model=gcn \
    dataset=parkinsons \
    seed=123 \
    logger.wandb.tags=[gcn,parkinsons,hpsearch,rerun] \
    logger.wandb.project=bgbench_dataset_grid_search_final \
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
    dataset.loader.parameters.adjacency_threshold=0.4 \
    model.readout.fc_dropout=0.1 \
    trainer.devices=\[4\] \
    &

python -m ogbench \
    model=gcn \
    dataset=parkinsons \
    seed=456 \
    logger.wandb.tags=[gcn,parkinsons,hpsearch,rerun] \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    experiment=omics_readout \
    dataset.loader.parameters.adjacency_method=string \
    dataset.loader.parameters.node_sample_ratio=0.8 \
    dataset.loader.parameters.method=correlation \
    optimizer.parameters.lr=0.001 \
    optimizer.parameters.weight_decay=0.0 \
    model.backbone.dropout=0.1 \
    model.feature_encoder.out_channels=64 \
    model.backbone.num_layers=4 \
    dataset.loader.parameters.adjacency_threshold=0.4 \
    model.readout.fc_dropout=0.1 \
    trainer.devices=\[5\] \
    &

python -m ogbench \
    model=gcn \
    dataset=parkinsons \
    seed=123 \
    logger.wandb.tags=[gcn,parkinsons,hpsearch,rerun] \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    experiment=omics_readout \
    dataset.loader.parameters.adjacency_method=string \
    dataset.loader.parameters.node_sample_ratio=0.8 \
    dataset.loader.parameters.method=distance_correlation \
    optimizer.parameters.lr=0.0001 \
    optimizer.parameters.weight_decay=0.0 \
    model.backbone.dropout=0.1 \
    model.feature_encoder.out_channels=64 \
    model.backbone.num_layers=2 \
    dataset.loader.parameters.adjacency_threshold=0.4 \
    model.readout.fc_dropout=0.1 \
    trainer.devices=\[6\] \
    &

python -m ogbench \
    model=gcn \
    dataset=parkinsons \
    seed=456 \
    logger.wandb.tags=[gcn,parkinsons,hpsearch,rerun] \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    experiment=omics_readout \
    dataset.loader.parameters.adjacency_method=string \
    dataset.loader.parameters.node_sample_ratio=0.8 \
    dataset.loader.parameters.method=distance_correlation \
    optimizer.parameters.lr=0.001 \
    optimizer.parameters.weight_decay=0.0 \
    model.backbone.dropout=0.1 \
    model.feature_encoder.out_channels=16 \
    model.backbone.num_layers=2 \
    dataset.loader.parameters.adjacency_threshold=0.4 \
    model.readout.fc_dropout=0.1 \
    trainer.devices=\[7\] \
    &

python -m ogbench \
    model=gcn \
    dataset=addneuromed \
    seed=123 \
    logger.wandb.tags=[gcn,addneuromed,hpsearch,rerun] \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    experiment=no_readout \
    dataset.loader.parameters.adjacency_method=wgcna \
    dataset.loader.parameters.node_sample_ratio=0.8 \
    dataset.loader.parameters.method=variance \
    optimizer.parameters.lr=0.0001 \
    optimizer.parameters.weight_decay=0.0 \
    model.backbone.dropout=0.1 \
    model.feature_encoder.out_channels=16 \
    model.backbone.num_layers=2 \
    dataset.loader.parameters.adjacency_threshold=0.2537 \
    trainer.devices=\[0\] \
    &

python -m ogbench \
    model=gcn \
    dataset=addneuromed \
    seed=123 \
    logger.wandb.tags=[gcn,addneuromed,hpsearch,rerun] \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    experiment=no_readout \
    dataset.loader.parameters.adjacency_method=wgcna \
    dataset.loader.parameters.node_sample_ratio=0.8 \
    dataset.loader.parameters.method=variance \
    optimizer.parameters.lr=0.001 \
    optimizer.parameters.weight_decay=0.0 \
    model.backbone.dropout=0.1 \
    model.feature_encoder.out_channels=16 \
    model.backbone.num_layers=4 \
    dataset.loader.parameters.adjacency_threshold=0.2537 \
    trainer.devices=\[1\] \
    &

python -m ogbench \
    model=gcn \
    dataset=addneuromed \
    seed=123 \
    logger.wandb.tags=[gcn,addneuromed,hpsearch,rerun] \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    experiment=no_readout \
    dataset.loader.parameters.adjacency_method=wgcna \
    dataset.loader.parameters.node_sample_ratio=0.8 \
    dataset.loader.parameters.method=correlation \
    optimizer.parameters.lr=0.0001 \
    optimizer.parameters.weight_decay=0.0 \
    model.backbone.dropout=0.1 \
    model.feature_encoder.out_channels=16 \
    model.backbone.num_layers=4 \
    dataset.loader.parameters.adjacency_threshold=0.1086 \
    trainer.devices=\[2\] \
    &

python -m ogbench \
    model=gcn \
    dataset=addneuromed \
    seed=456 \
    logger.wandb.tags=[gcn,addneuromed,hpsearch,rerun] \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    experiment=no_readout \
    dataset.loader.parameters.adjacency_method=wgcna \
    dataset.loader.parameters.node_sample_ratio=0.8 \
    dataset.loader.parameters.method=correlation \
    optimizer.parameters.lr=0.001 \
    optimizer.parameters.weight_decay=0.0 \
    model.backbone.dropout=0.1 \
    model.feature_encoder.out_channels=64 \
    model.backbone.num_layers=2 \
    dataset.loader.parameters.adjacency_threshold=0.1086 \
    trainer.devices=\[3\] \
    &

python -m ogbench \
    model=gcn \
    dataset=addneuromed \
    seed=456 \
    logger.wandb.tags=[gcn,addneuromed,hpsearch,rerun] \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    experiment=no_readout \
    dataset.loader.parameters.adjacency_method=wgcna \
    dataset.loader.parameters.node_sample_ratio=0.8 \
    dataset.loader.parameters.method=distance_correlation \
    optimizer.parameters.lr=0.0001 \
    optimizer.parameters.weight_decay=0.0 \
    model.backbone.dropout=0.1 \
    model.feature_encoder.out_channels=16 \
    model.backbone.num_layers=4 \
    dataset.loader.parameters.adjacency_threshold=0.1765 \
    trainer.devices=\[4\] \
    &

python -m ogbench \
    model=gcn \
    dataset=addneuromed \
    seed=42 \
    logger.wandb.tags=[gcn,addneuromed,hpsearch,rerun] \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    experiment=no_readout \
    dataset.loader.parameters.adjacency_method=string \
    dataset.loader.parameters.node_sample_ratio=1.0 \
    dataset.loader.parameters.method=variance \
    optimizer.parameters.lr=0.0001 \
    optimizer.parameters.weight_decay=0.0 \
    model.backbone.dropout=0.1 \
    model.feature_encoder.out_channels=16 \
    model.backbone.num_layers=4 \
    dataset.loader.parameters.adjacency_threshold=0.4 \
    trainer.devices=\[5\] \
    &

python -m ogbench \
    model=gcn \
    dataset=addneuromed \
    seed=456 \
    logger.wandb.tags=[gcn,addneuromed,hpsearch,rerun] \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    experiment=no_readout \
    dataset.loader.parameters.adjacency_method=string \
    dataset.loader.parameters.node_sample_ratio=1.0 \
    dataset.loader.parameters.method=correlation \
    optimizer.parameters.lr=0.001 \
    optimizer.parameters.weight_decay=0.0 \
    model.backbone.dropout=0.1 \
    model.feature_encoder.out_channels=64 \
    model.backbone.num_layers=2 \
    dataset.loader.parameters.adjacency_threshold=0.4 \
    trainer.devices=\[6\] \
    &

python -m ogbench \
    model=gcn \
    dataset=addneuromed \
    seed=456 \
    logger.wandb.tags=[gcn,addneuromed,hpsearch,rerun] \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    experiment=no_readout \
    dataset.loader.parameters.adjacency_method=string \
    dataset.loader.parameters.node_sample_ratio=0.8 \
    dataset.loader.parameters.method=variance \
    optimizer.parameters.lr=0.001 \
    optimizer.parameters.weight_decay=0.0 \
    model.backbone.dropout=0.1 \
    model.feature_encoder.out_channels=16 \
    model.backbone.num_layers=4 \
    dataset.loader.parameters.adjacency_threshold=0.4 \
    trainer.devices=\[7\] \
    &

python -m ogbench \
    model=gcn \
    dataset=addneuromed \
    seed=123 \
    logger.wandb.tags=[gcn,addneuromed,hpsearch,rerun] \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    experiment=no_readout \
    dataset.loader.parameters.adjacency_method=string \
    dataset.loader.parameters.node_sample_ratio=0.8 \
    dataset.loader.parameters.method=distance_correlation \
    optimizer.parameters.lr=0.001 \
    optimizer.parameters.weight_decay=0.0 \
    model.backbone.dropout=0.1 \
    model.feature_encoder.out_channels=16 \
    model.backbone.num_layers=2 \
    dataset.loader.parameters.adjacency_threshold=0.4 \
    trainer.devices=\[0\] \
    &

python -m ogbench \
    model=gcn \
    dataset=addneuromed \
    seed=42 \
    logger.wandb.tags=[gcn,addneuromed,hpsearch,rerun] \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    experiment=omics_readout \
    dataset.loader.parameters.adjacency_method=wgcna \
    dataset.loader.parameters.node_sample_ratio=1.0 \
    dataset.loader.parameters.method=variance \
    optimizer.parameters.lr=0.0001 \
    optimizer.parameters.weight_decay=0.0 \
    model.backbone.dropout=0.1 \
    model.feature_encoder.out_channels=16 \
    model.backbone.num_layers=2 \
    dataset.loader.parameters.adjacency_threshold=0.2996 \
    model.readout.fc_dropout=0.1 \
    trainer.devices=\[1\] \
    &

python -m ogbench \
    model=gcn \
    dataset=addneuromed \
    seed=123 \
    logger.wandb.tags=[gcn,addneuromed,hpsearch,rerun] \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    experiment=omics_readout \
    dataset.loader.parameters.adjacency_method=wgcna \
    dataset.loader.parameters.node_sample_ratio=1.0 \
    dataset.loader.parameters.method=correlation \
    optimizer.parameters.lr=0.0001 \
    optimizer.parameters.weight_decay=0.0 \
    model.backbone.dropout=0.1 \
    model.feature_encoder.out_channels=64 \
    model.backbone.num_layers=2 \
    dataset.loader.parameters.adjacency_threshold=0.0997 \
    model.readout.fc_dropout=0.1 \
    trainer.devices=\[2\] \
    &

python -m ogbench \
    model=gcn \
    dataset=addneuromed \
    seed=123 \
    logger.wandb.tags=[gcn,addneuromed,hpsearch,rerun] \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    experiment=omics_readout \
    dataset.loader.parameters.adjacency_method=wgcna \
    dataset.loader.parameters.node_sample_ratio=1.0 \
    dataset.loader.parameters.method=correlation \
    optimizer.parameters.lr=0.001 \
    optimizer.parameters.weight_decay=0.0 \
    model.backbone.dropout=0.1 \
    model.feature_encoder.out_channels=16 \
    model.backbone.num_layers=2 \
    dataset.loader.parameters.adjacency_threshold=0.0997 \
    model.readout.fc_dropout=0.1 \
    trainer.devices=\[3\] \
    &

python -m ogbench \
    model=gcn \
    dataset=addneuromed \
    seed=456 \
    logger.wandb.tags=[gcn,addneuromed,hpsearch,rerun] \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    experiment=omics_readout \
    dataset.loader.parameters.adjacency_method=wgcna \
    dataset.loader.parameters.node_sample_ratio=1.0 \
    dataset.loader.parameters.method=distance_correlation \
    optimizer.parameters.lr=0.001 \
    optimizer.parameters.weight_decay=0.0 \
    model.backbone.dropout=0.1 \
    model.feature_encoder.out_channels=16 \
    model.backbone.num_layers=4 \
    dataset.loader.parameters.adjacency_threshold=0.0066 \
    model.readout.fc_dropout=0.1 \
    trainer.devices=\[4\] \
    &

python -m ogbench \
    model=gcn \
    dataset=addneuromed \
    seed=42 \
    logger.wandb.tags=[gcn,addneuromed,hpsearch,rerun] \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    experiment=omics_readout \
    dataset.loader.parameters.adjacency_method=wgcna \
    dataset.loader.parameters.node_sample_ratio=0.8 \
    dataset.loader.parameters.method=distance_correlation \
    optimizer.parameters.lr=0.001 \
    optimizer.parameters.weight_decay=0.0 \
    model.backbone.dropout=0.1 \
    model.feature_encoder.out_channels=16 \
    model.backbone.num_layers=4 \
    dataset.loader.parameters.adjacency_threshold=0.1765 \
    model.readout.fc_dropout=0.1 \
    trainer.devices=\[5\] \
    &

python -m ogbench \
    model=gcn \
    dataset=addneuromed \
    seed=42 \
    logger.wandb.tags=[gcn,addneuromed,hpsearch,rerun] \
    logger.wandb.project=bgbench_dataset_grid_search_final \
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
    dataset.loader.parameters.adjacency_threshold=0.1765 \
    model.readout.fc_dropout=0.1 \
    trainer.devices=\[6\] \
    &

python -m ogbench \
    model=gcn \
    dataset=addneuromed \
    seed=456 \
    logger.wandb.tags=[gcn,addneuromed,hpsearch,rerun] \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    experiment=omics_readout \
    dataset.loader.parameters.adjacency_method=string \
    dataset.loader.parameters.node_sample_ratio=1.0 \
    dataset.loader.parameters.method=variance \
    optimizer.parameters.lr=0.001 \
    optimizer.parameters.weight_decay=0.0 \
    model.backbone.dropout=0.1 \
    model.feature_encoder.out_channels=16 \
    model.backbone.num_layers=2 \
    dataset.loader.parameters.adjacency_threshold=0.4 \
    model.readout.fc_dropout=0.1 \
    trainer.devices=\[7\] \
    &

python -m ogbench \
    model=gcn \
    dataset=addneuromed \
    seed=123 \
    logger.wandb.tags=[gcn,addneuromed,hpsearch,rerun] \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    experiment=omics_readout \
    dataset.loader.parameters.adjacency_method=string \
    dataset.loader.parameters.node_sample_ratio=1.0 \
    dataset.loader.parameters.method=variance \
    optimizer.parameters.lr=0.001 \
    optimizer.parameters.weight_decay=0.0 \
    model.backbone.dropout=0.1 \
    model.feature_encoder.out_channels=16 \
    model.backbone.num_layers=4 \
    dataset.loader.parameters.adjacency_threshold=0.4 \
    model.readout.fc_dropout=0.1 \
    trainer.devices=\[0\] \
    &

python -m ogbench \
    model=gcn \
    dataset=addneuromed \
    seed=123 \
    logger.wandb.tags=[gcn,addneuromed,hpsearch,rerun] \
    logger.wandb.project=bgbench_dataset_grid_search_final \
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
    dataset.loader.parameters.adjacency_threshold=0.4 \
    model.readout.fc_dropout=0.1 \
    trainer.devices=\[1\] \
    &

python -m ogbench \
    model=gcn \
    dataset=addneuromed \
    seed=456 \
    logger.wandb.tags=[gcn,addneuromed,hpsearch,rerun] \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    experiment=omics_readout \
    dataset.loader.parameters.adjacency_method=string \
    dataset.loader.parameters.node_sample_ratio=0.8 \
    dataset.loader.parameters.method=variance \
    optimizer.parameters.lr=0.0001 \
    optimizer.parameters.weight_decay=0.0 \
    model.backbone.dropout=0.1 \
    model.feature_encoder.out_channels=64 \
    model.backbone.num_layers=4 \
    dataset.loader.parameters.adjacency_threshold=0.4 \
    model.readout.fc_dropout=0.1 \
    trainer.devices=\[2\] \
    &

python -m ogbench \
    model=gcn \
    dataset=motrpac \
    seed=123 \
    logger.wandb.tags=[gcn,motrpac,hpsearch,rerun] \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    experiment=no_readout \
    dataset.loader.parameters.adjacency_method=wgcna \
    dataset.loader.parameters.node_sample_ratio=1.0 \
    dataset.loader.parameters.method=random \
    optimizer.parameters.lr=0.001 \
    optimizer.parameters.weight_decay=0.0 \
    model.backbone.dropout=0.1 \
    model.feature_encoder.out_channels=16 \
    model.backbone.num_layers=2 \
    dataset.loader.parameters.adjacency_threshold=0.0631 \
    trainer.devices=\[3\] \
    &

python -m ogbench \
    model=gcn \
    dataset=motrpac \
    seed=42 \
    logger.wandb.tags=[gcn,motrpac,hpsearch,rerun] \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    experiment=no_readout \
    dataset.loader.parameters.adjacency_method=wgcna \
    dataset.loader.parameters.node_sample_ratio=1.0 \
    dataset.loader.parameters.method=correlation \
    optimizer.parameters.lr=0.001 \
    optimizer.parameters.weight_decay=0.0 \
    model.backbone.dropout=0.1 \
    model.feature_encoder.out_channels=64 \
    model.backbone.num_layers=2 \
    dataset.loader.parameters.adjacency_threshold=0.1323 \
    trainer.devices=\[4\] \
    &

python -m ogbench \
    model=gcn \
    dataset=motrpac \
    seed=123 \
    logger.wandb.tags=[gcn,motrpac,hpsearch,rerun] \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    experiment=no_readout \
    dataset.loader.parameters.adjacency_method=wgcna \
    dataset.loader.parameters.node_sample_ratio=1.0 \
    dataset.loader.parameters.method=distance_correlation \
    optimizer.parameters.lr=0.001 \
    optimizer.parameters.weight_decay=0.0 \
    model.backbone.dropout=0.1 \
    model.feature_encoder.out_channels=16 \
    model.backbone.num_layers=4 \
    dataset.loader.parameters.adjacency_threshold=0.1061 \
    trainer.devices=\[5\] \
    &

python -m ogbench \
    model=gcn \
    dataset=motrpac \
    seed=42 \
    logger.wandb.tags=[gcn,motrpac,hpsearch,rerun] \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    experiment=no_readout \
    dataset.loader.parameters.adjacency_method=string \
    dataset.loader.parameters.node_sample_ratio=0.8 \
    dataset.loader.parameters.method=correlation \
    optimizer.parameters.lr=0.0001 \
    optimizer.parameters.weight_decay=0.0 \
    model.backbone.dropout=0.1 \
    model.feature_encoder.out_channels=16 \
    model.backbone.num_layers=4 \
    dataset.loader.parameters.adjacency_threshold=0.4 \
    trainer.devices=\[6\] \
    &

python -m ogbench \
    model=gcn \
    dataset=motrpac \
    seed=456 \
    logger.wandb.tags=[gcn,motrpac,hpsearch,rerun] \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    experiment=omics_readout \
    dataset.loader.parameters.adjacency_method=wgcna \
    dataset.loader.parameters.node_sample_ratio=1.0 \
    dataset.loader.parameters.method=variance \
    optimizer.parameters.lr=0.0001 \
    optimizer.parameters.weight_decay=0.0 \
    model.backbone.dropout=0.1 \
    model.feature_encoder.out_channels=64 \
    model.backbone.num_layers=2 \
    dataset.loader.parameters.adjacency_threshold=0.1179 \
    model.readout.fc_dropout=0.1 \
    trainer.devices=\[7\] \
    &

python -m ogbench \
    model=gcn \
    dataset=motrpac \
    seed=42 \
    logger.wandb.tags=[gcn,motrpac,hpsearch,rerun] \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    experiment=omics_readout \
    dataset.loader.parameters.adjacency_method=string \
    dataset.loader.parameters.node_sample_ratio=1.0 \
    dataset.loader.parameters.method=variance \
    optimizer.parameters.lr=0.001 \
    optimizer.parameters.weight_decay=0.0 \
    model.backbone.dropout=0.1 \
    model.feature_encoder.out_channels=16 \
    model.backbone.num_layers=4 \
    dataset.loader.parameters.adjacency_threshold=0.4 \
    model.readout.fc_dropout=0.1 \
    trainer.devices=\[0\] \
    &

python -m ogbench \
    model=gcn \
    dataset=motrpac \
    seed=123 \
    logger.wandb.tags=[gcn,motrpac,hpsearch,rerun] \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    experiment=omics_readout \
    dataset.loader.parameters.adjacency_method=string \
    dataset.loader.parameters.node_sample_ratio=0.8 \
    dataset.loader.parameters.method=variance \
    optimizer.parameters.lr=0.0001 \
    optimizer.parameters.weight_decay=0.0 \
    model.backbone.dropout=0.1 \
    model.feature_encoder.out_channels=64 \
    model.backbone.num_layers=4 \
    dataset.loader.parameters.adjacency_threshold=0.4 \
    model.readout.fc_dropout=0.1 \
    trainer.devices=\[1\] \
    &

python -m ogbench \
    model=gcn \
    dataset=motrpac \
    seed=42 \
    logger.wandb.tags=[gcn,motrpac,hpsearch,rerun] \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    experiment=omics_readout \
    dataset.loader.parameters.adjacency_method=string \
    dataset.loader.parameters.node_sample_ratio=0.8 \
    dataset.loader.parameters.method=correlation \
    optimizer.parameters.lr=0.001 \
    optimizer.parameters.weight_decay=0.0 \
    model.backbone.dropout=0.1 \
    model.feature_encoder.out_channels=16 \
    model.backbone.num_layers=2 \
    dataset.loader.parameters.adjacency_threshold=0.4 \
    model.readout.fc_dropout=0.1 \
    trainer.devices=\[2\] \
    &

python -m ogbench \
    model=gcn \
    dataset=motrpac \
    seed=123 \
    logger.wandb.tags=[gcn,motrpac,hpsearch,rerun] \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    experiment=omics_readout \
    dataset.loader.parameters.adjacency_method=string \
    dataset.loader.parameters.node_sample_ratio=0.8 \
    dataset.loader.parameters.method=correlation \
    optimizer.parameters.lr=0.001 \
    optimizer.parameters.weight_decay=0.0 \
    model.backbone.dropout=0.1 \
    model.feature_encoder.out_channels=16 \
    model.backbone.num_layers=4 \
    dataset.loader.parameters.adjacency_threshold=0.4 \
    model.readout.fc_dropout=0.1 \
    trainer.devices=\[3\] \
    &

python -m ogbench \
    model=gcn \
    dataset=brca \
    seed=123 \
    logger.wandb.tags=[gcn,brca,hpsearch,rerun] \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    experiment=no_readout \
    dataset.loader.parameters.adjacency_method=wgcna \
    dataset.loader.parameters.node_sample_ratio=1.0 \
    dataset.loader.parameters.method=random \
    optimizer.parameters.lr=0.0001 \
    optimizer.parameters.weight_decay=0.0 \
    model.backbone.dropout=0.1 \
    model.feature_encoder.out_channels=16 \
    model.backbone.num_layers=2 \
    dataset.loader.parameters.adjacency_threshold=0.0012 \
    trainer.devices=\[4\] \
    &

python -m ogbench \
    model=gcn \
    dataset=brca \
    seed=456 \
    logger.wandb.tags=[gcn,brca,hpsearch,rerun] \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    experiment=no_readout \
    dataset.loader.parameters.adjacency_method=wgcna \
    dataset.loader.parameters.node_sample_ratio=0.8 \
    dataset.loader.parameters.method=distance_correlation \
    optimizer.parameters.lr=0.0001 \
    optimizer.parameters.weight_decay=0.0 \
    model.backbone.dropout=0.1 \
    model.feature_encoder.out_channels=16 \
    model.backbone.num_layers=4 \
    dataset.loader.parameters.adjacency_threshold=0.0006 \
    trainer.devices=\[5\] \
    &

python -m ogbench \
    model=gcn \
    dataset=brca \
    seed=456 \
    logger.wandb.tags=[gcn,brca,hpsearch,rerun] \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    experiment=no_readout \
    dataset.loader.parameters.adjacency_method=string \
    dataset.loader.parameters.node_sample_ratio=1.0 \
    dataset.loader.parameters.method=variance \
    optimizer.parameters.lr=0.0001 \
    optimizer.parameters.weight_decay=0.0 \
    model.backbone.dropout=0.1 \
    model.feature_encoder.out_channels=64 \
    model.backbone.num_layers=4 \
    dataset.loader.parameters.adjacency_threshold=0.4 \
    trainer.devices=\[6\] \
    &

python -m ogbench \
    model=gcn \
    dataset=brca \
    seed=123 \
    logger.wandb.tags=[gcn,brca,hpsearch,rerun] \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    experiment=no_readout \
    dataset.loader.parameters.adjacency_method=string \
    dataset.loader.parameters.node_sample_ratio=1.0 \
    dataset.loader.parameters.method=random \
    optimizer.parameters.lr=0.0001 \
    optimizer.parameters.weight_decay=0.0 \
    model.backbone.dropout=0.1 \
    model.feature_encoder.out_channels=64 \
    model.backbone.num_layers=2 \
    dataset.loader.parameters.adjacency_threshold=0.4 \
    trainer.devices=\[7\] \
    &

python -m ogbench \
    model=gcn \
    dataset=brca \
    seed=123 \
    logger.wandb.tags=[gcn,brca,hpsearch,rerun] \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    experiment=no_readout \
    dataset.loader.parameters.adjacency_method=string \
    dataset.loader.parameters.node_sample_ratio=1.0 \
    dataset.loader.parameters.method=correlation \
    optimizer.parameters.lr=0.001 \
    optimizer.parameters.weight_decay=0.0 \
    model.backbone.dropout=0.1 \
    model.feature_encoder.out_channels=64 \
    model.backbone.num_layers=4 \
    dataset.loader.parameters.adjacency_threshold=0.4 \
    trainer.devices=\[0\] \
    &

python -m ogbench \
    model=gcn \
    dataset=brca \
    seed=42 \
    logger.wandb.tags=[gcn,brca,hpsearch,rerun] \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    experiment=no_readout \
    dataset.loader.parameters.adjacency_method=string \
    dataset.loader.parameters.node_sample_ratio=1.0 \
    dataset.loader.parameters.method=distance_correlation \
    optimizer.parameters.lr=0.001 \
    optimizer.parameters.weight_decay=0.0 \
    model.backbone.dropout=0.1 \
    model.feature_encoder.out_channels=64 \
    model.backbone.num_layers=2 \
    dataset.loader.parameters.adjacency_threshold=0.4 \
    trainer.devices=\[1\] \
    &

python -m ogbench \
    model=gcn \
    dataset=brca \
    seed=456 \
    logger.wandb.tags=[gcn,brca,hpsearch,rerun] \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    experiment=no_readout \
    dataset.loader.parameters.adjacency_method=string \
    dataset.loader.parameters.node_sample_ratio=0.8 \
    dataset.loader.parameters.method=variance \
    optimizer.parameters.lr=0.0001 \
    optimizer.parameters.weight_decay=0.0 \
    model.backbone.dropout=0.1 \
    model.feature_encoder.out_channels=64 \
    model.backbone.num_layers=2 \
    dataset.loader.parameters.adjacency_threshold=0.4 \
    trainer.devices=\[2\] \
    &

python -m ogbench \
    model=gcn \
    dataset=brca \
    seed=456 \
    logger.wandb.tags=[gcn,brca,hpsearch,rerun] \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    experiment=no_readout \
    dataset.loader.parameters.adjacency_method=string \
    dataset.loader.parameters.node_sample_ratio=0.8 \
    dataset.loader.parameters.method=variance \
    optimizer.parameters.lr=0.001 \
    optimizer.parameters.weight_decay=0.0 \
    model.backbone.dropout=0.1 \
    model.feature_encoder.out_channels=16 \
    model.backbone.num_layers=2 \
    dataset.loader.parameters.adjacency_threshold=0.4 \
    trainer.devices=\[3\] \
    &

python -m ogbench \
    model=gcn \
    dataset=brca \
    seed=42 \
    logger.wandb.tags=[gcn,brca,hpsearch,rerun] \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    experiment=no_readout \
    dataset.loader.parameters.adjacency_method=string \
    dataset.loader.parameters.node_sample_ratio=0.8 \
    dataset.loader.parameters.method=correlation \
    optimizer.parameters.lr=0.001 \
    optimizer.parameters.weight_decay=0.0 \
    model.backbone.dropout=0.1 \
    model.feature_encoder.out_channels=64 \
    model.backbone.num_layers=4 \
    dataset.loader.parameters.adjacency_threshold=0.4 \
    trainer.devices=\[4\] \
    &

python -m ogbench \
    model=gcn \
    dataset=brca \
    seed=123 \
    logger.wandb.tags=[gcn,brca,hpsearch,rerun] \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    experiment=omics_readout \
    dataset.loader.parameters.adjacency_method=wgcna \
    dataset.loader.parameters.node_sample_ratio=1.0 \
    dataset.loader.parameters.method=variance \
    optimizer.parameters.lr=0.001 \
    optimizer.parameters.weight_decay=0.0 \
    model.backbone.dropout=0.1 \
    model.feature_encoder.out_channels=16 \
    model.backbone.num_layers=2 \
    dataset.loader.parameters.adjacency_threshold=0.0009 \
    model.readout.fc_dropout=0.1 \
    trainer.devices=\[5\] \
    &

python -m ogbench \
    model=gcn \
    dataset=brca \
    seed=123 \
    logger.wandb.tags=[gcn,brca,hpsearch,rerun] \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    experiment=omics_readout \
    dataset.loader.parameters.adjacency_method=wgcna \
    dataset.loader.parameters.node_sample_ratio=1.0 \
    dataset.loader.parameters.method=correlation \
    optimizer.parameters.lr=0.0001 \
    optimizer.parameters.weight_decay=0.0 \
    model.backbone.dropout=0.1 \
    model.feature_encoder.out_channels=64 \
    model.backbone.num_layers=2 \
    dataset.loader.parameters.adjacency_threshold=0.0008 \
    model.readout.fc_dropout=0.1 \
    trainer.devices=\[6\] \
    &

python -m ogbench \
    model=gcn \
    dataset=brca \
    seed=456 \
    logger.wandb.tags=[gcn,brca,hpsearch,rerun] \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    experiment=omics_readout \
    dataset.loader.parameters.adjacency_method=wgcna \
    dataset.loader.parameters.node_sample_ratio=1.0 \
    dataset.loader.parameters.method=correlation \
    optimizer.parameters.lr=0.001 \
    optimizer.parameters.weight_decay=0.0 \
    model.backbone.dropout=0.1 \
    model.feature_encoder.out_channels=16 \
    model.backbone.num_layers=4 \
    dataset.loader.parameters.adjacency_threshold=0.0008 \
    model.readout.fc_dropout=0.1 \
    trainer.devices=\[7\] \
    &

python -m ogbench \
    model=gcn \
    dataset=brca \
    seed=42 \
    logger.wandb.tags=[gcn,brca,hpsearch,rerun] \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    experiment=omics_readout \
    dataset.loader.parameters.adjacency_method=wgcna \
    dataset.loader.parameters.node_sample_ratio=1.0 \
    dataset.loader.parameters.method=distance_correlation \
    optimizer.parameters.lr=0.0001 \
    optimizer.parameters.weight_decay=0.0 \
    model.backbone.dropout=0.1 \
    model.feature_encoder.out_channels=16 \
    model.backbone.num_layers=2 \
    dataset.loader.parameters.adjacency_threshold=0.0008 \
    model.readout.fc_dropout=0.1 \
    trainer.devices=\[0\] \
    &

python -m ogbench \
    model=gcn \
    dataset=brca \
    seed=456 \
    logger.wandb.tags=[gcn,brca,hpsearch,rerun] \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    experiment=omics_readout \
    dataset.loader.parameters.adjacency_method=wgcna \
    dataset.loader.parameters.node_sample_ratio=1.0 \
    dataset.loader.parameters.method=distance_correlation \
    optimizer.parameters.lr=0.0001 \
    optimizer.parameters.weight_decay=0.0 \
    model.backbone.dropout=0.1 \
    model.feature_encoder.out_channels=16 \
    model.backbone.num_layers=2 \
    dataset.loader.parameters.adjacency_threshold=0.0008 \
    model.readout.fc_dropout=0.1 \
    trainer.devices=\[1\] \
    &

python -m ogbench \
    model=gcn \
    dataset=brca \
    seed=42 \
    logger.wandb.tags=[gcn,brca,hpsearch,rerun] \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    experiment=omics_readout \
    dataset.loader.parameters.adjacency_method=wgcna \
    dataset.loader.parameters.node_sample_ratio=1.0 \
    dataset.loader.parameters.method=distance_correlation \
    optimizer.parameters.lr=0.001 \
    optimizer.parameters.weight_decay=0.0 \
    model.backbone.dropout=0.1 \
    model.feature_encoder.out_channels=64 \
    model.backbone.num_layers=4 \
    dataset.loader.parameters.adjacency_threshold=0.0008 \
    model.readout.fc_dropout=0.1 \
    trainer.devices=\[2\] \
    &

python -m ogbench \
    model=gcn \
    dataset=brca \
    seed=123 \
    logger.wandb.tags=[gcn,brca,hpsearch,rerun] \
    logger.wandb.project=bgbench_dataset_grid_search_final \
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
    dataset.loader.parameters.adjacency_threshold=0.0005 \
    model.readout.fc_dropout=0.1 \
    trainer.devices=\[3\] \
    &

python -m ogbench \
    model=gcn \
    dataset=brca \
    seed=456 \
    logger.wandb.tags=[gcn,brca,hpsearch,rerun] \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    experiment=omics_readout \
    dataset.loader.parameters.adjacency_method=wgcna \
    dataset.loader.parameters.node_sample_ratio=0.8 \
    dataset.loader.parameters.method=distance_correlation \
    optimizer.parameters.lr=0.0001 \
    optimizer.parameters.weight_decay=0.0 \
    model.backbone.dropout=0.1 \
    model.feature_encoder.out_channels=64 \
    model.backbone.num_layers=4 \
    dataset.loader.parameters.adjacency_threshold=0.0006 \
    model.readout.fc_dropout=0.1 \
    trainer.devices=\[4\] \
    &

python -m ogbench \
    model=gcn \
    dataset=brca \
    seed=42 \
    logger.wandb.tags=[gcn,brca,hpsearch,rerun] \
    logger.wandb.project=bgbench_dataset_grid_search_final \
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
    dataset.loader.parameters.adjacency_threshold=0.0006 \
    model.readout.fc_dropout=0.1 \
    trainer.devices=\[5\] \
    &

python -m ogbench \
    model=gcn \
    dataset=brca \
    seed=42 \
    logger.wandb.tags=[gcn,brca,hpsearch,rerun] \
    logger.wandb.project=bgbench_dataset_grid_search_final \
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
    trainer.devices=\[6\] \
    &

python -m ogbench \
    model=gcn \
    dataset=brca \
    seed=123 \
    logger.wandb.tags=[gcn,brca,hpsearch,rerun] \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    experiment=omics_readout \
    dataset.loader.parameters.adjacency_method=string \
    dataset.loader.parameters.node_sample_ratio=1.0 \
    dataset.loader.parameters.method=correlation \
    optimizer.parameters.lr=0.001 \
    optimizer.parameters.weight_decay=0.0 \
    model.backbone.dropout=0.1 \
    model.feature_encoder.out_channels=64 \
    model.backbone.num_layers=4 \
    dataset.loader.parameters.adjacency_threshold=0.4 \
    model.readout.fc_dropout=0.1 \
    trainer.devices=\[7\] \
    &

python -m ogbench \
    model=gcn \
    dataset=brca \
    seed=42 \
    logger.wandb.tags=[gcn,brca,hpsearch,rerun] \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    experiment=omics_readout \
    dataset.loader.parameters.adjacency_method=string \
    dataset.loader.parameters.node_sample_ratio=0.8 \
    dataset.loader.parameters.method=variance \
    optimizer.parameters.lr=0.0001 \
    optimizer.parameters.weight_decay=0.0 \
    model.backbone.dropout=0.1 \
    model.feature_encoder.out_channels=16 \
    model.backbone.num_layers=2 \
    dataset.loader.parameters.adjacency_threshold=0.4 \
    model.readout.fc_dropout=0.1 \
    trainer.devices=\[0\] \
    &

python -m ogbench \
    model=gin \
    dataset=parkinsons \
    seed=456 \
    logger.wandb.tags=[gin,parkinsons,hpsearch,rerun] \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    experiment=no_readout \
    dataset.loader.parameters.adjacency_method=wgcna \
    dataset.loader.parameters.node_sample_ratio=1.0 \
    dataset.loader.parameters.method=random \
    optimizer.parameters.lr=0.001 \
    optimizer.parameters.weight_decay=0.0 \
    model.backbone.dropout=0.1 \
    model.feature_encoder.out_channels=16 \
    model.backbone.num_layers=2 \
    dataset.loader.parameters.adjacency_threshold=0.0013 \
    trainer.devices=\[1\] \
    &

python -m ogbench \
    model=gin \
    dataset=parkinsons \
    seed=456 \
    logger.wandb.tags=[gin,parkinsons,hpsearch,rerun] \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    experiment=no_readout \
    dataset.loader.parameters.adjacency_method=wgcna \
    dataset.loader.parameters.node_sample_ratio=1.0 \
    dataset.loader.parameters.method=correlation \
    optimizer.parameters.lr=0.0001 \
    optimizer.parameters.weight_decay=0.0 \
    model.backbone.dropout=0.1 \
    model.feature_encoder.out_channels=16 \
    model.backbone.num_layers=2 \
    dataset.loader.parameters.adjacency_threshold=0.0005 \
    trainer.devices=\[2\] \
    &

python -m ogbench \
    model=gin \
    dataset=parkinsons \
    seed=456 \
    logger.wandb.tags=[gin,parkinsons,hpsearch,rerun] \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    experiment=no_readout \
    dataset.loader.parameters.adjacency_method=wgcna \
    dataset.loader.parameters.node_sample_ratio=1.0 \
    dataset.loader.parameters.method=correlation \
    optimizer.parameters.lr=0.0001 \
    optimizer.parameters.weight_decay=0.0 \
    model.backbone.dropout=0.1 \
    model.feature_encoder.out_channels=64 \
    model.backbone.num_layers=2 \
    dataset.loader.parameters.adjacency_threshold=0.0005 \
    trainer.devices=\[3\] \
    &

python -m ogbench \
    model=gin \
    dataset=parkinsons \
    seed=123 \
    logger.wandb.tags=[gin,parkinsons,hpsearch,rerun] \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    experiment=no_readout \
    dataset.loader.parameters.adjacency_method=wgcna \
    dataset.loader.parameters.node_sample_ratio=0.8 \
    dataset.loader.parameters.method=random \
    optimizer.parameters.lr=0.0001 \
    optimizer.parameters.weight_decay=0.0 \
    model.backbone.dropout=0.1 \
    model.feature_encoder.out_channels=16 \
    model.backbone.num_layers=2 \
    dataset.loader.parameters.adjacency_threshold=0.0216 \
    trainer.devices=\[4\] \
    &

python -m ogbench \
    model=gin \
    dataset=parkinsons \
    seed=123 \
    logger.wandb.tags=[gin,parkinsons,hpsearch,rerun] \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    experiment=no_readout \
    dataset.loader.parameters.adjacency_method=string \
    dataset.loader.parameters.node_sample_ratio=0.8 \
    dataset.loader.parameters.method=random \
    optimizer.parameters.lr=0.0001 \
    optimizer.parameters.weight_decay=0.0 \
    model.backbone.dropout=0.1 \
    model.feature_encoder.out_channels=64 \
    model.backbone.num_layers=4 \
    dataset.loader.parameters.adjacency_threshold=0.4 \
    trainer.devices=\[5\] \
    &

python -m ogbench \
    model=gin \
    dataset=parkinsons \
    seed=42 \
    logger.wandb.tags=[gin,parkinsons,hpsearch,rerun] \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    experiment=no_readout \
    dataset.loader.parameters.adjacency_method=string \
    dataset.loader.parameters.node_sample_ratio=0.8 \
    dataset.loader.parameters.method=distance_correlation \
    optimizer.parameters.lr=0.001 \
    optimizer.parameters.weight_decay=0.0 \
    model.backbone.dropout=0.1 \
    model.feature_encoder.out_channels=16 \
    model.backbone.num_layers=4 \
    dataset.loader.parameters.adjacency_threshold=0.4 \
    trainer.devices=\[6\] \
    &

python -m ogbench \
    model=gin \
    dataset=parkinsons \
    seed=42 \
    logger.wandb.tags=[gin,parkinsons,hpsearch,rerun] \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    experiment=no_readout \
    dataset.loader.parameters.adjacency_method=string \
    dataset.loader.parameters.node_sample_ratio=0.8 \
    dataset.loader.parameters.method=distance_correlation \
    optimizer.parameters.lr=0.001 \
    optimizer.parameters.weight_decay=0.0 \
    model.backbone.dropout=0.1 \
    model.feature_encoder.out_channels=64 \
    model.backbone.num_layers=4 \
    dataset.loader.parameters.adjacency_threshold=0.4 \
    trainer.devices=\[7\] \
    &

python -m ogbench \
    model=gin \
    dataset=parkinsons \
    seed=42 \
    logger.wandb.tags=[gin,parkinsons,hpsearch,rerun] \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    experiment=omics_readout \
    dataset.loader.parameters.adjacency_method=wgcna \
    dataset.loader.parameters.node_sample_ratio=1.0 \
    dataset.loader.parameters.method=variance \
    optimizer.parameters.lr=0.001 \
    optimizer.parameters.weight_decay=0.0 \
    model.backbone.dropout=0.1 \
    model.feature_encoder.out_channels=64 \
    model.backbone.num_layers=2 \
    dataset.loader.parameters.adjacency_threshold=0.0346 \
    model.readout.fc_dropout=0.1 \
    trainer.devices=\[0\] \
    &

python -m ogbench \
    model=gin \
    dataset=parkinsons \
    seed=456 \
    logger.wandb.tags=[gin,parkinsons,hpsearch,rerun] \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    experiment=omics_readout \
    dataset.loader.parameters.adjacency_method=wgcna \
    dataset.loader.parameters.node_sample_ratio=0.8 \
    dataset.loader.parameters.method=random \
    optimizer.parameters.lr=0.001 \
    optimizer.parameters.weight_decay=0.0 \
    model.backbone.dropout=0.1 \
    model.feature_encoder.out_channels=16 \
    model.backbone.num_layers=4 \
    dataset.loader.parameters.adjacency_threshold=0.0216 \
    model.readout.fc_dropout=0.1 \
    trainer.devices=\[1\] \
    &

python -m ogbench \
    model=gin \
    dataset=parkinsons \
    seed=42 \
    logger.wandb.tags=[gin,parkinsons,hpsearch,rerun] \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    experiment=omics_readout \
    dataset.loader.parameters.adjacency_method=wgcna \
    dataset.loader.parameters.node_sample_ratio=0.8 \
    dataset.loader.parameters.method=distance_correlation \
    optimizer.parameters.lr=0.0001 \
    optimizer.parameters.weight_decay=0.0 \
    model.backbone.dropout=0.1 \
    model.feature_encoder.out_channels=16 \
    model.backbone.num_layers=4 \
    dataset.loader.parameters.adjacency_threshold=0.0054 \
    model.readout.fc_dropout=0.1 \
    trainer.devices=\[2\] \
    &

python -m ogbench \
    model=gin \
    dataset=parkinsons \
    seed=456 \
    logger.wandb.tags=[gin,parkinsons,hpsearch,rerun] \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    experiment=omics_readout \
    dataset.loader.parameters.adjacency_method=string \
    dataset.loader.parameters.node_sample_ratio=1.0 \
    dataset.loader.parameters.method=variance \
    optimizer.parameters.lr=0.001 \
    optimizer.parameters.weight_decay=0.0 \
    model.backbone.dropout=0.1 \
    model.feature_encoder.out_channels=64 \
    model.backbone.num_layers=4 \
    dataset.loader.parameters.adjacency_threshold=0.4 \
    model.readout.fc_dropout=0.1 \
    trainer.devices=\[3\] \
    &

python -m ogbench \
    model=gin \
    dataset=parkinsons \
    seed=123 \
    logger.wandb.tags=[gin,parkinsons,hpsearch,rerun] \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    experiment=omics_readout \
    dataset.loader.parameters.adjacency_method=string \
    dataset.loader.parameters.node_sample_ratio=1.0 \
    dataset.loader.parameters.method=correlation \
    optimizer.parameters.lr=0.0001 \
    optimizer.parameters.weight_decay=0.0 \
    model.backbone.dropout=0.1 \
    model.feature_encoder.out_channels=16 \
    model.backbone.num_layers=4 \
    dataset.loader.parameters.adjacency_threshold=0.4 \
    model.readout.fc_dropout=0.1 \
    trainer.devices=\[4\] \
    &

python -m ogbench \
    model=gin \
    dataset=parkinsons \
    seed=123 \
    logger.wandb.tags=[gin,parkinsons,hpsearch,rerun] \
    logger.wandb.project=bgbench_dataset_grid_search_final \
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
    trainer.devices=\[5\] \
    &

python -m ogbench \
    model=gin \
    dataset=addneuromed \
    seed=42 \
    logger.wandb.tags=[gin,addneuromed,hpsearch,rerun] \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    experiment=no_readout \
    dataset.loader.parameters.adjacency_method=wgcna \
    dataset.loader.parameters.node_sample_ratio=1.0 \
    dataset.loader.parameters.method=correlation \
    optimizer.parameters.lr=0.0001 \
    optimizer.parameters.weight_decay=0.0 \
    model.backbone.dropout=0.1 \
    model.feature_encoder.out_channels=64 \
    model.backbone.num_layers=2 \
    dataset.loader.parameters.adjacency_threshold=0.0997 \
    trainer.devices=\[6\] \
    &

python -m ogbench \
    model=gin \
    dataset=addneuromed \
    seed=456 \
    logger.wandb.tags=[gin,addneuromed,hpsearch,rerun] \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    experiment=no_readout \
    dataset.loader.parameters.adjacency_method=wgcna \
    dataset.loader.parameters.node_sample_ratio=1.0 \
    dataset.loader.parameters.method=correlation \
    optimizer.parameters.lr=0.0001 \
    optimizer.parameters.weight_decay=0.0 \
    model.backbone.dropout=0.1 \
    model.feature_encoder.out_channels=64 \
    model.backbone.num_layers=4 \
    dataset.loader.parameters.adjacency_threshold=0.0997 \
    trainer.devices=\[7\] \
    &

python -m ogbench \
    model=gin \
    dataset=addneuromed \
    seed=456 \
    logger.wandb.tags=[gin,addneuromed,hpsearch,rerun] \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    experiment=no_readout \
    dataset.loader.parameters.adjacency_method=string \
    dataset.loader.parameters.node_sample_ratio=1.0 \
    dataset.loader.parameters.method=random \
    optimizer.parameters.lr=0.001 \
    optimizer.parameters.weight_decay=0.0 \
    model.backbone.dropout=0.1 \
    model.feature_encoder.out_channels=16 \
    model.backbone.num_layers=4 \
    dataset.loader.parameters.adjacency_threshold=0.4 \
    trainer.devices=\[0\] \
    &

python -m ogbench \
    model=gin \
    dataset=addneuromed \
    seed=42 \
    logger.wandb.tags=[gin,addneuromed,hpsearch,rerun] \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    experiment=no_readout \
    dataset.loader.parameters.adjacency_method=string \
    dataset.loader.parameters.node_sample_ratio=1.0 \
    dataset.loader.parameters.method=random \
    optimizer.parameters.lr=0.001 \
    optimizer.parameters.weight_decay=0.0 \
    model.backbone.dropout=0.1 \
    model.feature_encoder.out_channels=64 \
    model.backbone.num_layers=4 \
    dataset.loader.parameters.adjacency_threshold=0.4 \
    trainer.devices=\[1\] \
    &

python -m ogbench \
    model=gin \
    dataset=addneuromed \
    seed=42 \
    logger.wandb.tags=[gin,addneuromed,hpsearch,rerun] \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    experiment=no_readout \
    dataset.loader.parameters.adjacency_method=string \
    dataset.loader.parameters.node_sample_ratio=1.0 \
    dataset.loader.parameters.method=correlation \
    optimizer.parameters.lr=0.001 \
    optimizer.parameters.weight_decay=0.0 \
    model.backbone.dropout=0.1 \
    model.feature_encoder.out_channels=64 \
    model.backbone.num_layers=2 \
    dataset.loader.parameters.adjacency_threshold=0.4 \
    trainer.devices=\[2\] \
    &

python -m ogbench \
    model=gin \
    dataset=addneuromed \
    seed=456 \
    logger.wandb.tags=[gin,addneuromed,hpsearch,rerun] \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    experiment=no_readout \
    dataset.loader.parameters.adjacency_method=string \
    dataset.loader.parameters.node_sample_ratio=1.0 \
    dataset.loader.parameters.method=distance_correlation \
    optimizer.parameters.lr=0.001 \
    optimizer.parameters.weight_decay=0.0 \
    model.backbone.dropout=0.1 \
    model.feature_encoder.out_channels=16 \
    model.backbone.num_layers=4 \
    dataset.loader.parameters.adjacency_threshold=0.4 \
    trainer.devices=\[3\] \
    &

python -m ogbench \
    model=gin \
    dataset=addneuromed \
    seed=42 \
    logger.wandb.tags=[gin,addneuromed,hpsearch,rerun] \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    experiment=no_readout \
    dataset.loader.parameters.adjacency_method=string \
    dataset.loader.parameters.node_sample_ratio=0.8 \
    dataset.loader.parameters.method=distance_correlation \
    optimizer.parameters.lr=0.0001 \
    optimizer.parameters.weight_decay=0.0 \
    model.backbone.dropout=0.1 \
    model.feature_encoder.out_channels=16 \
    model.backbone.num_layers=2 \
    dataset.loader.parameters.adjacency_threshold=0.4 \
    trainer.devices=\[4\] \
    &

python -m ogbench \
    model=gin \
    dataset=addneuromed \
    seed=123 \
    logger.wandb.tags=[gin,addneuromed,hpsearch,rerun] \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    experiment=no_readout \
    dataset.loader.parameters.adjacency_method=string \
    dataset.loader.parameters.node_sample_ratio=0.8 \
    dataset.loader.parameters.method=distance_correlation \
    optimizer.parameters.lr=0.001 \
    optimizer.parameters.weight_decay=0.0 \
    model.backbone.dropout=0.1 \
    model.feature_encoder.out_channels=16 \
    model.backbone.num_layers=2 \
    dataset.loader.parameters.adjacency_threshold=0.4 \
    trainer.devices=\[5\] \
    &

python -m ogbench \
    model=gin \
    dataset=addneuromed \
    seed=123 \
    logger.wandb.tags=[gin,addneuromed,hpsearch,rerun] \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    experiment=omics_readout \
    dataset.loader.parameters.adjacency_method=wgcna \
    dataset.loader.parameters.node_sample_ratio=1.0 \
    dataset.loader.parameters.method=correlation \
    optimizer.parameters.lr=0.0001 \
    optimizer.parameters.weight_decay=0.0 \
    model.backbone.dropout=0.1 \
    model.feature_encoder.out_channels=64 \
    model.backbone.num_layers=4 \
    dataset.loader.parameters.adjacency_threshold=0.0997 \
    model.readout.fc_dropout=0.1 \
    trainer.devices=\[6\] \
    &

python -m ogbench \
    model=gin \
    dataset=addneuromed \
    seed=42 \
    logger.wandb.tags=[gin,addneuromed,hpsearch,rerun] \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    experiment=omics_readout \
    dataset.loader.parameters.adjacency_method=wgcna \
    dataset.loader.parameters.node_sample_ratio=1.0 \
    dataset.loader.parameters.method=correlation \
    optimizer.parameters.lr=0.001 \
    optimizer.parameters.weight_decay=0.0 \
    model.backbone.dropout=0.1 \
    model.feature_encoder.out_channels=16 \
    model.backbone.num_layers=2 \
    dataset.loader.parameters.adjacency_threshold=0.0997 \
    model.readout.fc_dropout=0.1 \
    trainer.devices=\[7\] \
    &

python -m ogbench \
    model=gin \
    dataset=addneuromed \
    seed=456 \
    logger.wandb.tags=[gin,addneuromed,hpsearch,rerun] \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    experiment=omics_readout \
    dataset.loader.parameters.adjacency_method=wgcna \
    dataset.loader.parameters.node_sample_ratio=0.8 \
    dataset.loader.parameters.method=random \
    optimizer.parameters.lr=0.0001 \
    optimizer.parameters.weight_decay=0.0 \
    model.backbone.dropout=0.1 \
    model.feature_encoder.out_channels=16 \
    model.backbone.num_layers=2 \
    dataset.loader.parameters.adjacency_threshold=0.0005 \
    model.readout.fc_dropout=0.1 \
    trainer.devices=\[0\] \
    &

python -m ogbench \
    model=gin \
    dataset=addneuromed \
    seed=42 \
    logger.wandb.tags=[gin,addneuromed,hpsearch,rerun] \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    experiment=omics_readout \
    dataset.loader.parameters.adjacency_method=wgcna \
    dataset.loader.parameters.node_sample_ratio=0.8 \
    dataset.loader.parameters.method=random \
    optimizer.parameters.lr=0.001 \
    optimizer.parameters.weight_decay=0.0 \
    model.backbone.dropout=0.1 \
    model.feature_encoder.out_channels=16 \
    model.backbone.num_layers=4 \
    dataset.loader.parameters.adjacency_threshold=0.0005 \
    model.readout.fc_dropout=0.1 \
    trainer.devices=\[1\] \
    &

python -m ogbench \
    model=gin \
    dataset=addneuromed \
    seed=42 \
    logger.wandb.tags=[gin,addneuromed,hpsearch,rerun] \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    experiment=omics_readout \
    dataset.loader.parameters.adjacency_method=string \
    dataset.loader.parameters.node_sample_ratio=1.0 \
    dataset.loader.parameters.method=variance \
    optimizer.parameters.lr=0.0001 \
    optimizer.parameters.weight_decay=0.0 \
    model.backbone.dropout=0.1 \
    model.feature_encoder.out_channels=64 \
    model.backbone.num_layers=4 \
    dataset.loader.parameters.adjacency_threshold=0.4 \
    model.readout.fc_dropout=0.1 \
    trainer.devices=\[2\] \
    &

python -m ogbench \
    model=gin \
    dataset=addneuromed \
    seed=123 \
    logger.wandb.tags=[gin,addneuromed,hpsearch,rerun] \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    experiment=omics_readout \
    dataset.loader.parameters.adjacency_method=string \
    dataset.loader.parameters.node_sample_ratio=1.0 \
    dataset.loader.parameters.method=correlation \
    optimizer.parameters.lr=0.0001 \
    optimizer.parameters.weight_decay=0.0 \
    model.backbone.dropout=0.1 \
    model.feature_encoder.out_channels=16 \
    model.backbone.num_layers=2 \
    dataset.loader.parameters.adjacency_threshold=0.4 \
    model.readout.fc_dropout=0.1 \
    trainer.devices=\[3\] \
    &

python -m ogbench \
    model=gin \
    dataset=addneuromed \
    seed=123 \
    logger.wandb.tags=[gin,addneuromed,hpsearch,rerun] \
    logger.wandb.project=bgbench_dataset_grid_search_final \
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
    trainer.devices=\[4\] \
    &

python -m ogbench \
    model=gin \
    dataset=motrpac \
    seed=123 \
    logger.wandb.tags=[gin,motrpac,hpsearch,rerun] \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    experiment=no_readout \
    dataset.loader.parameters.adjacency_method=string \
    dataset.loader.parameters.node_sample_ratio=1.0 \
    dataset.loader.parameters.method=random \
    optimizer.parameters.lr=0.0001 \
    optimizer.parameters.weight_decay=0.0 \
    model.backbone.dropout=0.1 \
    model.feature_encoder.out_channels=64 \
    model.backbone.num_layers=4 \
    dataset.loader.parameters.adjacency_threshold=0.4 \
    trainer.devices=\[5\] \
    &

python -m ogbench \
    model=gin \
    dataset=motrpac \
    seed=123 \
    logger.wandb.tags=[gin,motrpac,hpsearch,rerun] \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    experiment=no_readout \
    dataset.loader.parameters.adjacency_method=string \
    dataset.loader.parameters.node_sample_ratio=1.0 \
    dataset.loader.parameters.method=distance_correlation \
    optimizer.parameters.lr=0.001 \
    optimizer.parameters.weight_decay=0.0 \
    model.backbone.dropout=0.1 \
    model.feature_encoder.out_channels=16 \
    model.backbone.num_layers=4 \
    dataset.loader.parameters.adjacency_threshold=0.4 \
    trainer.devices=\[6\] \
    &

python -m ogbench \
    model=gin \
    dataset=motrpac \
    seed=456 \
    logger.wandb.tags=[gin,motrpac,hpsearch,rerun] \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    experiment=no_readout \
    dataset.loader.parameters.adjacency_method=string \
    dataset.loader.parameters.node_sample_ratio=0.8 \
    dataset.loader.parameters.method=variance \
    optimizer.parameters.lr=0.0001 \
    optimizer.parameters.weight_decay=0.0 \
    model.backbone.dropout=0.1 \
    model.feature_encoder.out_channels=64 \
    model.backbone.num_layers=2 \
    dataset.loader.parameters.adjacency_threshold=0.4 \
    trainer.devices=\[7\] \
    &

python -m ogbench \
    model=gin \
    dataset=motrpac \
    seed=42 \
    logger.wandb.tags=[gin,motrpac,hpsearch,rerun] \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    experiment=no_readout \
    dataset.loader.parameters.adjacency_method=string \
    dataset.loader.parameters.node_sample_ratio=0.8 \
    dataset.loader.parameters.method=random \
    optimizer.parameters.lr=0.001 \
    optimizer.parameters.weight_decay=0.0 \
    model.backbone.dropout=0.1 \
    model.feature_encoder.out_channels=16 \
    model.backbone.num_layers=2 \
    dataset.loader.parameters.adjacency_threshold=0.4 \
    trainer.devices=\[0\] \
    &

python -m ogbench \
    model=gin \
    dataset=motrpac \
    seed=456 \
    logger.wandb.tags=[gin,motrpac,hpsearch,rerun] \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    experiment=no_readout \
    dataset.loader.parameters.adjacency_method=string \
    dataset.loader.parameters.node_sample_ratio=0.8 \
    dataset.loader.parameters.method=correlation \
    optimizer.parameters.lr=0.0001 \
    optimizer.parameters.weight_decay=0.0 \
    model.backbone.dropout=0.1 \
    model.feature_encoder.out_channels=64 \
    model.backbone.num_layers=2 \
    dataset.loader.parameters.adjacency_threshold=0.4 \
    trainer.devices=\[1\] \
    &

python -m ogbench \
    model=gin \
    dataset=motrpac \
    seed=456 \
    logger.wandb.tags=[gin,motrpac,hpsearch,rerun] \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    experiment=omics_readout \
    dataset.loader.parameters.adjacency_method=wgcna \
    dataset.loader.parameters.node_sample_ratio=0.8 \
    dataset.loader.parameters.method=correlation \
    optimizer.parameters.lr=0.0001 \
    optimizer.parameters.weight_decay=0.0 \
    model.backbone.dropout=0.1 \
    model.feature_encoder.out_channels=16 \
    model.backbone.num_layers=2 \
    dataset.loader.parameters.adjacency_threshold=0.1105 \
    model.readout.fc_dropout=0.1 \
    trainer.devices=\[2\] \
    &

python -m ogbench \
    model=gin \
    dataset=motrpac \
    seed=456 \
    logger.wandb.tags=[gin,motrpac,hpsearch,rerun] \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    experiment=omics_readout \
    dataset.loader.parameters.adjacency_method=string \
    dataset.loader.parameters.node_sample_ratio=1.0 \
    dataset.loader.parameters.method=variance \
    optimizer.parameters.lr=0.0001 \
    optimizer.parameters.weight_decay=0.0 \
    model.backbone.dropout=0.1 \
    model.feature_encoder.out_channels=16 \
    model.backbone.num_layers=4 \
    dataset.loader.parameters.adjacency_threshold=0.4 \
    model.readout.fc_dropout=0.1 \
    trainer.devices=\[3\] \
    &

python -m ogbench \
    model=gin \
    dataset=motrpac \
    seed=123 \
    logger.wandb.tags=[gin,motrpac,hpsearch,rerun] \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    experiment=omics_readout \
    dataset.loader.parameters.adjacency_method=string \
    dataset.loader.parameters.node_sample_ratio=0.8 \
    dataset.loader.parameters.method=variance \
    optimizer.parameters.lr=0.001 \
    optimizer.parameters.weight_decay=0.0 \
    model.backbone.dropout=0.1 \
    model.feature_encoder.out_channels=16 \
    model.backbone.num_layers=2 \
    dataset.loader.parameters.adjacency_threshold=0.4 \
    model.readout.fc_dropout=0.1 \
    trainer.devices=\[4\] \
    &

python -m ogbench \
    model=gin \
    dataset=motrpac \
    seed=456 \
    logger.wandb.tags=[gin,motrpac,hpsearch,rerun] \
    logger.wandb.project=bgbench_dataset_grid_search_final \
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
    trainer.devices=\[5\] \
    &

python -m ogbench \
    model=gin \
    dataset=brca \
    seed=456 \
    logger.wandb.tags=[gin,brca,hpsearch,rerun] \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    experiment=no_readout \
    dataset.loader.parameters.adjacency_method=wgcna \
    dataset.loader.parameters.node_sample_ratio=1.0 \
    dataset.loader.parameters.method=distance_correlation \
    optimizer.parameters.lr=0.001 \
    optimizer.parameters.weight_decay=0.0 \
    model.backbone.dropout=0.1 \
    model.feature_encoder.out_channels=64 \
    model.backbone.num_layers=2 \
    dataset.loader.parameters.adjacency_threshold=0.0008 \
    trainer.devices=\[6\] \
    &

python -m ogbench \
    model=gin \
    dataset=brca \
    seed=42 \
    logger.wandb.tags=[gin,brca,hpsearch,rerun] \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    experiment=no_readout \
    dataset.loader.parameters.adjacency_method=wgcna \
    dataset.loader.parameters.node_sample_ratio=0.8 \
    dataset.loader.parameters.method=variance \
    optimizer.parameters.lr=0.001 \
    optimizer.parameters.weight_decay=0.0 \
    model.backbone.dropout=0.1 \
    model.feature_encoder.out_channels=16 \
    model.backbone.num_layers=4 \
    dataset.loader.parameters.adjacency_threshold=0.0005 \
    trainer.devices=\[7\] \
    &

python -m ogbench \
    model=gin \
    dataset=brca \
    seed=456 \
    logger.wandb.tags=[gin,brca,hpsearch,rerun] \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    experiment=no_readout \
    dataset.loader.parameters.adjacency_method=wgcna \
    dataset.loader.parameters.node_sample_ratio=0.8 \
    dataset.loader.parameters.method=distance_correlation \
    optimizer.parameters.lr=0.001 \
    optimizer.parameters.weight_decay=0.0 \
    model.backbone.dropout=0.1 \
    model.feature_encoder.out_channels=16 \
    model.backbone.num_layers=4 \
    dataset.loader.parameters.adjacency_threshold=0.0006 \
    trainer.devices=\[0\] \
    &

python -m ogbench \
    model=gin \
    dataset=brca \
    seed=456 \
    logger.wandb.tags=[gin,brca,hpsearch,rerun] \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    experiment=no_readout \
    dataset.loader.parameters.adjacency_method=string \
    dataset.loader.parameters.node_sample_ratio=1.0 \
    dataset.loader.parameters.method=variance \
    optimizer.parameters.lr=0.0001 \
    optimizer.parameters.weight_decay=0.0 \
    model.backbone.dropout=0.1 \
    model.feature_encoder.out_channels=64 \
    model.backbone.num_layers=2 \
    dataset.loader.parameters.adjacency_threshold=0.4 \
    trainer.devices=\[1\] \
    &

python -m ogbench \
    model=gin \
    dataset=brca \
    seed=42 \
    logger.wandb.tags=[gin,brca,hpsearch,rerun] \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    experiment=no_readout \
    dataset.loader.parameters.adjacency_method=string \
    dataset.loader.parameters.node_sample_ratio=1.0 \
    dataset.loader.parameters.method=random \
    optimizer.parameters.lr=0.001 \
    optimizer.parameters.weight_decay=0.0 \
    model.backbone.dropout=0.1 \
    model.feature_encoder.out_channels=16 \
    model.backbone.num_layers=2 \
    dataset.loader.parameters.adjacency_threshold=0.4 \
    trainer.devices=\[2\] \
    &

python -m ogbench \
    model=gin \
    dataset=brca \
    seed=42 \
    logger.wandb.tags=[gin,brca,hpsearch,rerun] \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    experiment=no_readout \
    dataset.loader.parameters.adjacency_method=string \
    dataset.loader.parameters.node_sample_ratio=1.0 \
    dataset.loader.parameters.method=correlation \
    optimizer.parameters.lr=0.0001 \
    optimizer.parameters.weight_decay=0.0 \
    model.backbone.dropout=0.1 \
    model.feature_encoder.out_channels=16 \
    model.backbone.num_layers=4 \
    dataset.loader.parameters.adjacency_threshold=0.4 \
    trainer.devices=\[3\] \
    &

python -m ogbench \
    model=gin \
    dataset=brca \
    seed=456 \
    logger.wandb.tags=[gin,brca,hpsearch,rerun] \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    experiment=no_readout \
    dataset.loader.parameters.adjacency_method=string \
    dataset.loader.parameters.node_sample_ratio=1.0 \
    dataset.loader.parameters.method=correlation \
    optimizer.parameters.lr=0.0001 \
    optimizer.parameters.weight_decay=0.0 \
    model.backbone.dropout=0.1 \
    model.feature_encoder.out_channels=64 \
    model.backbone.num_layers=4 \
    dataset.loader.parameters.adjacency_threshold=0.4 \
    trainer.devices=\[4\] \
    &

python -m ogbench \
    model=gin \
    dataset=brca \
    seed=42 \
    logger.wandb.tags=[gin,brca,hpsearch,rerun] \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    experiment=no_readout \
    dataset.loader.parameters.adjacency_method=string \
    dataset.loader.parameters.node_sample_ratio=1.0 \
    dataset.loader.parameters.method=distance_correlation \
    optimizer.parameters.lr=0.0001 \
    optimizer.parameters.weight_decay=0.0 \
    model.backbone.dropout=0.1 \
    model.feature_encoder.out_channels=16 \
    model.backbone.num_layers=4 \
    dataset.loader.parameters.adjacency_threshold=0.4 \
    trainer.devices=\[5\] \
    &

python -m ogbench \
    model=gin \
    dataset=brca \
    seed=123 \
    logger.wandb.tags=[gin,brca,hpsearch,rerun] \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    experiment=no_readout \
    dataset.loader.parameters.adjacency_method=string \
    dataset.loader.parameters.node_sample_ratio=1.0 \
    dataset.loader.parameters.method=distance_correlation \
    optimizer.parameters.lr=0.001 \
    optimizer.parameters.weight_decay=0.0 \
    model.backbone.dropout=0.1 \
    model.feature_encoder.out_channels=64 \
    model.backbone.num_layers=2 \
    dataset.loader.parameters.adjacency_threshold=0.4 \
    trainer.devices=\[6\] \
    &

python -m ogbench \
    model=gin \
    dataset=brca \
    seed=42 \
    logger.wandb.tags=[gin,brca,hpsearch,rerun] \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    experiment=no_readout \
    dataset.loader.parameters.adjacency_method=string \
    dataset.loader.parameters.node_sample_ratio=0.8 \
    dataset.loader.parameters.method=variance \
    optimizer.parameters.lr=0.0001 \
    optimizer.parameters.weight_decay=0.0 \
    model.backbone.dropout=0.1 \
    model.feature_encoder.out_channels=16 \
    model.backbone.num_layers=4 \
    dataset.loader.parameters.adjacency_threshold=0.4 \
    trainer.devices=\[7\] \
    &

python -m ogbench \
    model=gin \
    dataset=brca \
    seed=42 \
    logger.wandb.tags=[gin,brca,hpsearch,rerun] \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    experiment=omics_readout \
    dataset.loader.parameters.adjacency_method=wgcna \
    dataset.loader.parameters.node_sample_ratio=1.0 \
    dataset.loader.parameters.method=variance \
    optimizer.parameters.lr=0.0001 \
    optimizer.parameters.weight_decay=0.0 \
    model.backbone.dropout=0.1 \
    model.feature_encoder.out_channels=64 \
    model.backbone.num_layers=2 \
    dataset.loader.parameters.adjacency_threshold=0.0009 \
    model.readout.fc_dropout=0.1 \
    trainer.devices=\[0\] \
    &

python -m ogbench \
    model=gin \
    dataset=brca \
    seed=42 \
    logger.wandb.tags=[gin,brca,hpsearch,rerun] \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    experiment=omics_readout \
    dataset.loader.parameters.adjacency_method=wgcna \
    dataset.loader.parameters.node_sample_ratio=1.0 \
    dataset.loader.parameters.method=random \
    optimizer.parameters.lr=0.0001 \
    optimizer.parameters.weight_decay=0.0 \
    model.backbone.dropout=0.1 \
    model.feature_encoder.out_channels=16 \
    model.backbone.num_layers=4 \
    dataset.loader.parameters.adjacency_threshold=0.0012 \
    model.readout.fc_dropout=0.1 \
    trainer.devices=\[1\] \
    &

python -m ogbench \
    model=gin \
    dataset=brca \
    seed=456 \
    logger.wandb.tags=[gin,brca,hpsearch,rerun] \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    experiment=omics_readout \
    dataset.loader.parameters.adjacency_method=wgcna \
    dataset.loader.parameters.node_sample_ratio=1.0 \
    dataset.loader.parameters.method=random \
    optimizer.parameters.lr=0.0001 \
    optimizer.parameters.weight_decay=0.0 \
    model.backbone.dropout=0.1 \
    model.feature_encoder.out_channels=64 \
    model.backbone.num_layers=2 \
    dataset.loader.parameters.adjacency_threshold=0.0012 \
    model.readout.fc_dropout=0.1 \
    trainer.devices=\[2\] \
    &

python -m ogbench \
    model=gin \
    dataset=brca \
    seed=42 \
    logger.wandb.tags=[gin,brca,hpsearch,rerun] \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    experiment=omics_readout \
    dataset.loader.parameters.adjacency_method=wgcna \
    dataset.loader.parameters.node_sample_ratio=1.0 \
    dataset.loader.parameters.method=distance_correlation \
    optimizer.parameters.lr=0.001 \
    optimizer.parameters.weight_decay=0.0 \
    model.backbone.dropout=0.1 \
    model.feature_encoder.out_channels=16 \
    model.backbone.num_layers=4 \
    dataset.loader.parameters.adjacency_threshold=0.0008 \
    model.readout.fc_dropout=0.1 \
    trainer.devices=\[3\] \
    &

python -m ogbench \
    model=gin \
    dataset=brca \
    seed=42 \
    logger.wandb.tags=[gin,brca,hpsearch,rerun] \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    experiment=omics_readout \
    dataset.loader.parameters.adjacency_method=wgcna \
    dataset.loader.parameters.node_sample_ratio=0.8 \
    dataset.loader.parameters.method=correlation \
    optimizer.parameters.lr=0.001 \
    optimizer.parameters.weight_decay=0.0 \
    model.backbone.dropout=0.1 \
    model.feature_encoder.out_channels=16 \
    model.backbone.num_layers=2 \
    dataset.loader.parameters.adjacency_threshold=0.0005 \
    model.readout.fc_dropout=0.1 \
    trainer.devices=\[4\] \
    &

python -m ogbench \
    model=gin \
    dataset=brca \
    seed=123 \
    logger.wandb.tags=[gin,brca,hpsearch,rerun] \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    experiment=omics_readout \
    dataset.loader.parameters.adjacency_method=wgcna \
    dataset.loader.parameters.node_sample_ratio=0.8 \
    dataset.loader.parameters.method=correlation \
    optimizer.parameters.lr=0.001 \
    optimizer.parameters.weight_decay=0.0 \
    model.backbone.dropout=0.1 \
    model.feature_encoder.out_channels=64 \
    model.backbone.num_layers=4 \
    dataset.loader.parameters.adjacency_threshold=0.0005 \
    model.readout.fc_dropout=0.1 \
    trainer.devices=\[5\] \
    &

python -m ogbench \
    model=gin \
    dataset=brca \
    seed=456 \
    logger.wandb.tags=[gin,brca,hpsearch,rerun] \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    experiment=omics_readout \
    dataset.loader.parameters.adjacency_method=string \
    dataset.loader.parameters.node_sample_ratio=1.0 \
    dataset.loader.parameters.method=variance \
    optimizer.parameters.lr=0.0001 \
    optimizer.parameters.weight_decay=0.0 \
    model.backbone.dropout=0.1 \
    model.feature_encoder.out_channels=64 \
    model.backbone.num_layers=4 \
    dataset.loader.parameters.adjacency_threshold=0.4 \
    model.readout.fc_dropout=0.1 \
    trainer.devices=\[6\] \
    &

python -m ogbench \
    model=gin \
    dataset=brca \
    seed=42 \
    logger.wandb.tags=[gin,brca,hpsearch,rerun] \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    experiment=omics_readout \
    dataset.loader.parameters.adjacency_method=string \
    dataset.loader.parameters.node_sample_ratio=1.0 \
    dataset.loader.parameters.method=variance \
    optimizer.parameters.lr=0.001 \
    optimizer.parameters.weight_decay=0.0 \
    model.backbone.dropout=0.1 \
    model.feature_encoder.out_channels=64 \
    model.backbone.num_layers=2 \
    dataset.loader.parameters.adjacency_threshold=0.4 \
    model.readout.fc_dropout=0.1 \
    trainer.devices=\[7\] \
    &

python -m ogbench \
    model=gin \
    dataset=brca \
    seed=456 \
    logger.wandb.tags=[gin,brca,hpsearch,rerun] \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    experiment=omics_readout \
    dataset.loader.parameters.adjacency_method=string \
    dataset.loader.parameters.node_sample_ratio=1.0 \
    dataset.loader.parameters.method=random \
    optimizer.parameters.lr=0.0001 \
    optimizer.parameters.weight_decay=0.0 \
    model.backbone.dropout=0.1 \
    model.feature_encoder.out_channels=64 \
    model.backbone.num_layers=2 \
    dataset.loader.parameters.adjacency_threshold=0.4 \
    model.readout.fc_dropout=0.1 \
    trainer.devices=\[0\] \
    &

python -m ogbench \
    model=gin \
    dataset=brca \
    seed=42 \
    logger.wandb.tags=[gin,brca,hpsearch,rerun] \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    experiment=omics_readout \
    dataset.loader.parameters.adjacency_method=string \
    dataset.loader.parameters.node_sample_ratio=1.0 \
    dataset.loader.parameters.method=correlation \
    optimizer.parameters.lr=0.0001 \
    optimizer.parameters.weight_decay=0.0 \
    model.backbone.dropout=0.1 \
    model.feature_encoder.out_channels=16 \
    model.backbone.num_layers=2 \
    dataset.loader.parameters.adjacency_threshold=0.4 \
    model.readout.fc_dropout=0.1 \
    trainer.devices=\[1\] \
    &

python -m ogbench \
    model=gin \
    dataset=brca \
    seed=456 \
    logger.wandb.tags=[gin,brca,hpsearch,rerun] \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    experiment=omics_readout \
    dataset.loader.parameters.adjacency_method=string \
    dataset.loader.parameters.node_sample_ratio=0.8 \
    dataset.loader.parameters.method=correlation \
    optimizer.parameters.lr=0.001 \
    optimizer.parameters.weight_decay=0.0 \
    model.backbone.dropout=0.1 \
    model.feature_encoder.out_channels=16 \
    model.backbone.num_layers=2 \
    dataset.loader.parameters.adjacency_threshold=0.4 \
    model.readout.fc_dropout=0.1 \
    trainer.devices=\[2\] \
    &

python -m ogbench \
    model=gatv2 \
    dataset=parkinsons \
    seed=123 \
    logger.wandb.tags=[gatv2,parkinsons,hpsearch,rerun] \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    experiment=no_readout \
    dataset.loader.parameters.adjacency_method=wgcna \
    dataset.loader.parameters.node_sample_ratio=1.0 \
    dataset.loader.parameters.method=variance \
    optimizer.parameters.lr=0.001 \
    optimizer.parameters.weight_decay=0.0 \
    model.backbone.dropout=0.1 \
    model.feature_encoder.out_channels=16 \
    model.backbone.num_layers=2 \
    model.backbone.heads=2 \
    dataset.loader.parameters.adjacency_threshold=0.0346 \
    trainer.devices=\[3\] \
    &

python -m ogbench \
    model=gatv2 \
    dataset=parkinsons \
    seed=456 \
    logger.wandb.tags=[gatv2,parkinsons,hpsearch,rerun] \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    experiment=no_readout \
    dataset.loader.parameters.adjacency_method=wgcna \
    dataset.loader.parameters.node_sample_ratio=1.0 \
    dataset.loader.parameters.method=variance \
    optimizer.parameters.lr=0.001 \
    optimizer.parameters.weight_decay=0.0 \
    model.backbone.dropout=0.1 \
    model.feature_encoder.out_channels=16 \
    model.backbone.num_layers=4 \
    model.backbone.heads=4 \
    dataset.loader.parameters.adjacency_threshold=0.0346 \
    trainer.devices=\[4\] \
    &

python -m ogbench \
    model=gatv2 \
    dataset=parkinsons \
    seed=123 \
    logger.wandb.tags=[gatv2,parkinsons,hpsearch,rerun] \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    experiment=no_readout \
    dataset.loader.parameters.adjacency_method=wgcna \
    dataset.loader.parameters.node_sample_ratio=0.8 \
    dataset.loader.parameters.method=random \
    optimizer.parameters.lr=0.001 \
    optimizer.parameters.weight_decay=0.0 \
    model.backbone.dropout=0.1 \
    model.feature_encoder.out_channels=64 \
    model.backbone.num_layers=4 \
    model.backbone.heads=4 \
    dataset.loader.parameters.adjacency_threshold=0.0216 \
    trainer.devices=\[5\] \
    &

python -m ogbench \
    model=gatv2 \
    dataset=parkinsons \
    seed=42 \
    logger.wandb.tags=[gatv2,parkinsons,hpsearch,rerun] \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    experiment=no_readout \
    dataset.loader.parameters.adjacency_method=string \
    dataset.loader.parameters.node_sample_ratio=1.0 \
    dataset.loader.parameters.method=variance \
    optimizer.parameters.lr=0.0001 \
    optimizer.parameters.weight_decay=0.0 \
    model.backbone.dropout=0.1 \
    model.feature_encoder.out_channels=16 \
    model.backbone.num_layers=4 \
    model.backbone.heads=2 \
    dataset.loader.parameters.adjacency_threshold=0.4 \
    trainer.devices=\[6\] \
    &

python -m ogbench \
    model=gatv2 \
    dataset=parkinsons \
    seed=123 \
    logger.wandb.tags=[gatv2,parkinsons,hpsearch,rerun] \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    experiment=no_readout \
    dataset.loader.parameters.adjacency_method=string \
    dataset.loader.parameters.node_sample_ratio=1.0 \
    dataset.loader.parameters.method=correlation \
    optimizer.parameters.lr=0.0001 \
    optimizer.parameters.weight_decay=0.0 \
    model.backbone.dropout=0.1 \
    model.feature_encoder.out_channels=16 \
    model.backbone.num_layers=2 \
    model.backbone.heads=2 \
    dataset.loader.parameters.adjacency_threshold=0.4 \
    trainer.devices=\[7\] \
    &

python -m ogbench \
    model=gatv2 \
    dataset=parkinsons \
    seed=123 \
    logger.wandb.tags=[gatv2,parkinsons,hpsearch,rerun] \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    experiment=no_readout \
    dataset.loader.parameters.adjacency_method=string \
    dataset.loader.parameters.node_sample_ratio=1.0 \
    dataset.loader.parameters.method=distance_correlation \
    optimizer.parameters.lr=0.0001 \
    optimizer.parameters.weight_decay=0.0 \
    model.backbone.dropout=0.1 \
    model.feature_encoder.out_channels=16 \
    model.backbone.num_layers=2 \
    model.backbone.heads=2 \
    dataset.loader.parameters.adjacency_threshold=0.4 \
    trainer.devices=\[0\] \
    &

python -m ogbench \
    model=gatv2 \
    dataset=parkinsons \
    seed=456 \
    logger.wandb.tags=[gatv2,parkinsons,hpsearch,rerun] \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    experiment=no_readout \
    dataset.loader.parameters.adjacency_method=string \
    dataset.loader.parameters.node_sample_ratio=0.8 \
    dataset.loader.parameters.method=distance_correlation \
    optimizer.parameters.lr=0.0001 \
    optimizer.parameters.weight_decay=0.0 \
    model.backbone.dropout=0.1 \
    model.feature_encoder.out_channels=16 \
    model.backbone.num_layers=2 \
    model.backbone.heads=4 \
    dataset.loader.parameters.adjacency_threshold=0.4 \
    trainer.devices=\[1\] \
    &

python -m ogbench \
    model=gatv2 \
    dataset=parkinsons \
    seed=42 \
    logger.wandb.tags=[gatv2,parkinsons,hpsearch,rerun] \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    experiment=omics_readout \
    dataset.loader.parameters.adjacency_method=wgcna \
    dataset.loader.parameters.node_sample_ratio=1.0 \
    dataset.loader.parameters.method=variance \
    optimizer.parameters.lr=0.001 \
    optimizer.parameters.weight_decay=0.0 \
    model.backbone.dropout=0.1 \
    model.feature_encoder.out_channels=64 \
    model.backbone.num_layers=2 \
    model.backbone.heads=2 \
    dataset.loader.parameters.adjacency_threshold=0.0346 \
    model.readout.fc_dropout=0.1 \
    trainer.devices=\[2\] \
    &

python -m ogbench \
    model=gatv2 \
    dataset=parkinsons \
    seed=123 \
    logger.wandb.tags=[gatv2,parkinsons,hpsearch,rerun] \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    experiment=omics_readout \
    dataset.loader.parameters.adjacency_method=wgcna \
    dataset.loader.parameters.node_sample_ratio=0.8 \
    dataset.loader.parameters.method=distance_correlation \
    optimizer.parameters.lr=0.0001 \
    optimizer.parameters.weight_decay=0.0 \
    model.backbone.dropout=0.1 \
    model.feature_encoder.out_channels=16 \
    model.backbone.num_layers=2 \
    model.backbone.heads=2 \
    dataset.loader.parameters.adjacency_threshold=0.0054 \
    model.readout.fc_dropout=0.1 \
    trainer.devices=\[3\] \
    &

python -m ogbench \
    model=gatv2 \
    dataset=parkinsons \
    seed=123 \
    logger.wandb.tags=[gatv2,parkinsons,hpsearch,rerun] \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    experiment=omics_readout \
    dataset.loader.parameters.adjacency_method=wgcna \
    dataset.loader.parameters.node_sample_ratio=0.8 \
    dataset.loader.parameters.method=distance_correlation \
    optimizer.parameters.lr=0.001 \
    optimizer.parameters.weight_decay=0.0 \
    model.backbone.dropout=0.1 \
    model.feature_encoder.out_channels=16 \
    model.backbone.num_layers=4 \
    model.backbone.heads=4 \
    dataset.loader.parameters.adjacency_threshold=0.0054 \
    model.readout.fc_dropout=0.1 \
    trainer.devices=\[4\] \
    &

python -m ogbench \
    model=gatv2 \
    dataset=parkinsons \
    seed=123 \
    logger.wandb.tags=[gatv2,parkinsons,hpsearch,rerun] \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    experiment=omics_readout \
    dataset.loader.parameters.adjacency_method=string \
    dataset.loader.parameters.node_sample_ratio=1.0 \
    dataset.loader.parameters.method=random \
    optimizer.parameters.lr=0.001 \
    optimizer.parameters.weight_decay=0.0 \
    model.backbone.dropout=0.1 \
    model.feature_encoder.out_channels=16 \
    model.backbone.num_layers=4 \
    model.backbone.heads=2 \
    dataset.loader.parameters.adjacency_threshold=0.4 \
    model.readout.fc_dropout=0.1 \
    trainer.devices=\[5\] \
    &

python -m ogbench \
    model=gatv2 \
    dataset=parkinsons \
    seed=456 \
    logger.wandb.tags=[gatv2,parkinsons,hpsearch,rerun] \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    experiment=omics_readout \
    dataset.loader.parameters.adjacency_method=string \
    dataset.loader.parameters.node_sample_ratio=0.8 \
    dataset.loader.parameters.method=variance \
    optimizer.parameters.lr=0.0001 \
    optimizer.parameters.weight_decay=0.0 \
    model.backbone.dropout=0.1 \
    model.feature_encoder.out_channels=16 \
    model.backbone.num_layers=4 \
    model.backbone.heads=2 \
    dataset.loader.parameters.adjacency_threshold=0.4 \
    model.readout.fc_dropout=0.1 \
    trainer.devices=\[6\] \
    &

python -m ogbench \
    model=gatv2 \
    dataset=parkinsons \
    seed=123 \
    logger.wandb.tags=[gatv2,parkinsons,hpsearch,rerun] \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    experiment=omics_readout \
    dataset.loader.parameters.adjacency_method=string \
    dataset.loader.parameters.node_sample_ratio=0.8 \
    dataset.loader.parameters.method=variance \
    optimizer.parameters.lr=0.0001 \
    optimizer.parameters.weight_decay=0.0 \
    model.backbone.dropout=0.1 \
    model.feature_encoder.out_channels=64 \
    model.backbone.num_layers=2 \
    model.backbone.heads=2 \
    dataset.loader.parameters.adjacency_threshold=0.4 \
    model.readout.fc_dropout=0.1 \
    trainer.devices=\[7\] \
    &

python -m ogbench \
    model=gatv2 \
    dataset=parkinsons \
    seed=123 \
    logger.wandb.tags=[gatv2,parkinsons,hpsearch,rerun] \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    experiment=omics_readout \
    dataset.loader.parameters.adjacency_method=string \
    dataset.loader.parameters.node_sample_ratio=0.8 \
    dataset.loader.parameters.method=variance \
    optimizer.parameters.lr=0.0001 \
    optimizer.parameters.weight_decay=0.0 \
    model.backbone.dropout=0.1 \
    model.feature_encoder.out_channels=64 \
    model.backbone.num_layers=4 \
    model.backbone.heads=4 \
    dataset.loader.parameters.adjacency_threshold=0.4 \
    model.readout.fc_dropout=0.1 \
    trainer.devices=\[0\] \
    &

python -m ogbench \
    model=gatv2 \
    dataset=parkinsons \
    seed=456 \
    logger.wandb.tags=[gatv2,parkinsons,hpsearch,rerun] \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    experiment=omics_readout \
    dataset.loader.parameters.adjacency_method=string \
    dataset.loader.parameters.node_sample_ratio=0.8 \
    dataset.loader.parameters.method=correlation \
    optimizer.parameters.lr=0.0001 \
    optimizer.parameters.weight_decay=0.0 \
    model.backbone.dropout=0.1 \
    model.feature_encoder.out_channels=64 \
    model.backbone.num_layers=2 \
    model.backbone.heads=2 \
    dataset.loader.parameters.adjacency_threshold=0.4 \
    model.readout.fc_dropout=0.1 \
    trainer.devices=\[1\] \
    &

python -m ogbench \
    model=gatv2 \
    dataset=addneuromed \
    seed=42 \
    logger.wandb.tags=[gatv2,addneuromed,hpsearch,rerun] \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    experiment=no_readout \
    dataset.loader.parameters.adjacency_method=string \
    dataset.loader.parameters.node_sample_ratio=1.0 \
    dataset.loader.parameters.method=correlation \
    optimizer.parameters.lr=0.0001 \
    optimizer.parameters.weight_decay=0.0 \
    model.backbone.dropout=0.1 \
    model.feature_encoder.out_channels=64 \
    model.backbone.num_layers=2 \
    model.backbone.heads=4 \
    dataset.loader.parameters.adjacency_threshold=0.4 \
    trainer.devices=\[2\] \
    &

python -m ogbench \
    model=gatv2 \
    dataset=addneuromed \
    seed=42 \
    logger.wandb.tags=[gatv2,addneuromed,hpsearch,rerun] \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    experiment=no_readout \
    dataset.loader.parameters.adjacency_method=string \
    dataset.loader.parameters.node_sample_ratio=1.0 \
    dataset.loader.parameters.method=correlation \
    optimizer.parameters.lr=0.0001 \
    optimizer.parameters.weight_decay=0.0 \
    model.backbone.dropout=0.1 \
    model.feature_encoder.out_channels=64 \
    model.backbone.num_layers=4 \
    model.backbone.heads=4 \
    dataset.loader.parameters.adjacency_threshold=0.4 \
    trainer.devices=\[3\] \
    &

python -m ogbench \
    model=gatv2 \
    dataset=addneuromed \
    seed=42 \
    logger.wandb.tags=[gatv2,addneuromed,hpsearch,rerun] \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    experiment=no_readout \
    dataset.loader.parameters.adjacency_method=string \
    dataset.loader.parameters.node_sample_ratio=1.0 \
    dataset.loader.parameters.method=correlation \
    optimizer.parameters.lr=0.001 \
    optimizer.parameters.weight_decay=0.0 \
    model.backbone.dropout=0.1 \
    model.feature_encoder.out_channels=64 \
    model.backbone.num_layers=2 \
    model.backbone.heads=4 \
    dataset.loader.parameters.adjacency_threshold=0.4 \
    trainer.devices=\[4\] \
    &

python -m ogbench \
    model=gatv2 \
    dataset=addneuromed \
    seed=42 \
    logger.wandb.tags=[gatv2,addneuromed,hpsearch,rerun] \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    experiment=no_readout \
    dataset.loader.parameters.adjacency_method=string \
    dataset.loader.parameters.node_sample_ratio=0.8 \
    dataset.loader.parameters.method=random \
    optimizer.parameters.lr=0.0001 \
    optimizer.parameters.weight_decay=0.0 \
    model.backbone.dropout=0.1 \
    model.feature_encoder.out_channels=64 \
    model.backbone.num_layers=2 \
    model.backbone.heads=4 \
    dataset.loader.parameters.adjacency_threshold=0.4 \
    trainer.devices=\[5\] \
    &

python -m ogbench \
    model=gatv2 \
    dataset=addneuromed \
    seed=123 \
    logger.wandb.tags=[gatv2,addneuromed,hpsearch,rerun] \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    experiment=no_readout \
    dataset.loader.parameters.adjacency_method=string \
    dataset.loader.parameters.node_sample_ratio=0.8 \
    dataset.loader.parameters.method=correlation \
    optimizer.parameters.lr=0.001 \
    optimizer.parameters.weight_decay=0.0 \
    model.backbone.dropout=0.1 \
    model.feature_encoder.out_channels=16 \
    model.backbone.num_layers=4 \
    model.backbone.heads=2 \
    dataset.loader.parameters.adjacency_threshold=0.4 \
    trainer.devices=\[6\] \
    &

python -m ogbench \
    model=gatv2 \
    dataset=addneuromed \
    seed=123 \
    logger.wandb.tags=[gatv2,addneuromed,hpsearch,rerun] \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    experiment=no_readout \
    dataset.loader.parameters.adjacency_method=string \
    dataset.loader.parameters.node_sample_ratio=0.8 \
    dataset.loader.parameters.method=correlation \
    optimizer.parameters.lr=0.001 \
    optimizer.parameters.weight_decay=0.0 \
    model.backbone.dropout=0.1 \
    model.feature_encoder.out_channels=16 \
    model.backbone.num_layers=4 \
    model.backbone.heads=4 \
    dataset.loader.parameters.adjacency_threshold=0.4 \
    trainer.devices=\[7\] \
    &

python -m ogbench \
    model=gatv2 \
    dataset=addneuromed \
    seed=456 \
    logger.wandb.tags=[gatv2,addneuromed,hpsearch,rerun] \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    experiment=omics_readout \
    dataset.loader.parameters.adjacency_method=wgcna \
    dataset.loader.parameters.node_sample_ratio=0.8 \
    dataset.loader.parameters.method=distance_correlation \
    optimizer.parameters.lr=0.0001 \
    optimizer.parameters.weight_decay=0.0 \
    model.backbone.dropout=0.1 \
    model.feature_encoder.out_channels=16 \
    model.backbone.num_layers=2 \
    model.backbone.heads=4 \
    dataset.loader.parameters.adjacency_threshold=0.1765 \
    model.readout.fc_dropout=0.1 \
    trainer.devices=\[0\] \
    &

python -m ogbench \
    model=gatv2 \
    dataset=addneuromed \
    seed=42 \
    logger.wandb.tags=[gatv2,addneuromed,hpsearch,rerun] \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    experiment=omics_readout \
    dataset.loader.parameters.adjacency_method=string \
    dataset.loader.parameters.node_sample_ratio=1.0 \
    dataset.loader.parameters.method=random \
    optimizer.parameters.lr=0.0001 \
    optimizer.parameters.weight_decay=0.0 \
    model.backbone.dropout=0.1 \
    model.feature_encoder.out_channels=64 \
    model.backbone.num_layers=4 \
    model.backbone.heads=2 \
    dataset.loader.parameters.adjacency_threshold=0.4 \
    model.readout.fc_dropout=0.1 \
    trainer.devices=\[1\] \
    &

python -m ogbench \
    model=gatv2 \
    dataset=addneuromed \
    seed=42 \
    logger.wandb.tags=[gatv2,addneuromed,hpsearch,rerun] \
    logger.wandb.project=bgbench_dataset_grid_search_final \
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
    model.backbone.heads=2 \
    dataset.loader.parameters.adjacency_threshold=0.4 \
    model.readout.fc_dropout=0.1 \
    trainer.devices=\[2\] \
    &

python -m ogbench \
    model=gatv2 \
    dataset=addneuromed \
    seed=456 \
    logger.wandb.tags=[gatv2,addneuromed,hpsearch,rerun] \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    experiment=omics_readout \
    dataset.loader.parameters.adjacency_method=string \
    dataset.loader.parameters.node_sample_ratio=1.0 \
    dataset.loader.parameters.method=correlation \
    optimizer.parameters.lr=0.001 \
    optimizer.parameters.weight_decay=0.0 \
    model.backbone.dropout=0.1 \
    model.feature_encoder.out_channels=64 \
    model.backbone.num_layers=4 \
    model.backbone.heads=4 \
    dataset.loader.parameters.adjacency_threshold=0.4 \
    model.readout.fc_dropout=0.1 \
    trainer.devices=\[3\] \
    &

python -m ogbench \
    model=gatv2 \
    dataset=addneuromed \
    seed=42 \
    logger.wandb.tags=[gatv2,addneuromed,hpsearch,rerun] \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    experiment=omics_readout \
    dataset.loader.parameters.adjacency_method=string \
    dataset.loader.parameters.node_sample_ratio=0.8 \
    dataset.loader.parameters.method=variance \
    optimizer.parameters.lr=0.001 \
    optimizer.parameters.weight_decay=0.0 \
    model.backbone.dropout=0.1 \
    model.feature_encoder.out_channels=64 \
    model.backbone.num_layers=4 \
    model.backbone.heads=4 \
    dataset.loader.parameters.adjacency_threshold=0.4 \
    model.readout.fc_dropout=0.1 \
    trainer.devices=\[4\] \
    &

python -m ogbench \
    model=gatv2 \
    dataset=addneuromed \
    seed=42 \
    logger.wandb.tags=[gatv2,addneuromed,hpsearch,rerun] \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    experiment=omics_readout \
    dataset.loader.parameters.adjacency_method=string \
    dataset.loader.parameters.node_sample_ratio=0.8 \
    dataset.loader.parameters.method=correlation \
    optimizer.parameters.lr=0.0001 \
    optimizer.parameters.weight_decay=0.0 \
    model.backbone.dropout=0.1 \
    model.feature_encoder.out_channels=16 \
    model.backbone.num_layers=4 \
    model.backbone.heads=2 \
    dataset.loader.parameters.adjacency_threshold=0.4 \
    model.readout.fc_dropout=0.1 \
    trainer.devices=\[5\] \
    &

python -m ogbench \
    model=gatv2 \
    dataset=addneuromed \
    seed=42 \
    logger.wandb.tags=[gatv2,addneuromed,hpsearch,rerun] \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    experiment=omics_readout \
    dataset.loader.parameters.adjacency_method=string \
    dataset.loader.parameters.node_sample_ratio=0.8 \
    dataset.loader.parameters.method=correlation \
    optimizer.parameters.lr=0.001 \
    optimizer.parameters.weight_decay=0.0 \
    model.backbone.dropout=0.1 \
    model.feature_encoder.out_channels=64 \
    model.backbone.num_layers=2 \
    model.backbone.heads=2 \
    dataset.loader.parameters.adjacency_threshold=0.4 \
    model.readout.fc_dropout=0.1 \
    trainer.devices=\[6\] \
    &

python -m ogbench \
    model=gatv2 \
    dataset=motrpac \
    seed=123 \
    logger.wandb.tags=[gatv2,motrpac,hpsearch,rerun] \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    experiment=no_readout \
    dataset.loader.parameters.adjacency_method=wgcna \
    dataset.loader.parameters.node_sample_ratio=1.0 \
    dataset.loader.parameters.method=distance_correlation \
    optimizer.parameters.lr=0.001 \
    optimizer.parameters.weight_decay=0.0 \
    model.backbone.dropout=0.1 \
    model.feature_encoder.out_channels=16 \
    model.backbone.num_layers=2 \
    model.backbone.heads=4 \
    dataset.loader.parameters.adjacency_threshold=0.1061 \
    trainer.devices=\[7\] \
    &

python -m ogbench \
    model=gatv2 \
    dataset=motrpac \
    seed=42 \
    logger.wandb.tags=[gatv2,motrpac,hpsearch,rerun] \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    experiment=no_readout \
    dataset.loader.parameters.adjacency_method=wgcna \
    dataset.loader.parameters.node_sample_ratio=0.8 \
    dataset.loader.parameters.method=variance \
    optimizer.parameters.lr=0.0001 \
    optimizer.parameters.weight_decay=0.0 \
    model.backbone.dropout=0.1 \
    model.feature_encoder.out_channels=16 \
    model.backbone.num_layers=4 \
    model.backbone.heads=4 \
    dataset.loader.parameters.adjacency_threshold=0.0988 \
    trainer.devices=\[0\] \
    &

python -m ogbench \
    model=gatv2 \
    dataset=motrpac \
    seed=456 \
    logger.wandb.tags=[gatv2,motrpac,hpsearch,rerun] \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    experiment=no_readout \
    dataset.loader.parameters.adjacency_method=wgcna \
    dataset.loader.parameters.node_sample_ratio=0.8 \
    dataset.loader.parameters.method=variance \
    optimizer.parameters.lr=0.001 \
    optimizer.parameters.weight_decay=0.0 \
    model.backbone.dropout=0.1 \
    model.feature_encoder.out_channels=64 \
    model.backbone.num_layers=4 \
    model.backbone.heads=4 \
    dataset.loader.parameters.adjacency_threshold=0.0988 \
    trainer.devices=\[1\] \
    &

python -m ogbench \
    model=gatv2 \
    dataset=motrpac \
    seed=456 \
    logger.wandb.tags=[gatv2,motrpac,hpsearch,rerun] \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    experiment=no_readout \
    dataset.loader.parameters.adjacency_method=string \
    dataset.loader.parameters.node_sample_ratio=1.0 \
    dataset.loader.parameters.method=random \
    optimizer.parameters.lr=0.001 \
    optimizer.parameters.weight_decay=0.0 \
    model.backbone.dropout=0.1 \
    model.feature_encoder.out_channels=16 \
    model.backbone.num_layers=2 \
    model.backbone.heads=4 \
    dataset.loader.parameters.adjacency_threshold=0.4 \
    trainer.devices=\[2\] \
    &

python -m ogbench \
    model=gatv2 \
    dataset=motrpac \
    seed=42 \
    logger.wandb.tags=[gatv2,motrpac,hpsearch,rerun] \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    experiment=no_readout \
    dataset.loader.parameters.adjacency_method=string \
    dataset.loader.parameters.node_sample_ratio=1.0 \
    dataset.loader.parameters.method=random \
    optimizer.parameters.lr=0.001 \
    optimizer.parameters.weight_decay=0.0 \
    model.backbone.dropout=0.1 \
    model.feature_encoder.out_channels=64 \
    model.backbone.num_layers=4 \
    model.backbone.heads=4 \
    dataset.loader.parameters.adjacency_threshold=0.4 \
    trainer.devices=\[3\] \
    &

python -m ogbench \
    model=gatv2 \
    dataset=motrpac \
    seed=123 \
    logger.wandb.tags=[gatv2,motrpac,hpsearch,rerun] \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    experiment=no_readout \
    dataset.loader.parameters.adjacency_method=string \
    dataset.loader.parameters.node_sample_ratio=1.0 \
    dataset.loader.parameters.method=correlation \
    optimizer.parameters.lr=0.001 \
    optimizer.parameters.weight_decay=0.0 \
    model.backbone.dropout=0.1 \
    model.feature_encoder.out_channels=64 \
    model.backbone.num_layers=4 \
    model.backbone.heads=2 \
    dataset.loader.parameters.adjacency_threshold=0.4 \
    trainer.devices=\[4\] \
    &

python -m ogbench \
    model=gatv2 \
    dataset=motrpac \
    seed=42 \
    logger.wandb.tags=[gatv2,motrpac,hpsearch,rerun] \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    experiment=no_readout \
    dataset.loader.parameters.adjacency_method=string \
    dataset.loader.parameters.node_sample_ratio=0.8 \
    dataset.loader.parameters.method=random \
    optimizer.parameters.lr=0.0001 \
    optimizer.parameters.weight_decay=0.0 \
    model.backbone.dropout=0.1 \
    model.feature_encoder.out_channels=16 \
    model.backbone.num_layers=4 \
    model.backbone.heads=4 \
    dataset.loader.parameters.adjacency_threshold=0.4 \
    trainer.devices=\[5\] \
    &

python -m ogbench \
    model=gatv2 \
    dataset=motrpac \
    seed=456 \
    logger.wandb.tags=[gatv2,motrpac,hpsearch,rerun] \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    experiment=no_readout \
    dataset.loader.parameters.adjacency_method=string \
    dataset.loader.parameters.node_sample_ratio=0.8 \
    dataset.loader.parameters.method=random \
    optimizer.parameters.lr=0.001 \
    optimizer.parameters.weight_decay=0.0 \
    model.backbone.dropout=0.1 \
    model.feature_encoder.out_channels=64 \
    model.backbone.num_layers=2 \
    model.backbone.heads=2 \
    dataset.loader.parameters.adjacency_threshold=0.4 \
    trainer.devices=\[6\] \
    &

python -m ogbench \
    model=gatv2 \
    dataset=motrpac \
    seed=42 \
    logger.wandb.tags=[gatv2,motrpac,hpsearch,rerun] \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    experiment=omics_readout \
    dataset.loader.parameters.adjacency_method=wgcna \
    dataset.loader.parameters.node_sample_ratio=1.0 \
    dataset.loader.parameters.method=variance \
    optimizer.parameters.lr=0.0001 \
    optimizer.parameters.weight_decay=0.0 \
    model.backbone.dropout=0.1 \
    model.feature_encoder.out_channels=16 \
    model.backbone.num_layers=2 \
    model.backbone.heads=2 \
    dataset.loader.parameters.adjacency_threshold=0.1179 \
    model.readout.fc_dropout=0.1 \
    trainer.devices=\[7\] \
    &

python -m ogbench \
    model=gatv2 \
    dataset=motrpac \
    seed=42 \
    logger.wandb.tags=[gatv2,motrpac,hpsearch,rerun] \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    experiment=omics_readout \
    dataset.loader.parameters.adjacency_method=wgcna \
    dataset.loader.parameters.node_sample_ratio=1.0 \
    dataset.loader.parameters.method=variance \
    optimizer.parameters.lr=0.001 \
    optimizer.parameters.weight_decay=0.0 \
    model.backbone.dropout=0.1 \
    model.feature_encoder.out_channels=64 \
    model.backbone.num_layers=2 \
    model.backbone.heads=4 \
    dataset.loader.parameters.adjacency_threshold=0.1179 \
    model.readout.fc_dropout=0.1 \
    trainer.devices=\[0\] \
    &

python -m ogbench \
    model=gatv2 \
    dataset=motrpac \
    seed=123 \
    logger.wandb.tags=[gatv2,motrpac,hpsearch,rerun] \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    experiment=omics_readout \
    dataset.loader.parameters.adjacency_method=wgcna \
    dataset.loader.parameters.node_sample_ratio=1.0 \
    dataset.loader.parameters.method=correlation \
    optimizer.parameters.lr=0.0001 \
    optimizer.parameters.weight_decay=0.0 \
    model.backbone.dropout=0.1 \
    model.feature_encoder.out_channels=64 \
    model.backbone.num_layers=2 \
    model.backbone.heads=2 \
    dataset.loader.parameters.adjacency_threshold=0.1323 \
    model.readout.fc_dropout=0.1 \
    trainer.devices=\[1\] \
    &

python -m ogbench \
    model=gatv2 \
    dataset=motrpac \
    seed=42 \
    logger.wandb.tags=[gatv2,motrpac,hpsearch,rerun] \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    experiment=omics_readout \
    dataset.loader.parameters.adjacency_method=wgcna \
    dataset.loader.parameters.node_sample_ratio=1.0 \
    dataset.loader.parameters.method=distance_correlation \
    optimizer.parameters.lr=0.0001 \
    optimizer.parameters.weight_decay=0.0 \
    model.backbone.dropout=0.1 \
    model.feature_encoder.out_channels=16 \
    model.backbone.num_layers=4 \
    model.backbone.heads=2 \
    dataset.loader.parameters.adjacency_threshold=0.1061 \
    model.readout.fc_dropout=0.1 \
    trainer.devices=\[2\] \
    &

python -m ogbench \
    model=gatv2 \
    dataset=motrpac \
    seed=456 \
    logger.wandb.tags=[gatv2,motrpac,hpsearch,rerun] \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    experiment=omics_readout \
    dataset.loader.parameters.adjacency_method=string \
    dataset.loader.parameters.node_sample_ratio=1.0 \
    dataset.loader.parameters.method=variance \
    optimizer.parameters.lr=0.0001 \
    optimizer.parameters.weight_decay=0.0 \
    model.backbone.dropout=0.1 \
    model.feature_encoder.out_channels=16 \
    model.backbone.num_layers=2 \
    model.backbone.heads=4 \
    dataset.loader.parameters.adjacency_threshold=0.4 \
    model.readout.fc_dropout=0.1 \
    trainer.devices=\[3\] \
    &

python -m ogbench \
    model=gatv2 \
    dataset=motrpac \
    seed=123 \
    logger.wandb.tags=[gatv2,motrpac,hpsearch,rerun] \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    experiment=omics_readout \
    dataset.loader.parameters.adjacency_method=string \
    dataset.loader.parameters.node_sample_ratio=1.0 \
    dataset.loader.parameters.method=variance \
    optimizer.parameters.lr=0.001 \
    optimizer.parameters.weight_decay=0.0 \
    model.backbone.dropout=0.1 \
    model.feature_encoder.out_channels=16 \
    model.backbone.num_layers=4 \
    model.backbone.heads=4 \
    dataset.loader.parameters.adjacency_threshold=0.4 \
    model.readout.fc_dropout=0.1 \
    trainer.devices=\[4\] \
    &

python -m ogbench \
    model=gatv2 \
    dataset=motrpac \
    seed=456 \
    logger.wandb.tags=[gatv2,motrpac,hpsearch,rerun] \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    experiment=omics_readout \
    dataset.loader.parameters.adjacency_method=string \
    dataset.loader.parameters.node_sample_ratio=1.0 \
    dataset.loader.parameters.method=correlation \
    optimizer.parameters.lr=0.0001 \
    optimizer.parameters.weight_decay=0.0 \
    model.backbone.dropout=0.1 \
    model.feature_encoder.out_channels=16 \
    model.backbone.num_layers=2 \
    model.backbone.heads=2 \
    dataset.loader.parameters.adjacency_threshold=0.4 \
    model.readout.fc_dropout=0.1 \
    trainer.devices=\[5\] \
    &

python -m ogbench \
    model=gatv2 \
    dataset=motrpac \
    seed=456 \
    logger.wandb.tags=[gatv2,motrpac,hpsearch,rerun] \
    logger.wandb.project=bgbench_dataset_grid_search_final \
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
    model.backbone.heads=4 \
    dataset.loader.parameters.adjacency_threshold=0.4 \
    model.readout.fc_dropout=0.1 \
    trainer.devices=\[6\] \
    &

python -m ogbench \
    model=gatv2 \
    dataset=motrpac \
    seed=123 \
    logger.wandb.tags=[gatv2,motrpac,hpsearch,rerun] \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    experiment=omics_readout \
    dataset.loader.parameters.adjacency_method=string \
    dataset.loader.parameters.node_sample_ratio=1.0 \
    dataset.loader.parameters.method=correlation \
    optimizer.parameters.lr=0.001 \
    optimizer.parameters.weight_decay=0.0 \
    model.backbone.dropout=0.1 \
    model.feature_encoder.out_channels=64 \
    model.backbone.num_layers=4 \
    model.backbone.heads=4 \
    dataset.loader.parameters.adjacency_threshold=0.4 \
    model.readout.fc_dropout=0.1 \
    trainer.devices=\[7\] \
    &

python -m ogbench \
    model=gatv2 \
    dataset=motrpac \
    seed=123 \
    logger.wandb.tags=[gatv2,motrpac,hpsearch,rerun] \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    experiment=omics_readout \
    dataset.loader.parameters.adjacency_method=string \
    dataset.loader.parameters.node_sample_ratio=0.8 \
    dataset.loader.parameters.method=random \
    optimizer.parameters.lr=0.0001 \
    optimizer.parameters.weight_decay=0.0 \
    model.backbone.dropout=0.1 \
    model.feature_encoder.out_channels=16 \
    model.backbone.num_layers=2 \
    model.backbone.heads=4 \
    dataset.loader.parameters.adjacency_threshold=0.4 \
    model.readout.fc_dropout=0.1 \
    trainer.devices=\[0\] \
    &

python -m ogbench \
    model=gatv2 \
    dataset=motrpac \
    seed=123 \
    logger.wandb.tags=[gatv2,motrpac,hpsearch,rerun] \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    experiment=omics_readout \
    dataset.loader.parameters.adjacency_method=string \
    dataset.loader.parameters.node_sample_ratio=0.8 \
    dataset.loader.parameters.method=random \
    optimizer.parameters.lr=0.001 \
    optimizer.parameters.weight_decay=0.0 \
    model.backbone.dropout=0.1 \
    model.feature_encoder.out_channels=16 \
    model.backbone.num_layers=4 \
    model.backbone.heads=2 \
    dataset.loader.parameters.adjacency_threshold=0.4 \
    model.readout.fc_dropout=0.1 \
    trainer.devices=\[1\] \
    &

python -m ogbench \
    model=gatv2 \
    dataset=motrpac \
    seed=42 \
    logger.wandb.tags=[gatv2,motrpac,hpsearch,rerun] \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    experiment=omics_readout \
    dataset.loader.parameters.adjacency_method=string \
    dataset.loader.parameters.node_sample_ratio=0.8 \
    dataset.loader.parameters.method=distance_correlation \
    optimizer.parameters.lr=0.001 \
    optimizer.parameters.weight_decay=0.0 \
    model.backbone.dropout=0.1 \
    model.feature_encoder.out_channels=16 \
    model.backbone.num_layers=4 \
    model.backbone.heads=4 \
    dataset.loader.parameters.adjacency_threshold=0.4 \
    model.readout.fc_dropout=0.1 \
    trainer.devices=\[2\] \
    &

python -m ogbench \
    model=gatv2 \
    dataset=motrpac \
    seed=42 \
    logger.wandb.tags=[gatv2,motrpac,hpsearch,rerun] \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    experiment=omics_readout \
    dataset.loader.parameters.adjacency_method=string \
    dataset.loader.parameters.node_sample_ratio=0.8 \
    dataset.loader.parameters.method=distance_correlation \
    optimizer.parameters.lr=0.001 \
    optimizer.parameters.weight_decay=0.0 \
    model.backbone.dropout=0.1 \
    model.feature_encoder.out_channels=64 \
    model.backbone.num_layers=2 \
    model.backbone.heads=2 \
    dataset.loader.parameters.adjacency_threshold=0.4 \
    model.readout.fc_dropout=0.1 \
    trainer.devices=\[3\] \
    &

python -m ogbench \
    model=gatv2 \
    dataset=brca \
    seed=456 \
    logger.wandb.tags=[gatv2,brca,hpsearch,rerun] \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    experiment=no_readout \
    dataset.loader.parameters.adjacency_method=wgcna \
    dataset.loader.parameters.node_sample_ratio=1.0 \
    dataset.loader.parameters.method=random \
    optimizer.parameters.lr=0.001 \
    optimizer.parameters.weight_decay=0.0 \
    model.backbone.dropout=0.1 \
    model.feature_encoder.out_channels=16 \
    model.backbone.num_layers=2 \
    model.backbone.heads=2 \
    dataset.loader.parameters.adjacency_threshold=0.0012 \
    trainer.devices=\[4\] \
    &

python -m ogbench \
    model=gatv2 \
    dataset=brca \
    seed=456 \
    logger.wandb.tags=[gatv2,brca,hpsearch,rerun] \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    experiment=no_readout \
    dataset.loader.parameters.adjacency_method=wgcna \
    dataset.loader.parameters.node_sample_ratio=1.0 \
    dataset.loader.parameters.method=correlation \
    optimizer.parameters.lr=0.0001 \
    optimizer.parameters.weight_decay=0.0 \
    model.backbone.dropout=0.1 \
    model.feature_encoder.out_channels=16 \
    model.backbone.num_layers=2 \
    model.backbone.heads=4 \
    dataset.loader.parameters.adjacency_threshold=0.0008 \
    trainer.devices=\[5\] \
    &

python -m ogbench \
    model=gatv2 \
    dataset=brca \
    seed=42 \
    logger.wandb.tags=[gatv2,brca,hpsearch,rerun] \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    experiment=no_readout \
    dataset.loader.parameters.adjacency_method=wgcna \
    dataset.loader.parameters.node_sample_ratio=1.0 \
    dataset.loader.parameters.method=correlation \
    optimizer.parameters.lr=0.001 \
    optimizer.parameters.weight_decay=0.0 \
    model.backbone.dropout=0.1 \
    model.feature_encoder.out_channels=16 \
    model.backbone.num_layers=4 \
    model.backbone.heads=2 \
    dataset.loader.parameters.adjacency_threshold=0.0008 \
    trainer.devices=\[6\] \
    &

python -m ogbench \
    model=gatv2 \
    dataset=brca \
    seed=456 \
    logger.wandb.tags=[gatv2,brca,hpsearch,rerun] \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    experiment=no_readout \
    dataset.loader.parameters.adjacency_method=wgcna \
    dataset.loader.parameters.node_sample_ratio=1.0 \
    dataset.loader.parameters.method=distance_correlation \
    optimizer.parameters.lr=0.001 \
    optimizer.parameters.weight_decay=0.0 \
    model.backbone.dropout=0.1 \
    model.feature_encoder.out_channels=64 \
    model.backbone.num_layers=2 \
    model.backbone.heads=2 \
    dataset.loader.parameters.adjacency_threshold=0.0008 \
    trainer.devices=\[7\] \
    &

python -m ogbench \
    model=gatv2 \
    dataset=brca \
    seed=456 \
    logger.wandb.tags=[gatv2,brca,hpsearch,rerun] \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    experiment=no_readout \
    dataset.loader.parameters.adjacency_method=wgcna \
    dataset.loader.parameters.node_sample_ratio=0.8 \
    dataset.loader.parameters.method=correlation \
    optimizer.parameters.lr=0.0001 \
    optimizer.parameters.weight_decay=0.0 \
    model.backbone.dropout=0.1 \
    model.feature_encoder.out_channels=64 \
    model.backbone.num_layers=2 \
    model.backbone.heads=2 \
    dataset.loader.parameters.adjacency_threshold=0.0005 \
    trainer.devices=\[0\] \
    &

python -m ogbench \
    model=gatv2 \
    dataset=brca \
    seed=42 \
    logger.wandb.tags=[gatv2,brca,hpsearch,rerun] \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    experiment=no_readout \
    dataset.loader.parameters.adjacency_method=string \
    dataset.loader.parameters.node_sample_ratio=1.0 \
    dataset.loader.parameters.method=random \
    optimizer.parameters.lr=0.0001 \
    optimizer.parameters.weight_decay=0.0 \
    model.backbone.dropout=0.1 \
    model.feature_encoder.out_channels=16 \
    model.backbone.num_layers=4 \
    model.backbone.heads=4 \
    dataset.loader.parameters.adjacency_threshold=0.4 \
    trainer.devices=\[1\] \
    &

python -m ogbench \
    model=gatv2 \
    dataset=brca \
    seed=456 \
    logger.wandb.tags=[gatv2,brca,hpsearch,rerun] \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    experiment=no_readout \
    dataset.loader.parameters.adjacency_method=string \
    dataset.loader.parameters.node_sample_ratio=1.0 \
    dataset.loader.parameters.method=correlation \
    optimizer.parameters.lr=0.001 \
    optimizer.parameters.weight_decay=0.0 \
    model.backbone.dropout=0.1 \
    model.feature_encoder.out_channels=16 \
    model.backbone.num_layers=4 \
    model.backbone.heads=4 \
    dataset.loader.parameters.adjacency_threshold=0.4 \
    trainer.devices=\[2\] \
    &

python -m ogbench \
    model=gatv2 \
    dataset=brca \
    seed=123 \
    logger.wandb.tags=[gatv2,brca,hpsearch,rerun] \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    experiment=no_readout \
    dataset.loader.parameters.adjacency_method=string \
    dataset.loader.parameters.node_sample_ratio=1.0 \
    dataset.loader.parameters.method=distance_correlation \
    optimizer.parameters.lr=0.0001 \
    optimizer.parameters.weight_decay=0.0 \
    model.backbone.dropout=0.1 \
    model.feature_encoder.out_channels=16 \
    model.backbone.num_layers=2 \
    model.backbone.heads=4 \
    dataset.loader.parameters.adjacency_threshold=0.4 \
    trainer.devices=\[3\] \
    &

python -m ogbench \
    model=gatv2 \
    dataset=brca \
    seed=42 \
    logger.wandb.tags=[gatv2,brca,hpsearch,rerun] \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    experiment=no_readout \
    dataset.loader.parameters.adjacency_method=string \
    dataset.loader.parameters.node_sample_ratio=1.0 \
    dataset.loader.parameters.method=distance_correlation \
    optimizer.parameters.lr=0.0001 \
    optimizer.parameters.weight_decay=0.0 \
    model.backbone.dropout=0.1 \
    model.feature_encoder.out_channels=64 \
    model.backbone.num_layers=4 \
    model.backbone.heads=2 \
    dataset.loader.parameters.adjacency_threshold=0.4 \
    trainer.devices=\[4\] \
    &

python -m ogbench \
    model=gatv2 \
    dataset=brca \
    seed=456 \
    logger.wandb.tags=[gatv2,brca,hpsearch,rerun] \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    experiment=no_readout \
    dataset.loader.parameters.adjacency_method=string \
    dataset.loader.parameters.node_sample_ratio=0.8 \
    dataset.loader.parameters.method=random \
    optimizer.parameters.lr=0.001 \
    optimizer.parameters.weight_decay=0.0 \
    model.backbone.dropout=0.1 \
    model.feature_encoder.out_channels=16 \
    model.backbone.num_layers=2 \
    model.backbone.heads=4 \
    dataset.loader.parameters.adjacency_threshold=0.4 \
    trainer.devices=\[5\] \
    &

python -m ogbench \
    model=gatv2 \
    dataset=brca \
    seed=123 \
    logger.wandb.tags=[gatv2,brca,hpsearch,rerun] \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    experiment=omics_readout \
    dataset.loader.parameters.adjacency_method=wgcna \
    dataset.loader.parameters.node_sample_ratio=1.0 \
    dataset.loader.parameters.method=random \
    optimizer.parameters.lr=0.0001 \
    optimizer.parameters.weight_decay=0.0 \
    model.backbone.dropout=0.1 \
    model.feature_encoder.out_channels=16 \
    model.backbone.num_layers=4 \
    model.backbone.heads=2 \
    dataset.loader.parameters.adjacency_threshold=0.0012 \
    model.readout.fc_dropout=0.1 \
    trainer.devices=\[6\] \
    &

python -m ogbench \
    model=gatv2 \
    dataset=brca \
    seed=42 \
    logger.wandb.tags=[gatv2,brca,hpsearch,rerun] \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    experiment=omics_readout \
    dataset.loader.parameters.adjacency_method=wgcna \
    dataset.loader.parameters.node_sample_ratio=1.0 \
    dataset.loader.parameters.method=random \
    optimizer.parameters.lr=0.001 \
    optimizer.parameters.weight_decay=0.0 \
    model.backbone.dropout=0.1 \
    model.feature_encoder.out_channels=64 \
    model.backbone.num_layers=4 \
    model.backbone.heads=2 \
    dataset.loader.parameters.adjacency_threshold=0.0012 \
    model.readout.fc_dropout=0.1 \
    trainer.devices=\[7\] \
    &

python -m ogbench \
    model=gatv2 \
    dataset=brca \
    seed=42 \
    logger.wandb.tags=[gatv2,brca,hpsearch,rerun] \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    experiment=omics_readout \
    dataset.loader.parameters.adjacency_method=wgcna \
    dataset.loader.parameters.node_sample_ratio=1.0 \
    dataset.loader.parameters.method=distance_correlation \
    optimizer.parameters.lr=0.0001 \
    optimizer.parameters.weight_decay=0.0 \
    model.backbone.dropout=0.1 \
    model.feature_encoder.out_channels=64 \
    model.backbone.num_layers=2 \
    model.backbone.heads=4 \
    dataset.loader.parameters.adjacency_threshold=0.0008 \
    model.readout.fc_dropout=0.1 \
    trainer.devices=\[0\] \
    &

python -m ogbench \
    model=gatv2 \
    dataset=brca \
    seed=123 \
    logger.wandb.tags=[gatv2,brca,hpsearch,rerun] \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    experiment=omics_readout \
    dataset.loader.parameters.adjacency_method=string \
    dataset.loader.parameters.node_sample_ratio=1.0 \
    dataset.loader.parameters.method=random \
    optimizer.parameters.lr=0.001 \
    optimizer.parameters.weight_decay=0.0 \
    model.backbone.dropout=0.1 \
    model.feature_encoder.out_channels=64 \
    model.backbone.num_layers=2 \
    model.backbone.heads=2 \
    dataset.loader.parameters.adjacency_threshold=0.4 \
    model.readout.fc_dropout=0.1 \
    trainer.devices=\[1\] \
    &

python -m ogbench \
    model=gatv2 \
    dataset=brca \
    seed=42 \
    logger.wandb.tags=[gatv2,brca,hpsearch,rerun] \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    experiment=omics_readout \
    dataset.loader.parameters.adjacency_method=string \
    dataset.loader.parameters.node_sample_ratio=1.0 \
    dataset.loader.parameters.method=random \
    optimizer.parameters.lr=0.001 \
    optimizer.parameters.weight_decay=0.0 \
    model.backbone.dropout=0.1 \
    model.feature_encoder.out_channels=64 \
    model.backbone.num_layers=2 \
    model.backbone.heads=2 \
    dataset.loader.parameters.adjacency_threshold=0.4 \
    model.readout.fc_dropout=0.1 \
    trainer.devices=\[2\] \
    &

python -m ogbench \
    model=gatv2 \
    dataset=brca \
    seed=123 \
    logger.wandb.tags=[gatv2,brca,hpsearch,rerun] \
    logger.wandb.project=bgbench_dataset_grid_search_final \
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
    model.backbone.heads=4 \
    dataset.loader.parameters.adjacency_threshold=0.4 \
    model.readout.fc_dropout=0.1 \
    trainer.devices=\[3\] \
    &

python -m ogbench \
    model=gatv2 \
    dataset=brca \
    seed=456 \
    logger.wandb.tags=[gatv2,brca,hpsearch,rerun] \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    experiment=omics_readout \
    dataset.loader.parameters.adjacency_method=string \
    dataset.loader.parameters.node_sample_ratio=1.0 \
    dataset.loader.parameters.method=distance_correlation \
    optimizer.parameters.lr=0.001 \
    optimizer.parameters.weight_decay=0.0 \
    model.backbone.dropout=0.1 \
    model.feature_encoder.out_channels=16 \
    model.backbone.num_layers=4 \
    model.backbone.heads=2 \
    dataset.loader.parameters.adjacency_threshold=0.4 \
    model.readout.fc_dropout=0.1 \
    trainer.devices=\[4\] \
    &

python -m ogbench \
    model=gatv2 \
    dataset=brca \
    seed=42 \
    logger.wandb.tags=[gatv2,brca,hpsearch,rerun] \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    experiment=omics_readout \
    dataset.loader.parameters.adjacency_method=string \
    dataset.loader.parameters.node_sample_ratio=0.8 \
    dataset.loader.parameters.method=correlation \
    optimizer.parameters.lr=0.001 \
    optimizer.parameters.weight_decay=0.0 \
    model.backbone.dropout=0.1 \
    model.feature_encoder.out_channels=16 \
    model.backbone.num_layers=4 \
    model.backbone.heads=4 \
    dataset.loader.parameters.adjacency_threshold=0.4 \
    model.readout.fc_dropout=0.1 \
    trainer.devices=\[5\] \
    &

python -m ogbench \
    model=graph_sage \
    dataset=parkinsons \
    seed=42 \
    logger.wandb.tags=[graph_sage,parkinsons,hpsearch,rerun] \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    experiment=no_readout \
    dataset.loader.parameters.adjacency_method=wgcna \
    dataset.loader.parameters.node_sample_ratio=1.0 \
    dataset.loader.parameters.method=variance \
    optimizer.parameters.lr=0.0001 \
    optimizer.parameters.weight_decay=0.0 \
    model.backbone.dropout=0.1 \
    model.feature_encoder.out_channels=64 \
    model.backbone.num_layers=2 \
    dataset.loader.parameters.adjacency_threshold=0.0346 \
    trainer.devices=\[6\] \
    &

python -m ogbench \
    model=graph_sage \
    dataset=parkinsons \
    seed=123 \
    logger.wandb.tags=[graph_sage,parkinsons,hpsearch,rerun] \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    experiment=no_readout \
    dataset.loader.parameters.adjacency_method=wgcna \
    dataset.loader.parameters.node_sample_ratio=0.8 \
    dataset.loader.parameters.method=random \
    optimizer.parameters.lr=0.001 \
    optimizer.parameters.weight_decay=0.0 \
    model.backbone.dropout=0.1 \
    model.feature_encoder.out_channels=16 \
    model.backbone.num_layers=2 \
    dataset.loader.parameters.adjacency_threshold=0.0216 \
    trainer.devices=\[7\] \
    &

python -m ogbench \
    model=graph_sage \
    dataset=parkinsons \
    seed=123 \
    logger.wandb.tags=[graph_sage,parkinsons,hpsearch,rerun] \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    experiment=no_readout \
    dataset.loader.parameters.adjacency_method=wgcna \
    dataset.loader.parameters.node_sample_ratio=0.8 \
    dataset.loader.parameters.method=correlation \
    optimizer.parameters.lr=0.001 \
    optimizer.parameters.weight_decay=0.0 \
    model.backbone.dropout=0.1 \
    model.feature_encoder.out_channels=64 \
    model.backbone.num_layers=4 \
    dataset.loader.parameters.adjacency_threshold=0.0019 \
    trainer.devices=\[0\] \
    &

python -m ogbench \
    model=graph_sage \
    dataset=parkinsons \
    seed=456 \
    logger.wandb.tags=[graph_sage,parkinsons,hpsearch,rerun] \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    experiment=no_readout \
    dataset.loader.parameters.adjacency_method=wgcna \
    dataset.loader.parameters.node_sample_ratio=0.8 \
    dataset.loader.parameters.method=distance_correlation \
    optimizer.parameters.lr=0.001 \
    optimizer.parameters.weight_decay=0.0 \
    model.backbone.dropout=0.1 \
    model.feature_encoder.out_channels=16 \
    model.backbone.num_layers=4 \
    dataset.loader.parameters.adjacency_threshold=0.0054 \
    trainer.devices=\[1\] \
    &

python -m ogbench \
    model=graph_sage \
    dataset=parkinsons \
    seed=42 \
    logger.wandb.tags=[graph_sage,parkinsons,hpsearch,rerun] \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    experiment=no_readout \
    dataset.loader.parameters.adjacency_method=string \
    dataset.loader.parameters.node_sample_ratio=1.0 \
    dataset.loader.parameters.method=variance \
    optimizer.parameters.lr=0.001 \
    optimizer.parameters.weight_decay=0.0 \
    model.backbone.dropout=0.1 \
    model.feature_encoder.out_channels=16 \
    model.backbone.num_layers=4 \
    dataset.loader.parameters.adjacency_threshold=0.4 \
    trainer.devices=\[2\] \
    &

python -m ogbench \
    model=graph_sage \
    dataset=parkinsons \
    seed=42 \
    logger.wandb.tags=[graph_sage,parkinsons,hpsearch,rerun] \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    experiment=no_readout \
    dataset.loader.parameters.adjacency_method=string \
    dataset.loader.parameters.node_sample_ratio=1.0 \
    dataset.loader.parameters.method=random \
    optimizer.parameters.lr=0.0001 \
    optimizer.parameters.weight_decay=0.0 \
    model.backbone.dropout=0.1 \
    model.feature_encoder.out_channels=64 \
    model.backbone.num_layers=2 \
    dataset.loader.parameters.adjacency_threshold=0.4 \
    trainer.devices=\[3\] \
    &

python -m ogbench \
    model=graph_sage \
    dataset=parkinsons \
    seed=42 \
    logger.wandb.tags=[graph_sage,parkinsons,hpsearch,rerun] \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    experiment=no_readout \
    dataset.loader.parameters.adjacency_method=string \
    dataset.loader.parameters.node_sample_ratio=0.8 \
    dataset.loader.parameters.method=correlation \
    optimizer.parameters.lr=0.001 \
    optimizer.parameters.weight_decay=0.0 \
    model.backbone.dropout=0.1 \
    model.feature_encoder.out_channels=16 \
    model.backbone.num_layers=2 \
    dataset.loader.parameters.adjacency_threshold=0.4 \
    trainer.devices=\[4\] \
    &

python -m ogbench \
    model=graph_sage \
    dataset=parkinsons \
    seed=456 \
    logger.wandb.tags=[graph_sage,parkinsons,hpsearch,rerun] \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    experiment=no_readout \
    dataset.loader.parameters.adjacency_method=string \
    dataset.loader.parameters.node_sample_ratio=0.8 \
    dataset.loader.parameters.method=distance_correlation \
    optimizer.parameters.lr=0.001 \
    optimizer.parameters.weight_decay=0.0 \
    model.backbone.dropout=0.1 \
    model.feature_encoder.out_channels=16 \
    model.backbone.num_layers=2 \
    dataset.loader.parameters.adjacency_threshold=0.4 \
    trainer.devices=\[5\] \
    &

python -m ogbench \
    model=graph_sage \
    dataset=parkinsons \
    seed=456 \
    logger.wandb.tags=[graph_sage,parkinsons,hpsearch,rerun] \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    experiment=omics_readout \
    dataset.loader.parameters.adjacency_method=wgcna \
    dataset.loader.parameters.node_sample_ratio=1.0 \
    dataset.loader.parameters.method=distance_correlation \
    optimizer.parameters.lr=0.0001 \
    optimizer.parameters.weight_decay=0.0 \
    model.backbone.dropout=0.1 \
    model.feature_encoder.out_channels=64 \
    model.backbone.num_layers=4 \
    dataset.loader.parameters.adjacency_threshold=0.0061 \
    model.readout.fc_dropout=0.1 \
    trainer.devices=\[6\] \
    &

python -m ogbench \
    model=graph_sage \
    dataset=parkinsons \
    seed=123 \
    logger.wandb.tags=[graph_sage,parkinsons,hpsearch,rerun] \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    experiment=omics_readout \
    dataset.loader.parameters.adjacency_method=wgcna \
    dataset.loader.parameters.node_sample_ratio=0.8 \
    dataset.loader.parameters.method=correlation \
    optimizer.parameters.lr=0.001 \
    optimizer.parameters.weight_decay=0.0 \
    model.backbone.dropout=0.1 \
    model.feature_encoder.out_channels=16 \
    model.backbone.num_layers=4 \
    dataset.loader.parameters.adjacency_threshold=0.0019 \
    model.readout.fc_dropout=0.1 \
    trainer.devices=\[7\] \
    &

python -m ogbench \
    model=graph_sage \
    dataset=parkinsons \
    seed=42 \
    logger.wandb.tags=[graph_sage,parkinsons,hpsearch,rerun] \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    experiment=omics_readout \
    dataset.loader.parameters.adjacency_method=wgcna \
    dataset.loader.parameters.node_sample_ratio=0.8 \
    dataset.loader.parameters.method=distance_correlation \
    optimizer.parameters.lr=0.0001 \
    optimizer.parameters.weight_decay=0.0 \
    model.backbone.dropout=0.1 \
    model.feature_encoder.out_channels=16 \
    model.backbone.num_layers=4 \
    dataset.loader.parameters.adjacency_threshold=0.0054 \
    model.readout.fc_dropout=0.1 \
    trainer.devices=\[0\] \
    &

python -m ogbench \
    model=graph_sage \
    dataset=parkinsons \
    seed=123 \
    logger.wandb.tags=[graph_sage,parkinsons,hpsearch,rerun] \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    experiment=omics_readout \
    dataset.loader.parameters.adjacency_method=wgcna \
    dataset.loader.parameters.node_sample_ratio=0.8 \
    dataset.loader.parameters.method=distance_correlation \
    optimizer.parameters.lr=0.0001 \
    optimizer.parameters.weight_decay=0.0 \
    model.backbone.dropout=0.1 \
    model.feature_encoder.out_channels=64 \
    model.backbone.num_layers=4 \
    dataset.loader.parameters.adjacency_threshold=0.0054 \
    model.readout.fc_dropout=0.1 \
    trainer.devices=\[1\] \
    &

python -m ogbench \
    model=graph_sage \
    dataset=parkinsons \
    seed=42 \
    logger.wandb.tags=[graph_sage,parkinsons,hpsearch,rerun] \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    experiment=omics_readout \
    dataset.loader.parameters.adjacency_method=string \
    dataset.loader.parameters.node_sample_ratio=0.8 \
    dataset.loader.parameters.method=variance \
    optimizer.parameters.lr=0.0001 \
    optimizer.parameters.weight_decay=0.0 \
    model.backbone.dropout=0.1 \
    model.feature_encoder.out_channels=64 \
    model.backbone.num_layers=2 \
    dataset.loader.parameters.adjacency_threshold=0.4 \
    model.readout.fc_dropout=0.1 \
    trainer.devices=\[2\] \
    &

python -m ogbench \
    model=graph_sage \
    dataset=addneuromed \
    seed=123 \
    logger.wandb.tags=[graph_sage,addneuromed,hpsearch,rerun] \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    experiment=no_readout \
    dataset.loader.parameters.adjacency_method=wgcna \
    dataset.loader.parameters.node_sample_ratio=0.8 \
    dataset.loader.parameters.method=variance \
    optimizer.parameters.lr=0.0001 \
    optimizer.parameters.weight_decay=0.0 \
    model.backbone.dropout=0.1 \
    model.feature_encoder.out_channels=16 \
    model.backbone.num_layers=2 \
    dataset.loader.parameters.adjacency_threshold=0.2537 \
    trainer.devices=\[3\] \
    &

python -m ogbench \
    model=graph_sage \
    dataset=addneuromed \
    seed=123 \
    logger.wandb.tags=[graph_sage,addneuromed,hpsearch,rerun] \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    experiment=no_readout \
    dataset.loader.parameters.adjacency_method=wgcna \
    dataset.loader.parameters.node_sample_ratio=0.8 \
    dataset.loader.parameters.method=distance_correlation \
    optimizer.parameters.lr=0.0001 \
    optimizer.parameters.weight_decay=0.0 \
    model.backbone.dropout=0.1 \
    model.feature_encoder.out_channels=64 \
    model.backbone.num_layers=4 \
    dataset.loader.parameters.adjacency_threshold=0.1765 \
    trainer.devices=\[4\] \
    &

python -m ogbench \
    model=graph_sage \
    dataset=addneuromed \
    seed=123 \
    logger.wandb.tags=[graph_sage,addneuromed,hpsearch,rerun] \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    experiment=no_readout \
    dataset.loader.parameters.adjacency_method=wgcna \
    dataset.loader.parameters.node_sample_ratio=0.8 \
    dataset.loader.parameters.method=distance_correlation \
    optimizer.parameters.lr=0.001 \
    optimizer.parameters.weight_decay=0.0 \
    model.backbone.dropout=0.1 \
    model.feature_encoder.out_channels=64 \
    model.backbone.num_layers=4 \
    dataset.loader.parameters.adjacency_threshold=0.1765 \
    trainer.devices=\[5\] \
    &

python -m ogbench \
    model=graph_sage \
    dataset=addneuromed \
    seed=42 \
    logger.wandb.tags=[graph_sage,addneuromed,hpsearch,rerun] \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    experiment=no_readout \
    dataset.loader.parameters.adjacency_method=string \
    dataset.loader.parameters.node_sample_ratio=1.0 \
    dataset.loader.parameters.method=random \
    optimizer.parameters.lr=0.001 \
    optimizer.parameters.weight_decay=0.0 \
    model.backbone.dropout=0.1 \
    model.feature_encoder.out_channels=16 \
    model.backbone.num_layers=2 \
    dataset.loader.parameters.adjacency_threshold=0.4 \
    trainer.devices=\[6\] \
    &

python -m ogbench \
    model=graph_sage \
    dataset=addneuromed \
    seed=123 \
    logger.wandb.tags=[graph_sage,addneuromed,hpsearch,rerun] \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    experiment=no_readout \
    dataset.loader.parameters.adjacency_method=string \
    dataset.loader.parameters.node_sample_ratio=1.0 \
    dataset.loader.parameters.method=random \
    optimizer.parameters.lr=0.001 \
    optimizer.parameters.weight_decay=0.0 \
    model.backbone.dropout=0.1 \
    model.feature_encoder.out_channels=64 \
    model.backbone.num_layers=2 \
    dataset.loader.parameters.adjacency_threshold=0.4 \
    trainer.devices=\[7\] \
    &

python -m ogbench \
    model=graph_sage \
    dataset=addneuromed \
    seed=123 \
    logger.wandb.tags=[graph_sage,addneuromed,hpsearch,rerun] \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    experiment=no_readout \
    dataset.loader.parameters.adjacency_method=string \
    dataset.loader.parameters.node_sample_ratio=1.0 \
    dataset.loader.parameters.method=distance_correlation \
    optimizer.parameters.lr=0.0001 \
    optimizer.parameters.weight_decay=0.0 \
    model.backbone.dropout=0.1 \
    model.feature_encoder.out_channels=64 \
    model.backbone.num_layers=4 \
    dataset.loader.parameters.adjacency_threshold=0.4 \
    trainer.devices=\[0\] \
    &

python -m ogbench \
    model=graph_sage \
    dataset=addneuromed \
    seed=42 \
    logger.wandb.tags=[graph_sage,addneuromed,hpsearch,rerun] \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    experiment=no_readout \
    dataset.loader.parameters.adjacency_method=string \
    dataset.loader.parameters.node_sample_ratio=0.8 \
    dataset.loader.parameters.method=random \
    optimizer.parameters.lr=0.0001 \
    optimizer.parameters.weight_decay=0.0 \
    model.backbone.dropout=0.1 \
    model.feature_encoder.out_channels=16 \
    model.backbone.num_layers=4 \
    dataset.loader.parameters.adjacency_threshold=0.4 \
    trainer.devices=\[1\] \
    &

python -m ogbench \
    model=graph_sage \
    dataset=addneuromed \
    seed=123 \
    logger.wandb.tags=[graph_sage,addneuromed,hpsearch,rerun] \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    experiment=no_readout \
    dataset.loader.parameters.adjacency_method=string \
    dataset.loader.parameters.node_sample_ratio=0.8 \
    dataset.loader.parameters.method=random \
    optimizer.parameters.lr=0.001 \
    optimizer.parameters.weight_decay=0.0 \
    model.backbone.dropout=0.1 \
    model.feature_encoder.out_channels=16 \
    model.backbone.num_layers=4 \
    dataset.loader.parameters.adjacency_threshold=0.4 \
    trainer.devices=\[2\] \
    &

python -m ogbench \
    model=graph_sage \
    dataset=addneuromed \
    seed=456 \
    logger.wandb.tags=[graph_sage,addneuromed,hpsearch,rerun] \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    experiment=omics_readout \
    dataset.loader.parameters.adjacency_method=wgcna \
    dataset.loader.parameters.node_sample_ratio=1.0 \
    dataset.loader.parameters.method=variance \
    optimizer.parameters.lr=0.001 \
    optimizer.parameters.weight_decay=0.0 \
    model.backbone.dropout=0.1 \
    model.feature_encoder.out_channels=16 \
    model.backbone.num_layers=2 \
    dataset.loader.parameters.adjacency_threshold=0.2996 \
    model.readout.fc_dropout=0.1 \
    trainer.devices=\[3\] \
    &

python -m ogbench \
    model=graph_sage \
    dataset=addneuromed \
    seed=123 \
    logger.wandb.tags=[graph_sage,addneuromed,hpsearch,rerun] \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    experiment=omics_readout \
    dataset.loader.parameters.adjacency_method=wgcna \
    dataset.loader.parameters.node_sample_ratio=1.0 \
    dataset.loader.parameters.method=correlation \
    optimizer.parameters.lr=0.001 \
    optimizer.parameters.weight_decay=0.0 \
    model.backbone.dropout=0.1 \
    model.feature_encoder.out_channels=64 \
    model.backbone.num_layers=2 \
    dataset.loader.parameters.adjacency_threshold=0.0997 \
    model.readout.fc_dropout=0.1 \
    trainer.devices=\[4\] \
    &

python -m ogbench \
    model=graph_sage \
    dataset=addneuromed \
    seed=42 \
    logger.wandb.tags=[graph_sage,addneuromed,hpsearch,rerun] \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    experiment=omics_readout \
    dataset.loader.parameters.adjacency_method=wgcna \
    dataset.loader.parameters.node_sample_ratio=1.0 \
    dataset.loader.parameters.method=correlation \
    optimizer.parameters.lr=0.001 \
    optimizer.parameters.weight_decay=0.0 \
    model.backbone.dropout=0.1 \
    model.feature_encoder.out_channels=64 \
    model.backbone.num_layers=4 \
    dataset.loader.parameters.adjacency_threshold=0.0997 \
    model.readout.fc_dropout=0.1 \
    trainer.devices=\[5\] \
    &

python -m ogbench \
    model=graph_sage \
    dataset=addneuromed \
    seed=42 \
    logger.wandb.tags=[graph_sage,addneuromed,hpsearch,rerun] \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    experiment=omics_readout \
    dataset.loader.parameters.adjacency_method=wgcna \
    dataset.loader.parameters.node_sample_ratio=0.8 \
    dataset.loader.parameters.method=variance \
    optimizer.parameters.lr=0.0001 \
    optimizer.parameters.weight_decay=0.0 \
    model.backbone.dropout=0.1 \
    model.feature_encoder.out_channels=64 \
    model.backbone.num_layers=4 \
    dataset.loader.parameters.adjacency_threshold=0.2537 \
    model.readout.fc_dropout=0.1 \
    trainer.devices=\[6\] \
    &

python -m ogbench \
    model=graph_sage \
    dataset=addneuromed \
    seed=123 \
    logger.wandb.tags=[graph_sage,addneuromed,hpsearch,rerun] \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    experiment=omics_readout \
    dataset.loader.parameters.adjacency_method=string \
    dataset.loader.parameters.node_sample_ratio=0.8 \
    dataset.loader.parameters.method=variance \
    optimizer.parameters.lr=0.0001 \
    optimizer.parameters.weight_decay=0.0 \
    model.backbone.dropout=0.1 \
    model.feature_encoder.out_channels=64 \
    model.backbone.num_layers=2 \
    dataset.loader.parameters.adjacency_threshold=0.4 \
    model.readout.fc_dropout=0.1 \
    trainer.devices=\[7\] \
    &

python -m ogbench \
    model=graph_sage \
    dataset=addneuromed \
    seed=123 \
    logger.wandb.tags=[graph_sage,addneuromed,hpsearch,rerun] \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    experiment=omics_readout \
    dataset.loader.parameters.adjacency_method=string \
    dataset.loader.parameters.node_sample_ratio=0.8 \
    dataset.loader.parameters.method=correlation \
    optimizer.parameters.lr=0.0001 \
    optimizer.parameters.weight_decay=0.0 \
    model.backbone.dropout=0.1 \
    model.feature_encoder.out_channels=16 \
    model.backbone.num_layers=2 \
    dataset.loader.parameters.adjacency_threshold=0.4 \
    model.readout.fc_dropout=0.1 \
    trainer.devices=\[0\] \
    &

python -m ogbench \
    model=graph_sage \
    dataset=addneuromed \
    seed=456 \
    logger.wandb.tags=[graph_sage,addneuromed,hpsearch,rerun] \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    experiment=omics_readout \
    dataset.loader.parameters.adjacency_method=string \
    dataset.loader.parameters.node_sample_ratio=0.8 \
    dataset.loader.parameters.method=distance_correlation \
    optimizer.parameters.lr=0.0001 \
    optimizer.parameters.weight_decay=0.0 \
    model.backbone.dropout=0.1 \
    model.feature_encoder.out_channels=16 \
    model.backbone.num_layers=2 \
    dataset.loader.parameters.adjacency_threshold=0.4 \
    model.readout.fc_dropout=0.1 \
    trainer.devices=\[1\] \
    &

python -m ogbench \
    model=graph_sage \
    dataset=addneuromed \
    seed=42 \
    logger.wandb.tags=[graph_sage,addneuromed,hpsearch,rerun] \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    experiment=omics_readout \
    dataset.loader.parameters.adjacency_method=string \
    dataset.loader.parameters.node_sample_ratio=0.8 \
    dataset.loader.parameters.method=distance_correlation \
    optimizer.parameters.lr=0.001 \
    optimizer.parameters.weight_decay=0.0 \
    model.backbone.dropout=0.1 \
    model.feature_encoder.out_channels=16 \
    model.backbone.num_layers=4 \
    dataset.loader.parameters.adjacency_threshold=0.4 \
    model.readout.fc_dropout=0.1 \
    trainer.devices=\[2\] \
    &

python -m ogbench \
    model=graph_sage \
    dataset=addneuromed \
    seed=456 \
    logger.wandb.tags=[graph_sage,addneuromed,hpsearch,rerun] \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    experiment=omics_readout \
    dataset.loader.parameters.adjacency_method=string \
    dataset.loader.parameters.node_sample_ratio=0.8 \
    dataset.loader.parameters.method=distance_correlation \
    optimizer.parameters.lr=0.001 \
    optimizer.parameters.weight_decay=0.0 \
    model.backbone.dropout=0.1 \
    model.feature_encoder.out_channels=64 \
    model.backbone.num_layers=4 \
    dataset.loader.parameters.adjacency_threshold=0.4 \
    model.readout.fc_dropout=0.1 \
    trainer.devices=\[3\] \
    &

python -m ogbench \
    model=graph_sage \
    dataset=motrpac \
    seed=123 \
    logger.wandb.tags=[graph_sage,motrpac,hpsearch,rerun] \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    experiment=no_readout \
    dataset.loader.parameters.adjacency_method=wgcna \
    dataset.loader.parameters.node_sample_ratio=1.0 \
    dataset.loader.parameters.method=variance \
    optimizer.parameters.lr=0.0001 \
    optimizer.parameters.weight_decay=0.0 \
    model.backbone.dropout=0.1 \
    model.feature_encoder.out_channels=16 \
    model.backbone.num_layers=2 \
    dataset.loader.parameters.adjacency_threshold=0.1179 \
    trainer.devices=\[4\] \
    &

python -m ogbench \
    model=graph_sage \
    dataset=motrpac \
    seed=123 \
    logger.wandb.tags=[graph_sage,motrpac,hpsearch,rerun] \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    experiment=no_readout \
    dataset.loader.parameters.adjacency_method=wgcna \
    dataset.loader.parameters.node_sample_ratio=1.0 \
    dataset.loader.parameters.method=distance_correlation \
    optimizer.parameters.lr=0.001 \
    optimizer.parameters.weight_decay=0.0 \
    model.backbone.dropout=0.1 \
    model.feature_encoder.out_channels=16 \
    model.backbone.num_layers=4 \
    dataset.loader.parameters.adjacency_threshold=0.1061 \
    trainer.devices=\[5\] \
    &

python -m ogbench \
    model=graph_sage \
    dataset=motrpac \
    seed=123 \
    logger.wandb.tags=[graph_sage,motrpac,hpsearch,rerun] \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    experiment=no_readout \
    dataset.loader.parameters.adjacency_method=string \
    dataset.loader.parameters.node_sample_ratio=0.8 \
    dataset.loader.parameters.method=random \
    optimizer.parameters.lr=0.001 \
    optimizer.parameters.weight_decay=0.0 \
    model.backbone.dropout=0.1 \
    model.feature_encoder.out_channels=16 \
    model.backbone.num_layers=2 \
    dataset.loader.parameters.adjacency_threshold=0.4 \
    trainer.devices=\[6\] \
    &

python -m ogbench \
    model=graph_sage \
    dataset=motrpac \
    seed=42 \
    logger.wandb.tags=[graph_sage,motrpac,hpsearch,rerun] \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    experiment=no_readout \
    dataset.loader.parameters.adjacency_method=string \
    dataset.loader.parameters.node_sample_ratio=0.8 \
    dataset.loader.parameters.method=correlation \
    optimizer.parameters.lr=0.001 \
    optimizer.parameters.weight_decay=0.0 \
    model.backbone.dropout=0.1 \
    model.feature_encoder.out_channels=16 \
    model.backbone.num_layers=2 \
    dataset.loader.parameters.adjacency_threshold=0.4 \
    trainer.devices=\[7\] \
    &

python -m ogbench \
    model=graph_sage \
    dataset=motrpac \
    seed=456 \
    logger.wandb.tags=[graph_sage,motrpac,hpsearch,rerun] \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    experiment=omics_readout \
    dataset.loader.parameters.adjacency_method=wgcna \
    dataset.loader.parameters.node_sample_ratio=1.0 \
    dataset.loader.parameters.method=variance \
    optimizer.parameters.lr=0.001 \
    optimizer.parameters.weight_decay=0.0 \
    model.backbone.dropout=0.1 \
    model.feature_encoder.out_channels=64 \
    model.backbone.num_layers=4 \
    dataset.loader.parameters.adjacency_threshold=0.1179 \
    model.readout.fc_dropout=0.1 \
    trainer.devices=\[0\] \
    &

python -m ogbench \
    model=graph_sage \
    dataset=motrpac \
    seed=456 \
    logger.wandb.tags=[graph_sage,motrpac,hpsearch,rerun] \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    experiment=omics_readout \
    dataset.loader.parameters.adjacency_method=wgcna \
    dataset.loader.parameters.node_sample_ratio=1.0 \
    dataset.loader.parameters.method=correlation \
    optimizer.parameters.lr=0.0001 \
    optimizer.parameters.weight_decay=0.0 \
    model.backbone.dropout=0.1 \
    model.feature_encoder.out_channels=64 \
    model.backbone.num_layers=4 \
    dataset.loader.parameters.adjacency_threshold=0.1323 \
    model.readout.fc_dropout=0.1 \
    trainer.devices=\[1\] \
    &

python -m ogbench \
    model=graph_sage \
    dataset=motrpac \
    seed=456 \
    logger.wandb.tags=[graph_sage,motrpac,hpsearch,rerun] \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    experiment=omics_readout \
    dataset.loader.parameters.adjacency_method=wgcna \
    dataset.loader.parameters.node_sample_ratio=1.0 \
    dataset.loader.parameters.method=distance_correlation \
    optimizer.parameters.lr=0.001 \
    optimizer.parameters.weight_decay=0.0 \
    model.backbone.dropout=0.1 \
    model.feature_encoder.out_channels=16 \
    model.backbone.num_layers=4 \
    dataset.loader.parameters.adjacency_threshold=0.1061 \
    model.readout.fc_dropout=0.1 \
    trainer.devices=\[2\] \
    &

python -m ogbench \
    model=graph_sage \
    dataset=motrpac \
    seed=123 \
    logger.wandb.tags=[graph_sage,motrpac,hpsearch,rerun] \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    experiment=omics_readout \
    dataset.loader.parameters.adjacency_method=wgcna \
    dataset.loader.parameters.node_sample_ratio=0.8 \
    dataset.loader.parameters.method=variance \
    optimizer.parameters.lr=0.0001 \
    optimizer.parameters.weight_decay=0.0 \
    model.backbone.dropout=0.1 \
    model.feature_encoder.out_channels=16 \
    model.backbone.num_layers=4 \
    dataset.loader.parameters.adjacency_threshold=0.0988 \
    model.readout.fc_dropout=0.1 \
    trainer.devices=\[3\] \
    &

python -m ogbench \
    model=graph_sage \
    dataset=motrpac \
    seed=456 \
    logger.wandb.tags=[graph_sage,motrpac,hpsearch,rerun] \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    experiment=omics_readout \
    dataset.loader.parameters.adjacency_method=wgcna \
    dataset.loader.parameters.node_sample_ratio=0.8 \
    dataset.loader.parameters.method=variance \
    optimizer.parameters.lr=0.001 \
    optimizer.parameters.weight_decay=0.0 \
    model.backbone.dropout=0.1 \
    model.feature_encoder.out_channels=64 \
    model.backbone.num_layers=4 \
    dataset.loader.parameters.adjacency_threshold=0.0988 \
    model.readout.fc_dropout=0.1 \
    trainer.devices=\[4\] \
    &

python -m ogbench \
    model=graph_sage \
    dataset=motrpac \
    seed=42 \
    logger.wandb.tags=[graph_sage,motrpac,hpsearch,rerun] \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    experiment=omics_readout \
    dataset.loader.parameters.adjacency_method=string \
    dataset.loader.parameters.node_sample_ratio=1.0 \
    dataset.loader.parameters.method=random \
    optimizer.parameters.lr=0.001 \
    optimizer.parameters.weight_decay=0.0 \
    model.backbone.dropout=0.1 \
    model.feature_encoder.out_channels=64 \
    model.backbone.num_layers=2 \
    dataset.loader.parameters.adjacency_threshold=0.4 \
    model.readout.fc_dropout=0.1 \
    trainer.devices=\[5\] \
    &

python -m ogbench \
    model=graph_sage \
    dataset=motrpac \
    seed=456 \
    logger.wandb.tags=[graph_sage,motrpac,hpsearch,rerun] \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    experiment=omics_readout \
    dataset.loader.parameters.adjacency_method=string \
    dataset.loader.parameters.node_sample_ratio=1.0 \
    dataset.loader.parameters.method=correlation \
    optimizer.parameters.lr=0.0001 \
    optimizer.parameters.weight_decay=0.0 \
    model.backbone.dropout=0.1 \
    model.feature_encoder.out_channels=16 \
    model.backbone.num_layers=4 \
    dataset.loader.parameters.adjacency_threshold=0.4 \
    model.readout.fc_dropout=0.1 \
    trainer.devices=\[6\] \
    &

python -m ogbench \
    model=graph_sage \
    dataset=motrpac \
    seed=123 \
    logger.wandb.tags=[graph_sage,motrpac,hpsearch,rerun] \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    experiment=omics_readout \
    dataset.loader.parameters.adjacency_method=string \
    dataset.loader.parameters.node_sample_ratio=0.8 \
    dataset.loader.parameters.method=correlation \
    optimizer.parameters.lr=0.001 \
    optimizer.parameters.weight_decay=0.0 \
    model.backbone.dropout=0.1 \
    model.feature_encoder.out_channels=16 \
    model.backbone.num_layers=4 \
    dataset.loader.parameters.adjacency_threshold=0.4 \
    model.readout.fc_dropout=0.1 \
    trainer.devices=\[7\] \
    &

python -m ogbench \
    model=graph_sage \
    dataset=motrpac \
    seed=42 \
    logger.wandb.tags=[graph_sage,motrpac,hpsearch,rerun] \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    experiment=omics_readout \
    dataset.loader.parameters.adjacency_method=string \
    dataset.loader.parameters.node_sample_ratio=0.8 \
    dataset.loader.parameters.method=distance_correlation \
    optimizer.parameters.lr=0.0001 \
    optimizer.parameters.weight_decay=0.0 \
    model.backbone.dropout=0.1 \
    model.feature_encoder.out_channels=64 \
    model.backbone.num_layers=4 \
    dataset.loader.parameters.adjacency_threshold=0.4 \
    model.readout.fc_dropout=0.1 \
    trainer.devices=\[0\] \
    &

python -m ogbench \
    model=graph_sage \
    dataset=brca \
    seed=42 \
    logger.wandb.tags=[graph_sage,brca,hpsearch,rerun] \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    experiment=no_readout \
    dataset.loader.parameters.adjacency_method=wgcna \
    dataset.loader.parameters.node_sample_ratio=1.0 \
    dataset.loader.parameters.method=random \
    optimizer.parameters.lr=0.0001 \
    optimizer.parameters.weight_decay=0.0 \
    model.backbone.dropout=0.1 \
    model.feature_encoder.out_channels=16 \
    model.backbone.num_layers=2 \
    dataset.loader.parameters.adjacency_threshold=0.0012 \
    trainer.devices=\[1\] \
    &

python -m ogbench \
    model=graph_sage \
    dataset=brca \
    seed=123 \
    logger.wandb.tags=[graph_sage,brca,hpsearch,rerun] \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    experiment=no_readout \
    dataset.loader.parameters.adjacency_method=wgcna \
    dataset.loader.parameters.node_sample_ratio=0.8 \
    dataset.loader.parameters.method=correlation \
    optimizer.parameters.lr=0.0001 \
    optimizer.parameters.weight_decay=0.0 \
    model.backbone.dropout=0.1 \
    model.feature_encoder.out_channels=16 \
    model.backbone.num_layers=4 \
    dataset.loader.parameters.adjacency_threshold=0.0005 \
    trainer.devices=\[2\] \
    &

python -m ogbench \
    model=graph_sage \
    dataset=brca \
    seed=42 \
    logger.wandb.tags=[graph_sage,brca,hpsearch,rerun] \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    experiment=no_readout \
    dataset.loader.parameters.adjacency_method=wgcna \
    dataset.loader.parameters.node_sample_ratio=0.8 \
    dataset.loader.parameters.method=correlation \
    optimizer.parameters.lr=0.001 \
    optimizer.parameters.weight_decay=0.0 \
    model.backbone.dropout=0.1 \
    model.feature_encoder.out_channels=16 \
    model.backbone.num_layers=4 \
    dataset.loader.parameters.adjacency_threshold=0.0005 \
    trainer.devices=\[3\] \
    &

python -m ogbench \
    model=graph_sage \
    dataset=brca \
    seed=123 \
    logger.wandb.tags=[graph_sage,brca,hpsearch,rerun] \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    experiment=no_readout \
    dataset.loader.parameters.adjacency_method=string \
    dataset.loader.parameters.node_sample_ratio=1.0 \
    dataset.loader.parameters.method=correlation \
    optimizer.parameters.lr=0.001 \
    optimizer.parameters.weight_decay=0.0 \
    model.backbone.dropout=0.1 \
    model.feature_encoder.out_channels=64 \
    model.backbone.num_layers=2 \
    dataset.loader.parameters.adjacency_threshold=0.4 \
    trainer.devices=\[4\] \
    &

python -m ogbench \
    model=graph_sage \
    dataset=brca \
    seed=42 \
    logger.wandb.tags=[graph_sage,brca,hpsearch,rerun] \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    experiment=no_readout \
    dataset.loader.parameters.adjacency_method=string \
    dataset.loader.parameters.node_sample_ratio=1.0 \
    dataset.loader.parameters.method=distance_correlation \
    optimizer.parameters.lr=0.001 \
    optimizer.parameters.weight_decay=0.0 \
    model.backbone.dropout=0.1 \
    model.feature_encoder.out_channels=16 \
    model.backbone.num_layers=2 \
    dataset.loader.parameters.adjacency_threshold=0.4 \
    trainer.devices=\[5\] \
    &

python -m ogbench \
    model=graph_sage \
    dataset=brca \
    seed=123 \
    logger.wandb.tags=[graph_sage,brca,hpsearch,rerun] \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    experiment=no_readout \
    dataset.loader.parameters.adjacency_method=string \
    dataset.loader.parameters.node_sample_ratio=0.8 \
    dataset.loader.parameters.method=variance \
    optimizer.parameters.lr=0.0001 \
    optimizer.parameters.weight_decay=0.0 \
    model.backbone.dropout=0.1 \
    model.feature_encoder.out_channels=64 \
    model.backbone.num_layers=4 \
    dataset.loader.parameters.adjacency_threshold=0.4 \
    trainer.devices=\[6\] \
    &

python -m ogbench \
    model=graph_sage \
    dataset=brca \
    seed=42 \
    logger.wandb.tags=[graph_sage,brca,hpsearch,rerun] \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    experiment=no_readout \
    dataset.loader.parameters.adjacency_method=string \
    dataset.loader.parameters.node_sample_ratio=0.8 \
    dataset.loader.parameters.method=variance \
    optimizer.parameters.lr=0.001 \
    optimizer.parameters.weight_decay=0.0 \
    model.backbone.dropout=0.1 \
    model.feature_encoder.out_channels=64 \
    model.backbone.num_layers=2 \
    dataset.loader.parameters.adjacency_threshold=0.4 \
    trainer.devices=\[7\] \
    &

python -m ogbench \
    model=graph_sage \
    dataset=brca \
    seed=42 \
    logger.wandb.tags=[graph_sage,brca,hpsearch,rerun] \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    experiment=no_readout \
    dataset.loader.parameters.adjacency_method=string \
    dataset.loader.parameters.node_sample_ratio=0.8 \
    dataset.loader.parameters.method=correlation \
    optimizer.parameters.lr=0.001 \
    optimizer.parameters.weight_decay=0.0 \
    model.backbone.dropout=0.1 \
    model.feature_encoder.out_channels=64 \
    model.backbone.num_layers=4 \
    dataset.loader.parameters.adjacency_threshold=0.4 \
    trainer.devices=\[0\] \
    &

python -m ogbench \
    model=graph_sage \
    dataset=brca \
    seed=42 \
    logger.wandb.tags=[graph_sage,brca,hpsearch,rerun] \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    experiment=omics_readout \
    dataset.loader.parameters.adjacency_method=wgcna \
    dataset.loader.parameters.node_sample_ratio=1.0 \
    dataset.loader.parameters.method=random \
    optimizer.parameters.lr=0.0001 \
    optimizer.parameters.weight_decay=0.0 \
    model.backbone.dropout=0.1 \
    model.feature_encoder.out_channels=64 \
    model.backbone.num_layers=2 \
    dataset.loader.parameters.adjacency_threshold=0.0012 \
    model.readout.fc_dropout=0.1 \
    trainer.devices=\[1\] \
    &

python -m ogbench \
    model=graph_sage \
    dataset=brca \
    seed=456 \
    logger.wandb.tags=[graph_sage,brca,hpsearch,rerun] \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    experiment=omics_readout \
    dataset.loader.parameters.adjacency_method=wgcna \
    dataset.loader.parameters.node_sample_ratio=0.8 \
    dataset.loader.parameters.method=variance \
    optimizer.parameters.lr=0.0001 \
    optimizer.parameters.weight_decay=0.0 \
    model.backbone.dropout=0.1 \
    model.feature_encoder.out_channels=16 \
    model.backbone.num_layers=4 \
    dataset.loader.parameters.adjacency_threshold=0.0005 \
    model.readout.fc_dropout=0.1 \
    trainer.devices=\[2\] \
    &

python -m ogbench \
    model=graph_sage \
    dataset=brca \
    seed=42 \
    logger.wandb.tags=[graph_sage,brca,hpsearch,rerun] \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    experiment=omics_readout \
    dataset.loader.parameters.adjacency_method=wgcna \
    dataset.loader.parameters.node_sample_ratio=0.8 \
    dataset.loader.parameters.method=variance \
    optimizer.parameters.lr=0.001 \
    optimizer.parameters.weight_decay=0.0 \
    model.backbone.dropout=0.1 \
    model.feature_encoder.out_channels=16 \
    model.backbone.num_layers=2 \
    dataset.loader.parameters.adjacency_threshold=0.0005 \
    model.readout.fc_dropout=0.1 \
    trainer.devices=\[3\] \
    &

python -m ogbench \
    model=graph_sage \
    dataset=brca \
    seed=456 \
    logger.wandb.tags=[graph_sage,brca,hpsearch,rerun] \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    experiment=omics_readout \
    dataset.loader.parameters.adjacency_method=wgcna \
    dataset.loader.parameters.node_sample_ratio=0.8 \
    dataset.loader.parameters.method=random \
    optimizer.parameters.lr=0.001 \
    optimizer.parameters.weight_decay=0.0 \
    model.backbone.dropout=0.1 \
    model.feature_encoder.out_channels=16 \
    model.backbone.num_layers=4 \
    dataset.loader.parameters.adjacency_threshold=0.0005 \
    model.readout.fc_dropout=0.1 \
    trainer.devices=\[4\] \
    &

python -m ogbench \
    model=graph_sage \
    dataset=brca \
    seed=42 \
    logger.wandb.tags=[graph_sage,brca,hpsearch,rerun] \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    experiment=omics_readout \
    dataset.loader.parameters.adjacency_method=string \
    dataset.loader.parameters.node_sample_ratio=1.0 \
    dataset.loader.parameters.method=variance \
    optimizer.parameters.lr=0.0001 \
    optimizer.parameters.weight_decay=0.0 \
    model.backbone.dropout=0.1 \
    model.feature_encoder.out_channels=64 \
    model.backbone.num_layers=2 \
    dataset.loader.parameters.adjacency_threshold=0.4 \
    model.readout.fc_dropout=0.1 \
    trainer.devices=\[5\] \
    &

python -m ogbench \
    model=graph_sage \
    dataset=brca \
    seed=42 \
    logger.wandb.tags=[graph_sage,brca,hpsearch,rerun] \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    experiment=omics_readout \
    dataset.loader.parameters.adjacency_method=string \
    dataset.loader.parameters.node_sample_ratio=0.8 \
    dataset.loader.parameters.method=random \
    optimizer.parameters.lr=0.0001 \
    optimizer.parameters.weight_decay=0.0 \
    model.backbone.dropout=0.1 \
    model.feature_encoder.out_channels=64 \
    model.backbone.num_layers=2 \
    dataset.loader.parameters.adjacency_threshold=0.4 \
    model.readout.fc_dropout=0.1 \
    trainer.devices=\[6\] \
    &

python -m ogbench \
    model=graph_sage \
    dataset=brca \
    seed=42 \
    logger.wandb.tags=[graph_sage,brca,hpsearch,rerun] \
    logger.wandb.project=bgbench_dataset_grid_search_final \
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
    trainer.devices=\[7\] \
    &

python -m ogbench \
    model=graph_sage \
    dataset=brca \
    seed=456 \
    logger.wandb.tags=[graph_sage,brca,hpsearch,rerun] \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    experiment=omics_readout \
    dataset.loader.parameters.adjacency_method=string \
    dataset.loader.parameters.node_sample_ratio=0.8 \
    dataset.loader.parameters.method=random \
    optimizer.parameters.lr=0.001 \
    optimizer.parameters.weight_decay=0.0 \
    model.backbone.dropout=0.1 \
    model.feature_encoder.out_channels=16 \
    model.backbone.num_layers=2 \
    dataset.loader.parameters.adjacency_threshold=0.4 \
    model.readout.fc_dropout=0.1 \
    trainer.devices=\[0\] \
    &

python -m ogbench \
    model=graph_sage \
    dataset=brca \
    seed=42 \
    logger.wandb.tags=[graph_sage,brca,hpsearch,rerun] \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    experiment=omics_readout \
    dataset.loader.parameters.adjacency_method=string \
    dataset.loader.parameters.node_sample_ratio=0.8 \
    dataset.loader.parameters.method=correlation \
    optimizer.parameters.lr=0.001 \
    optimizer.parameters.weight_decay=0.0 \
    model.backbone.dropout=0.1 \
    model.feature_encoder.out_channels=64 \
    model.backbone.num_layers=2 \
    dataset.loader.parameters.adjacency_threshold=0.4 \
    model.readout.fc_dropout=0.1 \
    trainer.devices=\[1\] \
    &

python -m ogbench \
    model=chebnet \
    dataset=parkinsons \
    seed=42 \
    logger.wandb.tags=[chebnet,parkinsons,hpsearch,rerun] \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    experiment=no_readout \
    dataset.loader.parameters.adjacency_method=wgcna \
    dataset.loader.parameters.node_sample_ratio=1.0 \
    dataset.loader.parameters.method=variance \
    optimizer.parameters.lr=0.001 \
    optimizer.parameters.weight_decay=0.0 \
    model.backbone.dropout=0.1 \
    model.feature_encoder.out_channels=16 \
    model.backbone.num_layers=2 \
    dataset.loader.parameters.adjacency_threshold=0.0346 \
    trainer.devices=\[2\] \
    &

python -m ogbench \
    model=chebnet \
    dataset=parkinsons \
    seed=123 \
    logger.wandb.tags=[chebnet,parkinsons,hpsearch,rerun] \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    experiment=no_readout \
    dataset.loader.parameters.adjacency_method=wgcna \
    dataset.loader.parameters.node_sample_ratio=1.0 \
    dataset.loader.parameters.method=random \
    optimizer.parameters.lr=0.0001 \
    optimizer.parameters.weight_decay=0.0 \
    model.backbone.dropout=0.1 \
    model.feature_encoder.out_channels=64 \
    model.backbone.num_layers=2 \
    dataset.loader.parameters.adjacency_threshold=0.0013 \
    trainer.devices=\[3\] \
    &

python -m ogbench \
    model=chebnet \
    dataset=parkinsons \
    seed=456 \
    logger.wandb.tags=[chebnet,parkinsons,hpsearch,rerun] \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    experiment=no_readout \
    dataset.loader.parameters.adjacency_method=wgcna \
    dataset.loader.parameters.node_sample_ratio=1.0 \
    dataset.loader.parameters.method=random \
    optimizer.parameters.lr=0.0001 \
    optimizer.parameters.weight_decay=0.0 \
    model.backbone.dropout=0.1 \
    model.feature_encoder.out_channels=64 \
    model.backbone.num_layers=4 \
    dataset.loader.parameters.adjacency_threshold=0.0013 \
    trainer.devices=\[4\] \
    &

python -m ogbench \
    model=chebnet \
    dataset=parkinsons \
    seed=42 \
    logger.wandb.tags=[chebnet,parkinsons,hpsearch,rerun] \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    experiment=no_readout \
    dataset.loader.parameters.adjacency_method=wgcna \
    dataset.loader.parameters.node_sample_ratio=0.8 \
    dataset.loader.parameters.method=variance \
    optimizer.parameters.lr=0.0001 \
    optimizer.parameters.weight_decay=0.0 \
    model.backbone.dropout=0.1 \
    model.feature_encoder.out_channels=16 \
    model.backbone.num_layers=2 \
    dataset.loader.parameters.adjacency_threshold=0.031 \
    trainer.devices=\[5\] \
    &

python -m ogbench \
    model=chebnet \
    dataset=parkinsons \
    seed=123 \
    logger.wandb.tags=[chebnet,parkinsons,hpsearch,rerun] \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    experiment=no_readout \
    dataset.loader.parameters.adjacency_method=wgcna \
    dataset.loader.parameters.node_sample_ratio=0.8 \
    dataset.loader.parameters.method=correlation \
    optimizer.parameters.lr=0.0001 \
    optimizer.parameters.weight_decay=0.0 \
    model.backbone.dropout=0.1 \
    model.feature_encoder.out_channels=64 \
    model.backbone.num_layers=2 \
    dataset.loader.parameters.adjacency_threshold=0.0019 \
    trainer.devices=\[6\] \
    &

python -m ogbench \
    model=chebnet \
    dataset=parkinsons \
    seed=42 \
    logger.wandb.tags=[chebnet,parkinsons,hpsearch,rerun] \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    experiment=no_readout \
    dataset.loader.parameters.adjacency_method=wgcna \
    dataset.loader.parameters.node_sample_ratio=0.8 \
    dataset.loader.parameters.method=correlation \
    optimizer.parameters.lr=0.001 \
    optimizer.parameters.weight_decay=0.0 \
    model.backbone.dropout=0.1 \
    model.feature_encoder.out_channels=16 \
    model.backbone.num_layers=2 \
    dataset.loader.parameters.adjacency_threshold=0.0019 \
    trainer.devices=\[7\] \
    &

python -m ogbench \
    model=chebnet \
    dataset=parkinsons \
    seed=123 \
    logger.wandb.tags=[chebnet,parkinsons,hpsearch,rerun] \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    experiment=no_readout \
    dataset.loader.parameters.adjacency_method=wgcna \
    dataset.loader.parameters.node_sample_ratio=0.8 \
    dataset.loader.parameters.method=distance_correlation \
    optimizer.parameters.lr=0.0001 \
    optimizer.parameters.weight_decay=0.0 \
    model.backbone.dropout=0.1 \
    model.feature_encoder.out_channels=16 \
    model.backbone.num_layers=2 \
    dataset.loader.parameters.adjacency_threshold=0.0054 \
    trainer.devices=\[0\] \
    &

python -m ogbench \
    model=chebnet \
    dataset=parkinsons \
    seed=456 \
    logger.wandb.tags=[chebnet,parkinsons,hpsearch,rerun] \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    experiment=no_readout \
    dataset.loader.parameters.adjacency_method=string \
    dataset.loader.parameters.node_sample_ratio=1.0 \
    dataset.loader.parameters.method=random \
    optimizer.parameters.lr=0.0001 \
    optimizer.parameters.weight_decay=0.0 \
    model.backbone.dropout=0.1 \
    model.feature_encoder.out_channels=16 \
    model.backbone.num_layers=4 \
    dataset.loader.parameters.adjacency_threshold=0.4 \
    trainer.devices=\[1\] \
    &

python -m ogbench \
    model=chebnet \
    dataset=parkinsons \
    seed=456 \
    logger.wandb.tags=[chebnet,parkinsons,hpsearch,rerun] \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    experiment=no_readout \
    dataset.loader.parameters.adjacency_method=string \
    dataset.loader.parameters.node_sample_ratio=1.0 \
    dataset.loader.parameters.method=random \
    optimizer.parameters.lr=0.001 \
    optimizer.parameters.weight_decay=0.0 \
    model.backbone.dropout=0.1 \
    model.feature_encoder.out_channels=64 \
    model.backbone.num_layers=2 \
    dataset.loader.parameters.adjacency_threshold=0.4 \
    trainer.devices=\[2\] \
    &

python -m ogbench \
    model=chebnet \
    dataset=parkinsons \
    seed=456 \
    logger.wandb.tags=[chebnet,parkinsons,hpsearch,rerun] \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    experiment=no_readout \
    dataset.loader.parameters.adjacency_method=string \
    dataset.loader.parameters.node_sample_ratio=1.0 \
    dataset.loader.parameters.method=distance_correlation \
    optimizer.parameters.lr=0.001 \
    optimizer.parameters.weight_decay=0.0 \
    model.backbone.dropout=0.1 \
    model.feature_encoder.out_channels=64 \
    model.backbone.num_layers=2 \
    dataset.loader.parameters.adjacency_threshold=0.4 \
    trainer.devices=\[3\] \
    &

python -m ogbench \
    model=chebnet \
    dataset=parkinsons \
    seed=42 \
    logger.wandb.tags=[chebnet,parkinsons,hpsearch,rerun] \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    experiment=omics_readout \
    dataset.loader.parameters.adjacency_method=wgcna \
    dataset.loader.parameters.node_sample_ratio=1.0 \
    dataset.loader.parameters.method=correlation \
    optimizer.parameters.lr=0.001 \
    optimizer.parameters.weight_decay=0.0 \
    model.backbone.dropout=0.1 \
    model.feature_encoder.out_channels=16 \
    model.backbone.num_layers=2 \
    dataset.loader.parameters.adjacency_threshold=0.0005 \
    model.readout.fc_dropout=0.1 \
    trainer.devices=\[4\] \
    &

python -m ogbench \
    model=chebnet \
    dataset=parkinsons \
    seed=123 \
    logger.wandb.tags=[chebnet,parkinsons,hpsearch,rerun] \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    experiment=omics_readout \
    dataset.loader.parameters.adjacency_method=wgcna \
    dataset.loader.parameters.node_sample_ratio=1.0 \
    dataset.loader.parameters.method=correlation \
    optimizer.parameters.lr=0.001 \
    optimizer.parameters.weight_decay=0.0 \
    model.backbone.dropout=0.1 \
    model.feature_encoder.out_channels=16 \
    model.backbone.num_layers=4 \
    dataset.loader.parameters.adjacency_threshold=0.0005 \
    model.readout.fc_dropout=0.1 \
    trainer.devices=\[5\] \
    &

python -m ogbench \
    model=chebnet \
    dataset=parkinsons \
    seed=456 \
    logger.wandb.tags=[chebnet,parkinsons,hpsearch,rerun] \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    experiment=omics_readout \
    dataset.loader.parameters.adjacency_method=wgcna \
    dataset.loader.parameters.node_sample_ratio=0.8 \
    dataset.loader.parameters.method=random \
    optimizer.parameters.lr=0.001 \
    optimizer.parameters.weight_decay=0.0 \
    model.backbone.dropout=0.1 \
    model.feature_encoder.out_channels=64 \
    model.backbone.num_layers=4 \
    dataset.loader.parameters.adjacency_threshold=0.0216 \
    model.readout.fc_dropout=0.1 \
    trainer.devices=\[6\] \
    &

python -m ogbench \
    model=chebnet \
    dataset=parkinsons \
    seed=456 \
    logger.wandb.tags=[chebnet,parkinsons,hpsearch,rerun] \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    experiment=omics_readout \
    dataset.loader.parameters.adjacency_method=wgcna \
    dataset.loader.parameters.node_sample_ratio=0.8 \
    dataset.loader.parameters.method=correlation \
    optimizer.parameters.lr=0.0001 \
    optimizer.parameters.weight_decay=0.0 \
    model.backbone.dropout=0.1 \
    model.feature_encoder.out_channels=16 \
    model.backbone.num_layers=2 \
    dataset.loader.parameters.adjacency_threshold=0.0019 \
    model.readout.fc_dropout=0.1 \
    trainer.devices=\[7\] \
    &

python -m ogbench \
    model=chebnet \
    dataset=parkinsons \
    seed=456 \
    logger.wandb.tags=[chebnet,parkinsons,hpsearch,rerun] \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    experiment=omics_readout \
    dataset.loader.parameters.adjacency_method=wgcna \
    dataset.loader.parameters.node_sample_ratio=0.8 \
    dataset.loader.parameters.method=correlation \
    optimizer.parameters.lr=0.001 \
    optimizer.parameters.weight_decay=0.0 \
    model.backbone.dropout=0.1 \
    model.feature_encoder.out_channels=16 \
    model.backbone.num_layers=2 \
    dataset.loader.parameters.adjacency_threshold=0.0019 \
    model.readout.fc_dropout=0.1 \
    trainer.devices=\[0\] \
    &

python -m ogbench \
    model=chebnet \
    dataset=parkinsons \
    seed=456 \
    logger.wandb.tags=[chebnet,parkinsons,hpsearch,rerun] \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    experiment=omics_readout \
    dataset.loader.parameters.adjacency_method=wgcna \
    dataset.loader.parameters.node_sample_ratio=0.8 \
    dataset.loader.parameters.method=distance_correlation \
    optimizer.parameters.lr=0.001 \
    optimizer.parameters.weight_decay=0.0 \
    model.backbone.dropout=0.1 \
    model.feature_encoder.out_channels=16 \
    model.backbone.num_layers=4 \
    dataset.loader.parameters.adjacency_threshold=0.0054 \
    model.readout.fc_dropout=0.1 \
    trainer.devices=\[1\] \
    &

python -m ogbench \
    model=chebnet \
    dataset=parkinsons \
    seed=123 \
    logger.wandb.tags=[chebnet,parkinsons,hpsearch,rerun] \
    logger.wandb.project=bgbench_dataset_grid_search_final \
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
    dataset.loader.parameters.adjacency_threshold=0.0054 \
    model.readout.fc_dropout=0.1 \
    trainer.devices=\[2\] \
    &

python -m ogbench \
    model=chebnet \
    dataset=parkinsons \
    seed=456 \
    logger.wandb.tags=[chebnet,parkinsons,hpsearch,rerun] \
    logger.wandb.project=bgbench_dataset_grid_search_final \
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
    dataset.loader.parameters.adjacency_threshold=0.0054 \
    model.readout.fc_dropout=0.1 \
    trainer.devices=\[3\] \
    &

python -m ogbench \
    model=chebnet \
    dataset=parkinsons \
    seed=123 \
    logger.wandb.tags=[chebnet,parkinsons,hpsearch,rerun] \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    experiment=omics_readout \
    dataset.loader.parameters.adjacency_method=wgcna \
    dataset.loader.parameters.node_sample_ratio=0.8 \
    dataset.loader.parameters.method=distance_correlation \
    optimizer.parameters.lr=0.001 \
    optimizer.parameters.weight_decay=0.0 \
    model.backbone.dropout=0.1 \
    model.feature_encoder.out_channels=64 \
    model.backbone.num_layers=4 \
    dataset.loader.parameters.adjacency_threshold=0.0054 \
    model.readout.fc_dropout=0.1 \
    trainer.devices=\[4\] \
    &

python -m ogbench \
    model=chebnet \
    dataset=parkinsons \
    seed=42 \
    logger.wandb.tags=[chebnet,parkinsons,hpsearch,rerun] \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    experiment=omics_readout \
    dataset.loader.parameters.adjacency_method=wgcna \
    dataset.loader.parameters.node_sample_ratio=0.8 \
    dataset.loader.parameters.method=distance_correlation \
    optimizer.parameters.lr=0.001 \
    optimizer.parameters.weight_decay=0.0 \
    model.backbone.dropout=0.1 \
    model.feature_encoder.out_channels=64 \
    model.backbone.num_layers=4 \
    dataset.loader.parameters.adjacency_threshold=0.0054 \
    model.readout.fc_dropout=0.1 \
    trainer.devices=\[5\] \
    &

python -m ogbench \
    model=chebnet \
    dataset=parkinsons \
    seed=456 \
    logger.wandb.tags=[chebnet,parkinsons,hpsearch,rerun] \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    experiment=omics_readout \
    dataset.loader.parameters.adjacency_method=wgcna \
    dataset.loader.parameters.node_sample_ratio=0.8 \
    dataset.loader.parameters.method=distance_correlation \
    optimizer.parameters.lr=0.001 \
    optimizer.parameters.weight_decay=0.0 \
    model.backbone.dropout=0.1 \
    model.feature_encoder.out_channels=64 \
    model.backbone.num_layers=4 \
    dataset.loader.parameters.adjacency_threshold=0.0054 \
    model.readout.fc_dropout=0.1 \
    trainer.devices=\[6\] \
    &

python -m ogbench \
    model=chebnet \
    dataset=parkinsons \
    seed=42 \
    logger.wandb.tags=[chebnet,parkinsons,hpsearch,rerun] \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    experiment=omics_readout \
    dataset.loader.parameters.adjacency_method=string \
    dataset.loader.parameters.node_sample_ratio=1.0 \
    dataset.loader.parameters.method=variance \
    optimizer.parameters.lr=0.0001 \
    optimizer.parameters.weight_decay=0.0 \
    model.backbone.dropout=0.1 \
    model.feature_encoder.out_channels=16 \
    model.backbone.num_layers=2 \
    dataset.loader.parameters.adjacency_threshold=0.4 \
    model.readout.fc_dropout=0.1 \
    trainer.devices=\[7\] \
    &

python -m ogbench \
    model=chebnet \
    dataset=parkinsons \
    seed=123 \
    logger.wandb.tags=[chebnet,parkinsons,hpsearch,rerun] \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    experiment=omics_readout \
    dataset.loader.parameters.adjacency_method=string \
    dataset.loader.parameters.node_sample_ratio=1.0 \
    dataset.loader.parameters.method=variance \
    optimizer.parameters.lr=0.0001 \
    optimizer.parameters.weight_decay=0.0 \
    model.backbone.dropout=0.1 \
    model.feature_encoder.out_channels=16 \
    model.backbone.num_layers=2 \
    dataset.loader.parameters.adjacency_threshold=0.4 \
    model.readout.fc_dropout=0.1 \
    trainer.devices=\[0\] \
    &

python -m ogbench \
    model=chebnet \
    dataset=parkinsons \
    seed=42 \
    logger.wandb.tags=[chebnet,parkinsons,hpsearch,rerun] \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    experiment=omics_readout \
    dataset.loader.parameters.adjacency_method=string \
    dataset.loader.parameters.node_sample_ratio=1.0 \
    dataset.loader.parameters.method=variance \
    optimizer.parameters.lr=0.0001 \
    optimizer.parameters.weight_decay=0.0 \
    model.backbone.dropout=0.1 \
    model.feature_encoder.out_channels=16 \
    model.backbone.num_layers=4 \
    dataset.loader.parameters.adjacency_threshold=0.4 \
    model.readout.fc_dropout=0.1 \
    trainer.devices=\[1\] \
    &

python -m ogbench \
    model=chebnet \
    dataset=parkinsons \
    seed=123 \
    logger.wandb.tags=[chebnet,parkinsons,hpsearch,rerun] \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    experiment=omics_readout \
    dataset.loader.parameters.adjacency_method=string \
    dataset.loader.parameters.node_sample_ratio=1.0 \
    dataset.loader.parameters.method=variance \
    optimizer.parameters.lr=0.0001 \
    optimizer.parameters.weight_decay=0.0 \
    model.backbone.dropout=0.1 \
    model.feature_encoder.out_channels=16 \
    model.backbone.num_layers=4 \
    dataset.loader.parameters.adjacency_threshold=0.4 \
    model.readout.fc_dropout=0.1 \
    trainer.devices=\[2\] \
    &

python -m ogbench \
    model=chebnet \
    dataset=parkinsons \
    seed=456 \
    logger.wandb.tags=[chebnet,parkinsons,hpsearch,rerun] \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    experiment=omics_readout \
    dataset.loader.parameters.adjacency_method=string \
    dataset.loader.parameters.node_sample_ratio=1.0 \
    dataset.loader.parameters.method=variance \
    optimizer.parameters.lr=0.0001 \
    optimizer.parameters.weight_decay=0.0 \
    model.backbone.dropout=0.1 \
    model.feature_encoder.out_channels=16 \
    model.backbone.num_layers=4 \
    dataset.loader.parameters.adjacency_threshold=0.4 \
    model.readout.fc_dropout=0.1 \
    trainer.devices=\[3\] \
    &

python -m ogbench \
    model=chebnet \
    dataset=parkinsons \
    seed=42 \
    logger.wandb.tags=[chebnet,parkinsons,hpsearch,rerun] \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    experiment=omics_readout \
    dataset.loader.parameters.adjacency_method=string \
    dataset.loader.parameters.node_sample_ratio=1.0 \
    dataset.loader.parameters.method=variance \
    optimizer.parameters.lr=0.0001 \
    optimizer.parameters.weight_decay=0.0 \
    model.backbone.dropout=0.1 \
    model.feature_encoder.out_channels=64 \
    model.backbone.num_layers=2 \
    dataset.loader.parameters.adjacency_threshold=0.4 \
    model.readout.fc_dropout=0.1 \
    trainer.devices=\[4\] \
    &

python -m ogbench \
    model=chebnet \
    dataset=parkinsons \
    seed=123 \
    logger.wandb.tags=[chebnet,parkinsons,hpsearch,rerun] \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    experiment=omics_readout \
    dataset.loader.parameters.adjacency_method=string \
    dataset.loader.parameters.node_sample_ratio=1.0 \
    dataset.loader.parameters.method=variance \
    optimizer.parameters.lr=0.0001 \
    optimizer.parameters.weight_decay=0.0 \
    model.backbone.dropout=0.1 \
    model.feature_encoder.out_channels=64 \
    model.backbone.num_layers=2 \
    dataset.loader.parameters.adjacency_threshold=0.4 \
    model.readout.fc_dropout=0.1 \
    trainer.devices=\[5\] \
    &

python -m ogbench \
    model=chebnet \
    dataset=parkinsons \
    seed=456 \
    logger.wandb.tags=[chebnet,parkinsons,hpsearch,rerun] \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    experiment=omics_readout \
    dataset.loader.parameters.adjacency_method=string \
    dataset.loader.parameters.node_sample_ratio=1.0 \
    dataset.loader.parameters.method=variance \
    optimizer.parameters.lr=0.0001 \
    optimizer.parameters.weight_decay=0.0 \
    model.backbone.dropout=0.1 \
    model.feature_encoder.out_channels=64 \
    model.backbone.num_layers=2 \
    dataset.loader.parameters.adjacency_threshold=0.4 \
    model.readout.fc_dropout=0.1 \
    trainer.devices=\[6\] \
    &
# SKIP run_id=iisj5gvh (no Hydra args list in W&B summary or metadata)
# SKIP run_id=tmexw7x9 (no Hydra args list in W&B summary or metadata)
# SKIP run_id=3p6hu840 (no Hydra args list in W&B summary or metadata)
# SKIP run_id=4a7ehcf6 (no Hydra args list in W&B summary or metadata)
# SKIP run_id=gskf4oel (no Hydra args list in W&B summary or metadata)
# SKIP run_id=971yaeuz (no Hydra args list in W&B summary or metadata)
# SKIP run_id=j8qzgw4c (no Hydra args list in W&B summary or metadata)
# SKIP run_id=94qwoi09 (no Hydra args list in W&B summary or metadata)
# SKIP run_id=2lclhmj0 (no Hydra args list in W&B summary or metadata)
# SKIP run_id=oe065qcg (no Hydra args list in W&B summary or metadata)
# SKIP run_id=k8x3mbh7 (no Hydra args list in W&B summary or metadata)
# SKIP run_id=etc0fza2 (no Hydra args list in W&B summary or metadata)
# SKIP run_id=egavc79a (no Hydra args list in W&B summary or metadata)
# SKIP run_id=vo336ngk (no Hydra args list in W&B summary or metadata)
# SKIP run_id=9ig2kzwd (no Hydra args list in W&B summary or metadata)
# SKIP run_id=lxyn1rxr (no Hydra args list in W&B summary or metadata)
# SKIP run_id=qcyeb9vz (no Hydra args list in W&B summary or metadata)
# SKIP run_id=w4182x1z (no Hydra args list in W&B summary or metadata)
# SKIP run_id=4ovhfq5o (no Hydra args list in W&B summary or metadata)
# SKIP run_id=2e70qmrp (no Hydra args list in W&B summary or metadata)
# SKIP run_id=hru4eou9 (no Hydra args list in W&B summary or metadata)

python -m ogbench \
    model=chebnet \
    dataset=parkinsons \
    seed=123 \
    logger.wandb.tags=[chebnet,parkinsons,hpsearch,rerun] \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    experiment=omics_readout \
    dataset.loader.parameters.adjacency_method=string \
    dataset.loader.parameters.node_sample_ratio=0.8 \
    dataset.loader.parameters.method=correlation \
    optimizer.parameters.lr=0.0001 \
    optimizer.parameters.weight_decay=0.0 \
    model.backbone.dropout=0.1 \
    model.feature_encoder.out_channels=64 \
    model.backbone.num_layers=2 \
    dataset.loader.parameters.adjacency_threshold=0.4 \
    model.readout.fc_dropout=0.1 \
    trainer.devices=\[4\] \
    &

python -m ogbench \
    model=chebnet \
    dataset=parkinsons \
    seed=123 \
    logger.wandb.tags=[chebnet,parkinsons,hpsearch,rerun] \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    experiment=omics_readout \
    dataset.loader.parameters.adjacency_method=string \
    dataset.loader.parameters.node_sample_ratio=0.8 \
    dataset.loader.parameters.method=correlation \
    optimizer.parameters.lr=0.001 \
    optimizer.parameters.weight_decay=0.0 \
    model.backbone.dropout=0.1 \
    model.feature_encoder.out_channels=16 \
    model.backbone.num_layers=4 \
    dataset.loader.parameters.adjacency_threshold=0.4 \
    model.readout.fc_dropout=0.1 \
    trainer.devices=\[5\] \
    &

python -m ogbench \
    model=chebnet \
    dataset=parkinsons \
    seed=456 \
    logger.wandb.tags=[chebnet,parkinsons,hpsearch,rerun] \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    experiment=omics_readout \
    dataset.loader.parameters.adjacency_method=string \
    dataset.loader.parameters.node_sample_ratio=0.8 \
    dataset.loader.parameters.method=distance_correlation \
    optimizer.parameters.lr=0.001 \
    optimizer.parameters.weight_decay=0.0 \
    model.backbone.dropout=0.1 \
    model.feature_encoder.out_channels=16 \
    model.backbone.num_layers=4 \
    dataset.loader.parameters.adjacency_threshold=0.4 \
    model.readout.fc_dropout=0.1 \
    trainer.devices=\[6\] \
    &

python -m ogbench \
    model=chebnet \
    dataset=addneuromed \
    seed=42 \
    logger.wandb.tags=[chebnet,addneuromed,hpsearch,rerun] \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    experiment=no_readout \
    dataset.loader.parameters.adjacency_method=wgcna \
    dataset.loader.parameters.node_sample_ratio=1.0 \
    dataset.loader.parameters.method=variance \
    optimizer.parameters.lr=0.0001 \
    optimizer.parameters.weight_decay=0.0 \
    model.backbone.dropout=0.1 \
    model.feature_encoder.out_channels=16 \
    model.backbone.num_layers=2 \
    dataset.loader.parameters.adjacency_threshold=0.2996 \
    trainer.devices=\[7\] \
    &

python -m ogbench \
    model=chebnet \
    dataset=addneuromed \
    seed=123 \
    logger.wandb.tags=[chebnet,addneuromed,hpsearch,rerun] \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    experiment=no_readout \
    dataset.loader.parameters.adjacency_method=wgcna \
    dataset.loader.parameters.node_sample_ratio=1.0 \
    dataset.loader.parameters.method=variance \
    optimizer.parameters.lr=0.001 \
    optimizer.parameters.weight_decay=0.0 \
    model.backbone.dropout=0.1 \
    model.feature_encoder.out_channels=64 \
    model.backbone.num_layers=2 \
    dataset.loader.parameters.adjacency_threshold=0.2996 \
    trainer.devices=\[0\] \
    &

python -m ogbench \
    model=chebnet \
    dataset=addneuromed \
    seed=456 \
    logger.wandb.tags=[chebnet,addneuromed,hpsearch,rerun] \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    experiment=no_readout \
    dataset.loader.parameters.adjacency_method=wgcna \
    dataset.loader.parameters.node_sample_ratio=1.0 \
    dataset.loader.parameters.method=variance \
    optimizer.parameters.lr=0.001 \
    optimizer.parameters.weight_decay=0.0 \
    model.backbone.dropout=0.1 \
    model.feature_encoder.out_channels=64 \
    model.backbone.num_layers=4 \
    dataset.loader.parameters.adjacency_threshold=0.2996 \
    trainer.devices=\[1\] \
    &

python -m ogbench \
    model=chebnet \
    dataset=addneuromed \
    seed=42 \
    logger.wandb.tags=[chebnet,addneuromed,hpsearch,rerun] \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    experiment=no_readout \
    dataset.loader.parameters.adjacency_method=string \
    dataset.loader.parameters.node_sample_ratio=0.8 \
    dataset.loader.parameters.method=correlation \
    optimizer.parameters.lr=0.0001 \
    optimizer.parameters.weight_decay=0.0 \
    model.backbone.dropout=0.1 \
    model.feature_encoder.out_channels=64 \
    model.backbone.num_layers=4 \
    dataset.loader.parameters.adjacency_threshold=0.4 \
    trainer.devices=\[2\] \
    &

python -m ogbench \
    model=chebnet \
    dataset=addneuromed \
    seed=123 \
    logger.wandb.tags=[chebnet,addneuromed,hpsearch,rerun] \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    experiment=no_readout \
    dataset.loader.parameters.adjacency_method=string \
    dataset.loader.parameters.node_sample_ratio=0.8 \
    dataset.loader.parameters.method=correlation \
    optimizer.parameters.lr=0.001 \
    optimizer.parameters.weight_decay=0.0 \
    model.backbone.dropout=0.1 \
    model.feature_encoder.out_channels=16 \
    model.backbone.num_layers=4 \
    dataset.loader.parameters.adjacency_threshold=0.4 \
    trainer.devices=\[3\] \
    &

python -m ogbench \
    model=chebnet \
    dataset=addneuromed \
    seed=456 \
    logger.wandb.tags=[chebnet,addneuromed,hpsearch,rerun] \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    experiment=omics_readout \
    dataset.loader.parameters.adjacency_method=wgcna \
    dataset.loader.parameters.node_sample_ratio=0.8 \
    dataset.loader.parameters.method=random \
    optimizer.parameters.lr=0.0001 \
    optimizer.parameters.weight_decay=0.0 \
    model.backbone.dropout=0.1 \
    model.feature_encoder.out_channels=64 \
    model.backbone.num_layers=4 \
    dataset.loader.parameters.adjacency_threshold=0.0005 \
    model.readout.fc_dropout=0.1 \
    trainer.devices=\[4\] \
    &

python -m ogbench \
    model=chebnet \
    dataset=motrpac \
    seed=42 \
    logger.wandb.tags=[chebnet,motrpac,hpsearch,rerun] \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    experiment=no_readout \
    dataset.loader.parameters.adjacency_method=wgcna \
    dataset.loader.parameters.node_sample_ratio=0.8 \
    dataset.loader.parameters.method=random \
    optimizer.parameters.lr=0.0001 \
    optimizer.parameters.weight_decay=0.0 \
    model.backbone.dropout=0.1 \
    model.feature_encoder.out_channels=16 \
    model.backbone.num_layers=2 \
    dataset.loader.parameters.adjacency_threshold=0.0817 \
    trainer.devices=\[5\] \
    &

python -m ogbench \
    model=chebnet \
    dataset=motrpac \
    seed=42 \
    logger.wandb.tags=[chebnet,motrpac,hpsearch,rerun] \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    experiment=no_readout \
    dataset.loader.parameters.adjacency_method=wgcna \
    dataset.loader.parameters.node_sample_ratio=0.8 \
    dataset.loader.parameters.method=distance_correlation \
    optimizer.parameters.lr=0.0001 \
    optimizer.parameters.weight_decay=0.0 \
    model.backbone.dropout=0.1 \
    model.feature_encoder.out_channels=16 \
    model.backbone.num_layers=2 \
    dataset.loader.parameters.adjacency_threshold=0.0911 \
    trainer.devices=\[6\] \
    &

python -m ogbench \
    model=chebnet \
    dataset=motrpac \
    seed=42 \
    logger.wandb.tags=[chebnet,motrpac,hpsearch,rerun] \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    experiment=no_readout \
    dataset.loader.parameters.adjacency_method=string \
    dataset.loader.parameters.node_sample_ratio=1.0 \
    dataset.loader.parameters.method=variance \
    optimizer.parameters.lr=0.0001 \
    optimizer.parameters.weight_decay=0.0 \
    model.backbone.dropout=0.1 \
    model.feature_encoder.out_channels=16 \
    model.backbone.num_layers=2 \
    dataset.loader.parameters.adjacency_threshold=0.4 \
    trainer.devices=\[7\] \
    &

python -m ogbench \
    model=chebnet \
    dataset=motrpac \
    seed=42 \
    logger.wandb.tags=[chebnet,motrpac,hpsearch,rerun] \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    experiment=no_readout \
    dataset.loader.parameters.adjacency_method=string \
    dataset.loader.parameters.node_sample_ratio=1.0 \
    dataset.loader.parameters.method=random \
    optimizer.parameters.lr=0.0001 \
    optimizer.parameters.weight_decay=0.0 \
    model.backbone.dropout=0.1 \
    model.feature_encoder.out_channels=16 \
    model.backbone.num_layers=2 \
    dataset.loader.parameters.adjacency_threshold=0.4 \
    trainer.devices=\[0\] \
    &

python -m ogbench \
    model=chebnet \
    dataset=motrpac \
    seed=123 \
    logger.wandb.tags=[chebnet,motrpac,hpsearch,rerun] \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    experiment=no_readout \
    dataset.loader.parameters.adjacency_method=string \
    dataset.loader.parameters.node_sample_ratio=1.0 \
    dataset.loader.parameters.method=random \
    optimizer.parameters.lr=0.0001 \
    optimizer.parameters.weight_decay=0.0 \
    model.backbone.dropout=0.1 \
    model.feature_encoder.out_channels=16 \
    model.backbone.num_layers=2 \
    dataset.loader.parameters.adjacency_threshold=0.4 \
    trainer.devices=\[1\] \
    &

python -m ogbench \
    model=chebnet \
    dataset=motrpac \
    seed=456 \
    logger.wandb.tags=[chebnet,motrpac,hpsearch,rerun] \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    experiment=no_readout \
    dataset.loader.parameters.adjacency_method=string \
    dataset.loader.parameters.node_sample_ratio=1.0 \
    dataset.loader.parameters.method=correlation \
    optimizer.parameters.lr=0.001 \
    optimizer.parameters.weight_decay=0.0 \
    model.backbone.dropout=0.1 \
    model.feature_encoder.out_channels=16 \
    model.backbone.num_layers=2 \
    dataset.loader.parameters.adjacency_threshold=0.4 \
    trainer.devices=\[2\] \
    &

python -m ogbench \
    model=chebnet \
    dataset=motrpac \
    seed=123 \
    logger.wandb.tags=[chebnet,motrpac,hpsearch,rerun] \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    experiment=no_readout \
    dataset.loader.parameters.adjacency_method=string \
    dataset.loader.parameters.node_sample_ratio=1.0 \
    dataset.loader.parameters.method=correlation \
    optimizer.parameters.lr=0.001 \
    optimizer.parameters.weight_decay=0.0 \
    model.backbone.dropout=0.1 \
    model.feature_encoder.out_channels=64 \
    model.backbone.num_layers=4 \
    dataset.loader.parameters.adjacency_threshold=0.4 \
    trainer.devices=\[3\] \
    &

python -m ogbench \
    model=chebnet \
    dataset=motrpac \
    seed=123 \
    logger.wandb.tags=[chebnet,motrpac,hpsearch,rerun] \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    experiment=no_readout \
    dataset.loader.parameters.adjacency_method=string \
    dataset.loader.parameters.node_sample_ratio=0.8 \
    dataset.loader.parameters.method=variance \
    optimizer.parameters.lr=0.001 \
    optimizer.parameters.weight_decay=0.0 \
    model.backbone.dropout=0.1 \
    model.feature_encoder.out_channels=64 \
    model.backbone.num_layers=4 \
    dataset.loader.parameters.adjacency_threshold=0.4 \
    trainer.devices=\[4\] \
    &

python -m ogbench \
    model=chebnet \
    dataset=motrpac \
    seed=123 \
    logger.wandb.tags=[chebnet,motrpac,hpsearch,rerun] \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    experiment=no_readout \
    dataset.loader.parameters.adjacency_method=string \
    dataset.loader.parameters.node_sample_ratio=0.8 \
    dataset.loader.parameters.method=distance_correlation \
    optimizer.parameters.lr=0.001 \
    optimizer.parameters.weight_decay=0.0 \
    model.backbone.dropout=0.1 \
    model.feature_encoder.out_channels=16 \
    model.backbone.num_layers=4 \
    dataset.loader.parameters.adjacency_threshold=0.4 \
    trainer.devices=\[5\] \
    &

python -m ogbench \
    model=chebnet \
    dataset=motrpac \
    seed=456 \
    logger.wandb.tags=[chebnet,motrpac,hpsearch,rerun] \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    experiment=omics_readout \
    dataset.loader.parameters.adjacency_method=wgcna \
    dataset.loader.parameters.node_sample_ratio=1.0 \
    dataset.loader.parameters.method=random \
    optimizer.parameters.lr=0.0001 \
    optimizer.parameters.weight_decay=0.0 \
    model.backbone.dropout=0.1 \
    model.feature_encoder.out_channels=64 \
    model.backbone.num_layers=2 \
    dataset.loader.parameters.adjacency_threshold=0.0631 \
    model.readout.fc_dropout=0.1 \
    trainer.devices=\[6\] \
    &

python -m ogbench \
    model=chebnet \
    dataset=motrpac \
    seed=123 \
    logger.wandb.tags=[chebnet,motrpac,hpsearch,rerun] \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    experiment=omics_readout \
    dataset.loader.parameters.adjacency_method=wgcna \
    dataset.loader.parameters.node_sample_ratio=1.0 \
    dataset.loader.parameters.method=distance_correlation \
    optimizer.parameters.lr=0.001 \
    optimizer.parameters.weight_decay=0.0 \
    model.backbone.dropout=0.1 \
    model.feature_encoder.out_channels=64 \
    model.backbone.num_layers=4 \
    dataset.loader.parameters.adjacency_threshold=0.1061 \
    model.readout.fc_dropout=0.1 \
    trainer.devices=\[7\] \
    &

python -m ogbench \
    model=chebnet \
    dataset=motrpac \
    seed=123 \
    logger.wandb.tags=[chebnet,motrpac,hpsearch,rerun] \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    experiment=omics_readout \
    dataset.loader.parameters.adjacency_method=string \
    dataset.loader.parameters.node_sample_ratio=1.0 \
    dataset.loader.parameters.method=variance \
    optimizer.parameters.lr=0.001 \
    optimizer.parameters.weight_decay=0.0 \
    model.backbone.dropout=0.1 \
    model.feature_encoder.out_channels=16 \
    model.backbone.num_layers=4 \
    dataset.loader.parameters.adjacency_threshold=0.4 \
    model.readout.fc_dropout=0.1 \
    trainer.devices=\[0\] \
    &

python -m ogbench \
    model=chebnet \
    dataset=motrpac \
    seed=456 \
    logger.wandb.tags=[chebnet,motrpac,hpsearch,rerun] \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    experiment=omics_readout \
    dataset.loader.parameters.adjacency_method=string \
    dataset.loader.parameters.node_sample_ratio=1.0 \
    dataset.loader.parameters.method=correlation \
    optimizer.parameters.lr=0.0001 \
    optimizer.parameters.weight_decay=0.0 \
    model.backbone.dropout=0.1 \
    model.feature_encoder.out_channels=16 \
    model.backbone.num_layers=2 \
    dataset.loader.parameters.adjacency_threshold=0.4 \
    model.readout.fc_dropout=0.1 \
    trainer.devices=\[1\] \
    &

python -m ogbench \
    model=chebnet \
    dataset=motrpac \
    seed=123 \
    logger.wandb.tags=[chebnet,motrpac,hpsearch,rerun] \
    logger.wandb.project=bgbench_dataset_grid_search_final \
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
    trainer.devices=\[2\] \
    &

python -m ogbench \
    model=chebnet \
    dataset=motrpac \
    seed=42 \
    logger.wandb.tags=[chebnet,motrpac,hpsearch,rerun] \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    experiment=omics_readout \
    dataset.loader.parameters.adjacency_method=string \
    dataset.loader.parameters.node_sample_ratio=1.0 \
    dataset.loader.parameters.method=distance_correlation \
    optimizer.parameters.lr=0.001 \
    optimizer.parameters.weight_decay=0.0 \
    model.backbone.dropout=0.1 \
    model.feature_encoder.out_channels=16 \
    model.backbone.num_layers=4 \
    dataset.loader.parameters.adjacency_threshold=0.4 \
    model.readout.fc_dropout=0.1 \
    trainer.devices=\[3\] \
    &

python -m ogbench \
    model=chebnet \
    dataset=motrpac \
    seed=42 \
    logger.wandb.tags=[chebnet,motrpac,hpsearch,rerun] \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    experiment=omics_readout \
    dataset.loader.parameters.adjacency_method=string \
    dataset.loader.parameters.node_sample_ratio=0.8 \
    dataset.loader.parameters.method=random \
    optimizer.parameters.lr=0.001 \
    optimizer.parameters.weight_decay=0.0 \
    model.backbone.dropout=0.1 \
    model.feature_encoder.out_channels=16 \
    model.backbone.num_layers=2 \
    dataset.loader.parameters.adjacency_threshold=0.4 \
    model.readout.fc_dropout=0.1 \
    trainer.devices=\[4\] \
    &

python -m ogbench \
    model=chebnet \
    dataset=motrpac \
    seed=42 \
    logger.wandb.tags=[chebnet,motrpac,hpsearch,rerun] \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    experiment=omics_readout \
    dataset.loader.parameters.adjacency_method=string \
    dataset.loader.parameters.node_sample_ratio=0.8 \
    dataset.loader.parameters.method=distance_correlation \
    optimizer.parameters.lr=0.0001 \
    optimizer.parameters.weight_decay=0.0 \
    model.backbone.dropout=0.1 \
    model.feature_encoder.out_channels=16 \
    model.backbone.num_layers=4 \
    dataset.loader.parameters.adjacency_threshold=0.4 \
    model.readout.fc_dropout=0.1 \
    trainer.devices=\[5\] \
    &

python -m ogbench \
    model=chebnet \
    dataset=brca \
    seed=456 \
    logger.wandb.tags=[chebnet,brca,hpsearch,rerun] \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    experiment=no_readout \
    dataset.loader.parameters.adjacency_method=wgcna \
    dataset.loader.parameters.node_sample_ratio=1.0 \
    dataset.loader.parameters.method=variance \
    optimizer.parameters.lr=0.001 \
    optimizer.parameters.weight_decay=0.0 \
    model.backbone.dropout=0.1 \
    model.feature_encoder.out_channels=16 \
    model.backbone.num_layers=4 \
    dataset.loader.parameters.adjacency_threshold=0.0009 \
    trainer.devices=\[6\] \
    &

python -m ogbench \
    model=chebnet \
    dataset=brca \
    seed=456 \
    logger.wandb.tags=[chebnet,brca,hpsearch,rerun] \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    experiment=no_readout \
    dataset.loader.parameters.adjacency_method=wgcna \
    dataset.loader.parameters.node_sample_ratio=1.0 \
    dataset.loader.parameters.method=distance_correlation \
    optimizer.parameters.lr=0.001 \
    optimizer.parameters.weight_decay=0.0 \
    model.backbone.dropout=0.1 \
    model.feature_encoder.out_channels=16 \
    model.backbone.num_layers=4 \
    dataset.loader.parameters.adjacency_threshold=0.0008 \
    trainer.devices=\[7\] \
    &

python -m ogbench \
    model=chebnet \
    dataset=brca \
    seed=456 \
    logger.wandb.tags=[chebnet,brca,hpsearch,rerun] \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    experiment=no_readout \
    dataset.loader.parameters.adjacency_method=wgcna \
    dataset.loader.parameters.node_sample_ratio=0.8 \
    dataset.loader.parameters.method=correlation \
    optimizer.parameters.lr=0.0001 \
    optimizer.parameters.weight_decay=0.0 \
    model.backbone.dropout=0.1 \
    model.feature_encoder.out_channels=16 \
    model.backbone.num_layers=2 \
    dataset.loader.parameters.adjacency_threshold=0.0005 \
    trainer.devices=\[0\] \
    &

python -m ogbench \
    model=chebnet \
    dataset=brca \
    seed=42 \
    logger.wandb.tags=[chebnet,brca,hpsearch,rerun] \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    experiment=no_readout \
    dataset.loader.parameters.adjacency_method=string \
    dataset.loader.parameters.node_sample_ratio=1.0 \
    dataset.loader.parameters.method=correlation \
    optimizer.parameters.lr=0.001 \
    optimizer.parameters.weight_decay=0.0 \
    model.backbone.dropout=0.1 \
    model.feature_encoder.out_channels=16 \
    model.backbone.num_layers=2 \
    dataset.loader.parameters.adjacency_threshold=0.4 \
    trainer.devices=\[1\] \
    &

python -m ogbench \
    model=chebnet \
    dataset=brca \
    seed=456 \
    logger.wandb.tags=[chebnet,brca,hpsearch,rerun] \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    experiment=no_readout \
    dataset.loader.parameters.adjacency_method=string \
    dataset.loader.parameters.node_sample_ratio=1.0 \
    dataset.loader.parameters.method=distance_correlation \
    optimizer.parameters.lr=0.0001 \
    optimizer.parameters.weight_decay=0.0 \
    model.backbone.dropout=0.1 \
    model.feature_encoder.out_channels=64 \
    model.backbone.num_layers=2 \
    dataset.loader.parameters.adjacency_threshold=0.4 \
    trainer.devices=\[2\] \
    &

python -m ogbench \
    model=chebnet \
    dataset=brca \
    seed=123 \
    logger.wandb.tags=[chebnet,brca,hpsearch,rerun] \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    experiment=no_readout \
    dataset.loader.parameters.adjacency_method=string \
    dataset.loader.parameters.node_sample_ratio=1.0 \
    dataset.loader.parameters.method=distance_correlation \
    optimizer.parameters.lr=0.0001 \
    optimizer.parameters.weight_decay=0.0 \
    model.backbone.dropout=0.1 \
    model.feature_encoder.out_channels=64 \
    model.backbone.num_layers=4 \
    dataset.loader.parameters.adjacency_threshold=0.4 \
    trainer.devices=\[3\] \
    &

python -m ogbench \
    model=chebnet \
    dataset=brca \
    seed=123 \
    logger.wandb.tags=[chebnet,brca,hpsearch,rerun] \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    experiment=no_readout \
    dataset.loader.parameters.adjacency_method=string \
    dataset.loader.parameters.node_sample_ratio=0.8 \
    dataset.loader.parameters.method=variance \
    optimizer.parameters.lr=0.0001 \
    optimizer.parameters.weight_decay=0.0 \
    model.backbone.dropout=0.1 \
    model.feature_encoder.out_channels=16 \
    model.backbone.num_layers=2 \
    dataset.loader.parameters.adjacency_threshold=0.4 \
    trainer.devices=\[4\] \
    &

python -m ogbench \
    model=chebnet \
    dataset=brca \
    seed=42 \
    logger.wandb.tags=[chebnet,brca,hpsearch,rerun] \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    experiment=omics_readout \
    dataset.loader.parameters.adjacency_method=wgcna \
    dataset.loader.parameters.node_sample_ratio=1.0 \
    dataset.loader.parameters.method=random \
    optimizer.parameters.lr=0.0001 \
    optimizer.parameters.weight_decay=0.0 \
    model.backbone.dropout=0.1 \
    model.feature_encoder.out_channels=16 \
    model.backbone.num_layers=2 \
    dataset.loader.parameters.adjacency_threshold=0.0012 \
    model.readout.fc_dropout=0.1 \
    trainer.devices=\[5\] \
    &

python -m ogbench \
    model=chebnet \
    dataset=brca \
    seed=123 \
    logger.wandb.tags=[chebnet,brca,hpsearch,rerun] \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    experiment=omics_readout \
    dataset.loader.parameters.adjacency_method=wgcna \
    dataset.loader.parameters.node_sample_ratio=1.0 \
    dataset.loader.parameters.method=correlation \
    optimizer.parameters.lr=0.0001 \
    optimizer.parameters.weight_decay=0.0 \
    model.backbone.dropout=0.1 \
    model.feature_encoder.out_channels=64 \
    model.backbone.num_layers=2 \
    dataset.loader.parameters.adjacency_threshold=0.0008 \
    model.readout.fc_dropout=0.1 \
    trainer.devices=\[6\] \
    &

python -m ogbench \
    model=chebnet \
    dataset=brca \
    seed=42 \
    logger.wandb.tags=[chebnet,brca,hpsearch,rerun] \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    experiment=omics_readout \
    dataset.loader.parameters.adjacency_method=wgcna \
    dataset.loader.parameters.node_sample_ratio=0.8 \
    dataset.loader.parameters.method=variance \
    optimizer.parameters.lr=0.0001 \
    optimizer.parameters.weight_decay=0.0 \
    model.backbone.dropout=0.1 \
    model.feature_encoder.out_channels=16 \
    model.backbone.num_layers=2 \
    dataset.loader.parameters.adjacency_threshold=0.0005 \
    model.readout.fc_dropout=0.1 \
    trainer.devices=\[7\] \
    &

python -m ogbench \
    model=chebnet \
    dataset=brca \
    seed=456 \
    logger.wandb.tags=[chebnet,brca,hpsearch,rerun] \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    experiment=omics_readout \
    dataset.loader.parameters.adjacency_method=wgcna \
    dataset.loader.parameters.node_sample_ratio=0.8 \
    dataset.loader.parameters.method=correlation \
    optimizer.parameters.lr=0.001 \
    optimizer.parameters.weight_decay=0.0 \
    model.backbone.dropout=0.1 \
    model.feature_encoder.out_channels=64 \
    model.backbone.num_layers=2 \
    dataset.loader.parameters.adjacency_threshold=0.0005 \
    model.readout.fc_dropout=0.1 \
    trainer.devices=\[0\] \
    &

python -m ogbench \
    model=chebnet \
    dataset=brca \
    seed=456 \
    logger.wandb.tags=[chebnet,brca,hpsearch,rerun] \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    experiment=omics_readout \
    dataset.loader.parameters.adjacency_method=wgcna \
    dataset.loader.parameters.node_sample_ratio=0.8 \
    dataset.loader.parameters.method=distance_correlation \
    optimizer.parameters.lr=0.001 \
    optimizer.parameters.weight_decay=0.0 \
    model.backbone.dropout=0.1 \
    model.feature_encoder.out_channels=16 \
    model.backbone.num_layers=2 \
    dataset.loader.parameters.adjacency_threshold=0.0006 \
    model.readout.fc_dropout=0.1 \
    trainer.devices=\[1\] \
    &

python -m ogbench \
    model=chebnet \
    dataset=brca \
    seed=456 \
    logger.wandb.tags=[chebnet,brca,hpsearch,rerun] \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    experiment=omics_readout \
    dataset.loader.parameters.adjacency_method=string \
    dataset.loader.parameters.node_sample_ratio=1.0 \
    dataset.loader.parameters.method=variance \
    optimizer.parameters.lr=0.0001 \
    optimizer.parameters.weight_decay=0.0 \
    model.backbone.dropout=0.1 \
    model.feature_encoder.out_channels=64 \
    model.backbone.num_layers=4 \
    dataset.loader.parameters.adjacency_threshold=0.4 \
    model.readout.fc_dropout=0.1 \
    trainer.devices=\[2\] \
    &

python -m ogbench \
    model=chebnet \
    dataset=brca \
    seed=42 \
    logger.wandb.tags=[chebnet,brca,hpsearch,rerun] \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    experiment=omics_readout \
    dataset.loader.parameters.adjacency_method=string \
    dataset.loader.parameters.node_sample_ratio=1.0 \
    dataset.loader.parameters.method=random \
    optimizer.parameters.lr=0.0001 \
    optimizer.parameters.weight_decay=0.0 \
    model.backbone.dropout=0.1 \
    model.feature_encoder.out_channels=64 \
    model.backbone.num_layers=4 \
    dataset.loader.parameters.adjacency_threshold=0.4 \
    model.readout.fc_dropout=0.1 \
    trainer.devices=\[3\] \
    &

python -m ogbench \
    model=chebnet \
    dataset=brca \
    seed=123 \
    logger.wandb.tags=[chebnet,brca,hpsearch,rerun] \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    experiment=omics_readout \
    dataset.loader.parameters.adjacency_method=string \
    dataset.loader.parameters.node_sample_ratio=1.0 \
    dataset.loader.parameters.method=random \
    optimizer.parameters.lr=0.001 \
    optimizer.parameters.weight_decay=0.0 \
    model.backbone.dropout=0.1 \
    model.feature_encoder.out_channels=16 \
    model.backbone.num_layers=4 \
    dataset.loader.parameters.adjacency_threshold=0.4 \
    model.readout.fc_dropout=0.1 \
    trainer.devices=\[4\] \
    &

python -m ogbench \
    model=chebnet \
    dataset=brca \
    seed=123 \
    logger.wandb.tags=[chebnet,brca,hpsearch,rerun] \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    experiment=omics_readout \
    dataset.loader.parameters.adjacency_method=string \
    dataset.loader.parameters.node_sample_ratio=1.0 \
    dataset.loader.parameters.method=distance_correlation \
    optimizer.parameters.lr=0.001 \
    optimizer.parameters.weight_decay=0.0 \
    model.backbone.dropout=0.1 \
    model.feature_encoder.out_channels=64 \
    model.backbone.num_layers=4 \
    dataset.loader.parameters.adjacency_threshold=0.4 \
    model.readout.fc_dropout=0.1 \
    trainer.devices=\[5\] \
    &

python -m ogbench \
    model=chebnet \
    dataset=brca \
    seed=456 \
    logger.wandb.tags=[chebnet,brca,hpsearch,rerun] \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    experiment=omics_readout \
    dataset.loader.parameters.adjacency_method=string \
    dataset.loader.parameters.node_sample_ratio=0.8 \
    dataset.loader.parameters.method=correlation \
    optimizer.parameters.lr=0.0001 \
    optimizer.parameters.weight_decay=0.0 \
    model.backbone.dropout=0.1 \
    model.feature_encoder.out_channels=16 \
    model.backbone.num_layers=4 \
    dataset.loader.parameters.adjacency_threshold=0.4 \
    model.readout.fc_dropout=0.1 \
    trainer.devices=\[6\] \
    &

python -m ogbench \
    model=gps \
    dataset=parkinsons \
    seed=123 \
    logger.wandb.tags=[gps,parkinsons,hpsearch,rerun] \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    experiment=no_readout \
    dataset.loader.parameters.adjacency_method=string \
    dataset.loader.parameters.node_sample_ratio=1.0 \
    dataset.loader.parameters.method=random \
    optimizer.parameters.lr=0.0001 \
    optimizer.parameters.weight_decay=0.0 \
    model.backbone.dropout=0.1 \
    model.feature_encoder.out_channels=64 \
    model.backbone.num_layers=2 \
    model.encodings=[RWSE] \
    dataset.loader.parameters.adjacency_threshold=0.4 \
    trainer.devices=\[7\] \
    &

python -m ogbench \
    model=gps \
    dataset=parkinsons \
    seed=456 \
    logger.wandb.tags=[gps,parkinsons,hpsearch,rerun] \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    experiment=omics_readout \
    dataset.loader.parameters.adjacency_method=wgcna \
    dataset.loader.parameters.node_sample_ratio=1.0 \
    dataset.loader.parameters.method=variance \
    optimizer.parameters.lr=0.001 \
    optimizer.parameters.weight_decay=0.0 \
    model.backbone.dropout=0.1 \
    model.feature_encoder.out_channels=16 \
    model.backbone.num_layers=2 \
    model.encodings=[RWSE] \
    dataset.loader.parameters.adjacency_threshold=0.0346 \
    model.readout.fc_dropout=0.1 \
    trainer.devices=\[0\] \
    &

python -m ogbench \
    model=gps \
    dataset=addneuromed \
    seed=42 \
    logger.wandb.tags=[gps,addneuromed,hpsearch,rerun] \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    experiment=no_readout \
    dataset.loader.parameters.adjacency_method=wgcna \
    dataset.loader.parameters.node_sample_ratio=1.0 \
    dataset.loader.parameters.method=random \
    optimizer.parameters.lr=0.0001 \
    optimizer.parameters.weight_decay=0.0 \
    model.backbone.dropout=0.1 \
    model.feature_encoder.out_channels=64 \
    model.backbone.num_layers=4 \
    model.encodings=[RWSE] \
    dataset.loader.parameters.adjacency_threshold=0.0005 \
    trainer.devices=\[1\] \
    &

python -m ogbench \
    model=gps \
    dataset=addneuromed \
    seed=42 \
    logger.wandb.tags=[gps,addneuromed,hpsearch,rerun] \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    experiment=no_readout \
    dataset.loader.parameters.adjacency_method=string \
    dataset.loader.parameters.node_sample_ratio=0.8 \
    dataset.loader.parameters.method=distance_correlation \
    optimizer.parameters.lr=0.0001 \
    optimizer.parameters.weight_decay=0.0 \
    model.backbone.dropout=0.1 \
    model.feature_encoder.out_channels=16 \
    model.backbone.num_layers=2 \
    model.encodings=[LapPE] \
    dataset.loader.parameters.adjacency_threshold=0.4 \
    trainer.devices=\[2\] \
    &

python -m ogbench \
    model=gps \
    dataset=addneuromed \
    seed=456 \
    logger.wandb.tags=[gps,addneuromed,hpsearch,rerun] \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    experiment=no_readout \
    dataset.loader.parameters.adjacency_method=string \
    dataset.loader.parameters.node_sample_ratio=0.8 \
    dataset.loader.parameters.method=distance_correlation \
    optimizer.parameters.lr=0.0001 \
    optimizer.parameters.weight_decay=0.0 \
    model.backbone.dropout=0.1 \
    model.feature_encoder.out_channels=16 \
    model.backbone.num_layers=2 \
    model.encodings=[RWSE] \
    dataset.loader.parameters.adjacency_threshold=0.4 \
    trainer.devices=\[3\] \
    &

python -m ogbench \
    model=gps \
    dataset=addneuromed \
    seed=123 \
    logger.wandb.tags=[gps,addneuromed,hpsearch,rerun] \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    experiment=no_readout \
    dataset.loader.parameters.adjacency_method=string \
    dataset.loader.parameters.node_sample_ratio=0.8 \
    dataset.loader.parameters.method=distance_correlation \
    optimizer.parameters.lr=0.001 \
    optimizer.parameters.weight_decay=0.0 \
    model.backbone.dropout=0.1 \
    model.feature_encoder.out_channels=64 \
    model.backbone.num_layers=4 \
    model.encodings=[RWSE] \
    dataset.loader.parameters.adjacency_threshold=0.4 \
    trainer.devices=\[4\] \
    &

python -m ogbench \
    model=gps \
    dataset=addneuromed \
    seed=42 \
    logger.wandb.tags=[gps,addneuromed,hpsearch,rerun] \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    experiment=omics_readout \
    dataset.loader.parameters.adjacency_method=wgcna \
    dataset.loader.parameters.node_sample_ratio=1.0 \
    dataset.loader.parameters.method=correlation \
    optimizer.parameters.lr=0.001 \
    optimizer.parameters.weight_decay=0.0 \
    model.backbone.dropout=0.1 \
    model.feature_encoder.out_channels=64 \
    model.backbone.num_layers=4 \
    model.encodings=[LapPE] \
    dataset.loader.parameters.adjacency_threshold=0.0997 \
    model.readout.fc_dropout=0.1 \
    trainer.devices=\[5\] \
    &

python -m ogbench \
    model=gps \
    dataset=motrpac \
    seed=42 \
    logger.wandb.tags=[gps,motrpac,hpsearch,rerun] \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    experiment=no_readout \
    dataset.loader.parameters.adjacency_method=wgcna \
    dataset.loader.parameters.node_sample_ratio=1.0 \
    dataset.loader.parameters.method=random \
    optimizer.parameters.lr=0.001 \
    optimizer.parameters.weight_decay=0.0 \
    model.backbone.dropout=0.1 \
    model.feature_encoder.out_channels=64 \
    model.backbone.num_layers=2 \
    model.encodings=[LapPE] \
    dataset.loader.parameters.adjacency_threshold=0.0631 \
    trainer.devices=\[6\] \
    &

python -m ogbench \
    model=gps \
    dataset=motrpac \
    seed=456 \
    logger.wandb.tags=[gps,motrpac,hpsearch,rerun] \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    experiment=no_readout \
    dataset.loader.parameters.adjacency_method=wgcna \
    dataset.loader.parameters.node_sample_ratio=1.0 \
    dataset.loader.parameters.method=random \
    optimizer.parameters.lr=0.001 \
    optimizer.parameters.weight_decay=0.0 \
    model.backbone.dropout=0.1 \
    model.feature_encoder.out_channels=64 \
    model.backbone.num_layers=4 \
    model.encodings=[RWSE] \
    dataset.loader.parameters.adjacency_threshold=0.0631 \
    trainer.devices=\[7\] \
    &
# SKIP run_id=tajm6p3n (no Hydra args list in W&B summary or metadata)

python -m ogbench \
    model=gps \
    dataset=motrpac \
    seed=456 \
    logger.wandb.tags=[gps,motrpac,hpsearch,rerun] \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    experiment=no_readout \
    dataset.loader.parameters.adjacency_method=string \
    dataset.loader.parameters.node_sample_ratio=1.0 \
    dataset.loader.parameters.method=correlation \
    optimizer.parameters.lr=0.0001 \
    optimizer.parameters.weight_decay=0.0 \
    model.backbone.dropout=0.1 \
    model.feature_encoder.out_channels=64 \
    model.backbone.num_layers=4 \
    model.encodings=[RWSE] \
    dataset.loader.parameters.adjacency_threshold=0.4 \
    trainer.devices=\[1\] \
    &

python -m ogbench \
    model=gps \
    dataset=motrpac \
    seed=123 \
    logger.wandb.tags=[gps,motrpac,hpsearch,rerun] \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    experiment=no_readout \
    dataset.loader.parameters.adjacency_method=string \
    dataset.loader.parameters.node_sample_ratio=0.8 \
    dataset.loader.parameters.method=distance_correlation \
    optimizer.parameters.lr=0.001 \
    optimizer.parameters.weight_decay=0.0 \
    model.backbone.dropout=0.1 \
    model.feature_encoder.out_channels=16 \
    model.backbone.num_layers=4 \
    model.encodings=[LapPE] \
    dataset.loader.parameters.adjacency_threshold=0.4 \
    trainer.devices=\[2\] \
    &

python -m ogbench \
    model=gatv2 \
    dataset=parkinsons \
    seed=123 \
    logger.wandb.tags=[gatv2,parkinsons,hpsearch,rerun] \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    experiment=no_readout \
    dataset.loader.parameters.adjacency_method=wgcna \
    dataset.loader.parameters.node_sample_ratio=0.5 \
    dataset.loader.parameters.method=random \
    optimizer.parameters.lr=0.001 \
    optimizer.parameters.weight_decay=0.0 \
    model.backbone.dropout=0.1 \
    model.feature_encoder.out_channels=64 \
    model.backbone.num_layers=4 \
    model.backbone.heads=4 \
    dataset.loader.parameters.adjacency_threshold=0.0005 \
    trainer.devices=\[3\] \
    &

python -m ogbench \
    model=gatv2 \
    dataset=addneuromed \
    seed=42 \
    logger.wandb.tags=[gatv2,addneuromed,hpsearch,rerun] \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    experiment=no_readout \
    dataset.loader.parameters.adjacency_method=wgcna \
    dataset.loader.parameters.node_sample_ratio=0.5 \
    dataset.loader.parameters.method=random \
    optimizer.parameters.lr=0.0001 \
    optimizer.parameters.weight_decay=0.0 \
    model.backbone.dropout=0.1 \
    model.feature_encoder.out_channels=64 \
    model.backbone.num_layers=4 \
    model.backbone.heads=2 \
    dataset.loader.parameters.adjacency_threshold=0.0005 \
    trainer.devices=\[4\] \
    &

python -m ogbench \
    model=gatv2 \
    dataset=addneuromed \
    seed=42 \
    logger.wandb.tags=[gatv2,addneuromed,hpsearch,rerun] \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    experiment=no_readout \
    dataset.loader.parameters.adjacency_method=wgcna \
    dataset.loader.parameters.node_sample_ratio=0.5 \
    dataset.loader.parameters.method=random \
    optimizer.parameters.lr=0.0001 \
    optimizer.parameters.weight_decay=0.0 \
    model.backbone.dropout=0.1 \
    model.feature_encoder.out_channels=64 \
    model.backbone.num_layers=4 \
    model.backbone.heads=4 \
    dataset.loader.parameters.adjacency_threshold=0.0005 \
    trainer.devices=\[5\] \
    &

python -m ogbench \
    model=gatv2 \
    dataset=addneuromed \
    seed=42 \
    logger.wandb.tags=[gatv2,addneuromed,hpsearch,rerun] \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    experiment=no_readout \
    dataset.loader.parameters.adjacency_method=wgcna \
    dataset.loader.parameters.node_sample_ratio=0.5 \
    dataset.loader.parameters.method=random \
    optimizer.parameters.lr=0.001 \
    optimizer.parameters.weight_decay=0.0 \
    model.backbone.dropout=0.1 \
    model.feature_encoder.out_channels=64 \
    model.backbone.num_layers=4 \
    model.backbone.heads=2 \
    dataset.loader.parameters.adjacency_threshold=0.0005 \
    trainer.devices=\[6\] \
    &

python -m ogbench \
    model=gatv2 \
    dataset=addneuromed \
    seed=42 \
    logger.wandb.tags=[gatv2,addneuromed,hpsearch,rerun] \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    experiment=no_readout \
    dataset.loader.parameters.adjacency_method=wgcna \
    dataset.loader.parameters.node_sample_ratio=0.5 \
    dataset.loader.parameters.method=random \
    optimizer.parameters.lr=0.001 \
    optimizer.parameters.weight_decay=0.0 \
    model.backbone.dropout=0.1 \
    model.feature_encoder.out_channels=64 \
    model.backbone.num_layers=4 \
    model.backbone.heads=4 \
    dataset.loader.parameters.adjacency_threshold=0.0005 \
    trainer.devices=\[7\] \
    &

python -m ogbench \
    model=gatv2 \
    dataset=addneuromed \
    seed=123 \
    logger.wandb.tags=[gatv2,addneuromed,hpsearch,rerun] \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    experiment=no_readout \
    dataset.loader.parameters.adjacency_method=wgcna \
    dataset.loader.parameters.node_sample_ratio=0.5 \
    dataset.loader.parameters.method=random \
    optimizer.parameters.lr=0.001 \
    optimizer.parameters.weight_decay=0.0 \
    model.backbone.dropout=0.1 \
    model.feature_encoder.out_channels=64 \
    model.backbone.num_layers=4 \
    model.backbone.heads=4 \
    dataset.loader.parameters.adjacency_threshold=0.0005 \
    trainer.devices=\[0\] \
    &
# SKIP run_id=2ybj0ftl (no Hydra args list in W&B summary or metadata)
# SKIP run_id=9gfb25yg (no Hydra args list in W&B summary or metadata)
# SKIP run_id=0i97uqx4 (no Hydra args list in W&B summary or metadata)
# SKIP run_id=1tv2oiat (no Hydra args list in W&B summary or metadata)

python -m ogbench \
    model=gps \
    dataset=addneuromed \
    seed=456 \
    logger.wandb.tags=[gps,addneuromed,hpsearch,rerun] \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    experiment=no_readout \
    dataset.loader.parameters.adjacency_method=wgcna \
    dataset.loader.parameters.node_sample_ratio=0.5 \
    dataset.loader.parameters.method=random \
    optimizer.parameters.lr=0.001 \
    optimizer.parameters.weight_decay=0.0 \
    model.backbone.dropout=0.1 \
    model.feature_encoder.out_channels=64 \
    model.backbone.num_layers=4 \
    model.encodings=[LapPE] \
    dataset.loader.parameters.adjacency_threshold=0.0005 \
    trainer.devices=\[5\] \
    &

python -m ogbench \
    model=gps \
    dataset=addneuromed \
    seed=123 \
    logger.wandb.tags=[gps,addneuromed,hpsearch,rerun] \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    experiment=no_readout \
    dataset.loader.parameters.adjacency_method=wgcna \
    dataset.loader.parameters.node_sample_ratio=0.5 \
    dataset.loader.parameters.method=random \
    optimizer.parameters.lr=0.001 \
    optimizer.parameters.weight_decay=0.0 \
    model.backbone.dropout=0.1 \
    model.feature_encoder.out_channels=64 \
    model.backbone.num_layers=4 \
    model.encodings=[RWSE] \
    dataset.loader.parameters.adjacency_threshold=0.0005 \
    trainer.devices=\[6\] \
    &

python -m ogbench \
    model=gps \
    dataset=addneuromed \
    seed=123 \
    logger.wandb.tags=[gps,addneuromed,hpsearch,rerun] \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    experiment=no_readout \
    dataset.loader.parameters.adjacency_method=wgcna \
    dataset.loader.parameters.node_sample_ratio=0.5 \
    dataset.loader.parameters.method=correlation \
    optimizer.parameters.lr=0.001 \
    optimizer.parameters.weight_decay=0.0 \
    model.backbone.dropout=0.1 \
    model.feature_encoder.out_channels=16 \
    model.backbone.num_layers=4 \
    model.encodings=[RWSE] \
    dataset.loader.parameters.adjacency_threshold=0.0971 \
    trainer.devices=\[7\] \
    &

python -m ogbench \
    model=gps \
    dataset=addneuromed \
    seed=123 \
    logger.wandb.tags=[gps,addneuromed,hpsearch,rerun] \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    experiment=omics_readout \
    dataset.loader.parameters.adjacency_method=wgcna \
    dataset.loader.parameters.node_sample_ratio=0.5 \
    dataset.loader.parameters.method=correlation \
    optimizer.parameters.lr=0.0001 \
    optimizer.parameters.weight_decay=0.0 \
    model.backbone.dropout=0.1 \
    model.feature_encoder.out_channels=64 \
    model.backbone.num_layers=4 \
    model.encodings=[LapPE] \
    dataset.loader.parameters.adjacency_threshold=0.0971 \
    model.readout.fc_dropout=0.1 \
    trainer.devices=\[0\] \
    &

python -m ogbench \
    model=gatv2 \
    dataset=addneuromed \
    seed=456 \
    logger.wandb.tags=[gatv2,addneuromed,hpsearch,rerun] \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    experiment=no_readout \
    dataset.loader.parameters.adjacency_method=wgcna \
    dataset.loader.parameters.node_sample_ratio=0.3 \
    dataset.loader.parameters.method=variance \
    optimizer.parameters.lr=0.0001 \
    optimizer.parameters.weight_decay=0.0 \
    model.backbone.dropout=0.1 \
    model.feature_encoder.out_channels=16 \
    model.backbone.num_layers=4 \
    model.backbone.heads=4 \
    dataset.loader.parameters.adjacency_threshold=0.2378 \
    trainer.devices=\[1\] \
    &

python -m ogbench \
    model=gatv2 \
    dataset=addneuromed \
    seed=456 \
    logger.wandb.tags=[gatv2,addneuromed,hpsearch,rerun] \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    experiment=no_readout \
    dataset.loader.parameters.adjacency_method=wgcna \
    dataset.loader.parameters.node_sample_ratio=0.3 \
    dataset.loader.parameters.method=variance \
    optimizer.parameters.lr=0.0001 \
    optimizer.parameters.weight_decay=0.0 \
    model.backbone.dropout=0.1 \
    model.feature_encoder.out_channels=64 \
    model.backbone.num_layers=4 \
    model.backbone.heads=2 \
    dataset.loader.parameters.adjacency_threshold=0.2378 \
    trainer.devices=\[2\] \
    &

python -m ogbench \
    model=gatv2 \
    dataset=addneuromed \
    seed=42 \
    logger.wandb.tags=[gatv2,addneuromed,hpsearch,rerun] \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    experiment=no_readout \
    dataset.loader.parameters.adjacency_method=wgcna \
    dataset.loader.parameters.node_sample_ratio=0.3 \
    dataset.loader.parameters.method=variance \
    optimizer.parameters.lr=0.0001 \
    optimizer.parameters.weight_decay=0.0 \
    model.backbone.dropout=0.1 \
    model.feature_encoder.out_channels=64 \
    model.backbone.num_layers=4 \
    model.backbone.heads=4 \
    dataset.loader.parameters.adjacency_threshold=0.2378 \
    trainer.devices=\[3\] \
    &

python -m ogbench \
    model=gatv2 \
    dataset=addneuromed \
    seed=456 \
    logger.wandb.tags=[gatv2,addneuromed,hpsearch,rerun] \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    experiment=no_readout \
    dataset.loader.parameters.adjacency_method=wgcna \
    dataset.loader.parameters.node_sample_ratio=0.3 \
    dataset.loader.parameters.method=variance \
    optimizer.parameters.lr=0.0001 \
    optimizer.parameters.weight_decay=0.0 \
    model.backbone.dropout=0.1 \
    model.feature_encoder.out_channels=64 \
    model.backbone.num_layers=4 \
    model.backbone.heads=4 \
    dataset.loader.parameters.adjacency_threshold=0.2378 \
    trainer.devices=\[4\] \
    &
# SKIP run_id=49enw3sq (no Hydra args list in W&B summary or metadata)
# SKIP run_id=f61j9vnd (no Hydra args list in W&B summary or metadata)
# SKIP run_id=bk9s2ber (no Hydra args list in W&B summary or metadata)
# SKIP run_id=gej23096 (no Hydra args list in W&B summary or metadata)

python -m ogbench \
    model=gatv2 \
    dataset=addneuromed \
    seed=456 \
    logger.wandb.tags=[gatv2,addneuromed,hpsearch,rerun] \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    experiment=no_readout \
    dataset.loader.parameters.adjacency_method=wgcna \
    dataset.loader.parameters.node_sample_ratio=0.3 \
    dataset.loader.parameters.method=variance \
    optimizer.parameters.lr=0.001 \
    optimizer.parameters.weight_decay=0.0 \
    model.backbone.dropout=0.1 \
    model.feature_encoder.out_channels=64 \
    model.backbone.num_layers=4 \
    model.backbone.heads=2 \
    dataset.loader.parameters.adjacency_threshold=0.2378 \
    trainer.devices=\[1\] \
    &

python -m ogbench \
    model=gatv2 \
    dataset=addneuromed \
    seed=42 \
    logger.wandb.tags=[gatv2,addneuromed,hpsearch,rerun] \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    experiment=no_readout \
    dataset.loader.parameters.adjacency_method=wgcna \
    dataset.loader.parameters.node_sample_ratio=0.3 \
    dataset.loader.parameters.method=variance \
    optimizer.parameters.lr=0.001 \
    optimizer.parameters.weight_decay=0.0 \
    model.backbone.dropout=0.1 \
    model.feature_encoder.out_channels=64 \
    model.backbone.num_layers=4 \
    model.backbone.heads=4 \
    dataset.loader.parameters.adjacency_threshold=0.2378 \
    trainer.devices=\[2\] \
    &

python -m ogbench \
    model=gatv2 \
    dataset=addneuromed \
    seed=123 \
    logger.wandb.tags=[gatv2,addneuromed,hpsearch,rerun] \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    experiment=no_readout \
    dataset.loader.parameters.adjacency_method=wgcna \
    dataset.loader.parameters.node_sample_ratio=0.3 \
    dataset.loader.parameters.method=variance \
    optimizer.parameters.lr=0.001 \
    optimizer.parameters.weight_decay=0.0 \
    model.backbone.dropout=0.1 \
    model.feature_encoder.out_channels=64 \
    model.backbone.num_layers=4 \
    model.backbone.heads=4 \
    dataset.loader.parameters.adjacency_threshold=0.2378 \
    trainer.devices=\[3\] \
    &

python -m ogbench \
    model=gatv2 \
    dataset=addneuromed \
    seed=456 \
    logger.wandb.tags=[gatv2,addneuromed,hpsearch,rerun] \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    experiment=no_readout \
    dataset.loader.parameters.adjacency_method=wgcna \
    dataset.loader.parameters.node_sample_ratio=0.3 \
    dataset.loader.parameters.method=variance \
    optimizer.parameters.lr=0.001 \
    optimizer.parameters.weight_decay=0.0 \
    model.backbone.dropout=0.1 \
    model.feature_encoder.out_channels=64 \
    model.backbone.num_layers=4 \
    model.backbone.heads=4 \
    dataset.loader.parameters.adjacency_threshold=0.2378 \
    trainer.devices=\[4\] \
    &

python -m ogbench \
    model=gatv2 \
    dataset=addneuromed \
    seed=42 \
    logger.wandb.tags=[gatv2,addneuromed,hpsearch,rerun] \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    experiment=no_readout \
    dataset.loader.parameters.adjacency_method=wgcna \
    dataset.loader.parameters.node_sample_ratio=0.3 \
    dataset.loader.parameters.method=random \
    optimizer.parameters.lr=0.001 \
    optimizer.parameters.weight_decay=0.0 \
    model.backbone.dropout=0.1 \
    model.feature_encoder.out_channels=64 \
    model.backbone.num_layers=4 \
    model.backbone.heads=2 \
    dataset.loader.parameters.adjacency_threshold=0.0005 \
    trainer.devices=\[5\] \
    &

python -m ogbench \
    model=gatv2 \
    dataset=addneuromed \
    seed=123 \
    logger.wandb.tags=[gatv2,addneuromed,hpsearch,rerun] \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    experiment=no_readout \
    dataset.loader.parameters.adjacency_method=wgcna \
    dataset.loader.parameters.node_sample_ratio=0.3 \
    dataset.loader.parameters.method=random \
    optimizer.parameters.lr=0.001 \
    optimizer.parameters.weight_decay=0.0 \
    model.backbone.dropout=0.1 \
    model.feature_encoder.out_channels=64 \
    model.backbone.num_layers=4 \
    model.backbone.heads=2 \
    dataset.loader.parameters.adjacency_threshold=0.0005 \
    trainer.devices=\[6\] \
    &

python -m ogbench \
    model=gatv2 \
    dataset=addneuromed \
    seed=42 \
    logger.wandb.tags=[gatv2,addneuromed,hpsearch,rerun] \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    experiment=no_readout \
    dataset.loader.parameters.adjacency_method=wgcna \
    dataset.loader.parameters.node_sample_ratio=0.3 \
    dataset.loader.parameters.method=random \
    optimizer.parameters.lr=0.001 \
    optimizer.parameters.weight_decay=0.0 \
    model.backbone.dropout=0.1 \
    model.feature_encoder.out_channels=64 \
    model.backbone.num_layers=4 \
    model.backbone.heads=4 \
    dataset.loader.parameters.adjacency_threshold=0.0005 \
    trainer.devices=\[7\] \
    &

python -m ogbench \
    model=gps \
    dataset=motrpac \
    seed=456 \
    logger.wandb.tags=[gps,motrpac,hpsearch,rerun] \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    experiment=omics_readout \
    dataset.loader.parameters.adjacency_method=wgcna \
    dataset.loader.parameters.node_sample_ratio=0.5 \
    dataset.loader.parameters.method=distance_correlation \
    optimizer.parameters.lr=0.0001 \
    optimizer.parameters.weight_decay=0.0 \
    model.backbone.dropout=0.1 \
    model.feature_encoder.out_channels=16 \
    model.backbone.num_layers=4 \
    model.encodings=[LapPE] \
    dataset.loader.parameters.adjacency_threshold=0.0015 \
    model.readout.fc_dropout=0.1 \
    trainer.devices=\[0\] \
    &

python -m ogbench \
    model=gatv2 \
    dataset=addneuromed \
    seed=123 \
    logger.wandb.tags=[gatv2,addneuromed,hpsearch,rerun] \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    experiment=no_readout \
    dataset.loader.parameters.adjacency_method=wgcna \
    dataset.loader.parameters.node_sample_ratio=0.3 \
    dataset.loader.parameters.method=random \
    optimizer.parameters.lr=0.001 \
    optimizer.parameters.weight_decay=0.0 \
    model.backbone.dropout=0.1 \
    model.feature_encoder.out_channels=64 \
    model.backbone.num_layers=4 \
    model.backbone.heads=4 \
    dataset.loader.parameters.adjacency_threshold=0.0005 \
    trainer.devices=\[1\] \
    &

python -m ogbench \
    model=gatv2 \
    dataset=addneuromed \
    seed=456 \
    logger.wandb.tags=[gatv2,addneuromed,hpsearch,rerun] \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    experiment=no_readout \
    dataset.loader.parameters.adjacency_method=wgcna \
    dataset.loader.parameters.node_sample_ratio=0.3 \
    dataset.loader.parameters.method=random \
    optimizer.parameters.lr=0.001 \
    optimizer.parameters.weight_decay=0.0 \
    model.backbone.dropout=0.1 \
    model.feature_encoder.out_channels=64 \
    model.backbone.num_layers=4 \
    model.backbone.heads=4 \
    dataset.loader.parameters.adjacency_threshold=0.0005 \
    trainer.devices=\[2\] \
    &

python -m ogbench \
    model=gatv2 \
    dataset=addneuromed \
    seed=42 \
    logger.wandb.tags=[gatv2,addneuromed,hpsearch,rerun] \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    experiment=no_readout \
    dataset.loader.parameters.adjacency_method=wgcna \
    dataset.loader.parameters.node_sample_ratio=0.3 \
    dataset.loader.parameters.method=correlation \
    optimizer.parameters.lr=0.001 \
    optimizer.parameters.weight_decay=0.0 \
    model.backbone.dropout=0.1 \
    model.feature_encoder.out_channels=16 \
    model.backbone.num_layers=4 \
    model.backbone.heads=4 \
    dataset.loader.parameters.adjacency_threshold=0.0855 \
    trainer.devices=\[3\] \
    &

python -m ogbench \
    model=gps \
    dataset=motrpac \
    seed=42 \
    logger.wandb.tags=[gps,motrpac,hpsearch,rerun] \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    experiment=omics_readout \
    dataset.loader.parameters.adjacency_method=string \
    dataset.loader.parameters.node_sample_ratio=0.5 \
    dataset.loader.parameters.method=correlation \
    optimizer.parameters.lr=0.0001 \
    optimizer.parameters.weight_decay=0.0 \
    model.backbone.dropout=0.1 \
    model.feature_encoder.out_channels=16 \
    model.backbone.num_layers=4 \
    model.encodings=[RWSE] \
    dataset.loader.parameters.adjacency_threshold=0.4 \
    model.readout.fc_dropout=0.1 \
    trainer.devices=\[4\] \
    &

python -m ogbench \
    model=gatv2 \
    dataset=addneuromed \
    seed=42 \
    logger.wandb.tags=[gatv2,addneuromed,hpsearch,rerun] \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    experiment=no_readout \
    dataset.loader.parameters.adjacency_method=wgcna \
    dataset.loader.parameters.node_sample_ratio=0.3 \
    dataset.loader.parameters.method=correlation \
    optimizer.parameters.lr=0.001 \
    optimizer.parameters.weight_decay=0.0 \
    model.backbone.dropout=0.1 \
    model.feature_encoder.out_channels=64 \
    model.backbone.num_layers=4 \
    model.backbone.heads=2 \
    dataset.loader.parameters.adjacency_threshold=0.0855 \
    trainer.devices=\[5\] \
    &

python -m ogbench \
    model=gatv2 \
    dataset=addneuromed \
    seed=42 \
    logger.wandb.tags=[gatv2,addneuromed,hpsearch,rerun] \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    experiment=no_readout \
    dataset.loader.parameters.adjacency_method=wgcna \
    dataset.loader.parameters.node_sample_ratio=0.3 \
    dataset.loader.parameters.method=correlation \
    optimizer.parameters.lr=0.001 \
    optimizer.parameters.weight_decay=0.0 \
    model.backbone.dropout=0.1 \
    model.feature_encoder.out_channels=64 \
    model.backbone.num_layers=4 \
    model.backbone.heads=4 \
    dataset.loader.parameters.adjacency_threshold=0.0855 \
    trainer.devices=\[6\] \
    &

python -m ogbench \
    model=gatv4 \
    dataset=addneuromed \
    seed=42 \
    logger.wandb.tags=[gatv4,addneuromed,hpsearch,rerun] \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    experiment=omics_readout \
    dataset.loader.parameters.adjacency_method=wgcna \
    dataset.loader.parameters.node_sample_ratio=0.3 \
    dataset.loader.parameters.method=variance \
    optimizer.parameters.lr=0.0001 \
    optimizer.parameters.weight_decay=0.0 \
    model.backbone.dropout=0.1 \
    model.backbone.hidden_channels=[32,64] \
    model.backbone.heads=[8,8] \
    model.backbone.use_layer_norm=false \
    dataset.loader.parameters.adjacency_threshold=0.2378 \
    model.readout.fc_dropout=0.1 \
    trainer.devices=\[7\] \
    &

python -m ogbench \
    model=gatv2 \
    dataset=addneuromed \
    seed=456 \
    logger.wandb.tags=[gatv2,addneuromed,hpsearch,rerun] \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    experiment=no_readout \
    dataset.loader.parameters.adjacency_method=wgcna \
    dataset.loader.parameters.node_sample_ratio=0.3 \
    dataset.loader.parameters.method=correlation \
    optimizer.parameters.lr=0.001 \
    optimizer.parameters.weight_decay=0.0 \
    model.backbone.dropout=0.1 \
    model.feature_encoder.out_channels=64 \
    model.backbone.num_layers=4 \
    model.backbone.heads=4 \
    dataset.loader.parameters.adjacency_threshold=0.0855 \
    trainer.devices=\[0\] \
    &

python -m ogbench \
    model=gatv2 \
    dataset=addneuromed \
    seed=123 \
    logger.wandb.tags=[gatv2,addneuromed,hpsearch,rerun] \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    experiment=no_readout \
    dataset.loader.parameters.adjacency_method=wgcna \
    dataset.loader.parameters.node_sample_ratio=0.3 \
    dataset.loader.parameters.method=distance_correlation \
    optimizer.parameters.lr=0.0001 \
    optimizer.parameters.weight_decay=0.0 \
    model.backbone.dropout=0.1 \
    model.feature_encoder.out_channels=64 \
    model.backbone.num_layers=4 \
    model.backbone.heads=2 \
    dataset.loader.parameters.adjacency_threshold=0.1399 \
    trainer.devices=\[1\] \
    &

python -m ogbench \
    model=gatv2 \
    dataset=addneuromed \
    seed=123 \
    logger.wandb.tags=[gatv2,addneuromed,hpsearch,rerun] \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    experiment=no_readout \
    dataset.loader.parameters.adjacency_method=wgcna \
    dataset.loader.parameters.node_sample_ratio=0.3 \
    dataset.loader.parameters.method=distance_correlation \
    optimizer.parameters.lr=0.0001 \
    optimizer.parameters.weight_decay=0.0 \
    model.backbone.dropout=0.1 \
    model.feature_encoder.out_channels=64 \
    model.backbone.num_layers=4 \
    model.backbone.heads=4 \
    dataset.loader.parameters.adjacency_threshold=0.1399 \
    trainer.devices=\[2\] \
    &

python -m ogbench \
    model=gatv4 \
    dataset=addneuromed \
    seed=42 \
    logger.wandb.tags=[gatv4,addneuromed,hpsearch,rerun] \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    experiment=omics_readout \
    dataset.loader.parameters.adjacency_method=wgcna \
    dataset.loader.parameters.node_sample_ratio=0.3 \
    dataset.loader.parameters.method=random \
    optimizer.parameters.lr=0.0001 \
    optimizer.parameters.weight_decay=0.0 \
    model.backbone.dropout=0.1 \
    model.backbone.hidden_channels=[32,64] \
    model.backbone.heads=[8,8] \
    model.backbone.use_layer_norm=true \
    dataset.loader.parameters.adjacency_threshold=0.0005 \
    model.readout.fc_dropout=0.1 \
    trainer.devices=\[3\] \
    &

python -m ogbench \
    model=gps \
    dataset=brca \
    seed=42 \
    logger.wandb.tags=[gps,brca,hpsearch,rerun] \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    experiment=no_readout \
    dataset.loader.parameters.adjacency_method=wgcna \
    dataset.loader.parameters.node_sample_ratio=0.5 \
    dataset.loader.parameters.method=random \
    optimizer.parameters.lr=0.0001 \
    optimizer.parameters.weight_decay=0.0 \
    model.backbone.dropout=0.1 \
    model.feature_encoder.out_channels=16 \
    model.backbone.num_layers=4 \
    model.encodings=[LapPE] \
    dataset.loader.parameters.adjacency_threshold=0.0021 \
    trainer.devices=\[4\] \
    &

python -m ogbench \
    model=gatv2 \
    dataset=addneuromed \
    seed=42 \
    logger.wandb.tags=[gatv2,addneuromed,hpsearch,rerun] \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    experiment=no_readout \
    dataset.loader.parameters.adjacency_method=wgcna \
    dataset.loader.parameters.node_sample_ratio=0.3 \
    dataset.loader.parameters.method=distance_correlation \
    optimizer.parameters.lr=0.001 \
    optimizer.parameters.weight_decay=0.0 \
    model.backbone.dropout=0.1 \
    model.feature_encoder.out_channels=64 \
    model.backbone.num_layers=4 \
    model.backbone.heads=2 \
    dataset.loader.parameters.adjacency_threshold=0.1399 \
    trainer.devices=\[5\] \
    &
# SKIP run_id=8gwo4fad (no Hydra args list in W&B summary or metadata)
# SKIP run_id=ny6vllhm (no Hydra args list in W&B summary or metadata)
# SKIP run_id=hxa0jdv8 (no Hydra args list in W&B summary or metadata)
# SKIP run_id=4hufmuzf (no Hydra args list in W&B summary or metadata)
# SKIP run_id=pr0lxb7w (no Hydra args list in W&B summary or metadata)
# SKIP run_id=d8ljpixu (no Hydra args list in W&B summary or metadata)
# SKIP run_id=qaer39zg (no Hydra args list in W&B summary or metadata)
# SKIP run_id=obsfwmfq (no Hydra args list in W&B summary or metadata)
# SKIP run_id=ncv55tqz (no Hydra args list in W&B summary or metadata)
# SKIP run_id=wpulmkmo (no Hydra args list in W&B summary or metadata)
# SKIP run_id=lxnviffq (no Hydra args list in W&B summary or metadata)
# SKIP run_id=1a121hyz (no Hydra args list in W&B summary or metadata)
# SKIP run_id=gpblxfqq (no Hydra args list in W&B summary or metadata)
# SKIP run_id=58gukfsr (no Hydra args list in W&B summary or metadata)
# SKIP run_id=1oe98x3c (no Hydra args list in W&B summary or metadata)
# SKIP run_id=dlc7t0mb (no Hydra args list in W&B summary or metadata)
# SKIP run_id=25dc7k0w (no Hydra args list in W&B summary or metadata)
# SKIP run_id=n3odom7k (no Hydra args list in W&B summary or metadata)
# SKIP run_id=5pdtgkma (no Hydra args list in W&B summary or metadata)
# SKIP run_id=wwegoiy3 (no Hydra args list in W&B summary or metadata)
# SKIP run_id=zamz71uk (no Hydra args list in W&B summary or metadata)
# SKIP run_id=nw6tr8me (no Hydra args list in W&B summary or metadata)
# SKIP run_id=6sliynk1 (no Hydra args list in W&B summary or metadata)
# SKIP run_id=clk4z9x8 (no Hydra args list in W&B summary or metadata)

python -m ogbench \
    model=gatv4 \
    dataset=addneuromed \
    seed=123 \
    logger.wandb.tags=[gatv4,addneuromed,hpsearch,rerun] \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    experiment=omics_readout \
    dataset.loader.parameters.adjacency_method=wgcna \
    dataset.loader.parameters.node_sample_ratio=0.3 \
    dataset.loader.parameters.method=correlation \
    optimizer.parameters.lr=0.0001 \
    optimizer.parameters.weight_decay=0.0 \
    model.backbone.dropout=0.1 \
    model.backbone.hidden_channels=[32,64] \
    model.backbone.heads=[8,8] \
    model.backbone.use_layer_norm=true \
    dataset.loader.parameters.adjacency_threshold=0.0855 \
    model.readout.fc_dropout=0.1 \
    trainer.devices=\[6\] \
    &

python -m ogbench \
    model=gatv2 \
    dataset=addneuromed \
    seed=123 \
    logger.wandb.tags=[gatv2,addneuromed,hpsearch,rerun] \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    experiment=omics_readout \
    dataset.loader.parameters.adjacency_method=wgcna \
    dataset.loader.parameters.node_sample_ratio=0.3 \
    dataset.loader.parameters.method=variance \
    optimizer.parameters.lr=0.0001 \
    optimizer.parameters.weight_decay=0.0 \
    model.backbone.dropout=0.1 \
    model.feature_encoder.out_channels=64 \
    model.backbone.num_layers=4 \
    model.backbone.heads=4 \
    dataset.loader.parameters.adjacency_threshold=0.2378 \
    model.readout.fc_dropout=0.1 \
    trainer.devices=\[7\] \
    &

python -m ogbench \
    model=gatv2 \
    dataset=addneuromed \
    seed=123 \
    logger.wandb.tags=[gatv2,addneuromed,hpsearch,rerun] \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    experiment=omics_readout \
    dataset.loader.parameters.adjacency_method=wgcna \
    dataset.loader.parameters.node_sample_ratio=0.3 \
    dataset.loader.parameters.method=variance \
    optimizer.parameters.lr=0.001 \
    optimizer.parameters.weight_decay=0.0 \
    model.backbone.dropout=0.1 \
    model.feature_encoder.out_channels=16 \
    model.backbone.num_layers=4 \
    model.backbone.heads=4 \
    dataset.loader.parameters.adjacency_threshold=0.2378 \
    model.readout.fc_dropout=0.1 \
    trainer.devices=\[0\] \
    &

python -m ogbench \
    model=gatv2 \
    dataset=addneuromed \
    seed=123 \
    logger.wandb.tags=[gatv2,addneuromed,hpsearch,rerun] \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    experiment=omics_readout \
    dataset.loader.parameters.adjacency_method=wgcna \
    dataset.loader.parameters.node_sample_ratio=0.3 \
    dataset.loader.parameters.method=variance \
    optimizer.parameters.lr=0.001 \
    optimizer.parameters.weight_decay=0.0 \
    model.backbone.dropout=0.1 \
    model.feature_encoder.out_channels=64 \
    model.backbone.num_layers=4 \
    model.backbone.heads=2 \
    dataset.loader.parameters.adjacency_threshold=0.2378 \
    model.readout.fc_dropout=0.1 \
    trainer.devices=\[1\] \
    &

python -m ogbench \
    model=gatv2 \
    dataset=addneuromed \
    seed=123 \
    logger.wandb.tags=[gatv2,addneuromed,hpsearch,rerun] \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    experiment=omics_readout \
    dataset.loader.parameters.adjacency_method=wgcna \
    dataset.loader.parameters.node_sample_ratio=0.3 \
    dataset.loader.parameters.method=random \
    optimizer.parameters.lr=0.0001 \
    optimizer.parameters.weight_decay=0.0 \
    model.backbone.dropout=0.1 \
    model.feature_encoder.out_channels=64 \
    model.backbone.num_layers=4 \
    model.backbone.heads=2 \
    dataset.loader.parameters.adjacency_threshold=0.0005 \
    model.readout.fc_dropout=0.1 \
    trainer.devices=\[2\] \
    &

python -m ogbench \
    model=gatv2 \
    dataset=addneuromed \
    seed=456 \
    logger.wandb.tags=[gatv2,addneuromed,hpsearch,rerun] \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    experiment=omics_readout \
    dataset.loader.parameters.adjacency_method=wgcna \
    dataset.loader.parameters.node_sample_ratio=0.3 \
    dataset.loader.parameters.method=random \
    optimizer.parameters.lr=0.0001 \
    optimizer.parameters.weight_decay=0.0 \
    model.backbone.dropout=0.1 \
    model.feature_encoder.out_channels=64 \
    model.backbone.num_layers=4 \
    model.backbone.heads=2 \
    dataset.loader.parameters.adjacency_threshold=0.0005 \
    model.readout.fc_dropout=0.1 \
    trainer.devices=\[3\] \
    &

python -m ogbench \
    model=gatv2 \
    dataset=addneuromed \
    seed=123 \
    logger.wandb.tags=[gatv2,addneuromed,hpsearch,rerun] \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    experiment=omics_readout \
    dataset.loader.parameters.adjacency_method=wgcna \
    dataset.loader.parameters.node_sample_ratio=0.3 \
    dataset.loader.parameters.method=random \
    optimizer.parameters.lr=0.0001 \
    optimizer.parameters.weight_decay=0.0 \
    model.backbone.dropout=0.1 \
    model.feature_encoder.out_channels=64 \
    model.backbone.num_layers=4 \
    model.backbone.heads=4 \
    dataset.loader.parameters.adjacency_threshold=0.0005 \
    model.readout.fc_dropout=0.1 \
    trainer.devices=\[4\] \
    &

python -m ogbench \
    model=gatv2 \
    dataset=addneuromed \
    seed=123 \
    logger.wandb.tags=[gatv2,addneuromed,hpsearch,rerun] \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    experiment=omics_readout \
    dataset.loader.parameters.adjacency_method=wgcna \
    dataset.loader.parameters.node_sample_ratio=0.3 \
    dataset.loader.parameters.method=random \
    optimizer.parameters.lr=0.001 \
    optimizer.parameters.weight_decay=0.0 \
    model.backbone.dropout=0.1 \
    model.feature_encoder.out_channels=64 \
    model.backbone.num_layers=4 \
    model.backbone.heads=2 \
    dataset.loader.parameters.adjacency_threshold=0.0005 \
    model.readout.fc_dropout=0.1 \
    trainer.devices=\[5\] \
    &

python -m ogbench \
    model=gatv2 \
    dataset=addneuromed \
    seed=456 \
    logger.wandb.tags=[gatv2,addneuromed,hpsearch,rerun] \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    experiment=omics_readout \
    dataset.loader.parameters.adjacency_method=wgcna \
    dataset.loader.parameters.node_sample_ratio=0.3 \
    dataset.loader.parameters.method=correlation \
    optimizer.parameters.lr=0.0001 \
    optimizer.parameters.weight_decay=0.0 \
    model.backbone.dropout=0.1 \
    model.feature_encoder.out_channels=16 \
    model.backbone.num_layers=4 \
    model.backbone.heads=4 \
    dataset.loader.parameters.adjacency_threshold=0.0855 \
    model.readout.fc_dropout=0.1 \
    trainer.devices=\[6\] \
    &

python -m ogbench \
    model=gatv2 \
    dataset=addneuromed \
    seed=123 \
    logger.wandb.tags=[gatv2,addneuromed,hpsearch,rerun] \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    experiment=omics_readout \
    dataset.loader.parameters.adjacency_method=wgcna \
    dataset.loader.parameters.node_sample_ratio=0.3 \
    dataset.loader.parameters.method=correlation \
    optimizer.parameters.lr=0.0001 \
    optimizer.parameters.weight_decay=0.0 \
    model.backbone.dropout=0.1 \
    model.feature_encoder.out_channels=64 \
    model.backbone.num_layers=4 \
    model.backbone.heads=4 \
    dataset.loader.parameters.adjacency_threshold=0.0855 \
    model.readout.fc_dropout=0.1 \
    trainer.devices=\[7\] \
    &

python -m ogbench \
    model=gatv2 \
    dataset=addneuromed \
    seed=456 \
    logger.wandb.tags=[gatv2,addneuromed,hpsearch,rerun] \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    experiment=omics_readout \
    dataset.loader.parameters.adjacency_method=wgcna \
    dataset.loader.parameters.node_sample_ratio=0.3 \
    dataset.loader.parameters.method=correlation \
    optimizer.parameters.lr=0.001 \
    optimizer.parameters.weight_decay=0.0 \
    model.backbone.dropout=0.1 \
    model.feature_encoder.out_channels=64 \
    model.backbone.num_layers=4 \
    model.backbone.heads=4 \
    dataset.loader.parameters.adjacency_threshold=0.0855 \
    model.readout.fc_dropout=0.1 \
    trainer.devices=\[0\] \
    &

python -m ogbench \
    model=gatv2 \
    dataset=addneuromed \
    seed=123 \
    logger.wandb.tags=[gatv2,addneuromed,hpsearch,rerun] \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    experiment=omics_readout \
    dataset.loader.parameters.adjacency_method=wgcna \
    dataset.loader.parameters.node_sample_ratio=0.3 \
    dataset.loader.parameters.method=distance_correlation \
    optimizer.parameters.lr=0.0001 \
    optimizer.parameters.weight_decay=0.0 \
    model.backbone.dropout=0.1 \
    model.feature_encoder.out_channels=16 \
    model.backbone.num_layers=4 \
    model.backbone.heads=4 \
    dataset.loader.parameters.adjacency_threshold=0.1399 \
    model.readout.fc_dropout=0.1 \
    trainer.devices=\[1\] \
    &

python -m ogbench \
    model=gatv2 \
    dataset=addneuromed \
    seed=456 \
    logger.wandb.tags=[gatv2,addneuromed,hpsearch,rerun] \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    experiment=omics_readout \
    dataset.loader.parameters.adjacency_method=wgcna \
    dataset.loader.parameters.node_sample_ratio=0.3 \
    dataset.loader.parameters.method=distance_correlation \
    optimizer.parameters.lr=0.0001 \
    optimizer.parameters.weight_decay=0.0 \
    model.backbone.dropout=0.1 \
    model.feature_encoder.out_channels=64 \
    model.backbone.num_layers=4 \
    model.backbone.heads=4 \
    dataset.loader.parameters.adjacency_threshold=0.1399 \
    model.readout.fc_dropout=0.1 \
    trainer.devices=\[2\] \
    &

python -m ogbench \
    model=gatv2 \
    dataset=addneuromed \
    seed=456 \
    logger.wandb.tags=[gatv2,addneuromed,hpsearch,rerun] \
    logger.wandb.project=bgbench_dataset_grid_search_final \
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
    dataset.loader.parameters.adjacency_threshold=0.1399 \
    model.readout.fc_dropout=0.1 \
    trainer.devices=\[3\] \
    &

python -m ogbench \
    model=gatv4 \
    dataset=parkinsons \
    seed=456 \
    logger.wandb.tags=[gatv4,parkinsons,hpsearch,rerun] \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    experiment=no_readout \
    dataset.loader.parameters.adjacency_method=wgcna \
    dataset.loader.parameters.node_sample_ratio=0.5 \
    dataset.loader.parameters.method=random \
    optimizer.parameters.lr=0.0001 \
    optimizer.parameters.weight_decay=0.0 \
    model.backbone.dropout=0.1 \
    model.backbone.hidden_channels=[32,64] \
    model.backbone.heads=[8,8] \
    model.backbone.use_layer_norm=false \
    dataset.loader.parameters.adjacency_threshold=0.0005 \
    trainer.devices=\[4\] \
    &

python -m ogbench \
    model=gatv4 \
    dataset=parkinsons \
    seed=123 \
    logger.wandb.tags=[gatv4,parkinsons,hpsearch,rerun] \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    experiment=no_readout \
    dataset.loader.parameters.adjacency_method=wgcna \
    dataset.loader.parameters.node_sample_ratio=0.5 \
    dataset.loader.parameters.method=random \
    optimizer.parameters.lr=0.001 \
    optimizer.parameters.weight_decay=0.0 \
    model.backbone.dropout=0.1 \
    model.backbone.hidden_channels=[16,32] \
    model.backbone.heads=[8,8] \
    model.backbone.use_layer_norm=false \
    dataset.loader.parameters.adjacency_threshold=0.0005 \
    trainer.devices=\[5\] \
    &

python -m ogbench \
    model=gatv4 \
    dataset=parkinsons \
    seed=42 \
    logger.wandb.tags=[gatv4,parkinsons,hpsearch,rerun] \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    experiment=no_readout \
    dataset.loader.parameters.adjacency_method=wgcna \
    dataset.loader.parameters.node_sample_ratio=0.5 \
    dataset.loader.parameters.method=random \
    optimizer.parameters.lr=0.001 \
    optimizer.parameters.weight_decay=0.0 \
    model.backbone.dropout=0.1 \
    model.backbone.hidden_channels=[32,64] \
    model.backbone.heads=[4,4] \
    model.backbone.use_layer_norm=true \
    dataset.loader.parameters.adjacency_threshold=0.0005 \
    trainer.devices=\[6\] \
    &

python -m ogbench \
    model=gatv4 \
    dataset=parkinsons \
    seed=123 \
    logger.wandb.tags=[gatv4,parkinsons,hpsearch,rerun] \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    experiment=no_readout \
    dataset.loader.parameters.adjacency_method=wgcna \
    dataset.loader.parameters.node_sample_ratio=0.5 \
    dataset.loader.parameters.method=random \
    optimizer.parameters.lr=0.001 \
    optimizer.parameters.weight_decay=0.0 \
    model.backbone.dropout=0.1 \
    model.backbone.hidden_channels=[32,64] \
    model.backbone.heads=[4,4] \
    model.backbone.use_layer_norm=false \
    dataset.loader.parameters.adjacency_threshold=0.0005 \
    trainer.devices=\[7\] \
    &

python -m ogbench \
    model=gatv4 \
    dataset=parkinsons \
    seed=42 \
    logger.wandb.tags=[gatv4,parkinsons,hpsearch,rerun] \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    experiment=no_readout \
    dataset.loader.parameters.adjacency_method=wgcna \
    dataset.loader.parameters.node_sample_ratio=0.5 \
    dataset.loader.parameters.method=random \
    optimizer.parameters.lr=0.001 \
    optimizer.parameters.weight_decay=0.0 \
    model.backbone.dropout=0.1 \
    model.backbone.hidden_channels=[32,64] \
    model.backbone.heads=[8,8] \
    model.backbone.use_layer_norm=true \
    dataset.loader.parameters.adjacency_threshold=0.0005 \
    trainer.devices=\[0\] \
    &

python -m ogbench \
    model=gatv4 \
    dataset=parkinsons \
    seed=123 \
    logger.wandb.tags=[gatv4,parkinsons,hpsearch,rerun] \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    experiment=no_readout \
    dataset.loader.parameters.adjacency_method=wgcna \
    dataset.loader.parameters.node_sample_ratio=0.5 \
    dataset.loader.parameters.method=random \
    optimizer.parameters.lr=0.001 \
    optimizer.parameters.weight_decay=0.0 \
    model.backbone.dropout=0.1 \
    model.backbone.hidden_channels=[32,64] \
    model.backbone.heads=[8,8] \
    model.backbone.use_layer_norm=true \
    dataset.loader.parameters.adjacency_threshold=0.0005 \
    trainer.devices=\[1\] \
    &

python -m ogbench \
    model=gatv4 \
    dataset=parkinsons \
    seed=456 \
    logger.wandb.tags=[gatv4,parkinsons,hpsearch,rerun] \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    experiment=no_readout \
    dataset.loader.parameters.adjacency_method=wgcna \
    dataset.loader.parameters.node_sample_ratio=0.5 \
    dataset.loader.parameters.method=random \
    optimizer.parameters.lr=0.001 \
    optimizer.parameters.weight_decay=0.0 \
    model.backbone.dropout=0.1 \
    model.backbone.hidden_channels=[32,64] \
    model.backbone.heads=[8,8] \
    model.backbone.use_layer_norm=true \
    dataset.loader.parameters.adjacency_threshold=0.0005 \
    trainer.devices=\[2\] \
    &

python -m ogbench \
    model=gatv4 \
    dataset=parkinsons \
    seed=123 \
    logger.wandb.tags=[gatv4,parkinsons,hpsearch,rerun] \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    experiment=no_readout \
    dataset.loader.parameters.adjacency_method=wgcna \
    dataset.loader.parameters.node_sample_ratio=0.5 \
    dataset.loader.parameters.method=random \
    optimizer.parameters.lr=0.001 \
    optimizer.parameters.weight_decay=0.0 \
    model.backbone.dropout=0.1 \
    model.backbone.hidden_channels=[32,64] \
    model.backbone.heads=[8,8] \
    model.backbone.use_layer_norm=false \
    dataset.loader.parameters.adjacency_threshold=0.0005 \
    trainer.devices=\[3\] \
    &

python -m ogbench \
    model=gatv4 \
    dataset=parkinsons \
    seed=456 \
    logger.wandb.tags=[gatv4,parkinsons,hpsearch,rerun] \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    experiment=no_readout \
    dataset.loader.parameters.adjacency_method=wgcna \
    dataset.loader.parameters.node_sample_ratio=0.5 \
    dataset.loader.parameters.method=random \
    optimizer.parameters.lr=0.001 \
    optimizer.parameters.weight_decay=0.0 \
    model.backbone.dropout=0.1 \
    model.backbone.hidden_channels=[32,64] \
    model.backbone.heads=[8,8] \
    model.backbone.use_layer_norm=false \
    dataset.loader.parameters.adjacency_threshold=0.0005 \
    trainer.devices=\[4\] \
    &

python -m ogbench \
    model=gatv2 \
    dataset=motrpac \
    seed=456 \
    logger.wandb.tags=[gatv2,motrpac,hpsearch,rerun] \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    experiment=no_readout \
    dataset.loader.parameters.adjacency_method=wgcna \
    dataset.loader.parameters.node_sample_ratio=0.3 \
    dataset.loader.parameters.method=random \
    optimizer.parameters.lr=0.0001 \
    optimizer.parameters.weight_decay=0.0 \
    model.backbone.dropout=0.1 \
    model.feature_encoder.out_channels=64 \
    model.backbone.num_layers=4 \
    model.backbone.heads=4 \
    dataset.loader.parameters.adjacency_threshold=0.0582 \
    trainer.devices=\[5\] \
    &

python -m ogbench \
    model=gatv4 \
    dataset=parkinsons \
    seed=42 \
    logger.wandb.tags=[gatv4,parkinsons,hpsearch,rerun] \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    experiment=omics_readout \
    dataset.loader.parameters.adjacency_method=wgcna \
    dataset.loader.parameters.node_sample_ratio=0.5 \
    dataset.loader.parameters.method=random \
    optimizer.parameters.lr=0.0001 \
    optimizer.parameters.weight_decay=0.0 \
    model.backbone.dropout=0.1 \
    model.backbone.hidden_channels=[32,64] \
    model.backbone.heads=[8,8] \
    model.backbone.use_layer_norm=true \
    dataset.loader.parameters.adjacency_threshold=0.0005 \
    model.readout.fc_dropout=0.1 \
    trainer.devices=\[6\] \
    &

python -m ogbench \
    model=gatv4 \
    dataset=parkinsons \
    seed=123 \
    logger.wandb.tags=[gatv4,parkinsons,hpsearch,rerun] \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    experiment=omics_readout \
    dataset.loader.parameters.adjacency_method=wgcna \
    dataset.loader.parameters.node_sample_ratio=0.5 \
    dataset.loader.parameters.method=random \
    optimizer.parameters.lr=0.0001 \
    optimizer.parameters.weight_decay=0.0 \
    model.backbone.dropout=0.1 \
    model.backbone.hidden_channels=[32,64] \
    model.backbone.heads=[8,8] \
    model.backbone.use_layer_norm=true \
    dataset.loader.parameters.adjacency_threshold=0.0005 \
    model.readout.fc_dropout=0.1 \
    trainer.devices=\[7\] \
    &

python -m ogbench \
    model=gatv4 \
    dataset=parkinsons \
    seed=456 \
    logger.wandb.tags=[gatv4,parkinsons,hpsearch,rerun] \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    experiment=omics_readout \
    dataset.loader.parameters.adjacency_method=wgcna \
    dataset.loader.parameters.node_sample_ratio=0.5 \
    dataset.loader.parameters.method=random \
    optimizer.parameters.lr=0.0001 \
    optimizer.parameters.weight_decay=0.0 \
    model.backbone.dropout=0.1 \
    model.backbone.hidden_channels=[32,64] \
    model.backbone.heads=[8,8] \
    model.backbone.use_layer_norm=true \
    dataset.loader.parameters.adjacency_threshold=0.0005 \
    model.readout.fc_dropout=0.1 \
    trainer.devices=\[0\] \
    &

python -m ogbench \
    model=gatv4 \
    dataset=parkinsons \
    seed=42 \
    logger.wandb.tags=[gatv4,parkinsons,hpsearch,rerun] \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    experiment=omics_readout \
    dataset.loader.parameters.adjacency_method=wgcna \
    dataset.loader.parameters.node_sample_ratio=0.5 \
    dataset.loader.parameters.method=random \
    optimizer.parameters.lr=0.0001 \
    optimizer.parameters.weight_decay=0.0 \
    model.backbone.dropout=0.1 \
    model.backbone.hidden_channels=[32,64] \
    model.backbone.heads=[8,8] \
    model.backbone.use_layer_norm=false \
    dataset.loader.parameters.adjacency_threshold=0.0005 \
    model.readout.fc_dropout=0.1 \
    trainer.devices=\[1\] \
    &

python -m ogbench \
    model=gatv4 \
    dataset=parkinsons \
    seed=456 \
    logger.wandb.tags=[gatv4,parkinsons,hpsearch,rerun] \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    experiment=omics_readout \
    dataset.loader.parameters.adjacency_method=wgcna \
    dataset.loader.parameters.node_sample_ratio=0.5 \
    dataset.loader.parameters.method=random \
    optimizer.parameters.lr=0.0001 \
    optimizer.parameters.weight_decay=0.0 \
    model.backbone.dropout=0.1 \
    model.backbone.hidden_channels=[32,64] \
    model.backbone.heads=[8,8] \
    model.backbone.use_layer_norm=false \
    dataset.loader.parameters.adjacency_threshold=0.0005 \
    model.readout.fc_dropout=0.1 \
    trainer.devices=\[2\] \
    &

python -m ogbench \
    model=gatv4 \
    dataset=parkinsons \
    seed=456 \
    logger.wandb.tags=[gatv4,parkinsons,hpsearch,rerun] \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    experiment=omics_readout \
    dataset.loader.parameters.adjacency_method=wgcna \
    dataset.loader.parameters.node_sample_ratio=0.5 \
    dataset.loader.parameters.method=random \
    optimizer.parameters.lr=0.001 \
    optimizer.parameters.weight_decay=0.0 \
    model.backbone.dropout=0.1 \
    model.backbone.hidden_channels=[32,64] \
    model.backbone.heads=[8,8] \
    model.backbone.use_layer_norm=true \
    dataset.loader.parameters.adjacency_threshold=0.0005 \
    model.readout.fc_dropout=0.1 \
    trainer.devices=\[3\] \
    &

python -m ogbench \
    model=gatv4 \
    dataset=parkinsons \
    seed=42 \
    logger.wandb.tags=[gatv4,parkinsons,hpsearch,rerun] \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    experiment=omics_readout \
    dataset.loader.parameters.adjacency_method=wgcna \
    dataset.loader.parameters.node_sample_ratio=0.5 \
    dataset.loader.parameters.method=random \
    optimizer.parameters.lr=0.001 \
    optimizer.parameters.weight_decay=0.0 \
    model.backbone.dropout=0.1 \
    model.backbone.hidden_channels=[32,64] \
    model.backbone.heads=[8,8] \
    model.backbone.use_layer_norm=false \
    dataset.loader.parameters.adjacency_threshold=0.0005 \
    model.readout.fc_dropout=0.1 \
    trainer.devices=\[4\] \
    &

python -m ogbench \
    model=gatv4 \
    dataset=parkinsons \
    seed=123 \
    logger.wandb.tags=[gatv4,parkinsons,hpsearch,rerun] \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    experiment=omics_readout \
    dataset.loader.parameters.adjacency_method=wgcna \
    dataset.loader.parameters.node_sample_ratio=0.5 \
    dataset.loader.parameters.method=random \
    optimizer.parameters.lr=0.001 \
    optimizer.parameters.weight_decay=0.0 \
    model.backbone.dropout=0.1 \
    model.backbone.hidden_channels=[32,64] \
    model.backbone.heads=[8,8] \
    model.backbone.use_layer_norm=false \
    dataset.loader.parameters.adjacency_threshold=0.0005 \
    model.readout.fc_dropout=0.1 \
    trainer.devices=\[5\] \
    &

python -m ogbench \
    model=gatv2 \
    dataset=motrpac \
    seed=123 \
    logger.wandb.tags=[gatv2,motrpac,hpsearch,rerun] \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    experiment=no_readout \
    dataset.loader.parameters.adjacency_method=wgcna \
    dataset.loader.parameters.node_sample_ratio=0.3 \
    dataset.loader.parameters.method=random \
    optimizer.parameters.lr=0.001 \
    optimizer.parameters.weight_decay=0.0 \
    model.backbone.dropout=0.1 \
    model.feature_encoder.out_channels=64 \
    model.backbone.num_layers=4 \
    model.backbone.heads=2 \
    dataset.loader.parameters.adjacency_threshold=0.0582 \
    trainer.devices=\[6\] \
    &

python -m ogbench \
    model=gatv4 \
    dataset=addneuromed \
    seed=456 \
    logger.wandb.tags=[gatv4,addneuromed,hpsearch,rerun] \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    experiment=no_readout \
    dataset.loader.parameters.adjacency_method=wgcna \
    dataset.loader.parameters.node_sample_ratio=0.5 \
    dataset.loader.parameters.method=variance \
    optimizer.parameters.lr=0.0001 \
    optimizer.parameters.weight_decay=0.0 \
    model.backbone.dropout=0.1 \
    model.backbone.hidden_channels=[32,64] \
    model.backbone.heads=[8,8] \
    model.backbone.use_layer_norm=true \
    dataset.loader.parameters.adjacency_threshold=0.1786 \
    trainer.devices=\[7\] \
    &

python -m ogbench \
    model=gatv4 \
    dataset=addneuromed \
    seed=123 \
    logger.wandb.tags=[gatv4,addneuromed,hpsearch,rerun] \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    experiment=no_readout \
    dataset.loader.parameters.adjacency_method=wgcna \
    dataset.loader.parameters.node_sample_ratio=0.5 \
    dataset.loader.parameters.method=variance \
    optimizer.parameters.lr=0.0001 \
    optimizer.parameters.weight_decay=0.0 \
    model.backbone.dropout=0.1 \
    model.backbone.hidden_channels=[32,64] \
    model.backbone.heads=[8,8] \
    model.backbone.use_layer_norm=false \
    dataset.loader.parameters.adjacency_threshold=0.1786 \
    trainer.devices=\[0\] \
    &

python -m ogbench \
    model=gatv4 \
    dataset=addneuromed \
    seed=42 \
    logger.wandb.tags=[gatv4,addneuromed,hpsearch,rerun] \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    experiment=no_readout \
    dataset.loader.parameters.adjacency_method=wgcna \
    dataset.loader.parameters.node_sample_ratio=0.5 \
    dataset.loader.parameters.method=variance \
    optimizer.parameters.lr=0.001 \
    optimizer.parameters.weight_decay=0.0 \
    model.backbone.dropout=0.1 \
    model.backbone.hidden_channels=[32,64] \
    model.backbone.heads=[8,8] \
    model.backbone.use_layer_norm=false \
    dataset.loader.parameters.adjacency_threshold=0.1786 \
    trainer.devices=\[1\] \
    &

python -m ogbench \
    model=gatv4 \
    dataset=addneuromed \
    seed=123 \
    logger.wandb.tags=[gatv4,addneuromed,hpsearch,rerun] \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    experiment=no_readout \
    dataset.loader.parameters.adjacency_method=wgcna \
    dataset.loader.parameters.node_sample_ratio=0.5 \
    dataset.loader.parameters.method=variance \
    optimizer.parameters.lr=0.001 \
    optimizer.parameters.weight_decay=0.0 \
    model.backbone.dropout=0.1 \
    model.backbone.hidden_channels=[32,64] \
    model.backbone.heads=[8,8] \
    model.backbone.use_layer_norm=false \
    dataset.loader.parameters.adjacency_threshold=0.1786 \
    trainer.devices=\[2\] \
    &

python -m ogbench \
    model=gatv4 \
    dataset=addneuromed \
    seed=456 \
    logger.wandb.tags=[gatv4,addneuromed,hpsearch,rerun] \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    experiment=no_readout \
    dataset.loader.parameters.adjacency_method=wgcna \
    dataset.loader.parameters.node_sample_ratio=0.5 \
    dataset.loader.parameters.method=random \
    optimizer.parameters.lr=0.0001 \
    optimizer.parameters.weight_decay=0.0 \
    model.backbone.dropout=0.1 \
    model.backbone.hidden_channels=[16,32] \
    model.backbone.heads=[8,8] \
    model.backbone.use_layer_norm=false \
    dataset.loader.parameters.adjacency_threshold=0.0005 \
    trainer.devices=\[3\] \
    &

python -m ogbench \
    model=gatv4 \
    dataset=addneuromed \
    seed=456 \
    logger.wandb.tags=[gatv4,addneuromed,hpsearch,rerun] \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    experiment=no_readout \
    dataset.loader.parameters.adjacency_method=wgcna \
    dataset.loader.parameters.node_sample_ratio=0.5 \
    dataset.loader.parameters.method=random \
    optimizer.parameters.lr=0.0001 \
    optimizer.parameters.weight_decay=0.0 \
    model.backbone.dropout=0.1 \
    model.backbone.hidden_channels=[32,64] \
    model.backbone.heads=[4,4] \
    model.backbone.use_layer_norm=false \
    dataset.loader.parameters.adjacency_threshold=0.0005 \
    trainer.devices=\[4\] \
    &

python -m ogbench \
    model=gatv4 \
    dataset=addneuromed \
    seed=123 \
    logger.wandb.tags=[gatv4,addneuromed,hpsearch,rerun] \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    experiment=no_readout \
    dataset.loader.parameters.adjacency_method=wgcna \
    dataset.loader.parameters.node_sample_ratio=0.5 \
    dataset.loader.parameters.method=random \
    optimizer.parameters.lr=0.0001 \
    optimizer.parameters.weight_decay=0.0 \
    model.backbone.dropout=0.1 \
    model.backbone.hidden_channels=[32,64] \
    model.backbone.heads=[8,8] \
    model.backbone.use_layer_norm=true \
    dataset.loader.parameters.adjacency_threshold=0.0005 \
    trainer.devices=\[5\] \
    &

python -m ogbench \
    model=gatv4 \
    dataset=addneuromed \
    seed=456 \
    logger.wandb.tags=[gatv4,addneuromed,hpsearch,rerun] \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    experiment=no_readout \
    dataset.loader.parameters.adjacency_method=wgcna \
    dataset.loader.parameters.node_sample_ratio=0.5 \
    dataset.loader.parameters.method=random \
    optimizer.parameters.lr=0.0001 \
    optimizer.parameters.weight_decay=0.0 \
    model.backbone.dropout=0.1 \
    model.backbone.hidden_channels=[32,64] \
    model.backbone.heads=[8,8] \
    model.backbone.use_layer_norm=true \
    dataset.loader.parameters.adjacency_threshold=0.0005 \
    trainer.devices=\[6\] \
    &

python -m ogbench \
    model=gatv4 \
    dataset=addneuromed \
    seed=42 \
    logger.wandb.tags=[gatv4,addneuromed,hpsearch,rerun] \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    experiment=no_readout \
    dataset.loader.parameters.adjacency_method=wgcna \
    dataset.loader.parameters.node_sample_ratio=0.5 \
    dataset.loader.parameters.method=random \
    optimizer.parameters.lr=0.0001 \
    optimizer.parameters.weight_decay=0.0 \
    model.backbone.dropout=0.1 \
    model.backbone.hidden_channels=[32,64] \
    model.backbone.heads=[8,8] \
    model.backbone.use_layer_norm=false \
    dataset.loader.parameters.adjacency_threshold=0.0005 \
    trainer.devices=\[7\] \
    &

python -m ogbench \
    model=gatv4 \
    dataset=addneuromed \
    seed=123 \
    logger.wandb.tags=[gatv4,addneuromed,hpsearch,rerun] \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    experiment=no_readout \
    dataset.loader.parameters.adjacency_method=wgcna \
    dataset.loader.parameters.node_sample_ratio=0.5 \
    dataset.loader.parameters.method=random \
    optimizer.parameters.lr=0.0001 \
    optimizer.parameters.weight_decay=0.0 \
    model.backbone.dropout=0.1 \
    model.backbone.hidden_channels=[32,64] \
    model.backbone.heads=[8,8] \
    model.backbone.use_layer_norm=false \
    dataset.loader.parameters.adjacency_threshold=0.0005 \
    trainer.devices=\[0\] \
    &

python -m ogbench \
    model=gatv4 \
    dataset=addneuromed \
    seed=456 \
    logger.wandb.tags=[gatv4,addneuromed,hpsearch,rerun] \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    experiment=no_readout \
    dataset.loader.parameters.adjacency_method=wgcna \
    dataset.loader.parameters.node_sample_ratio=0.5 \
    dataset.loader.parameters.method=random \
    optimizer.parameters.lr=0.0001 \
    optimizer.parameters.weight_decay=0.0 \
    model.backbone.dropout=0.1 \
    model.backbone.hidden_channels=[32,64] \
    model.backbone.heads=[8,8] \
    model.backbone.use_layer_norm=false \
    dataset.loader.parameters.adjacency_threshold=0.0005 \
    trainer.devices=\[1\] \
    &

python -m ogbench \
    model=gatv4 \
    dataset=addneuromed \
    seed=123 \
    logger.wandb.tags=[gatv4,addneuromed,hpsearch,rerun] \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    experiment=no_readout \
    dataset.loader.parameters.adjacency_method=wgcna \
    dataset.loader.parameters.node_sample_ratio=0.5 \
    dataset.loader.parameters.method=random \
    optimizer.parameters.lr=0.001 \
    optimizer.parameters.weight_decay=0.0 \
    model.backbone.dropout=0.1 \
    model.backbone.hidden_channels=[16,32] \
    model.backbone.heads=[8,8] \
    model.backbone.use_layer_norm=true \
    dataset.loader.parameters.adjacency_threshold=0.0005 \
    trainer.devices=\[2\] \
    &

python -m ogbench \
    model=gatv4 \
    dataset=addneuromed \
    seed=123 \
    logger.wandb.tags=[gatv4,addneuromed,hpsearch,rerun] \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    experiment=no_readout \
    dataset.loader.parameters.adjacency_method=wgcna \
    dataset.loader.parameters.node_sample_ratio=0.5 \
    dataset.loader.parameters.method=random \
    optimizer.parameters.lr=0.001 \
    optimizer.parameters.weight_decay=0.0 \
    model.backbone.dropout=0.1 \
    model.backbone.hidden_channels=[32,64] \
    model.backbone.heads=[8,8] \
    model.backbone.use_layer_norm=true \
    dataset.loader.parameters.adjacency_threshold=0.0005 \
    trainer.devices=\[3\] \
    &

python -m ogbench \
    model=gatv4 \
    dataset=addneuromed \
    seed=456 \
    logger.wandb.tags=[gatv4,addneuromed,hpsearch,rerun] \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    experiment=no_readout \
    dataset.loader.parameters.adjacency_method=wgcna \
    dataset.loader.parameters.node_sample_ratio=0.5 \
    dataset.loader.parameters.method=random \
    optimizer.parameters.lr=0.001 \
    optimizer.parameters.weight_decay=0.0 \
    model.backbone.dropout=0.1 \
    model.backbone.hidden_channels=[32,64] \
    model.backbone.heads=[8,8] \
    model.backbone.use_layer_norm=true \
    dataset.loader.parameters.adjacency_threshold=0.0005 \
    trainer.devices=\[4\] \
    &

python -m ogbench \
    model=gatv4 \
    dataset=addneuromed \
    seed=42 \
    logger.wandb.tags=[gatv4,addneuromed,hpsearch,rerun] \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    experiment=no_readout \
    dataset.loader.parameters.adjacency_method=wgcna \
    dataset.loader.parameters.node_sample_ratio=0.5 \
    dataset.loader.parameters.method=random \
    optimizer.parameters.lr=0.001 \
    optimizer.parameters.weight_decay=0.0 \
    model.backbone.dropout=0.1 \
    model.backbone.hidden_channels=[32,64] \
    model.backbone.heads=[8,8] \
    model.backbone.use_layer_norm=false \
    dataset.loader.parameters.adjacency_threshold=0.0005 \
    trainer.devices=\[5\] \
    &

python -m ogbench \
    model=gatv4 \
    dataset=addneuromed \
    seed=123 \
    logger.wandb.tags=[gatv4,addneuromed,hpsearch,rerun] \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    experiment=no_readout \
    dataset.loader.parameters.adjacency_method=wgcna \
    dataset.loader.parameters.node_sample_ratio=0.5 \
    dataset.loader.parameters.method=random \
    optimizer.parameters.lr=0.001 \
    optimizer.parameters.weight_decay=0.0 \
    model.backbone.dropout=0.1 \
    model.backbone.hidden_channels=[32,64] \
    model.backbone.heads=[8,8] \
    model.backbone.use_layer_norm=false \
    dataset.loader.parameters.adjacency_threshold=0.0005 \
    trainer.devices=\[6\] \
    &

python -m ogbench \
    model=gatv4 \
    dataset=addneuromed \
    seed=456 \
    logger.wandb.tags=[gatv4,addneuromed,hpsearch,rerun] \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    experiment=no_readout \
    dataset.loader.parameters.adjacency_method=wgcna \
    dataset.loader.parameters.node_sample_ratio=0.5 \
    dataset.loader.parameters.method=random \
    optimizer.parameters.lr=0.001 \
    optimizer.parameters.weight_decay=0.0 \
    model.backbone.dropout=0.1 \
    model.backbone.hidden_channels=[32,64] \
    model.backbone.heads=[8,8] \
    model.backbone.use_layer_norm=false \
    dataset.loader.parameters.adjacency_threshold=0.0005 \
    trainer.devices=\[7\] \
    &

python -m ogbench \
    model=gatv4 \
    dataset=addneuromed \
    seed=42 \
    logger.wandb.tags=[gatv4,addneuromed,hpsearch,rerun] \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    experiment=omics_readout \
    dataset.loader.parameters.adjacency_method=wgcna \
    dataset.loader.parameters.node_sample_ratio=0.5 \
    dataset.loader.parameters.method=variance \
    optimizer.parameters.lr=0.001 \
    optimizer.parameters.weight_decay=0.0 \
    model.backbone.dropout=0.1 \
    model.backbone.hidden_channels=[32,64] \
    model.backbone.heads=[8,8] \
    model.backbone.use_layer_norm=false \
    dataset.loader.parameters.adjacency_threshold=0.1786 \
    model.readout.fc_dropout=0.1 \
    trainer.devices=\[0\] \
    &

python -m ogbench \
    model=gatv4 \
    dataset=addneuromed \
    seed=42 \
    logger.wandb.tags=[gatv4,addneuromed,hpsearch,rerun] \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    experiment=omics_readout \
    dataset.loader.parameters.adjacency_method=wgcna \
    dataset.loader.parameters.node_sample_ratio=0.5 \
    dataset.loader.parameters.method=random \
    optimizer.parameters.lr=0.0001 \
    optimizer.parameters.weight_decay=0.0 \
    model.backbone.dropout=0.1 \
    model.backbone.hidden_channels=[32,64] \
    model.backbone.heads=[4,4] \
    model.backbone.use_layer_norm=false \
    dataset.loader.parameters.adjacency_threshold=0.0005 \
    model.readout.fc_dropout=0.1 \
    trainer.devices=\[1\] \
    &

python -m ogbench \
    model=gatv4 \
    dataset=addneuromed \
    seed=42 \
    logger.wandb.tags=[gatv4,addneuromed,hpsearch,rerun] \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    experiment=omics_readout \
    dataset.loader.parameters.adjacency_method=wgcna \
    dataset.loader.parameters.node_sample_ratio=0.5 \
    dataset.loader.parameters.method=random \
    optimizer.parameters.lr=0.0001 \
    optimizer.parameters.weight_decay=0.0 \
    model.backbone.dropout=0.1 \
    model.backbone.hidden_channels=[32,64] \
    model.backbone.heads=[8,8] \
    model.backbone.use_layer_norm=true \
    dataset.loader.parameters.adjacency_threshold=0.0005 \
    model.readout.fc_dropout=0.1 \
    trainer.devices=\[2\] \
    &

python -m ogbench \
    model=gatv4 \
    dataset=addneuromed \
    seed=123 \
    logger.wandb.tags=[gatv4,addneuromed,hpsearch,rerun] \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    experiment=omics_readout \
    dataset.loader.parameters.adjacency_method=wgcna \
    dataset.loader.parameters.node_sample_ratio=0.5 \
    dataset.loader.parameters.method=random \
    optimizer.parameters.lr=0.0001 \
    optimizer.parameters.weight_decay=0.0 \
    model.backbone.dropout=0.1 \
    model.backbone.hidden_channels=[32,64] \
    model.backbone.heads=[8,8] \
    model.backbone.use_layer_norm=true \
    dataset.loader.parameters.adjacency_threshold=0.0005 \
    model.readout.fc_dropout=0.1 \
    trainer.devices=\[3\] \
    &

python -m ogbench \
    model=gatv4 \
    dataset=addneuromed \
    seed=456 \
    logger.wandb.tags=[gatv4,addneuromed,hpsearch,rerun] \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    experiment=omics_readout \
    dataset.loader.parameters.adjacency_method=wgcna \
    dataset.loader.parameters.node_sample_ratio=0.5 \
    dataset.loader.parameters.method=random \
    optimizer.parameters.lr=0.0001 \
    optimizer.parameters.weight_decay=0.0 \
    model.backbone.dropout=0.1 \
    model.backbone.hidden_channels=[32,64] \
    model.backbone.heads=[8,8] \
    model.backbone.use_layer_norm=true \
    dataset.loader.parameters.adjacency_threshold=0.0005 \
    model.readout.fc_dropout=0.1 \
    trainer.devices=\[4\] \
    &

python -m ogbench \
    model=gatv4 \
    dataset=addneuromed \
    seed=42 \
    logger.wandb.tags=[gatv4,addneuromed,hpsearch,rerun] \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    experiment=omics_readout \
    dataset.loader.parameters.adjacency_method=wgcna \
    dataset.loader.parameters.node_sample_ratio=0.5 \
    dataset.loader.parameters.method=random \
    optimizer.parameters.lr=0.0001 \
    optimizer.parameters.weight_decay=0.0 \
    model.backbone.dropout=0.1 \
    model.backbone.hidden_channels=[32,64] \
    model.backbone.heads=[8,8] \
    model.backbone.use_layer_norm=false \
    dataset.loader.parameters.adjacency_threshold=0.0005 \
    model.readout.fc_dropout=0.1 \
    trainer.devices=\[5\] \
    &

python -m ogbench \
    model=gatv2 \
    dataset=motrpac \
    seed=42 \
    logger.wandb.tags=[gatv2,motrpac,hpsearch,rerun] \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    experiment=omics_readout \
    dataset.loader.parameters.adjacency_method=wgcna \
    dataset.loader.parameters.node_sample_ratio=0.3 \
    dataset.loader.parameters.method=distance_correlation \
    optimizer.parameters.lr=0.0001 \
    optimizer.parameters.weight_decay=0.0 \
    model.backbone.dropout=0.1 \
    model.feature_encoder.out_channels=64 \
    model.backbone.num_layers=2 \
    model.backbone.heads=2 \
    dataset.loader.parameters.adjacency_threshold=0.0035 \
    model.readout.fc_dropout=0.1 \
    trainer.devices=\[6\] \
    &

python -m ogbench \
    model=gatv2 \
    dataset=motrpac \
    seed=123 \
    logger.wandb.tags=[gatv2,motrpac,hpsearch,rerun] \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    experiment=omics_readout \
    dataset.loader.parameters.adjacency_method=wgcna \
    dataset.loader.parameters.node_sample_ratio=0.3 \
    dataset.loader.parameters.method=distance_correlation \
    optimizer.parameters.lr=0.0001 \
    optimizer.parameters.weight_decay=0.0 \
    model.backbone.dropout=0.1 \
    model.feature_encoder.out_channels=64 \
    model.backbone.num_layers=2 \
    model.backbone.heads=2 \
    dataset.loader.parameters.adjacency_threshold=0.0035 \
    model.readout.fc_dropout=0.1 \
    trainer.devices=\[7\] \
    &

python -m ogbench \
    model=gatv4 \
    dataset=addneuromed \
    seed=123 \
    logger.wandb.tags=[gatv4,addneuromed,hpsearch,rerun] \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    experiment=omics_readout \
    dataset.loader.parameters.adjacency_method=wgcna \
    dataset.loader.parameters.node_sample_ratio=0.5 \
    dataset.loader.parameters.method=random \
    optimizer.parameters.lr=0.0001 \
    optimizer.parameters.weight_decay=0.0 \
    model.backbone.dropout=0.1 \
    model.backbone.hidden_channels=[32,64] \
    model.backbone.heads=[8,8] \
    model.backbone.use_layer_norm=false \
    dataset.loader.parameters.adjacency_threshold=0.0005 \
    model.readout.fc_dropout=0.1 \
    trainer.devices=\[0\] \
    &

python -m ogbench \
    model=gatv2 \
    dataset=motrpac \
    seed=456 \
    logger.wandb.tags=[gatv2,motrpac,hpsearch,rerun] \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    experiment=omics_readout \
    dataset.loader.parameters.adjacency_method=wgcna \
    dataset.loader.parameters.node_sample_ratio=0.3 \
    dataset.loader.parameters.method=distance_correlation \
    optimizer.parameters.lr=0.0001 \
    optimizer.parameters.weight_decay=0.0 \
    model.backbone.dropout=0.1 \
    model.feature_encoder.out_channels=64 \
    model.backbone.num_layers=2 \
    model.backbone.heads=2 \
    dataset.loader.parameters.adjacency_threshold=0.0035 \
    model.readout.fc_dropout=0.1 \
    trainer.devices=\[1\] \
    &

python -m ogbench \
    model=gatv4 \
    dataset=addneuromed \
    seed=456 \
    logger.wandb.tags=[gatv4,addneuromed,hpsearch,rerun] \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    experiment=omics_readout \
    dataset.loader.parameters.adjacency_method=wgcna \
    dataset.loader.parameters.node_sample_ratio=0.5 \
    dataset.loader.parameters.method=random \
    optimizer.parameters.lr=0.0001 \
    optimizer.parameters.weight_decay=0.0 \
    model.backbone.dropout=0.1 \
    model.backbone.hidden_channels=[32,64] \
    model.backbone.heads=[8,8] \
    model.backbone.use_layer_norm=false \
    dataset.loader.parameters.adjacency_threshold=0.0005 \
    model.readout.fc_dropout=0.1 \
    trainer.devices=\[2\] \
    &

python -m ogbench \
    model=gatv2 \
    dataset=motrpac \
    seed=42 \
    logger.wandb.tags=[gatv2,motrpac,hpsearch,rerun] \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    experiment=omics_readout \
    dataset.loader.parameters.adjacency_method=wgcna \
    dataset.loader.parameters.node_sample_ratio=0.3 \
    dataset.loader.parameters.method=distance_correlation \
    optimizer.parameters.lr=0.0001 \
    optimizer.parameters.weight_decay=0.0 \
    model.backbone.dropout=0.1 \
    model.feature_encoder.out_channels=64 \
    model.backbone.num_layers=2 \
    model.backbone.heads=4 \
    dataset.loader.parameters.adjacency_threshold=0.0035 \
    model.readout.fc_dropout=0.1 \
    trainer.devices=\[3\] \
    &

python -m ogbench \
    model=gatv2 \
    dataset=motrpac \
    seed=123 \
    logger.wandb.tags=[gatv2,motrpac,hpsearch,rerun] \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    experiment=omics_readout \
    dataset.loader.parameters.adjacency_method=wgcna \
    dataset.loader.parameters.node_sample_ratio=0.3 \
    dataset.loader.parameters.method=distance_correlation \
    optimizer.parameters.lr=0.0001 \
    optimizer.parameters.weight_decay=0.0 \
    model.backbone.dropout=0.1 \
    model.feature_encoder.out_channels=64 \
    model.backbone.num_layers=2 \
    model.backbone.heads=4 \
    dataset.loader.parameters.adjacency_threshold=0.0035 \
    model.readout.fc_dropout=0.1 \
    trainer.devices=\[4\] \
    &

python -m ogbench \
    model=gatv2 \
    dataset=motrpac \
    seed=456 \
    logger.wandb.tags=[gatv2,motrpac,hpsearch,rerun] \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    experiment=omics_readout \
    dataset.loader.parameters.adjacency_method=wgcna \
    dataset.loader.parameters.node_sample_ratio=0.3 \
    dataset.loader.parameters.method=distance_correlation \
    optimizer.parameters.lr=0.0001 \
    optimizer.parameters.weight_decay=0.0 \
    model.backbone.dropout=0.1 \
    model.feature_encoder.out_channels=64 \
    model.backbone.num_layers=2 \
    model.backbone.heads=4 \
    dataset.loader.parameters.adjacency_threshold=0.0035 \
    model.readout.fc_dropout=0.1 \
    trainer.devices=\[5\] \
    &

python -m ogbench \
    model=gatv2 \
    dataset=motrpac \
    seed=42 \
    logger.wandb.tags=[gatv2,motrpac,hpsearch,rerun] \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    experiment=omics_readout \
    dataset.loader.parameters.adjacency_method=wgcna \
    dataset.loader.parameters.node_sample_ratio=0.3 \
    dataset.loader.parameters.method=distance_correlation \
    optimizer.parameters.lr=0.0001 \
    optimizer.parameters.weight_decay=0.0 \
    model.backbone.dropout=0.1 \
    model.feature_encoder.out_channels=64 \
    model.backbone.num_layers=4 \
    model.backbone.heads=2 \
    dataset.loader.parameters.adjacency_threshold=0.0035 \
    model.readout.fc_dropout=0.1 \
    trainer.devices=\[6\] \
    &

python -m ogbench \
    model=gatv2 \
    dataset=motrpac \
    seed=123 \
    logger.wandb.tags=[gatv2,motrpac,hpsearch,rerun] \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    experiment=omics_readout \
    dataset.loader.parameters.adjacency_method=wgcna \
    dataset.loader.parameters.node_sample_ratio=0.3 \
    dataset.loader.parameters.method=distance_correlation \
    optimizer.parameters.lr=0.0001 \
    optimizer.parameters.weight_decay=0.0 \
    model.backbone.dropout=0.1 \
    model.feature_encoder.out_channels=64 \
    model.backbone.num_layers=4 \
    model.backbone.heads=2 \
    dataset.loader.parameters.adjacency_threshold=0.0035 \
    model.readout.fc_dropout=0.1 \
    trainer.devices=\[7\] \
    &

python -m ogbench \
    model=gatv4 \
    dataset=addneuromed \
    seed=42 \
    logger.wandb.tags=[gatv4,addneuromed,hpsearch,rerun] \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    experiment=omics_readout \
    dataset.loader.parameters.adjacency_method=wgcna \
    dataset.loader.parameters.node_sample_ratio=0.5 \
    dataset.loader.parameters.method=random \
    optimizer.parameters.lr=0.001 \
    optimizer.parameters.weight_decay=0.0 \
    model.backbone.dropout=0.1 \
    model.backbone.hidden_channels=[16,32] \
    model.backbone.heads=[8,8] \
    model.backbone.use_layer_norm=true \
    dataset.loader.parameters.adjacency_threshold=0.0005 \
    model.readout.fc_dropout=0.1 \
    trainer.devices=\[0\] \
    &

python -m ogbench \
    model=gatv4 \
    dataset=addneuromed \
    seed=123 \
    logger.wandb.tags=[gatv4,addneuromed,hpsearch,rerun] \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    experiment=omics_readout \
    dataset.loader.parameters.adjacency_method=wgcna \
    dataset.loader.parameters.node_sample_ratio=0.5 \
    dataset.loader.parameters.method=random \
    optimizer.parameters.lr=0.001 \
    optimizer.parameters.weight_decay=0.0 \
    model.backbone.dropout=0.1 \
    model.backbone.hidden_channels=[16,32] \
    model.backbone.heads=[8,8] \
    model.backbone.use_layer_norm=true \
    dataset.loader.parameters.adjacency_threshold=0.0005 \
    model.readout.fc_dropout=0.1 \
    trainer.devices=\[1\] \
    &

python -m ogbench \
    model=gatv4 \
    dataset=addneuromed \
    seed=456 \
    logger.wandb.tags=[gatv4,addneuromed,hpsearch,rerun] \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    experiment=omics_readout \
    dataset.loader.parameters.adjacency_method=wgcna \
    dataset.loader.parameters.node_sample_ratio=0.5 \
    dataset.loader.parameters.method=random \
    optimizer.parameters.lr=0.001 \
    optimizer.parameters.weight_decay=0.0 \
    model.backbone.dropout=0.1 \
    model.backbone.hidden_channels=[16,32] \
    model.backbone.heads=[8,8] \
    model.backbone.use_layer_norm=true \
    dataset.loader.parameters.adjacency_threshold=0.0005 \
    model.readout.fc_dropout=0.1 \
    trainer.devices=\[2\] \
    &

python -m ogbench \
    model=gatv4 \
    dataset=addneuromed \
    seed=42 \
    logger.wandb.tags=[gatv4,addneuromed,hpsearch,rerun] \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    experiment=omics_readout \
    dataset.loader.parameters.adjacency_method=wgcna \
    dataset.loader.parameters.node_sample_ratio=0.5 \
    dataset.loader.parameters.method=random \
    optimizer.parameters.lr=0.001 \
    optimizer.parameters.weight_decay=0.0 \
    model.backbone.dropout=0.1 \
    model.backbone.hidden_channels=[16,32] \
    model.backbone.heads=[8,8] \
    model.backbone.use_layer_norm=false \
    dataset.loader.parameters.adjacency_threshold=0.0005 \
    model.readout.fc_dropout=0.1 \
    trainer.devices=\[3\] \
    &

python -m ogbench \
    model=gatv4 \
    dataset=addneuromed \
    seed=123 \
    logger.wandb.tags=[gatv4,addneuromed,hpsearch,rerun] \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    experiment=omics_readout \
    dataset.loader.parameters.adjacency_method=wgcna \
    dataset.loader.parameters.node_sample_ratio=0.5 \
    dataset.loader.parameters.method=random \
    optimizer.parameters.lr=0.001 \
    optimizer.parameters.weight_decay=0.0 \
    model.backbone.dropout=0.1 \
    model.backbone.hidden_channels=[16,32] \
    model.backbone.heads=[8,8] \
    model.backbone.use_layer_norm=false \
    dataset.loader.parameters.adjacency_threshold=0.0005 \
    model.readout.fc_dropout=0.1 \
    trainer.devices=\[4\] \
    &

python -m ogbench \
    model=gatv4 \
    dataset=addneuromed \
    seed=456 \
    logger.wandb.tags=[gatv4,addneuromed,hpsearch,rerun] \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    experiment=omics_readout \
    dataset.loader.parameters.adjacency_method=wgcna \
    dataset.loader.parameters.node_sample_ratio=0.5 \
    dataset.loader.parameters.method=random \
    optimizer.parameters.lr=0.001 \
    optimizer.parameters.weight_decay=0.0 \
    model.backbone.dropout=0.1 \
    model.backbone.hidden_channels=[16,32] \
    model.backbone.heads=[8,8] \
    model.backbone.use_layer_norm=false \
    dataset.loader.parameters.adjacency_threshold=0.0005 \
    model.readout.fc_dropout=0.1 \
    trainer.devices=\[5\] \
    &

python -m ogbench \
    model=gatv4 \
    dataset=addneuromed \
    seed=42 \
    logger.wandb.tags=[gatv4,addneuromed,hpsearch,rerun] \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    experiment=omics_readout \
    dataset.loader.parameters.adjacency_method=wgcna \
    dataset.loader.parameters.node_sample_ratio=0.5 \
    dataset.loader.parameters.method=random \
    optimizer.parameters.lr=0.001 \
    optimizer.parameters.weight_decay=0.0 \
    model.backbone.dropout=0.1 \
    model.backbone.hidden_channels=[32,64] \
    model.backbone.heads=[4,4] \
    model.backbone.use_layer_norm=true \
    dataset.loader.parameters.adjacency_threshold=0.0005 \
    model.readout.fc_dropout=0.1 \
    trainer.devices=\[6\] \
    &

python -m ogbench \
    model=gatv4 \
    dataset=addneuromed \
    seed=123 \
    logger.wandb.tags=[gatv4,addneuromed,hpsearch,rerun] \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    experiment=omics_readout \
    dataset.loader.parameters.adjacency_method=wgcna \
    dataset.loader.parameters.node_sample_ratio=0.5 \
    dataset.loader.parameters.method=random \
    optimizer.parameters.lr=0.001 \
    optimizer.parameters.weight_decay=0.0 \
    model.backbone.dropout=0.1 \
    model.backbone.hidden_channels=[32,64] \
    model.backbone.heads=[4,4] \
    model.backbone.use_layer_norm=true \
    dataset.loader.parameters.adjacency_threshold=0.0005 \
    model.readout.fc_dropout=0.1 \
    trainer.devices=\[7\] \
    &

python -m ogbench \
    model=gatv4 \
    dataset=addneuromed \
    seed=456 \
    logger.wandb.tags=[gatv4,addneuromed,hpsearch,rerun] \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    experiment=omics_readout \
    dataset.loader.parameters.adjacency_method=wgcna \
    dataset.loader.parameters.node_sample_ratio=0.5 \
    dataset.loader.parameters.method=random \
    optimizer.parameters.lr=0.001 \
    optimizer.parameters.weight_decay=0.0 \
    model.backbone.dropout=0.1 \
    model.backbone.hidden_channels=[32,64] \
    model.backbone.heads=[4,4] \
    model.backbone.use_layer_norm=true \
    dataset.loader.parameters.adjacency_threshold=0.0005 \
    model.readout.fc_dropout=0.1 \
    trainer.devices=\[0\] \
    &

python -m ogbench \
    model=gatv4 \
    dataset=addneuromed \
    seed=42 \
    logger.wandb.tags=[gatv4,addneuromed,hpsearch,rerun] \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    experiment=omics_readout \
    dataset.loader.parameters.adjacency_method=wgcna \
    dataset.loader.parameters.node_sample_ratio=0.5 \
    dataset.loader.parameters.method=random \
    optimizer.parameters.lr=0.001 \
    optimizer.parameters.weight_decay=0.0 \
    model.backbone.dropout=0.1 \
    model.backbone.hidden_channels=[32,64] \
    model.backbone.heads=[4,4] \
    model.backbone.use_layer_norm=false \
    dataset.loader.parameters.adjacency_threshold=0.0005 \
    model.readout.fc_dropout=0.1 \
    trainer.devices=\[1\] \
    &

python -m ogbench \
    model=gatv4 \
    dataset=addneuromed \
    seed=123 \
    logger.wandb.tags=[gatv4,addneuromed,hpsearch,rerun] \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    experiment=omics_readout \
    dataset.loader.parameters.adjacency_method=wgcna \
    dataset.loader.parameters.node_sample_ratio=0.5 \
    dataset.loader.parameters.method=random \
    optimizer.parameters.lr=0.001 \
    optimizer.parameters.weight_decay=0.0 \
    model.backbone.dropout=0.1 \
    model.backbone.hidden_channels=[32,64] \
    model.backbone.heads=[4,4] \
    model.backbone.use_layer_norm=false \
    dataset.loader.parameters.adjacency_threshold=0.0005 \
    model.readout.fc_dropout=0.1 \
    trainer.devices=\[2\] \
    &

python -m ogbench \
    model=gatv4 \
    dataset=addneuromed \
    seed=42 \
    logger.wandb.tags=[gatv4,addneuromed,hpsearch,rerun] \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    experiment=omics_readout \
    dataset.loader.parameters.adjacency_method=wgcna \
    dataset.loader.parameters.node_sample_ratio=0.5 \
    dataset.loader.parameters.method=random \
    optimizer.parameters.lr=0.001 \
    optimizer.parameters.weight_decay=0.0 \
    model.backbone.dropout=0.1 \
    model.backbone.hidden_channels=[32,64] \
    model.backbone.heads=[8,8] \
    model.backbone.use_layer_norm=true \
    dataset.loader.parameters.adjacency_threshold=0.0005 \
    model.readout.fc_dropout=0.1 \
    trainer.devices=\[3\] \
    &

python -m ogbench \
    model=gatv4 \
    dataset=addneuromed \
    seed=123 \
    logger.wandb.tags=[gatv4,addneuromed,hpsearch,rerun] \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    experiment=omics_readout \
    dataset.loader.parameters.adjacency_method=wgcna \
    dataset.loader.parameters.node_sample_ratio=0.5 \
    dataset.loader.parameters.method=random \
    optimizer.parameters.lr=0.001 \
    optimizer.parameters.weight_decay=0.0 \
    model.backbone.dropout=0.1 \
    model.backbone.hidden_channels=[32,64] \
    model.backbone.heads=[8,8] \
    model.backbone.use_layer_norm=true \
    dataset.loader.parameters.adjacency_threshold=0.0005 \
    model.readout.fc_dropout=0.1 \
    trainer.devices=\[4\] \
    &

python -m ogbench \
    model=gatv4 \
    dataset=addneuromed \
    seed=456 \
    logger.wandb.tags=[gatv4,addneuromed,hpsearch,rerun] \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    experiment=omics_readout \
    dataset.loader.parameters.adjacency_method=wgcna \
    dataset.loader.parameters.node_sample_ratio=0.5 \
    dataset.loader.parameters.method=random \
    optimizer.parameters.lr=0.001 \
    optimizer.parameters.weight_decay=0.0 \
    model.backbone.dropout=0.1 \
    model.backbone.hidden_channels=[32,64] \
    model.backbone.heads=[8,8] \
    model.backbone.use_layer_norm=true \
    dataset.loader.parameters.adjacency_threshold=0.0005 \
    model.readout.fc_dropout=0.1 \
    trainer.devices=\[5\] \
    &

python -m ogbench \
    model=gatv4 \
    dataset=addneuromed \
    seed=123 \
    logger.wandb.tags=[gatv4,addneuromed,hpsearch,rerun] \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    experiment=omics_readout \
    dataset.loader.parameters.adjacency_method=wgcna \
    dataset.loader.parameters.node_sample_ratio=0.5 \
    dataset.loader.parameters.method=random \
    optimizer.parameters.lr=0.001 \
    optimizer.parameters.weight_decay=0.0 \
    model.backbone.dropout=0.1 \
    model.backbone.hidden_channels=[32,64] \
    model.backbone.heads=[8,8] \
    model.backbone.use_layer_norm=false \
    dataset.loader.parameters.adjacency_threshold=0.0005 \
    model.readout.fc_dropout=0.1 \
    trainer.devices=\[6\] \
    &

python -m ogbench \
    model=gatv4 \
    dataset=addneuromed \
    seed=456 \
    logger.wandb.tags=[gatv4,addneuromed,hpsearch,rerun] \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    experiment=omics_readout \
    dataset.loader.parameters.adjacency_method=wgcna \
    dataset.loader.parameters.node_sample_ratio=0.5 \
    dataset.loader.parameters.method=random \
    optimizer.parameters.lr=0.001 \
    optimizer.parameters.weight_decay=0.0 \
    model.backbone.dropout=0.1 \
    model.backbone.hidden_channels=[32,64] \
    model.backbone.heads=[8,8] \
    model.backbone.use_layer_norm=false \
    dataset.loader.parameters.adjacency_threshold=0.0005 \
    model.readout.fc_dropout=0.1 \
    trainer.devices=\[7\] \
    &

python -m ogbench \
    model=gatv4 \
    dataset=addneuromed \
    seed=123 \
    logger.wandb.tags=[gatv4,addneuromed,hpsearch,rerun] \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    experiment=omics_readout \
    dataset.loader.parameters.adjacency_method=wgcna \
    dataset.loader.parameters.node_sample_ratio=0.5 \
    dataset.loader.parameters.method=correlation \
    optimizer.parameters.lr=0.0001 \
    optimizer.parameters.weight_decay=0.0 \
    model.backbone.dropout=0.1 \
    model.backbone.hidden_channels=[32,64] \
    model.backbone.heads=[8,8] \
    model.backbone.use_layer_norm=true \
    dataset.loader.parameters.adjacency_threshold=0.0971 \
    model.readout.fc_dropout=0.1 \
    trainer.devices=\[0\] \
    &

python -m ogbench \
    model=gatv4 \
    dataset=addneuromed \
    seed=456 \
    logger.wandb.tags=[gatv4,addneuromed,hpsearch,rerun] \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    experiment=omics_readout \
    dataset.loader.parameters.adjacency_method=wgcna \
    dataset.loader.parameters.node_sample_ratio=0.5 \
    dataset.loader.parameters.method=correlation \
    optimizer.parameters.lr=0.0001 \
    optimizer.parameters.weight_decay=0.0 \
    model.backbone.dropout=0.1 \
    model.backbone.hidden_channels=[32,64] \
    model.backbone.heads=[8,8] \
    model.backbone.use_layer_norm=true \
    dataset.loader.parameters.adjacency_threshold=0.0971 \
    model.readout.fc_dropout=0.1 \
    trainer.devices=\[1\] \
    &

python -m ogbench \
    model=gatv4 \
    dataset=motrpac \
    seed=123 \
    logger.wandb.tags=[gatv4,motrpac,hpsearch,rerun] \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    experiment=omics_readout \
    dataset.loader.parameters.adjacency_method=wgcna \
    dataset.loader.parameters.node_sample_ratio=0.5 \
    dataset.loader.parameters.method=variance \
    optimizer.parameters.lr=0.001 \
    optimizer.parameters.weight_decay=0.0 \
    model.backbone.dropout=0.1 \
    model.backbone.hidden_channels=[32,64] \
    model.backbone.heads=[8,8] \
    model.backbone.use_layer_norm=false \
    dataset.loader.parameters.adjacency_threshold=0.0572 \
    model.readout.fc_dropout=0.1 \
    trainer.devices=\[2\] \
    &

python -m ogbench \
    model=gatv2 \
    dataset=brca \
    seed=42 \
    logger.wandb.tags=[gatv2,brca,hpsearch,rerun] \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    experiment=omics_readout \
    dataset.loader.parameters.adjacency_method=wgcna \
    dataset.loader.parameters.node_sample_ratio=0.3 \
    dataset.loader.parameters.method=variance \
    optimizer.parameters.lr=0.001 \
    optimizer.parameters.weight_decay=0.0 \
    model.backbone.dropout=0.1 \
    model.feature_encoder.out_channels=64 \
    model.backbone.num_layers=4 \
    model.backbone.heads=4 \
    dataset.loader.parameters.adjacency_threshold=0.0008 \
    model.readout.fc_dropout=0.1 \
    trainer.devices=\[3\] \
    &

python -m ogbench \
    model=gatv4 \
    dataset=brca \
    seed=123 \
    logger.wandb.tags=[gatv4,brca,hpsearch,rerun] \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    experiment=no_readout \
    dataset.loader.parameters.adjacency_method=wgcna \
    dataset.loader.parameters.node_sample_ratio=0.5 \
    dataset.loader.parameters.method=random \
    optimizer.parameters.lr=0.001 \
    optimizer.parameters.weight_decay=0.0 \
    model.backbone.dropout=0.1 \
    model.backbone.hidden_channels=[32,64] \
    model.backbone.heads=[8,8] \
    model.backbone.use_layer_norm=false \
    dataset.loader.parameters.adjacency_threshold=0.0021 \
    trainer.devices=\[4\] \
    &

python -m ogbench \
    model=gatv4 \
    dataset=brca \
    seed=123 \
    logger.wandb.tags=[gatv4,brca,hpsearch,rerun] \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    experiment=no_readout \
    dataset.loader.parameters.adjacency_method=wgcna \
    dataset.loader.parameters.node_sample_ratio=0.5 \
    dataset.loader.parameters.method=correlation \
    optimizer.parameters.lr=0.0001 \
    optimizer.parameters.weight_decay=0.0 \
    model.backbone.dropout=0.1 \
    model.backbone.hidden_channels=[32,64] \
    model.backbone.heads=[8,8] \
    model.backbone.use_layer_norm=false \
    dataset.loader.parameters.adjacency_threshold=0.0009 \
    trainer.devices=\[5\] \
    &

python -m ogbench \
    model=gatv4 \
    dataset=brca \
    seed=42 \
    logger.wandb.tags=[gatv4,brca,hpsearch,rerun] \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    experiment=no_readout \
    dataset.loader.parameters.adjacency_method=wgcna \
    dataset.loader.parameters.node_sample_ratio=0.5 \
    dataset.loader.parameters.method=correlation \
    optimizer.parameters.lr=0.001 \
    optimizer.parameters.weight_decay=0.0 \
    model.backbone.dropout=0.1 \
    model.backbone.hidden_channels=[32,64] \
    model.backbone.heads=[8,8] \
    model.backbone.use_layer_norm=false \
    dataset.loader.parameters.adjacency_threshold=0.0009 \
    trainer.devices=\[6\] \
    &

python -m ogbench \
    model=gatv4 \
    dataset=brca \
    seed=123 \
    logger.wandb.tags=[gatv4,brca,hpsearch,rerun] \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    experiment=no_readout \
    dataset.loader.parameters.adjacency_method=wgcna \
    dataset.loader.parameters.node_sample_ratio=0.5 \
    dataset.loader.parameters.method=correlation \
    optimizer.parameters.lr=0.001 \
    optimizer.parameters.weight_decay=0.0 \
    model.backbone.dropout=0.1 \
    model.backbone.hidden_channels=[32,64] \
    model.backbone.heads=[8,8] \
    model.backbone.use_layer_norm=false \
    dataset.loader.parameters.adjacency_threshold=0.0009 \
    trainer.devices=\[7\] \
    &

python -m ogbench \
    model=gatv4 \
    dataset=brca \
    seed=456 \
    logger.wandb.tags=[gatv4,brca,hpsearch,rerun] \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    experiment=omics_readout \
    dataset.loader.parameters.adjacency_method=wgcna \
    dataset.loader.parameters.node_sample_ratio=0.5 \
    dataset.loader.parameters.method=variance \
    optimizer.parameters.lr=0.0001 \
    optimizer.parameters.weight_decay=0.0 \
    model.backbone.dropout=0.1 \
    model.backbone.hidden_channels=[32,64] \
    model.backbone.heads=[8,8] \
    model.backbone.use_layer_norm=true \
    dataset.loader.parameters.adjacency_threshold=0.0015 \
    model.readout.fc_dropout=0.1 \
    trainer.devices=\[0\] \
    &

python -m ogbench \
    model=gatv4 \
    dataset=brca \
    seed=42 \
    logger.wandb.tags=[gatv4,brca,hpsearch,rerun] \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    experiment=omics_readout \
    dataset.loader.parameters.adjacency_method=wgcna \
    dataset.loader.parameters.node_sample_ratio=0.5 \
    dataset.loader.parameters.method=variance \
    optimizer.parameters.lr=0.0001 \
    optimizer.parameters.weight_decay=0.0 \
    model.backbone.dropout=0.1 \
    model.backbone.hidden_channels=[32,64] \
    model.backbone.heads=[8,8] \
    model.backbone.use_layer_norm=false \
    dataset.loader.parameters.adjacency_threshold=0.0015 \
    model.readout.fc_dropout=0.1 \
    trainer.devices=\[1\] \
    &

python -m ogbench \
    model=gatv4 \
    dataset=brca \
    seed=123 \
    logger.wandb.tags=[gatv4,brca,hpsearch,rerun] \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    experiment=omics_readout \
    dataset.loader.parameters.adjacency_method=wgcna \
    dataset.loader.parameters.node_sample_ratio=0.5 \
    dataset.loader.parameters.method=variance \
    optimizer.parameters.lr=0.0001 \
    optimizer.parameters.weight_decay=0.0 \
    model.backbone.dropout=0.1 \
    model.backbone.hidden_channels=[32,64] \
    model.backbone.heads=[8,8] \
    model.backbone.use_layer_norm=false \
    dataset.loader.parameters.adjacency_threshold=0.0015 \
    model.readout.fc_dropout=0.1 \
    trainer.devices=\[2\] \
    &

python -m ogbench \
    model=gatv2 \
    dataset=brca \
    seed=456 \
    logger.wandb.tags=[gatv2,brca,hpsearch,rerun] \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    experiment=omics_readout \
    dataset.loader.parameters.adjacency_method=wgcna \
    dataset.loader.parameters.node_sample_ratio=0.3 \
    dataset.loader.parameters.method=distance_correlation \
    optimizer.parameters.lr=0.0001 \
    optimizer.parameters.weight_decay=0.0 \
    model.backbone.dropout=0.1 \
    model.feature_encoder.out_channels=64 \
    model.backbone.num_layers=4 \
    model.backbone.heads=4 \
    dataset.loader.parameters.adjacency_threshold=0.0005 \
    model.readout.fc_dropout=0.1 \
    trainer.devices=\[3\] \
    &

python -m ogbench \
    model=gatv4 \
    dataset=brca \
    seed=456 \
    logger.wandb.tags=[gatv4,brca,hpsearch,rerun] \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    experiment=omics_readout \
    dataset.loader.parameters.adjacency_method=wgcna \
    dataset.loader.parameters.node_sample_ratio=0.5 \
    dataset.loader.parameters.method=random \
    optimizer.parameters.lr=0.0001 \
    optimizer.parameters.weight_decay=0.0 \
    model.backbone.dropout=0.1 \
    model.backbone.hidden_channels=[32,64] \
    model.backbone.heads=[8,8] \
    model.backbone.use_layer_norm=true \
    dataset.loader.parameters.adjacency_threshold=0.0021 \
    model.readout.fc_dropout=0.1 \
    trainer.devices=\[4\] \
    &

python -m ogbench \
    model=gatv4 \
    dataset=brca \
    seed=456 \
    logger.wandb.tags=[gatv4,brca,hpsearch,rerun] \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    experiment=omics_readout \
    dataset.loader.parameters.adjacency_method=wgcna \
    dataset.loader.parameters.node_sample_ratio=0.5 \
    dataset.loader.parameters.method=random \
    optimizer.parameters.lr=0.0001 \
    optimizer.parameters.weight_decay=0.0 \
    model.backbone.dropout=0.1 \
    model.backbone.hidden_channels=[32,64] \
    model.backbone.heads=[8,8] \
    model.backbone.use_layer_norm=false \
    dataset.loader.parameters.adjacency_threshold=0.0021 \
    model.readout.fc_dropout=0.1 \
    trainer.devices=\[5\] \
    &

python -m ogbench \
    model=gatv4 \
    dataset=brca \
    seed=456 \
    logger.wandb.tags=[gatv4,brca,hpsearch,rerun] \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    experiment=omics_readout \
    dataset.loader.parameters.adjacency_method=wgcna \
    dataset.loader.parameters.node_sample_ratio=0.5 \
    dataset.loader.parameters.method=correlation \
    optimizer.parameters.lr=0.0001 \
    optimizer.parameters.weight_decay=0.0 \
    model.backbone.dropout=0.1 \
    model.backbone.hidden_channels=[32,64] \
    model.backbone.heads=[8,8] \
    model.backbone.use_layer_norm=true \
    dataset.loader.parameters.adjacency_threshold=0.0009 \
    model.readout.fc_dropout=0.1 \
    trainer.devices=\[6\] \
    &

python -m ogbench \
    model=gatv4 \
    dataset=brca \
    seed=42 \
    logger.wandb.tags=[gatv4,brca,hpsearch,rerun] \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    experiment=omics_readout \
    dataset.loader.parameters.adjacency_method=wgcna \
    dataset.loader.parameters.node_sample_ratio=0.5 \
    dataset.loader.parameters.method=correlation \
    optimizer.parameters.lr=0.0001 \
    optimizer.parameters.weight_decay=0.0 \
    model.backbone.dropout=0.1 \
    model.backbone.hidden_channels=[32,64] \
    model.backbone.heads=[8,8] \
    model.backbone.use_layer_norm=false \
    dataset.loader.parameters.adjacency_threshold=0.0009 \
    model.readout.fc_dropout=0.1 \
    trainer.devices=\[7\] \
    &

python -m ogbench \
    model=gatv4 \
    dataset=brca \
    seed=123 \
    logger.wandb.tags=[gatv4,brca,hpsearch,rerun] \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    experiment=omics_readout \
    dataset.loader.parameters.adjacency_method=wgcna \
    dataset.loader.parameters.node_sample_ratio=0.5 \
    dataset.loader.parameters.method=correlation \
    optimizer.parameters.lr=0.0001 \
    optimizer.parameters.weight_decay=0.0 \
    model.backbone.dropout=0.1 \
    model.backbone.hidden_channels=[32,64] \
    model.backbone.heads=[8,8] \
    model.backbone.use_layer_norm=false \
    dataset.loader.parameters.adjacency_threshold=0.0009 \
    model.readout.fc_dropout=0.1 \
    trainer.devices=\[0\] \
    &

python -m ogbench \
    model=gatv4 \
    dataset=brca \
    seed=456 \
    logger.wandb.tags=[gatv4,brca,hpsearch,rerun] \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    experiment=omics_readout \
    dataset.loader.parameters.adjacency_method=wgcna \
    dataset.loader.parameters.node_sample_ratio=0.5 \
    dataset.loader.parameters.method=correlation \
    optimizer.parameters.lr=0.0001 \
    optimizer.parameters.weight_decay=0.0 \
    model.backbone.dropout=0.1 \
    model.backbone.hidden_channels=[32,64] \
    model.backbone.heads=[8,8] \
    model.backbone.use_layer_norm=false \
    dataset.loader.parameters.adjacency_threshold=0.0009 \
    model.readout.fc_dropout=0.1 \
    trainer.devices=\[1\] \
    &

python -m ogbench \
    model=gatv4 \
    dataset=brca \
    seed=456 \
    logger.wandb.tags=[gatv4,brca,hpsearch,rerun] \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    experiment=omics_readout \
    dataset.loader.parameters.adjacency_method=wgcna \
    dataset.loader.parameters.node_sample_ratio=0.5 \
    dataset.loader.parameters.method=correlation \
    optimizer.parameters.lr=0.001 \
    optimizer.parameters.weight_decay=0.0 \
    model.backbone.dropout=0.1 \
    model.backbone.hidden_channels=[32,64] \
    model.backbone.heads=[8,8] \
    model.backbone.use_layer_norm=true \
    dataset.loader.parameters.adjacency_threshold=0.0009 \
    model.readout.fc_dropout=0.1 \
    trainer.devices=\[2\] \
    &

python -m ogbench \
    model=gatv4 \
    dataset=brca \
    seed=42 \
    logger.wandb.tags=[gatv4,brca,hpsearch,rerun] \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    experiment=omics_readout \
    dataset.loader.parameters.adjacency_method=wgcna \
    dataset.loader.parameters.node_sample_ratio=0.5 \
    dataset.loader.parameters.method=correlation \
    optimizer.parameters.lr=0.001 \
    optimizer.parameters.weight_decay=0.0 \
    model.backbone.dropout=0.1 \
    model.backbone.hidden_channels=[32,64] \
    model.backbone.heads=[8,8] \
    model.backbone.use_layer_norm=false \
    dataset.loader.parameters.adjacency_threshold=0.0009 \
    model.readout.fc_dropout=0.1 \
    trainer.devices=\[3\] \
    &

python -m ogbench \
    model=gps \
    dataset=addneuromed \
    seed=123 \
    logger.wandb.tags=[gps,addneuromed,hpsearch,rerun] \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    experiment=no_readout \
    dataset.loader.parameters.adjacency_method=wgcna \
    dataset.loader.parameters.node_sample_ratio=0.3 \
    dataset.loader.parameters.method=random \
    optimizer.parameters.lr=0.001 \
    optimizer.parameters.weight_decay=0.0 \
    model.backbone.dropout=0.1 \
    model.feature_encoder.out_channels=16 \
    model.backbone.num_layers=4 \
    model.encodings=[RWSE] \
    dataset.loader.parameters.adjacency_threshold=0.0005 \
    trainer.devices=\[4\] \
    &

python -m ogbench \
    model=gps \
    dataset=addneuromed \
    seed=123 \
    logger.wandb.tags=[gps,addneuromed,hpsearch,rerun] \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    experiment=no_readout \
    dataset.loader.parameters.adjacency_method=wgcna \
    dataset.loader.parameters.node_sample_ratio=0.3 \
    dataset.loader.parameters.method=distance_correlation \
    optimizer.parameters.lr=0.001 \
    optimizer.parameters.weight_decay=0.0 \
    model.backbone.dropout=0.1 \
    model.feature_encoder.out_channels=64 \
    model.backbone.num_layers=4 \
    model.encodings=[LapPE] \
    dataset.loader.parameters.adjacency_threshold=0.1399 \
    trainer.devices=\[5\] \
    &

python -m ogbench \
    model=gps \
    dataset=addneuromed \
    seed=42 \
    logger.wandb.tags=[gps,addneuromed,hpsearch,rerun] \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    experiment=no_readout \
    dataset.loader.parameters.adjacency_method=string \
    dataset.loader.parameters.node_sample_ratio=0.3 \
    dataset.loader.parameters.method=variance \
    optimizer.parameters.lr=0.001 \
    optimizer.parameters.weight_decay=0.0 \
    model.backbone.dropout=0.1 \
    model.feature_encoder.out_channels=16 \
    model.backbone.num_layers=4 \
    model.encodings=[RWSE] \
    dataset.loader.parameters.adjacency_threshold=0.4 \
    trainer.devices=\[6\] \
    &

wait

# --- skipped (inspect in W&B UI or re-export) ---
# iisj5gvh	no args in summary/metadata
# tmexw7x9	no args in summary/metadata
# 3p6hu840	no args in summary/metadata
# 4a7ehcf6	no args in summary/metadata
# gskf4oel	no args in summary/metadata
# 971yaeuz	no args in summary/metadata
# j8qzgw4c	no args in summary/metadata
# 94qwoi09	no args in summary/metadata
# 2lclhmj0	no args in summary/metadata
# oe065qcg	no args in summary/metadata
# k8x3mbh7	no args in summary/metadata
# etc0fza2	no args in summary/metadata
# egavc79a	no args in summary/metadata
# vo336ngk	no args in summary/metadata
# 9ig2kzwd	no args in summary/metadata
# lxyn1rxr	no args in summary/metadata
# qcyeb9vz	no args in summary/metadata
# w4182x1z	no args in summary/metadata
# 4ovhfq5o	no args in summary/metadata
# 2e70qmrp	no args in summary/metadata
# hru4eou9	no args in summary/metadata
# tajm6p3n	no args in summary/metadata
# 2ybj0ftl	no args in summary/metadata
# 9gfb25yg	no args in summary/metadata
# 0i97uqx4	no args in summary/metadata
# 1tv2oiat	no args in summary/metadata
# 49enw3sq	no args in summary/metadata
# f61j9vnd	no args in summary/metadata
# bk9s2ber	no args in summary/metadata
# gej23096	no args in summary/metadata
# 8gwo4fad	no args in summary/metadata
# ny6vllhm	no args in summary/metadata
# hxa0jdv8	no args in summary/metadata
# 4hufmuzf	no args in summary/metadata
# pr0lxb7w	no args in summary/metadata
# d8ljpixu	no args in summary/metadata
# qaer39zg	no args in summary/metadata
# obsfwmfq	no args in summary/metadata
# ncv55tqz	no args in summary/metadata
# wpulmkmo	no args in summary/metadata
# lxnviffq	no args in summary/metadata
# 1a121hyz	no args in summary/metadata
# gpblxfqq	no args in summary/metadata
# 58gukfsr	no args in summary/metadata
# 1oe98x3c	no args in summary/metadata
# dlc7t0mb	no args in summary/metadata
# 25dc7k0w	no args in summary/metadata
# n3odom7k	no args in summary/metadata
# 5pdtgkma	no args in summary/metadata
# wwegoiy3	no args in summary/metadata
# zamz71uk	no args in summary/metadata
# nw6tr8me	no args in summary/metadata
# 6sliynk1	no args in summary/metadata
# clk4z9x8	no args in summary/metadata
