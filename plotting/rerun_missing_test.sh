#!/bin/bash
# Re-run training: rows missing best_test_f1_macro in the export CSV.
# Overrides from W&B (summary or metadata ``args``).
# format=cluster  n_gpus=8  launcher='python -m ogbench'  background=True

python -m ogbench \
    model=gcn \
    dataset=parkinsons \
    seed=42 \
    logger.wandb.tags=[gcn,parkinsons,hpsearch] \
    hydra.run.dir=${paths.log_dir}/${task_name}/runs/hpsearch_0031 \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    paths.root_dir=/scratch/lcornelis/bgbench \
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
    logger.wandb.tags=[gcn,parkinsons,hpsearch] \
    hydra.run.dir=${paths.log_dir}/${task_name}/runs/hpsearch_0133 \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    paths.root_dir=/scratch/lcornelis/bgbench \
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
    logger.wandb.tags=[gcn,parkinsons,hpsearch] \
    hydra.run.dir=${paths.log_dir}/${task_name}/runs/hpsearch_0149 \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    paths.root_dir=/scratch/lcornelis/bgbench \
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
    logger.wandb.tags=[gcn,parkinsons,hpsearch] \
    hydra.run.dir=${paths.log_dir}/${task_name}/runs/hpsearch_0174 \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    paths.root_dir=/scratch/lcornelis/bgbench \
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
    logger.wandb.tags=[gcn,parkinsons,hpsearch] \
    hydra.run.dir=${paths.log_dir}/${task_name}/runs/hpsearch_0202 \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    paths.root_dir=/scratch/lcornelis/bgbench \
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
    logger.wandb.tags=[gcn,parkinsons,hpsearch] \
    hydra.run.dir=${paths.log_dir}/${task_name}/runs/hpsearch_0248 \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    paths.root_dir=/scratch/lcornelis/bgbench \
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
    logger.wandb.tags=[gcn,parkinsons,hpsearch] \
    hydra.run.dir=${paths.log_dir}/${task_name}/runs/hpsearch_0255 \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    paths.root_dir=/scratch/lcornelis/bgbench \
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
    logger.wandb.tags=[gcn,parkinsons,hpsearch] \
    hydra.run.dir=${paths.log_dir}/${task_name}/runs/hpsearch_0269 \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    paths.root_dir=/scratch/lcornelis/bgbench \
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
    logger.wandb.tags=[gcn,parkinsons,hpsearch] \
    hydra.run.dir=${paths.log_dir}/${task_name}/runs/hpsearch_0279 \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    paths.root_dir=/scratch/lcornelis/bgbench \
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
    logger.wandb.tags=[gcn,parkinsons,hpsearch] \
    hydra.run.dir=${paths.log_dir}/${task_name}/runs/hpsearch_0289 \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    paths.root_dir=/scratch/lcornelis/bgbench \
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
    logger.wandb.tags=[gcn,parkinsons,hpsearch] \
    hydra.run.dir=${paths.log_dir}/${task_name}/runs/hpsearch_0439 \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    paths.root_dir=/scratch/lcornelis/bgbench \
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
    logger.wandb.tags=[gcn,parkinsons,hpsearch] \
    hydra.run.dir=${paths.log_dir}/${task_name}/runs/hpsearch_0616 \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    paths.root_dir=/scratch/lcornelis/bgbench \
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
    logger.wandb.tags=[gcn,parkinsons,hpsearch] \
    hydra.run.dir=${paths.log_dir}/${task_name}/runs/hpsearch_0668 \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    paths.root_dir=/scratch/lcornelis/bgbench \
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
    logger.wandb.tags=[gcn,parkinsons,hpsearch] \
    hydra.run.dir=${paths.log_dir}/${task_name}/runs/hpsearch_0744 \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    paths.root_dir=/scratch/lcornelis/bgbench \
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
    logger.wandb.tags=[gcn,parkinsons,hpsearch] \
    hydra.run.dir=${paths.log_dir}/${task_name}/runs/hpsearch_0752 \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    paths.root_dir=/scratch/lcornelis/bgbench \
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
    logger.wandb.tags=[gcn,parkinsons,hpsearch] \
    hydra.run.dir=${paths.log_dir}/${task_name}/runs/hpsearch_0759 \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    paths.root_dir=/scratch/lcornelis/bgbench \
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
    logger.wandb.tags=[gcn,addneuromed,hpsearch] \
    hydra.run.dir=${paths.log_dir}/${task_name}/runs/hpsearch_0866 \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    paths.root_dir=/scratch/lcornelis/bgbench \
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
    logger.wandb.tags=[gcn,addneuromed,hpsearch] \
    hydra.run.dir=${paths.log_dir}/${task_name}/runs/hpsearch_0881 \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    paths.root_dir=/scratch/lcornelis/bgbench \
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
    logger.wandb.tags=[gcn,addneuromed,hpsearch] \
    hydra.run.dir=${paths.log_dir}/${task_name}/runs/hpsearch_0917 \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    paths.root_dir=/scratch/lcornelis/bgbench \
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
    logger.wandb.tags=[gcn,addneuromed,hpsearch] \
    hydra.run.dir=${paths.log_dir}/${task_name}/runs/hpsearch_0933 \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    paths.root_dir=/scratch/lcornelis/bgbench \
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
    logger.wandb.tags=[gcn,addneuromed,hpsearch] \
    hydra.run.dir=${paths.log_dir}/${task_name}/runs/hpsearch_0942 \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    paths.root_dir=/scratch/lcornelis/bgbench \
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
    logger.wandb.tags=[gcn,addneuromed,hpsearch] \
    hydra.run.dir=${paths.log_dir}/${task_name}/runs/hpsearch_0964 \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    paths.root_dir=/scratch/lcornelis/bgbench \
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
    logger.wandb.tags=[gcn,addneuromed,hpsearch] \
    hydra.run.dir=${paths.log_dir}/${task_name}/runs/hpsearch_1029 \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    paths.root_dir=/scratch/lcornelis/bgbench \
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
    logger.wandb.tags=[gcn,addneuromed,hpsearch] \
    hydra.run.dir=${paths.log_dir}/${task_name}/runs/hpsearch_1074 \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    paths.root_dir=/scratch/lcornelis/bgbench \
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
    logger.wandb.tags=[gcn,addneuromed,hpsearch] \
    hydra.run.dir=${paths.log_dir}/${task_name}/runs/hpsearch_1142 \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    paths.root_dir=/scratch/lcornelis/bgbench \
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
    logger.wandb.tags=[gcn,addneuromed,hpsearch] \
    hydra.run.dir=${paths.log_dir}/${task_name}/runs/hpsearch_1153 \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    paths.root_dir=/scratch/lcornelis/bgbench \
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
    logger.wandb.tags=[gcn,addneuromed,hpsearch] \
    hydra.run.dir=${paths.log_dir}/${task_name}/runs/hpsearch_1208 \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    paths.root_dir=/scratch/lcornelis/bgbench \
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
    logger.wandb.tags=[gcn,addneuromed,hpsearch] \
    hydra.run.dir=${paths.log_dir}/${task_name}/runs/hpsearch_1214 \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    paths.root_dir=/scratch/lcornelis/bgbench \
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
    logger.wandb.tags=[gcn,addneuromed,hpsearch] \
    hydra.run.dir=${paths.log_dir}/${task_name}/runs/hpsearch_1242 \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    paths.root_dir=/scratch/lcornelis/bgbench \
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
    logger.wandb.tags=[gcn,addneuromed,hpsearch] \
    hydra.run.dir=${paths.log_dir}/${task_name}/runs/hpsearch_1336 \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    paths.root_dir=/scratch/lcornelis/bgbench \
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
    logger.wandb.tags=[gcn,addneuromed,hpsearch] \
    hydra.run.dir=${paths.log_dir}/${task_name}/runs/hpsearch_1339 \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    paths.root_dir=/scratch/lcornelis/bgbench \
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
    logger.wandb.tags=[gcn,addneuromed,hpsearch] \
    hydra.run.dir=${paths.log_dir}/${task_name}/runs/hpsearch_1359 \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    paths.root_dir=/scratch/lcornelis/bgbench \
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
    logger.wandb.tags=[gcn,addneuromed,hpsearch] \
    hydra.run.dir=${paths.log_dir}/${task_name}/runs/hpsearch_1361 \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    paths.root_dir=/scratch/lcornelis/bgbench \
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
    logger.wandb.tags=[gcn,addneuromed,hpsearch] \
    hydra.run.dir=${paths.log_dir}/${task_name}/runs/hpsearch_1412 \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    paths.root_dir=/scratch/lcornelis/bgbench \
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
    logger.wandb.tags=[gcn,addneuromed,hpsearch] \
    hydra.run.dir=${paths.log_dir}/${task_name}/runs/hpsearch_1452 \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    paths.root_dir=/scratch/lcornelis/bgbench \
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
    logger.wandb.tags=[gcn,motrpac,hpsearch] \
    hydra.run.dir=${paths.log_dir}/${task_name}/runs/hpsearch_1574 \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    paths.root_dir=/scratch/lcornelis/bgbench \
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
    logger.wandb.tags=[gcn,motrpac,hpsearch] \
    hydra.run.dir=${paths.log_dir}/${task_name}/runs/hpsearch_1603 \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    paths.root_dir=/scratch/lcornelis/bgbench \
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
    logger.wandb.tags=[gcn,motrpac,hpsearch] \
    hydra.run.dir=${paths.log_dir}/${task_name}/runs/hpsearch_1625 \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    paths.root_dir=/scratch/lcornelis/bgbench \
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
    logger.wandb.tags=[gcn,motrpac,hpsearch] \
    hydra.run.dir=${paths.log_dir}/${task_name}/runs/hpsearch_1876 \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    paths.root_dir=/scratch/lcornelis/bgbench \
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
    logger.wandb.tags=[gcn,motrpac,hpsearch] \
    hydra.run.dir=${paths.log_dir}/${task_name}/runs/hpsearch_1929 \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    paths.root_dir=/scratch/lcornelis/bgbench \
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
    logger.wandb.tags=[gcn,motrpac,hpsearch] \
    hydra.run.dir=${paths.log_dir}/${task_name}/runs/hpsearch_2128 \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    paths.root_dir=/scratch/lcornelis/bgbench \
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
    logger.wandb.tags=[gcn,motrpac,hpsearch] \
    hydra.run.dir=${paths.log_dir}/${task_name}/runs/hpsearch_2219 \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    paths.root_dir=/scratch/lcornelis/bgbench \
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
    logger.wandb.tags=[gcn,motrpac,hpsearch] \
    hydra.run.dir=${paths.log_dir}/${task_name}/runs/hpsearch_2269 \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    paths.root_dir=/scratch/lcornelis/bgbench \
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
    logger.wandb.tags=[gcn,motrpac,hpsearch] \
    hydra.run.dir=${paths.log_dir}/${task_name}/runs/hpsearch_2273 \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    paths.root_dir=/scratch/lcornelis/bgbench \
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
    logger.wandb.tags=[gcn,brca,hpsearch] \
    hydra.run.dir=${paths.log_dir}/${task_name}/runs/hpsearch_2330 \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    paths.root_dir=/scratch/lcornelis/bgbench \
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
    logger.wandb.tags=[gcn,brca,hpsearch] \
    hydra.run.dir=${paths.log_dir}/${task_name}/runs/hpsearch_2478 \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    paths.root_dir=/scratch/lcornelis/bgbench \
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
    logger.wandb.tags=[gcn,brca,hpsearch] \
    hydra.run.dir=${paths.log_dir}/${task_name}/runs/hpsearch_2508 \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    paths.root_dir=/scratch/lcornelis/bgbench \
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
    logger.wandb.tags=[gcn,brca,hpsearch] \
    hydra.run.dir=${paths.log_dir}/${task_name}/runs/hpsearch_2528 \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    paths.root_dir=/scratch/lcornelis/bgbench \
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
    logger.wandb.tags=[gcn,brca,hpsearch] \
    hydra.run.dir=${paths.log_dir}/${task_name}/runs/hpsearch_2567 \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    paths.root_dir=/scratch/lcornelis/bgbench \
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
    logger.wandb.tags=[gcn,brca,hpsearch] \
    hydra.run.dir=${paths.log_dir}/${task_name}/runs/hpsearch_2587 \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    paths.root_dir=/scratch/lcornelis/bgbench \
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
    logger.wandb.tags=[gcn,brca,hpsearch] \
    hydra.run.dir=${paths.log_dir}/${task_name}/runs/hpsearch_2601 \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    paths.root_dir=/scratch/lcornelis/bgbench \
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
    logger.wandb.tags=[gcn,brca,hpsearch] \
    hydra.run.dir=${paths.log_dir}/${task_name}/runs/hpsearch_2607 \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    paths.root_dir=/scratch/lcornelis/bgbench \
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
    logger.wandb.tags=[gcn,brca,hpsearch] \
    hydra.run.dir=${paths.log_dir}/${task_name}/runs/hpsearch_2662 \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    paths.root_dir=/scratch/lcornelis/bgbench \
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
    logger.wandb.tags=[gcn,brca,hpsearch] \
    hydra.run.dir=${paths.log_dir}/${task_name}/runs/hpsearch_2702 \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    paths.root_dir=/scratch/lcornelis/bgbench \
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
    logger.wandb.tags=[gcn,brca,hpsearch] \
    hydra.run.dir=${paths.log_dir}/${task_name}/runs/hpsearch_2744 \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    paths.root_dir=/scratch/lcornelis/bgbench \
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
    logger.wandb.tags=[gcn,brca,hpsearch] \
    hydra.run.dir=${paths.log_dir}/${task_name}/runs/hpsearch_2754 \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    paths.root_dir=/scratch/lcornelis/bgbench \
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
    logger.wandb.tags=[gcn,brca,hpsearch] \
    hydra.run.dir=${paths.log_dir}/${task_name}/runs/hpsearch_2761 \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    paths.root_dir=/scratch/lcornelis/bgbench \
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
    logger.wandb.tags=[gcn,brca,hpsearch] \
    hydra.run.dir=${paths.log_dir}/${task_name}/runs/hpsearch_2763 \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    paths.root_dir=/scratch/lcornelis/bgbench \
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
    logger.wandb.tags=[gcn,brca,hpsearch] \
    hydra.run.dir=${paths.log_dir}/${task_name}/runs/hpsearch_2782 \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    paths.root_dir=/scratch/lcornelis/bgbench \
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
    logger.wandb.tags=[gcn,brca,hpsearch] \
    hydra.run.dir=${paths.log_dir}/${task_name}/runs/hpsearch_2828 \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    paths.root_dir=/scratch/lcornelis/bgbench \
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
    logger.wandb.tags=[gcn,brca,hpsearch] \
    hydra.run.dir=${paths.log_dir}/${task_name}/runs/hpsearch_2868 \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    paths.root_dir=/scratch/lcornelis/bgbench \
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
    logger.wandb.tags=[gcn,brca,hpsearch] \
    hydra.run.dir=${paths.log_dir}/${task_name}/runs/hpsearch_2875 \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    paths.root_dir=/scratch/lcornelis/bgbench \
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
    logger.wandb.tags=[gcn,brca,hpsearch] \
    hydra.run.dir=${paths.log_dir}/${task_name}/runs/hpsearch_2941 \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    paths.root_dir=/scratch/lcornelis/bgbench \
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
    logger.wandb.tags=[gcn,brca,hpsearch] \
    hydra.run.dir=${paths.log_dir}/${task_name}/runs/hpsearch_2951 \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    paths.root_dir=/scratch/lcornelis/bgbench \
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
    logger.wandb.tags=[gcn,brca,hpsearch] \
    hydra.run.dir=${paths.log_dir}/${task_name}/runs/hpsearch_2977 \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    paths.root_dir=/scratch/lcornelis/bgbench \
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
    logger.wandb.tags=[gin,parkinsons,hpsearch] \
    hydra.run.dir=${paths.log_dir}/${task_name}/runs/hpsearch_3111 \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    paths.root_dir=/scratch/lcornelis/bgbench \
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
    logger.wandb.tags=[gin,parkinsons,hpsearch] \
    hydra.run.dir=${paths.log_dir}/${task_name}/runs/hpsearch_3123 \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    paths.root_dir=/scratch/lcornelis/bgbench \
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
    logger.wandb.tags=[gin,parkinsons,hpsearch] \
    hydra.run.dir=${paths.log_dir}/${task_name}/runs/hpsearch_3129 \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    paths.root_dir=/scratch/lcornelis/bgbench \
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
    logger.wandb.tags=[gin,parkinsons,hpsearch] \
    hydra.run.dir=${paths.log_dir}/${task_name}/runs/hpsearch_3194 \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    paths.root_dir=/scratch/lcornelis/bgbench \
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
    logger.wandb.tags=[gin,parkinsons,hpsearch] \
    hydra.run.dir=${paths.log_dir}/${task_name}/runs/hpsearch_3395 \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    paths.root_dir=/scratch/lcornelis/bgbench \
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
    logger.wandb.tags=[gin,parkinsons,hpsearch] \
    hydra.run.dir=${paths.log_dir}/${task_name}/runs/hpsearch_3448 \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    paths.root_dir=/scratch/lcornelis/bgbench \
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
    logger.wandb.tags=[gin,parkinsons,hpsearch] \
    hydra.run.dir=${paths.log_dir}/${task_name}/runs/hpsearch_3454 \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    paths.root_dir=/scratch/lcornelis/bgbench \
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
    logger.wandb.tags=[gin,parkinsons,hpsearch] \
    hydra.run.dir=${paths.log_dir}/${task_name}/runs/hpsearch_3475 \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    paths.root_dir=/scratch/lcornelis/bgbench \
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
    logger.wandb.tags=[gin,parkinsons,hpsearch] \
    hydra.run.dir=${paths.log_dir}/${task_name}/runs/hpsearch_3594 \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    paths.root_dir=/scratch/lcornelis/bgbench \
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
    logger.wandb.tags=[gin,parkinsons,hpsearch] \
    hydra.run.dir=${paths.log_dir}/${task_name}/runs/hpsearch_3628 \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    paths.root_dir=/scratch/lcornelis/bgbench \
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
    logger.wandb.tags=[gin,parkinsons,hpsearch] \
    hydra.run.dir=${paths.log_dir}/${task_name}/runs/hpsearch_3672 \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    paths.root_dir=/scratch/lcornelis/bgbench \
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
    logger.wandb.tags=[gin,parkinsons,hpsearch] \
    hydra.run.dir=${paths.log_dir}/${task_name}/runs/hpsearch_3701 \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    paths.root_dir=/scratch/lcornelis/bgbench \
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
    logger.wandb.tags=[gin,parkinsons,hpsearch] \
    hydra.run.dir=${paths.log_dir}/${task_name}/runs/hpsearch_3734 \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    paths.root_dir=/scratch/lcornelis/bgbench \
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
    logger.wandb.tags=[gin,addneuromed,hpsearch] \
    hydra.run.dir=${paths.log_dir}/${task_name}/runs/hpsearch_3895 \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    paths.root_dir=/scratch/lcornelis/bgbench \
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
    logger.wandb.tags=[gin,addneuromed,hpsearch] \
    hydra.run.dir=${paths.log_dir}/${task_name}/runs/hpsearch_3900 \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    paths.root_dir=/scratch/lcornelis/bgbench \
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
    logger.wandb.tags=[gin,addneuromed,hpsearch] \
    hydra.run.dir=${paths.log_dir}/${task_name}/runs/hpsearch_4074 \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    paths.root_dir=/scratch/lcornelis/bgbench \
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
    logger.wandb.tags=[gin,addneuromed,hpsearch] \
    hydra.run.dir=${paths.log_dir}/${task_name}/runs/hpsearch_4078 \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    paths.root_dir=/scratch/lcornelis/bgbench \
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
    logger.wandb.tags=[gin,addneuromed,hpsearch] \
    hydra.run.dir=${paths.log_dir}/${task_name}/runs/hpsearch_4099 \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    paths.root_dir=/scratch/lcornelis/bgbench \
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
    logger.wandb.tags=[gin,addneuromed,hpsearch] \
    hydra.run.dir=${paths.log_dir}/${task_name}/runs/hpsearch_4122 \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    paths.root_dir=/scratch/lcornelis/bgbench \
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
    logger.wandb.tags=[gin,addneuromed,hpsearch] \
    hydra.run.dir=${paths.log_dir}/${task_name}/runs/hpsearch_4201 \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    paths.root_dir=/scratch/lcornelis/bgbench \
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
    logger.wandb.tags=[gin,addneuromed,hpsearch] \
    hydra.run.dir=${paths.log_dir}/${task_name}/runs/hpsearch_4214 \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    paths.root_dir=/scratch/lcornelis/bgbench \
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
    logger.wandb.tags=[gin,addneuromed,hpsearch] \
    hydra.run.dir=${paths.log_dir}/${task_name}/runs/hpsearch_4283 \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    paths.root_dir=/scratch/lcornelis/bgbench \
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
    logger.wandb.tags=[gin,addneuromed,hpsearch] \
    hydra.run.dir=${paths.log_dir}/${task_name}/runs/hpsearch_4285 \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    paths.root_dir=/scratch/lcornelis/bgbench \
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
    logger.wandb.tags=[gin,addneuromed,hpsearch] \
    hydra.run.dir=${paths.log_dir}/${task_name}/runs/hpsearch_4347 \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    paths.root_dir=/scratch/lcornelis/bgbench \
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
    logger.wandb.tags=[gin,addneuromed,hpsearch] \
    hydra.run.dir=${paths.log_dir}/${task_name}/runs/hpsearch_4360 \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    paths.root_dir=/scratch/lcornelis/bgbench \
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
    logger.wandb.tags=[gin,addneuromed,hpsearch] \
    hydra.run.dir=${paths.log_dir}/${task_name}/runs/hpsearch_4426 \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    paths.root_dir=/scratch/lcornelis/bgbench \
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
    logger.wandb.tags=[gin,addneuromed,hpsearch] \
    hydra.run.dir=${paths.log_dir}/${task_name}/runs/hpsearch_4466 \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    paths.root_dir=/scratch/lcornelis/bgbench \
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
    logger.wandb.tags=[gin,addneuromed,hpsearch] \
    hydra.run.dir=${paths.log_dir}/${task_name}/runs/hpsearch_4502 \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    paths.root_dir=/scratch/lcornelis/bgbench \
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
    logger.wandb.tags=[gin,motrpac,hpsearch] \
    hydra.run.dir=${paths.log_dir}/${task_name}/runs/hpsearch_4835 \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    paths.root_dir=/scratch/lcornelis/bgbench \
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
    logger.wandb.tags=[gin,motrpac,hpsearch] \
    hydra.run.dir=${paths.log_dir}/${task_name}/runs/hpsearch_4889 \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    paths.root_dir=/scratch/lcornelis/bgbench \
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
    logger.wandb.tags=[gin,motrpac,hpsearch] \
    hydra.run.dir=${paths.log_dir}/${task_name}/runs/hpsearch_4905 \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    paths.root_dir=/scratch/lcornelis/bgbench \
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
    logger.wandb.tags=[gin,motrpac,hpsearch] \
    hydra.run.dir=${paths.log_dir}/${task_name}/runs/hpsearch_4933 \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    paths.root_dir=/scratch/lcornelis/bgbench \
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
    logger.wandb.tags=[gin,motrpac,hpsearch] \
    hydra.run.dir=${paths.log_dir}/${task_name}/runs/hpsearch_4953 \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    paths.root_dir=/scratch/lcornelis/bgbench \
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
    logger.wandb.tags=[gin,motrpac,hpsearch] \
    hydra.run.dir=${paths.log_dir}/${task_name}/runs/hpsearch_5139 \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    paths.root_dir=/scratch/lcornelis/bgbench \
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
    logger.wandb.tags=[gin,motrpac,hpsearch] \
    hydra.run.dir=${paths.log_dir}/${task_name}/runs/hpsearch_5190 \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    paths.root_dir=/scratch/lcornelis/bgbench \
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
    logger.wandb.tags=[gin,motrpac,hpsearch] \
    hydra.run.dir=${paths.log_dir}/${task_name}/runs/hpsearch_5294 \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    paths.root_dir=/scratch/lcornelis/bgbench \
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
    logger.wandb.tags=[gin,motrpac,hpsearch] \
    hydra.run.dir=${paths.log_dir}/${task_name}/runs/hpsearch_5316 \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    paths.root_dir=/scratch/lcornelis/bgbench \
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
    logger.wandb.tags=[gin,brca,hpsearch] \
    hydra.run.dir=${paths.log_dir}/${task_name}/runs/hpsearch_5469 \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    paths.root_dir=/scratch/lcornelis/bgbench \
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
    logger.wandb.tags=[gin,brca,hpsearch] \
    hydra.run.dir=${paths.log_dir}/${task_name}/runs/hpsearch_5488 \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    paths.root_dir=/scratch/lcornelis/bgbench \
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
    logger.wandb.tags=[gin,brca,hpsearch] \
    hydra.run.dir=${paths.log_dir}/${task_name}/runs/hpsearch_5562 \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    paths.root_dir=/scratch/lcornelis/bgbench \
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
    logger.wandb.tags=[gin,brca,hpsearch] \
    hydra.run.dir=${paths.log_dir}/${task_name}/runs/hpsearch_5577 \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    paths.root_dir=/scratch/lcornelis/bgbench \
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
    logger.wandb.tags=[gin,brca,hpsearch] \
    hydra.run.dir=${paths.log_dir}/${task_name}/runs/hpsearch_5605 \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    paths.root_dir=/scratch/lcornelis/bgbench \
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
    logger.wandb.tags=[gin,brca,hpsearch] \
    hydra.run.dir=${paths.log_dir}/${task_name}/runs/hpsearch_5620 \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    paths.root_dir=/scratch/lcornelis/bgbench \
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
    logger.wandb.tags=[gin,brca,hpsearch] \
    hydra.run.dir=${paths.log_dir}/${task_name}/runs/hpsearch_5628 \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    paths.root_dir=/scratch/lcornelis/bgbench \
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
    logger.wandb.tags=[gin,brca,hpsearch] \
    hydra.run.dir=${paths.log_dir}/${task_name}/runs/hpsearch_5644 \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    paths.root_dir=/scratch/lcornelis/bgbench \
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
    logger.wandb.tags=[gin,brca,hpsearch] \
    hydra.run.dir=${paths.log_dir}/${task_name}/runs/hpsearch_5660 \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    paths.root_dir=/scratch/lcornelis/bgbench \
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
    logger.wandb.tags=[gin,brca,hpsearch] \
    hydra.run.dir=${paths.log_dir}/${task_name}/runs/hpsearch_5668 \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    paths.root_dir=/scratch/lcornelis/bgbench \
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
    logger.wandb.tags=[gin,brca,hpsearch] \
    hydra.run.dir=${paths.log_dir}/${task_name}/runs/hpsearch_5767 \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    paths.root_dir=/scratch/lcornelis/bgbench \
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
    logger.wandb.tags=[gin,brca,hpsearch] \
    hydra.run.dir=${paths.log_dir}/${task_name}/runs/hpsearch_5788 \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    paths.root_dir=/scratch/lcornelis/bgbench \
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
    logger.wandb.tags=[gin,brca,hpsearch] \
    hydra.run.dir=${paths.log_dir}/${task_name}/runs/hpsearch_5793 \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    paths.root_dir=/scratch/lcornelis/bgbench \
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
    logger.wandb.tags=[gin,brca,hpsearch] \
    hydra.run.dir=${paths.log_dir}/${task_name}/runs/hpsearch_5848 \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    paths.root_dir=/scratch/lcornelis/bgbench \
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
    logger.wandb.tags=[gin,brca,hpsearch] \
    hydra.run.dir=${paths.log_dir}/${task_name}/runs/hpsearch_5917 \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    paths.root_dir=/scratch/lcornelis/bgbench \
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
    logger.wandb.tags=[gin,brca,hpsearch] \
    hydra.run.dir=${paths.log_dir}/${task_name}/runs/hpsearch_5927 \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    paths.root_dir=/scratch/lcornelis/bgbench \
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
    logger.wandb.tags=[gin,brca,hpsearch] \
    hydra.run.dir=${paths.log_dir}/${task_name}/runs/hpsearch_5964 \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    paths.root_dir=/scratch/lcornelis/bgbench \
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
    logger.wandb.tags=[gin,brca,hpsearch] \
    hydra.run.dir=${paths.log_dir}/${task_name}/runs/hpsearch_5971 \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    paths.root_dir=/scratch/lcornelis/bgbench \
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
    logger.wandb.tags=[gin,brca,hpsearch] \
    hydra.run.dir=${paths.log_dir}/${task_name}/runs/hpsearch_5985 \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    paths.root_dir=/scratch/lcornelis/bgbench \
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
    logger.wandb.tags=[gin,brca,hpsearch] \
    hydra.run.dir=${paths.log_dir}/${task_name}/runs/hpsearch_6001 \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    paths.root_dir=/scratch/lcornelis/bgbench \
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
    logger.wandb.tags=[gin,brca,hpsearch] \
    hydra.run.dir=${paths.log_dir}/${task_name}/runs/hpsearch_6111 \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    paths.root_dir=/scratch/lcornelis/bgbench \
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
    logger.wandb.tags=[gatv2,parkinsons,hpsearch] \
    hydra.run.dir=${paths.log_dir}/${task_name}/runs/hpsearch_6170 \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    paths.root_dir=/scratch/lcornelis/bgbench \
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
    logger.wandb.tags=[gatv2,parkinsons,hpsearch] \
    hydra.run.dir=${paths.log_dir}/${task_name}/runs/hpsearch_6180 \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    paths.root_dir=/scratch/lcornelis/bgbench \
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
    logger.wandb.tags=[gatv2,parkinsons,hpsearch] \
    hydra.run.dir=${paths.log_dir}/${task_name}/runs/hpsearch_6431 \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    paths.root_dir=/scratch/lcornelis/bgbench \
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
    logger.wandb.tags=[gatv2,parkinsons,hpsearch] \
    hydra.run.dir=${paths.log_dir}/${task_name}/runs/hpsearch_6535 \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    paths.root_dir=/scratch/lcornelis/bgbench \
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
    logger.wandb.tags=[gatv2,parkinsons,hpsearch] \
    hydra.run.dir=${paths.log_dir}/${task_name}/runs/hpsearch_6626 \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    paths.root_dir=/scratch/lcornelis/bgbench \
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
    logger.wandb.tags=[gatv2,parkinsons,hpsearch] \
    hydra.run.dir=${paths.log_dir}/${task_name}/runs/hpsearch_6674 \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    paths.root_dir=/scratch/lcornelis/bgbench \
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
    logger.wandb.tags=[gatv2,parkinsons,hpsearch] \
    hydra.run.dir=${paths.log_dir}/${task_name}/runs/hpsearch_6870 \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    paths.root_dir=/scratch/lcornelis/bgbench \
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
    logger.wandb.tags=[gatv2,parkinsons,hpsearch] \
    hydra.run.dir=${paths.log_dir}/${task_name}/runs/hpsearch_6949 \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    paths.root_dir=/scratch/lcornelis/bgbench \
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
    logger.wandb.tags=[gatv2,parkinsons,hpsearch] \
    hydra.run.dir=${paths.log_dir}/${task_name}/runs/hpsearch_7250 \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    paths.root_dir=/scratch/lcornelis/bgbench \
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
    logger.wandb.tags=[gatv2,parkinsons,hpsearch] \
    hydra.run.dir=${paths.log_dir}/${task_name}/runs/hpsearch_7283 \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    paths.root_dir=/scratch/lcornelis/bgbench \
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
    logger.wandb.tags=[gatv2,parkinsons,hpsearch] \
    hydra.run.dir=${paths.log_dir}/${task_name}/runs/hpsearch_7376 \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    paths.root_dir=/scratch/lcornelis/bgbench \
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
    logger.wandb.tags=[gatv2,parkinsons,hpsearch] \
    hydra.run.dir=${paths.log_dir}/${task_name}/runs/hpsearch_7497 \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    paths.root_dir=/scratch/lcornelis/bgbench \
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
    logger.wandb.tags=[gatv2,parkinsons,hpsearch] \
    hydra.run.dir=${paths.log_dir}/${task_name}/runs/hpsearch_7502 \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    paths.root_dir=/scratch/lcornelis/bgbench \
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
    logger.wandb.tags=[gatv2,parkinsons,hpsearch] \
    hydra.run.dir=${paths.log_dir}/${task_name}/runs/hpsearch_7511 \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    paths.root_dir=/scratch/lcornelis/bgbench \
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
    logger.wandb.tags=[gatv2,parkinsons,hpsearch] \
    hydra.run.dir=${paths.log_dir}/${task_name}/runs/hpsearch_7599 \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    paths.root_dir=/scratch/lcornelis/bgbench \
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
    logger.wandb.tags=[gatv2,addneuromed,hpsearch] \
    hydra.run.dir=${paths.log_dir}/${task_name}/runs/hpsearch_8176 \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    paths.root_dir=/scratch/lcornelis/bgbench \
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
    logger.wandb.tags=[gatv2,addneuromed,hpsearch] \
    hydra.run.dir=${paths.log_dir}/${task_name}/runs/hpsearch_8182 \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    paths.root_dir=/scratch/lcornelis/bgbench \
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
    logger.wandb.tags=[gatv2,addneuromed,hpsearch] \
    hydra.run.dir=${paths.log_dir}/${task_name}/runs/hpsearch_8200 \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    paths.root_dir=/scratch/lcornelis/bgbench \
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
    logger.wandb.tags=[gatv2,addneuromed,hpsearch] \
    hydra.run.dir=${paths.log_dir}/${task_name}/runs/hpsearch_8320 \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    paths.root_dir=/scratch/lcornelis/bgbench \
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
    logger.wandb.tags=[gatv2,addneuromed,hpsearch] \
    hydra.run.dir=${paths.log_dir}/${task_name}/runs/hpsearch_8384 \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    paths.root_dir=/scratch/lcornelis/bgbench \
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
    logger.wandb.tags=[gatv2,addneuromed,hpsearch] \
    hydra.run.dir=${paths.log_dir}/${task_name}/runs/hpsearch_8387 \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    paths.root_dir=/scratch/lcornelis/bgbench \
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
    logger.wandb.tags=[gatv2,addneuromed,hpsearch] \
    hydra.run.dir=${paths.log_dir}/${task_name}/runs/hpsearch_8790 \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    paths.root_dir=/scratch/lcornelis/bgbench \
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
    logger.wandb.tags=[gatv2,addneuromed,hpsearch] \
    hydra.run.dir=${paths.log_dir}/${task_name}/runs/hpsearch_8899 \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    paths.root_dir=/scratch/lcornelis/bgbench \
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
    logger.wandb.tags=[gatv2,addneuromed,hpsearch] \
    hydra.run.dir=${paths.log_dir}/${task_name}/runs/hpsearch_8905 \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    paths.root_dir=/scratch/lcornelis/bgbench \
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
    logger.wandb.tags=[gatv2,addneuromed,hpsearch] \
    hydra.run.dir=${paths.log_dir}/${task_name}/runs/hpsearch_8976 \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    paths.root_dir=/scratch/lcornelis/bgbench \
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
    logger.wandb.tags=[gatv2,addneuromed,hpsearch] \
    hydra.run.dir=${paths.log_dir}/${task_name}/runs/hpsearch_9070 \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    paths.root_dir=/scratch/lcornelis/bgbench \
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
    logger.wandb.tags=[gatv2,addneuromed,hpsearch] \
    hydra.run.dir=${paths.log_dir}/${task_name}/runs/hpsearch_9127 \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    paths.root_dir=/scratch/lcornelis/bgbench \
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
    logger.wandb.tags=[gatv2,addneuromed,hpsearch] \
    hydra.run.dir=${paths.log_dir}/${task_name}/runs/hpsearch_9157 \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    paths.root_dir=/scratch/lcornelis/bgbench \
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
    logger.wandb.tags=[gatv2,motrpac,hpsearch] \
    hydra.run.dir=${paths.log_dir}/${task_name}/runs/hpsearch_9389 \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    paths.root_dir=/scratch/lcornelis/bgbench \
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
    logger.wandb.tags=[gatv2,motrpac,hpsearch] \
    hydra.run.dir=${paths.log_dir}/${task_name}/runs/hpsearch_9418 \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    paths.root_dir=/scratch/lcornelis/bgbench \
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
    logger.wandb.tags=[gatv2,motrpac,hpsearch] \
    hydra.run.dir=${paths.log_dir}/${task_name}/runs/hpsearch_9456 \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    paths.root_dir=/scratch/lcornelis/bgbench \
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
    logger.wandb.tags=[gatv2,motrpac,hpsearch] \
    hydra.run.dir=${paths.log_dir}/${task_name}/runs/hpsearch_9678 \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    paths.root_dir=/scratch/lcornelis/bgbench \
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
    logger.wandb.tags=[gatv2,motrpac,hpsearch] \
    hydra.run.dir=${paths.log_dir}/${task_name}/runs/hpsearch_9694 \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    paths.root_dir=/scratch/lcornelis/bgbench \
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
    logger.wandb.tags=[gatv2,motrpac,hpsearch] \
    hydra.run.dir=${paths.log_dir}/${task_name}/runs/hpsearch_9740 \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    paths.root_dir=/scratch/lcornelis/bgbench \
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
    logger.wandb.tags=[gatv2,motrpac,hpsearch] \
    hydra.run.dir=${paths.log_dir}/${task_name}/runs/hpsearch_9850 \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    paths.root_dir=/scratch/lcornelis/bgbench \
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
    logger.wandb.tags=[gatv2,motrpac,hpsearch] \
    hydra.run.dir=${paths.log_dir}/${task_name}/runs/hpsearch_9879 \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    paths.root_dir=/scratch/lcornelis/bgbench \
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
    logger.wandb.tags=[gatv2,motrpac,hpsearch] \
    hydra.run.dir=${paths.log_dir}/${task_name}/runs/hpsearch_9985 \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    paths.root_dir=/scratch/lcornelis/bgbench \
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
    logger.wandb.tags=[gatv2,motrpac,hpsearch] \
    hydra.run.dir=${paths.log_dir}/${task_name}/runs/hpsearch_10024 \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    paths.root_dir=/scratch/lcornelis/bgbench \
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
    logger.wandb.tags=[gatv2,motrpac,hpsearch] \
    hydra.run.dir=${paths.log_dir}/${task_name}/runs/hpsearch_10094 \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    paths.root_dir=/scratch/lcornelis/bgbench \
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
    logger.wandb.tags=[gatv2,motrpac,hpsearch] \
    hydra.run.dir=${paths.log_dir}/${task_name}/runs/hpsearch_10135 \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    paths.root_dir=/scratch/lcornelis/bgbench \
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
    logger.wandb.tags=[gatv2,motrpac,hpsearch] \
    hydra.run.dir=${paths.log_dir}/${task_name}/runs/hpsearch_10374 \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    paths.root_dir=/scratch/lcornelis/bgbench \
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
    logger.wandb.tags=[gatv2,motrpac,hpsearch] \
    hydra.run.dir=${paths.log_dir}/${task_name}/runs/hpsearch_10403 \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    paths.root_dir=/scratch/lcornelis/bgbench \
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
    logger.wandb.tags=[gatv2,motrpac,hpsearch] \
    hydra.run.dir=${paths.log_dir}/${task_name}/runs/hpsearch_10467 \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    paths.root_dir=/scratch/lcornelis/bgbench \
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
    logger.wandb.tags=[gatv2,motrpac,hpsearch] \
    hydra.run.dir=${paths.log_dir}/${task_name}/runs/hpsearch_10506 \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    paths.root_dir=/scratch/lcornelis/bgbench \
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
    logger.wandb.tags=[gatv2,motrpac,hpsearch] \
    hydra.run.dir=${paths.log_dir}/${task_name}/runs/hpsearch_10511 \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    paths.root_dir=/scratch/lcornelis/bgbench \
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
    logger.wandb.tags=[gatv2,motrpac,hpsearch] \
    hydra.run.dir=${paths.log_dir}/${task_name}/runs/hpsearch_10613 \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    paths.root_dir=/scratch/lcornelis/bgbench \
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
    logger.wandb.tags=[gatv2,motrpac,hpsearch] \
    hydra.run.dir=${paths.log_dir}/${task_name}/runs/hpsearch_10640 \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    paths.root_dir=/scratch/lcornelis/bgbench \
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
    logger.wandb.tags=[gatv2,motrpac,hpsearch] \
    hydra.run.dir=${paths.log_dir}/${task_name}/runs/hpsearch_10738 \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    paths.root_dir=/scratch/lcornelis/bgbench \
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
    logger.wandb.tags=[gatv2,motrpac,hpsearch] \
    hydra.run.dir=${paths.log_dir}/${task_name}/runs/hpsearch_10741 \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    paths.root_dir=/scratch/lcornelis/bgbench \
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
    logger.wandb.tags=[gatv2,brca,hpsearch] \
    hydra.run.dir=${paths.log_dir}/${task_name}/runs/hpsearch_10827 \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    paths.root_dir=/scratch/lcornelis/bgbench \
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
    logger.wandb.tags=[gatv2,brca,hpsearch] \
    hydra.run.dir=${paths.log_dir}/${task_name}/runs/hpsearch_10854 \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    paths.root_dir=/scratch/lcornelis/bgbench \
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
    logger.wandb.tags=[gatv2,brca,hpsearch] \
    hydra.run.dir=${paths.log_dir}/${task_name}/runs/hpsearch_10879 \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    paths.root_dir=/scratch/lcornelis/bgbench \
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
    logger.wandb.tags=[gatv2,brca,hpsearch] \
    hydra.run.dir=${paths.log_dir}/${task_name}/runs/hpsearch_10935 \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    paths.root_dir=/scratch/lcornelis/bgbench \
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
    logger.wandb.tags=[gatv2,brca,hpsearch] \
    hydra.run.dir=${paths.log_dir}/${task_name}/runs/hpsearch_11055 \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    paths.root_dir=/scratch/lcornelis/bgbench \
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
    logger.wandb.tags=[gatv2,brca,hpsearch] \
    hydra.run.dir=${paths.log_dir}/${task_name}/runs/hpsearch_11194 \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    paths.root_dir=/scratch/lcornelis/bgbench \
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
    logger.wandb.tags=[gatv2,brca,hpsearch] \
    hydra.run.dir=${paths.log_dir}/${task_name}/runs/hpsearch_11268 \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    paths.root_dir=/scratch/lcornelis/bgbench \
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
    logger.wandb.tags=[gatv2,brca,hpsearch] \
    hydra.run.dir=${paths.log_dir}/${task_name}/runs/hpsearch_11285 \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    paths.root_dir=/scratch/lcornelis/bgbench \
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
    logger.wandb.tags=[gatv2,brca,hpsearch] \
    hydra.run.dir=${paths.log_dir}/${task_name}/runs/hpsearch_11299 \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    paths.root_dir=/scratch/lcornelis/bgbench \
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
    logger.wandb.tags=[gatv2,brca,hpsearch] \
    hydra.run.dir=${paths.log_dir}/${task_name}/runs/hpsearch_11406 \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    paths.root_dir=/scratch/lcornelis/bgbench \
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
    logger.wandb.tags=[gatv2,brca,hpsearch] \
    hydra.run.dir=${paths.log_dir}/${task_name}/runs/hpsearch_11576 \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    paths.root_dir=/scratch/lcornelis/bgbench \
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
    logger.wandb.tags=[gatv2,brca,hpsearch] \
    hydra.run.dir=${paths.log_dir}/${task_name}/runs/hpsearch_11611 \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    paths.root_dir=/scratch/lcornelis/bgbench \
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
    logger.wandb.tags=[gatv2,brca,hpsearch] \
    hydra.run.dir=${paths.log_dir}/${task_name}/runs/hpsearch_11680 \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    paths.root_dir=/scratch/lcornelis/bgbench \
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
    logger.wandb.tags=[gatv2,brca,hpsearch] \
    hydra.run.dir=${paths.log_dir}/${task_name}/runs/hpsearch_11990 \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    paths.root_dir=/scratch/lcornelis/bgbench \
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
    logger.wandb.tags=[gatv2,brca,hpsearch] \
    hydra.run.dir=${paths.log_dir}/${task_name}/runs/hpsearch_11989 \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    paths.root_dir=/scratch/lcornelis/bgbench \
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
    logger.wandb.tags=[gatv2,brca,hpsearch] \
    hydra.run.dir=${paths.log_dir}/${task_name}/runs/hpsearch_12029 \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    paths.root_dir=/scratch/lcornelis/bgbench \
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
    logger.wandb.tags=[gatv2,brca,hpsearch] \
    hydra.run.dir=${paths.log_dir}/${task_name}/runs/hpsearch_12081 \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    paths.root_dir=/scratch/lcornelis/bgbench \
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
    logger.wandb.tags=[gatv2,brca,hpsearch] \
    hydra.run.dir=${paths.log_dir}/${task_name}/runs/hpsearch_12226 \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    paths.root_dir=/scratch/lcornelis/bgbench \
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
    logger.wandb.tags=[graph_sage,parkinsons,hpsearch] \
    hydra.run.dir=${paths.log_dir}/${task_name}/runs/hpsearch_12295 \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    paths.root_dir=/scratch/lcornelis/bgbench \
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
    logger.wandb.tags=[graph_sage,parkinsons,hpsearch] \
    hydra.run.dir=${paths.log_dir}/${task_name}/runs/hpsearch_12422 \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    paths.root_dir=/scratch/lcornelis/bgbench \
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
    logger.wandb.tags=[graph_sage,parkinsons,hpsearch] \
    hydra.run.dir=${paths.log_dir}/${task_name}/runs/hpsearch_12455 \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    paths.root_dir=/scratch/lcornelis/bgbench \
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
    logger.wandb.tags=[graph_sage,parkinsons,hpsearch] \
    hydra.run.dir=${paths.log_dir}/${task_name}/runs/hpsearch_12474 \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    paths.root_dir=/scratch/lcornelis/bgbench \
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
    logger.wandb.tags=[graph_sage,parkinsons,hpsearch] \
    hydra.run.dir=${paths.log_dir}/${task_name}/runs/hpsearch_12496 \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    paths.root_dir=/scratch/lcornelis/bgbench \
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
    logger.wandb.tags=[graph_sage,parkinsons,hpsearch] \
    hydra.run.dir=${paths.log_dir}/${task_name}/runs/hpsearch_12511 \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    paths.root_dir=/scratch/lcornelis/bgbench \
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
    logger.wandb.tags=[graph_sage,parkinsons,hpsearch] \
    hydra.run.dir=${paths.log_dir}/${task_name}/runs/hpsearch_12637 \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    paths.root_dir=/scratch/lcornelis/bgbench \
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
    logger.wandb.tags=[graph_sage,parkinsons,hpsearch] \
    hydra.run.dir=${paths.log_dir}/${task_name}/runs/hpsearch_12663 \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    paths.root_dir=/scratch/lcornelis/bgbench \
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
    logger.wandb.tags=[graph_sage,parkinsons,hpsearch] \
    hydra.run.dir=${paths.log_dir}/${task_name}/runs/hpsearch_12756 \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    paths.root_dir=/scratch/lcornelis/bgbench \
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
    logger.wandb.tags=[graph_sage,parkinsons,hpsearch] \
    hydra.run.dir=${paths.log_dir}/${task_name}/runs/hpsearch_12833 \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    paths.root_dir=/scratch/lcornelis/bgbench \
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
    logger.wandb.tags=[graph_sage,parkinsons,hpsearch] \
    hydra.run.dir=${paths.log_dir}/${task_name}/runs/hpsearch_12844 \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    paths.root_dir=/scratch/lcornelis/bgbench \
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
    logger.wandb.tags=[graph_sage,parkinsons,hpsearch] \
    hydra.run.dir=${paths.log_dir}/${task_name}/runs/hpsearch_12851 \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    paths.root_dir=/scratch/lcornelis/bgbench \
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
    logger.wandb.tags=[graph_sage,parkinsons,hpsearch] \
    hydra.run.dir=${paths.log_dir}/${task_name}/runs/hpsearch_12967 \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    paths.root_dir=/scratch/lcornelis/bgbench \
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
    logger.wandb.tags=[graph_sage,addneuromed,hpsearch] \
    hydra.run.dir=${paths.log_dir}/${task_name}/runs/hpsearch_13154 \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    paths.root_dir=/scratch/lcornelis/bgbench \
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
    logger.wandb.tags=[graph_sage,addneuromed,hpsearch] \
    hydra.run.dir=${paths.log_dir}/${task_name}/runs/hpsearch_13235 \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    paths.root_dir=/scratch/lcornelis/bgbench \
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
    logger.wandb.tags=[graph_sage,addneuromed,hpsearch] \
    hydra.run.dir=${paths.log_dir}/${task_name}/runs/hpsearch_13247 \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    paths.root_dir=/scratch/lcornelis/bgbench \
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
    logger.wandb.tags=[graph_sage,addneuromed,hpsearch] \
    hydra.run.dir=${paths.log_dir}/${task_name}/runs/hpsearch_13285 \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    paths.root_dir=/scratch/lcornelis/bgbench \
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
    logger.wandb.tags=[graph_sage,addneuromed,hpsearch] \
    hydra.run.dir=${paths.log_dir}/${task_name}/runs/hpsearch_13292 \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    paths.root_dir=/scratch/lcornelis/bgbench \
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
    logger.wandb.tags=[graph_sage,addneuromed,hpsearch] \
    hydra.run.dir=${paths.log_dir}/${task_name}/runs/hpsearch_13331 \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    paths.root_dir=/scratch/lcornelis/bgbench \
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
    logger.wandb.tags=[graph_sage,addneuromed,hpsearch] \
    hydra.run.dir=${paths.log_dir}/${task_name}/runs/hpsearch_13372 \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    paths.root_dir=/scratch/lcornelis/bgbench \
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
    logger.wandb.tags=[graph_sage,addneuromed,hpsearch] \
    hydra.run.dir=${paths.log_dir}/${task_name}/runs/hpsearch_13385 \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    paths.root_dir=/scratch/lcornelis/bgbench \
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
    logger.wandb.tags=[graph_sage,addneuromed,hpsearch] \
    hydra.run.dir=${paths.log_dir}/${task_name}/runs/hpsearch_13455 \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    paths.root_dir=/scratch/lcornelis/bgbench \
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
    logger.wandb.tags=[graph_sage,addneuromed,hpsearch] \
    hydra.run.dir=${paths.log_dir}/${task_name}/runs/hpsearch_13508 \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    paths.root_dir=/scratch/lcornelis/bgbench \
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
    logger.wandb.tags=[graph_sage,addneuromed,hpsearch] \
    hydra.run.dir=${paths.log_dir}/${task_name}/runs/hpsearch_13510 \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    paths.root_dir=/scratch/lcornelis/bgbench \
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
    logger.wandb.tags=[graph_sage,addneuromed,hpsearch] \
    hydra.run.dir=${paths.log_dir}/${task_name}/runs/hpsearch_13546 \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    paths.root_dir=/scratch/lcornelis/bgbench \
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
    logger.wandb.tags=[graph_sage,addneuromed,hpsearch] \
    hydra.run.dir=${paths.log_dir}/${task_name}/runs/hpsearch_13736 \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    paths.root_dir=/scratch/lcornelis/bgbench \
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
    logger.wandb.tags=[graph_sage,addneuromed,hpsearch] \
    hydra.run.dir=${paths.log_dir}/${task_name}/runs/hpsearch_13778 \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    paths.root_dir=/scratch/lcornelis/bgbench \
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
    logger.wandb.tags=[graph_sage,addneuromed,hpsearch] \
    hydra.run.dir=${paths.log_dir}/${task_name}/runs/hpsearch_13803 \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    paths.root_dir=/scratch/lcornelis/bgbench \
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
    logger.wandb.tags=[graph_sage,addneuromed,hpsearch] \
    hydra.run.dir=${paths.log_dir}/${task_name}/runs/hpsearch_13816 \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    paths.root_dir=/scratch/lcornelis/bgbench \
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
    logger.wandb.tags=[graph_sage,addneuromed,hpsearch] \
    hydra.run.dir=${paths.log_dir}/${task_name}/runs/hpsearch_13824 \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    paths.root_dir=/scratch/lcornelis/bgbench \
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
    logger.wandb.tags=[graph_sage,motrpac,hpsearch] \
    hydra.run.dir=${paths.log_dir}/${task_name}/runs/hpsearch_13826 \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    paths.root_dir=/scratch/lcornelis/bgbench \
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
    logger.wandb.tags=[graph_sage,motrpac,hpsearch] \
    hydra.run.dir=${paths.log_dir}/${task_name}/runs/hpsearch_13913 \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    paths.root_dir=/scratch/lcornelis/bgbench \
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
    logger.wandb.tags=[graph_sage,motrpac,hpsearch] \
    hydra.run.dir=${paths.log_dir}/${task_name}/runs/hpsearch_14150 \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    paths.root_dir=/scratch/lcornelis/bgbench \
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
    logger.wandb.tags=[graph_sage,motrpac,hpsearch] \
    hydra.run.dir=${paths.log_dir}/${task_name}/runs/hpsearch_14173 \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    paths.root_dir=/scratch/lcornelis/bgbench \
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
    logger.wandb.tags=[graph_sage,motrpac,hpsearch] \
    hydra.run.dir=${paths.log_dir}/${task_name}/runs/hpsearch_14232 \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    paths.root_dir=/scratch/lcornelis/bgbench \
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
    logger.wandb.tags=[graph_sage,motrpac,hpsearch] \
    hydra.run.dir=${paths.log_dir}/${task_name}/runs/hpsearch_14268 \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    paths.root_dir=/scratch/lcornelis/bgbench \
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
    logger.wandb.tags=[graph_sage,motrpac,hpsearch] \
    hydra.run.dir=${paths.log_dir}/${task_name}/runs/hpsearch_14298 \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    paths.root_dir=/scratch/lcornelis/bgbench \
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
    logger.wandb.tags=[graph_sage,motrpac,hpsearch] \
    hydra.run.dir=${paths.log_dir}/${task_name}/runs/hpsearch_14309 \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    paths.root_dir=/scratch/lcornelis/bgbench \
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
    logger.wandb.tags=[graph_sage,motrpac,hpsearch] \
    hydra.run.dir=${paths.log_dir}/${task_name}/runs/hpsearch_14328 \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    paths.root_dir=/scratch/lcornelis/bgbench \
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
    logger.wandb.tags=[graph_sage,motrpac,hpsearch] \
    hydra.run.dir=${paths.log_dir}/${task_name}/runs/hpsearch_14443 \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    paths.root_dir=/scratch/lcornelis/bgbench \
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
    logger.wandb.tags=[graph_sage,motrpac,hpsearch] \
    hydra.run.dir=${paths.log_dir}/${task_name}/runs/hpsearch_14454 \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    paths.root_dir=/scratch/lcornelis/bgbench \
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
    logger.wandb.tags=[graph_sage,motrpac,hpsearch] \
    hydra.run.dir=${paths.log_dir}/${task_name}/runs/hpsearch_14561 \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    paths.root_dir=/scratch/lcornelis/bgbench \
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
    logger.wandb.tags=[graph_sage,motrpac,hpsearch] \
    hydra.run.dir=${paths.log_dir}/${task_name}/runs/hpsearch_14578 \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    paths.root_dir=/scratch/lcornelis/bgbench \
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
    logger.wandb.tags=[graph_sage,brca,hpsearch] \
    hydra.run.dir=${paths.log_dir}/${task_name}/runs/hpsearch_14617 \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    paths.root_dir=/scratch/lcornelis/bgbench \
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
    logger.wandb.tags=[graph_sage,brca,hpsearch] \
    hydra.run.dir=${paths.log_dir}/${task_name}/runs/hpsearch_14741 \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    paths.root_dir=/scratch/lcornelis/bgbench \
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
    logger.wandb.tags=[graph_sage,brca,hpsearch] \
    hydra.run.dir=${paths.log_dir}/${task_name}/runs/hpsearch_14752 \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    paths.root_dir=/scratch/lcornelis/bgbench \
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
    logger.wandb.tags=[graph_sage,brca,hpsearch] \
    hydra.run.dir=${paths.log_dir}/${task_name}/runs/hpsearch_14852 \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    paths.root_dir=/scratch/lcornelis/bgbench \
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
    logger.wandb.tags=[graph_sage,brca,hpsearch] \
    hydra.run.dir=${paths.log_dir}/${task_name}/runs/hpsearch_14869 \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    paths.root_dir=/scratch/lcornelis/bgbench \
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
    logger.wandb.tags=[graph_sage,brca,hpsearch] \
    hydra.run.dir=${paths.log_dir}/${task_name}/runs/hpsearch_14891 \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    paths.root_dir=/scratch/lcornelis/bgbench \
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
    logger.wandb.tags=[graph_sage,brca,hpsearch] \
    hydra.run.dir=${paths.log_dir}/${task_name}/runs/hpsearch_14899 \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    paths.root_dir=/scratch/lcornelis/bgbench \
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
    logger.wandb.tags=[graph_sage,brca,hpsearch] \
    hydra.run.dir=${paths.log_dir}/${task_name}/runs/hpsearch_14950 \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    paths.root_dir=/scratch/lcornelis/bgbench \
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
    logger.wandb.tags=[graph_sage,brca,hpsearch] \
    hydra.run.dir=${paths.log_dir}/${task_name}/runs/hpsearch_15007 \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    paths.root_dir=/scratch/lcornelis/bgbench \
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
    logger.wandb.tags=[graph_sage,brca,hpsearch] \
    hydra.run.dir=${paths.log_dir}/${task_name}/runs/hpsearch_15078 \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    paths.root_dir=/scratch/lcornelis/bgbench \
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
    logger.wandb.tags=[graph_sage,brca,hpsearch] \
    hydra.run.dir=${paths.log_dir}/${task_name}/runs/hpsearch_15085 \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    paths.root_dir=/scratch/lcornelis/bgbench \
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
    logger.wandb.tags=[graph_sage,brca,hpsearch] \
    hydra.run.dir=${paths.log_dir}/${task_name}/runs/hpsearch_15114 \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    paths.root_dir=/scratch/lcornelis/bgbench \
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
    logger.wandb.tags=[graph_sage,brca,hpsearch] \
    hydra.run.dir=${paths.log_dir}/${task_name}/runs/hpsearch_15175 \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    paths.root_dir=/scratch/lcornelis/bgbench \
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
    logger.wandb.tags=[graph_sage,brca,hpsearch] \
    hydra.run.dir=${paths.log_dir}/${task_name}/runs/hpsearch_15295 \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    paths.root_dir=/scratch/lcornelis/bgbench \
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
    logger.wandb.tags=[graph_sage,brca,hpsearch] \
    hydra.run.dir=${paths.log_dir}/${task_name}/runs/hpsearch_15298 \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    paths.root_dir=/scratch/lcornelis/bgbench \
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
    logger.wandb.tags=[graph_sage,brca,hpsearch] \
    hydra.run.dir=${paths.log_dir}/${task_name}/runs/hpsearch_15303 \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    paths.root_dir=/scratch/lcornelis/bgbench \
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
    logger.wandb.tags=[graph_sage,brca,hpsearch] \
    hydra.run.dir=${paths.log_dir}/${task_name}/runs/hpsearch_15331 \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    paths.root_dir=/scratch/lcornelis/bgbench \
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
    logger.wandb.tags=[chebnet,parkinsons,hpsearch] \
    hydra.run.dir=${paths.log_dir}/${task_name}/runs/hpsearch_15373 \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    paths.root_dir=/scratch/lcornelis/bgbench \
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
    logger.wandb.tags=[chebnet,parkinsons,hpsearch] \
    hydra.run.dir=${paths.log_dir}/${task_name}/runs/hpsearch_15392 \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    paths.root_dir=/scratch/lcornelis/bgbench \
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
    logger.wandb.tags=[chebnet,parkinsons,hpsearch] \
    hydra.run.dir=${paths.log_dir}/${task_name}/runs/hpsearch_15396 \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    paths.root_dir=/scratch/lcornelis/bgbench \
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
    logger.wandb.tags=[chebnet,parkinsons,hpsearch] \
    hydra.run.dir=${paths.log_dir}/${task_name}/runs/hpsearch_15457 \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    paths.root_dir=/scratch/lcornelis/bgbench \
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
    logger.wandb.tags=[chebnet,parkinsons,hpsearch] \
    hydra.run.dir=${paths.log_dir}/${task_name}/runs/hpsearch_15512 \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    paths.root_dir=/scratch/lcornelis/bgbench \
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
    logger.wandb.tags=[chebnet,parkinsons,hpsearch] \
    hydra.run.dir=${paths.log_dir}/${task_name}/runs/hpsearch_15517 \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    paths.root_dir=/scratch/lcornelis/bgbench \
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
    logger.wandb.tags=[chebnet,parkinsons,hpsearch] \
    hydra.run.dir=${paths.log_dir}/${task_name}/runs/hpsearch_15530 \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    paths.root_dir=/scratch/lcornelis/bgbench \
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
    logger.wandb.tags=[chebnet,parkinsons,hpsearch] \
    hydra.run.dir=${paths.log_dir}/${task_name}/runs/hpsearch_15582 \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    paths.root_dir=/scratch/lcornelis/bgbench \
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
    logger.wandb.tags=[chebnet,parkinsons,hpsearch] \
    hydra.run.dir=${paths.log_dir}/${task_name}/runs/hpsearch_15597 \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    paths.root_dir=/scratch/lcornelis/bgbench \
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
    logger.wandb.tags=[chebnet,parkinsons,hpsearch] \
    hydra.run.dir=${paths.log_dir}/${task_name}/runs/hpsearch_15645 \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    paths.root_dir=/scratch/lcornelis/bgbench \
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
    logger.wandb.tags=[chebnet,parkinsons,hpsearch] \
    hydra.run.dir=${paths.log_dir}/${task_name}/runs/hpsearch_15805 \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    paths.root_dir=/scratch/lcornelis/bgbench \
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
    logger.wandb.tags=[chebnet,parkinsons,hpsearch] \
    hydra.run.dir=${paths.log_dir}/${task_name}/runs/hpsearch_15809 \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    paths.root_dir=/scratch/lcornelis/bgbench \
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
    logger.wandb.tags=[chebnet,parkinsons,hpsearch] \
    hydra.run.dir=${paths.log_dir}/${task_name}/runs/hpsearch_15888 \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    paths.root_dir=/scratch/lcornelis/bgbench \
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
    logger.wandb.tags=[chebnet,parkinsons,hpsearch] \
    hydra.run.dir=${paths.log_dir}/${task_name}/runs/hpsearch_15891 \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    paths.root_dir=/scratch/lcornelis/bgbench \
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
    logger.wandb.tags=[chebnet,parkinsons,hpsearch] \
    hydra.run.dir=${paths.log_dir}/${task_name}/runs/hpsearch_15903 \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    paths.root_dir=/scratch/lcornelis/bgbench \
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
    logger.wandb.tags=[chebnet,parkinsons,hpsearch] \
    hydra.run.dir=${paths.log_dir}/${task_name}/runs/hpsearch_15930 \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    paths.root_dir=/scratch/lcornelis/bgbench \
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
    logger.wandb.tags=[chebnet,parkinsons,hpsearch] \
    hydra.run.dir=${paths.log_dir}/${task_name}/runs/hpsearch_15932 \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    paths.root_dir=/scratch/lcornelis/bgbench \
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
    logger.wandb.tags=[chebnet,parkinsons,hpsearch] \
    hydra.run.dir=${paths.log_dir}/${task_name}/runs/hpsearch_15933 \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    paths.root_dir=/scratch/lcornelis/bgbench \
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
    logger.wandb.tags=[chebnet,parkinsons,hpsearch] \
    hydra.run.dir=${paths.log_dir}/${task_name}/runs/hpsearch_15935 \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    paths.root_dir=/scratch/lcornelis/bgbench \
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
    logger.wandb.tags=[chebnet,parkinsons,hpsearch] \
    hydra.run.dir=${paths.log_dir}/${task_name}/runs/hpsearch_15934 \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    paths.root_dir=/scratch/lcornelis/bgbench \
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
    logger.wandb.tags=[chebnet,parkinsons,hpsearch] \
    hydra.run.dir=${paths.log_dir}/${task_name}/runs/hpsearch_15936 \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    paths.root_dir=/scratch/lcornelis/bgbench \
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
    logger.wandb.tags=[chebnet,parkinsons,hpsearch] \
    hydra.run.dir=${paths.log_dir}/${task_name}/runs/hpsearch_15937 \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    paths.root_dir=/scratch/lcornelis/bgbench \
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
    logger.wandb.tags=[chebnet,parkinsons,hpsearch] \
    hydra.run.dir=${paths.log_dir}/${task_name}/runs/hpsearch_15938 \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    paths.root_dir=/scratch/lcornelis/bgbench \
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
    logger.wandb.tags=[chebnet,parkinsons,hpsearch] \
    hydra.run.dir=${paths.log_dir}/${task_name}/runs/hpsearch_15940 \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    paths.root_dir=/scratch/lcornelis/bgbench \
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
    logger.wandb.tags=[chebnet,parkinsons,hpsearch] \
    hydra.run.dir=${paths.log_dir}/${task_name}/runs/hpsearch_15941 \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    paths.root_dir=/scratch/lcornelis/bgbench \
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
    logger.wandb.tags=[chebnet,parkinsons,hpsearch] \
    hydra.run.dir=${paths.log_dir}/${task_name}/runs/hpsearch_15942 \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    paths.root_dir=/scratch/lcornelis/bgbench \
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
    logger.wandb.tags=[chebnet,parkinsons,hpsearch] \
    hydra.run.dir=${paths.log_dir}/${task_name}/runs/hpsearch_15943 \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    paths.root_dir=/scratch/lcornelis/bgbench \
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
    logger.wandb.tags=[chebnet,parkinsons,hpsearch] \
    hydra.run.dir=${paths.log_dir}/${task_name}/runs/hpsearch_15944 \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    paths.root_dir=/scratch/lcornelis/bgbench \
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
    logger.wandb.tags=[chebnet,parkinsons,hpsearch] \
    hydra.run.dir=${paths.log_dir}/${task_name}/runs/hpsearch_15945 \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    paths.root_dir=/scratch/lcornelis/bgbench \
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
    logger.wandb.tags=[chebnet,parkinsons,hpsearch] \
    hydra.run.dir=${paths.log_dir}/${task_name}/runs/hpsearch_16088 \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    paths.root_dir=/scratch/lcornelis/bgbench \
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
    logger.wandb.tags=[chebnet,parkinsons,hpsearch] \
    hydra.run.dir=${paths.log_dir}/${task_name}/runs/hpsearch_16097 \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    paths.root_dir=/scratch/lcornelis/bgbench \
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
    logger.wandb.tags=[chebnet,parkinsons,hpsearch] \
    hydra.run.dir=${paths.log_dir}/${task_name}/runs/hpsearch_16122 \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    paths.root_dir=/scratch/lcornelis/bgbench \
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
    logger.wandb.tags=[chebnet,addneuromed,hpsearch] \
    hydra.run.dir=${paths.log_dir}/${task_name}/runs/hpsearch_16129 \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    paths.root_dir=/scratch/lcornelis/bgbench \
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
    logger.wandb.tags=[chebnet,addneuromed,hpsearch] \
    hydra.run.dir=${paths.log_dir}/${task_name}/runs/hpsearch_16148 \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    paths.root_dir=/scratch/lcornelis/bgbench \
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
    logger.wandb.tags=[chebnet,addneuromed,hpsearch] \
    hydra.run.dir=${paths.log_dir}/${task_name}/runs/hpsearch_16152 \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    paths.root_dir=/scratch/lcornelis/bgbench \
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
    logger.wandb.tags=[chebnet,addneuromed,hpsearch] \
    hydra.run.dir=${paths.log_dir}/${task_name}/runs/hpsearch_16474 \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    paths.root_dir=/scratch/lcornelis/bgbench \
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
    logger.wandb.tags=[chebnet,addneuromed,hpsearch] \
    hydra.run.dir=${paths.log_dir}/${task_name}/runs/hpsearch_16481 \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    paths.root_dir=/scratch/lcornelis/bgbench \
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
    logger.wandb.tags=[chebnet,addneuromed,hpsearch] \
    hydra.run.dir=${paths.log_dir}/${task_name}/runs/hpsearch_16644 \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    paths.root_dir=/scratch/lcornelis/bgbench \
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
    logger.wandb.tags=[chebnet,motrpac,hpsearch] \
    hydra.run.dir=${paths.log_dir}/${task_name}/runs/hpsearch_17017 \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    paths.root_dir=/scratch/lcornelis/bgbench \
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
    logger.wandb.tags=[chebnet,motrpac,hpsearch] \
    hydra.run.dir=${paths.log_dir}/${task_name}/runs/hpsearch_17065 \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    paths.root_dir=/scratch/lcornelis/bgbench \
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
    logger.wandb.tags=[chebnet,motrpac,hpsearch] \
    hydra.run.dir=${paths.log_dir}/${task_name}/runs/hpsearch_17089 \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    paths.root_dir=/scratch/lcornelis/bgbench \
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
    logger.wandb.tags=[chebnet,motrpac,hpsearch] \
    hydra.run.dir=${paths.log_dir}/${task_name}/runs/hpsearch_17113 \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    paths.root_dir=/scratch/lcornelis/bgbench \
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
    logger.wandb.tags=[chebnet,motrpac,hpsearch] \
    hydra.run.dir=${paths.log_dir}/${task_name}/runs/hpsearch_17114 \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    paths.root_dir=/scratch/lcornelis/bgbench \
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
    logger.wandb.tags=[chebnet,motrpac,hpsearch] \
    hydra.run.dir=${paths.log_dir}/${task_name}/runs/hpsearch_17151 \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    paths.root_dir=/scratch/lcornelis/bgbench \
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
    logger.wandb.tags=[chebnet,motrpac,hpsearch] \
    hydra.run.dir=${paths.log_dir}/${task_name}/runs/hpsearch_17159 \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    paths.root_dir=/scratch/lcornelis/bgbench \
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
    logger.wandb.tags=[chebnet,motrpac,hpsearch] \
    hydra.run.dir=${paths.log_dir}/${task_name}/runs/hpsearch_17207 \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    paths.root_dir=/scratch/lcornelis/bgbench \
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
    logger.wandb.tags=[chebnet,motrpac,hpsearch] \
    hydra.run.dir=${paths.log_dir}/${task_name}/runs/hpsearch_17273 \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    paths.root_dir=/scratch/lcornelis/bgbench \
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
    logger.wandb.tags=[chebnet,motrpac,hpsearch] \
    hydra.run.dir=${paths.log_dir}/${task_name}/runs/hpsearch_17313 \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    paths.root_dir=/scratch/lcornelis/bgbench \
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
    logger.wandb.tags=[chebnet,motrpac,hpsearch] \
    hydra.run.dir=${paths.log_dir}/${task_name}/runs/hpsearch_17375 \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    paths.root_dir=/scratch/lcornelis/bgbench \
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
    logger.wandb.tags=[chebnet,motrpac,hpsearch] \
    hydra.run.dir=${paths.log_dir}/${task_name}/runs/hpsearch_17489 \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    paths.root_dir=/scratch/lcornelis/bgbench \
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
    logger.wandb.tags=[chebnet,motrpac,hpsearch] \
    hydra.run.dir=${paths.log_dir}/${task_name}/runs/hpsearch_17523 \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    paths.root_dir=/scratch/lcornelis/bgbench \
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
    logger.wandb.tags=[chebnet,motrpac,hpsearch] \
    hydra.run.dir=${paths.log_dir}/${task_name}/runs/hpsearch_17534 \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    paths.root_dir=/scratch/lcornelis/bgbench \
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
    logger.wandb.tags=[chebnet,motrpac,hpsearch] \
    hydra.run.dir=${paths.log_dir}/${task_name}/runs/hpsearch_17560 \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    paths.root_dir=/scratch/lcornelis/bgbench \
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
    logger.wandb.tags=[chebnet,motrpac,hpsearch] \
    hydra.run.dir=${paths.log_dir}/${task_name}/runs/hpsearch_17605 \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    paths.root_dir=/scratch/lcornelis/bgbench \
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
    logger.wandb.tags=[chebnet,motrpac,hpsearch] \
    hydra.run.dir=${paths.log_dir}/${task_name}/runs/hpsearch_17644 \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    paths.root_dir=/scratch/lcornelis/bgbench \
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
    logger.wandb.tags=[chebnet,brca,hpsearch] \
    hydra.run.dir=${paths.log_dir}/${task_name}/runs/hpsearch_17682 \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    paths.root_dir=/scratch/lcornelis/bgbench \
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
    logger.wandb.tags=[chebnet,brca,hpsearch] \
    hydra.run.dir=${paths.log_dir}/${task_name}/runs/hpsearch_17754 \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    paths.root_dir=/scratch/lcornelis/bgbench \
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
    logger.wandb.tags=[chebnet,brca,hpsearch] \
    hydra.run.dir=${paths.log_dir}/${task_name}/runs/hpsearch_17811 \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    paths.root_dir=/scratch/lcornelis/bgbench \
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
    logger.wandb.tags=[chebnet,brca,hpsearch] \
    hydra.run.dir=${paths.log_dir}/${task_name}/runs/hpsearch_17917 \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    paths.root_dir=/scratch/lcornelis/bgbench \
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
    logger.wandb.tags=[chebnet,brca,hpsearch] \
    hydra.run.dir=${paths.log_dir}/${task_name}/runs/hpsearch_17937 \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    paths.root_dir=/scratch/lcornelis/bgbench \
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
    logger.wandb.tags=[chebnet,brca,hpsearch] \
    hydra.run.dir=${paths.log_dir}/${task_name}/runs/hpsearch_17939 \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    paths.root_dir=/scratch/lcornelis/bgbench \
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
    logger.wandb.tags=[chebnet,brca,hpsearch] \
    hydra.run.dir=${paths.log_dir}/${task_name}/runs/hpsearch_17954 \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    paths.root_dir=/scratch/lcornelis/bgbench \
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
    logger.wandb.tags=[chebnet,brca,hpsearch] \
    hydra.run.dir=${paths.log_dir}/${task_name}/runs/hpsearch_18073 \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    paths.root_dir=/scratch/lcornelis/bgbench \
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
    logger.wandb.tags=[chebnet,brca,hpsearch] \
    hydra.run.dir=${paths.log_dir}/${task_name}/runs/hpsearch_18104 \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    paths.root_dir=/scratch/lcornelis/bgbench \
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
    logger.wandb.tags=[chebnet,brca,hpsearch] \
    hydra.run.dir=${paths.log_dir}/${task_name}/runs/hpsearch_18145 \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    paths.root_dir=/scratch/lcornelis/bgbench \
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
    logger.wandb.tags=[chebnet,brca,hpsearch] \
    hydra.run.dir=${paths.log_dir}/${task_name}/runs/hpsearch_18213 \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    paths.root_dir=/scratch/lcornelis/bgbench \
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
    logger.wandb.tags=[chebnet,brca,hpsearch] \
    hydra.run.dir=${paths.log_dir}/${task_name}/runs/hpsearch_18231 \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    paths.root_dir=/scratch/lcornelis/bgbench \
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
    logger.wandb.tags=[chebnet,brca,hpsearch] \
    hydra.run.dir=${paths.log_dir}/${task_name}/runs/hpsearch_18252 \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    paths.root_dir=/scratch/lcornelis/bgbench \
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
    logger.wandb.tags=[chebnet,brca,hpsearch] \
    hydra.run.dir=${paths.log_dir}/${task_name}/runs/hpsearch_18274 \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    paths.root_dir=/scratch/lcornelis/bgbench \
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
    logger.wandb.tags=[chebnet,brca,hpsearch] \
    hydra.run.dir=${paths.log_dir}/${task_name}/runs/hpsearch_18281 \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    paths.root_dir=/scratch/lcornelis/bgbench \
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
    logger.wandb.tags=[chebnet,brca,hpsearch] \
    hydra.run.dir=${paths.log_dir}/${task_name}/runs/hpsearch_18335 \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    paths.root_dir=/scratch/lcornelis/bgbench \
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
    logger.wandb.tags=[chebnet,brca,hpsearch] \
    hydra.run.dir=${paths.log_dir}/${task_name}/runs/hpsearch_18390 \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    paths.root_dir=/scratch/lcornelis/bgbench \
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
    logger.wandb.tags=[gps,parkinsons,hpsearch] \
    hydra.run.dir=${paths.log_dir}/${task_name}/runs/hpsearch_18881 \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    paths.root_dir=/scratch/lcornelis/bgbench \
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
    logger.wandb.tags=[gps,parkinsons,hpsearch] \
    hydra.run.dir=${paths.log_dir}/${task_name}/runs/hpsearch_19230 \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    paths.root_dir=/scratch/lcornelis/bgbench \
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
    logger.wandb.tags=[gps,addneuromed,hpsearch] \
    hydra.run.dir=${paths.log_dir}/${task_name}/runs/hpsearch_20038 \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    paths.root_dir=/scratch/lcornelis/bgbench \
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
    logger.wandb.tags=[gps,addneuromed,hpsearch] \
    hydra.run.dir=${paths.log_dir}/${task_name}/runs/hpsearch_20689 \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    paths.root_dir=/scratch/lcornelis/bgbench \
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
    logger.wandb.tags=[gps,addneuromed,hpsearch] \
    hydra.run.dir=${paths.log_dir}/${task_name}/runs/hpsearch_20694 \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    paths.root_dir=/scratch/lcornelis/bgbench \
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
    logger.wandb.tags=[gps,addneuromed,hpsearch] \
    hydra.run.dir=${paths.log_dir}/${task_name}/runs/hpsearch_20735 \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    paths.root_dir=/scratch/lcornelis/bgbench \
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
    logger.wandb.tags=[gps,addneuromed,hpsearch] \
    hydra.run.dir=${paths.log_dir}/${task_name}/runs/hpsearch_20875 \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    paths.root_dir=/scratch/lcornelis/bgbench \
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
    logger.wandb.tags=[gps,motrpac,hpsearch] \
    hydra.run.dir=${paths.log_dir}/${task_name}/runs/hpsearch_21589 \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    paths.root_dir=/scratch/lcornelis/bgbench \
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
    logger.wandb.tags=[gps,motrpac,hpsearch] \
    hydra.run.dir=${paths.log_dir}/${task_name}/runs/hpsearch_21600 \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    paths.root_dir=/scratch/lcornelis/bgbench \
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
    logger.wandb.tags=[gps,motrpac,hpsearch] \
    hydra.run.dir=${paths.log_dir}/${task_name}/runs/hpsearch_22008 \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    paths.root_dir=/scratch/lcornelis/bgbench \
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
    logger.wandb.tags=[gps,motrpac,hpsearch] \
    hydra.run.dir=${paths.log_dir}/${task_name}/runs/hpsearch_22256 \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    paths.root_dir=/scratch/lcornelis/bgbench \
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
    logger.wandb.tags=[gatv2,parkinsons,hpsearch] \
    hydra.run.dir=${paths.log_dir}/${task_name}/runs/hpsearch_3167 \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    paths.root_dir=/scratch/lcornelis/bgbench \
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
    logger.wandb.tags=[gatv2,addneuromed,hpsearch] \
    hydra.run.dir=${paths.log_dir}/${task_name}/runs/hpsearch_3907 \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    paths.root_dir=/scratch/lcornelis/bgbench \
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
    logger.wandb.tags=[gatv2,addneuromed,hpsearch] \
    hydra.run.dir=${paths.log_dir}/${task_name}/runs/hpsearch_3910 \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    paths.root_dir=/scratch/lcornelis/bgbench \
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
    logger.wandb.tags=[gatv2,addneuromed,hpsearch] \
    hydra.run.dir=${paths.log_dir}/${task_name}/runs/hpsearch_3931 \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    paths.root_dir=/scratch/lcornelis/bgbench \
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
    logger.wandb.tags=[gatv2,addneuromed,hpsearch] \
    hydra.run.dir=${paths.log_dir}/${task_name}/runs/hpsearch_3934 \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    paths.root_dir=/scratch/lcornelis/bgbench \
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
    logger.wandb.tags=[gatv2,addneuromed,hpsearch] \
    hydra.run.dir=${paths.log_dir}/${task_name}/runs/hpsearch_3935 \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    paths.root_dir=/scratch/lcornelis/bgbench \
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
    logger.wandb.tags=[gps,addneuromed,hpsearch] \
    hydra.run.dir=${paths.log_dir}/${task_name}/runs/hpsearch_3933 \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    paths.root_dir=/scratch/lcornelis/bgbench \
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
    logger.wandb.tags=[gps,addneuromed,hpsearch] \
    hydra.run.dir=${paths.log_dir}/${task_name}/runs/hpsearch_3935 \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    paths.root_dir=/scratch/lcornelis/bgbench \
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
    logger.wandb.tags=[gps,addneuromed,hpsearch] \
    hydra.run.dir=${paths.log_dir}/${task_name}/runs/hpsearch_3971 \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    paths.root_dir=/scratch/lcornelis/bgbench \
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
    logger.wandb.tags=[gps,addneuromed,hpsearch] \
    hydra.run.dir=${paths.log_dir}/${task_name}/runs/hpsearch_4340 \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    paths.root_dir=/scratch/lcornelis/bgbench \
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
    logger.wandb.tags=[gatv2,addneuromed,hpsearch] \
    hydra.run.dir=${paths.log_dir}/${task_name}/runs/hpsearch_3852 \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    paths.root_dir=/scratch/lcornelis/bgbench \
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
    logger.wandb.tags=[gatv2,addneuromed,hpsearch] \
    hydra.run.dir=${paths.log_dir}/${task_name}/runs/hpsearch_3861 \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    paths.root_dir=/scratch/lcornelis/bgbench \
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
    logger.wandb.tags=[gatv2,addneuromed,hpsearch] \
    hydra.run.dir=${paths.log_dir}/${task_name}/runs/hpsearch_3862 \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    paths.root_dir=/scratch/lcornelis/bgbench \
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
    logger.wandb.tags=[gatv2,addneuromed,hpsearch] \
    hydra.run.dir=${paths.log_dir}/${task_name}/runs/hpsearch_3864 \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    paths.root_dir=/scratch/lcornelis/bgbench \
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
    logger.wandb.tags=[gatv2,addneuromed,hpsearch] \
    hydra.run.dir=${paths.log_dir}/${task_name}/runs/hpsearch_3885 \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    paths.root_dir=/scratch/lcornelis/bgbench \
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
    logger.wandb.tags=[gatv2,addneuromed,hpsearch] \
    hydra.run.dir=${paths.log_dir}/${task_name}/runs/hpsearch_3886 \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    paths.root_dir=/scratch/lcornelis/bgbench \
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
    logger.wandb.tags=[gatv2,addneuromed,hpsearch] \
    hydra.run.dir=${paths.log_dir}/${task_name}/runs/hpsearch_3887 \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    paths.root_dir=/scratch/lcornelis/bgbench \
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
    logger.wandb.tags=[gatv2,addneuromed,hpsearch] \
    hydra.run.dir=${paths.log_dir}/${task_name}/runs/hpsearch_3888 \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    paths.root_dir=/scratch/lcornelis/bgbench \
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
    logger.wandb.tags=[gatv2,addneuromed,hpsearch] \
    hydra.run.dir=${paths.log_dir}/${task_name}/runs/hpsearch_3931 \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    paths.root_dir=/scratch/lcornelis/bgbench \
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
    logger.wandb.tags=[gatv2,addneuromed,hpsearch] \
    hydra.run.dir=${paths.log_dir}/${task_name}/runs/hpsearch_3932 \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    paths.root_dir=/scratch/lcornelis/bgbench \
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
    logger.wandb.tags=[gatv2,addneuromed,hpsearch] \
    hydra.run.dir=${paths.log_dir}/${task_name}/runs/hpsearch_3934 \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    paths.root_dir=/scratch/lcornelis/bgbench \
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
    logger.wandb.tags=[gps,motrpac,hpsearch] \
    hydra.run.dir=${paths.log_dir}/${task_name}/runs/hpsearch_5145 \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    paths.root_dir=/scratch/lcornelis/bgbench \
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
    logger.wandb.tags=[gatv2,addneuromed,hpsearch] \
    hydra.run.dir=${paths.log_dir}/${task_name}/runs/hpsearch_3935 \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    paths.root_dir=/scratch/lcornelis/bgbench \
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
    logger.wandb.tags=[gatv2,addneuromed,hpsearch] \
    hydra.run.dir=${paths.log_dir}/${task_name}/runs/hpsearch_3936 \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    paths.root_dir=/scratch/lcornelis/bgbench \
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
    logger.wandb.tags=[gatv2,addneuromed,hpsearch] \
    hydra.run.dir=${paths.log_dir}/${task_name}/runs/hpsearch_3970 \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    paths.root_dir=/scratch/lcornelis/bgbench \
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
    logger.wandb.tags=[gps,motrpac,hpsearch] \
    hydra.run.dir=${paths.log_dir}/${task_name}/runs/hpsearch_5290 \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    paths.root_dir=/scratch/lcornelis/bgbench \
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
    logger.wandb.tags=[gatv2,addneuromed,hpsearch] \
    hydra.run.dir=${paths.log_dir}/${task_name}/runs/hpsearch_3979 \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    paths.root_dir=/scratch/lcornelis/bgbench \
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
    logger.wandb.tags=[gatv2,addneuromed,hpsearch] \
    hydra.run.dir=${paths.log_dir}/${task_name}/runs/hpsearch_3982 \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    paths.root_dir=/scratch/lcornelis/bgbench \
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
    logger.wandb.tags=[gatv4,addneuromed,hpsearch] \
    hydra.run.dir=${paths.log_dir}/${task_name}/runs/hpsearch_1174 \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    paths.root_dir=/scratch/lcornelis/bgbench \
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
    logger.wandb.tags=[gatv2,addneuromed,hpsearch] \
    hydra.run.dir=${paths.log_dir}/${task_name}/runs/hpsearch_3984 \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    paths.root_dir=/scratch/lcornelis/bgbench \
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
    logger.wandb.tags=[gatv2,addneuromed,hpsearch] \
    hydra.run.dir=${paths.log_dir}/${task_name}/runs/hpsearch_4004 \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    paths.root_dir=/scratch/lcornelis/bgbench \
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
    logger.wandb.tags=[gatv2,addneuromed,hpsearch] \
    hydra.run.dir=${paths.log_dir}/${task_name}/runs/hpsearch_4007 \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    paths.root_dir=/scratch/lcornelis/bgbench \
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
    logger.wandb.tags=[gatv4,addneuromed,hpsearch] \
    hydra.run.dir=${paths.log_dir}/${task_name}/runs/hpsearch_1219 \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    paths.root_dir=/scratch/lcornelis/bgbench \
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
    logger.wandb.tags=[gps,brca,hpsearch] \
    hydra.run.dir=${paths.log_dir}/${task_name}/runs/hpsearch_5431 \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    paths.root_dir=/scratch/lcornelis/bgbench \
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
    logger.wandb.tags=[gatv2,addneuromed,hpsearch] \
    hydra.run.dir=${paths.log_dir}/${task_name}/runs/hpsearch_4027 \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    paths.root_dir=/scratch/lcornelis/bgbench \
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
    logger.wandb.tags=[gatv4,addneuromed,hpsearch] \
    hydra.run.dir=${paths.log_dir}/${task_name}/runs/hpsearch_1268 \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    paths.root_dir=/scratch/lcornelis/bgbench \
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
    logger.wandb.tags=[gatv2,addneuromed,hpsearch] \
    hydra.run.dir=${paths.log_dir}/${task_name}/runs/hpsearch_4247 \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    paths.root_dir=/scratch/lcornelis/bgbench \
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
    logger.wandb.tags=[gatv2,addneuromed,hpsearch] \
    hydra.run.dir=${paths.log_dir}/${task_name}/runs/hpsearch_4259 \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    paths.root_dir=/scratch/lcornelis/bgbench \
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
    logger.wandb.tags=[gatv2,addneuromed,hpsearch] \
    hydra.run.dir=${paths.log_dir}/${task_name}/runs/hpsearch_4268 \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    paths.root_dir=/scratch/lcornelis/bgbench \
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
    logger.wandb.tags=[gatv2,addneuromed,hpsearch] \
    hydra.run.dir=${paths.log_dir}/${task_name}/runs/hpsearch_4292 \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    paths.root_dir=/scratch/lcornelis/bgbench \
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
    logger.wandb.tags=[gatv2,addneuromed,hpsearch] \
    hydra.run.dir=${paths.log_dir}/${task_name}/runs/hpsearch_4293 \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    paths.root_dir=/scratch/lcornelis/bgbench \
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
    logger.wandb.tags=[gatv2,addneuromed,hpsearch] \
    hydra.run.dir=${paths.log_dir}/${task_name}/runs/hpsearch_4295 \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    paths.root_dir=/scratch/lcornelis/bgbench \
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
    logger.wandb.tags=[gatv2,addneuromed,hpsearch] \
    hydra.run.dir=${paths.log_dir}/${task_name}/runs/hpsearch_4316 \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    paths.root_dir=/scratch/lcornelis/bgbench \
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
    logger.wandb.tags=[gatv2,addneuromed,hpsearch] \
    hydra.run.dir=${paths.log_dir}/${task_name}/runs/hpsearch_4332 \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    paths.root_dir=/scratch/lcornelis/bgbench \
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
    logger.wandb.tags=[gatv2,addneuromed,hpsearch] \
    hydra.run.dir=${paths.log_dir}/${task_name}/runs/hpsearch_4343 \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    paths.root_dir=/scratch/lcornelis/bgbench \
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
    logger.wandb.tags=[gatv2,addneuromed,hpsearch] \
    hydra.run.dir=${paths.log_dir}/${task_name}/runs/hpsearch_4368 \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    paths.root_dir=/scratch/lcornelis/bgbench \
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
    logger.wandb.tags=[gatv2,addneuromed,hpsearch] \
    hydra.run.dir=${paths.log_dir}/${task_name}/runs/hpsearch_4379 \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    paths.root_dir=/scratch/lcornelis/bgbench \
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
    logger.wandb.tags=[gatv2,addneuromed,hpsearch] \
    hydra.run.dir=${paths.log_dir}/${task_name}/runs/hpsearch_4392 \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    paths.root_dir=/scratch/lcornelis/bgbench \
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
    logger.wandb.tags=[gatv2,addneuromed,hpsearch] \
    hydra.run.dir=${paths.log_dir}/${task_name}/runs/hpsearch_4413 \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    paths.root_dir=/scratch/lcornelis/bgbench \
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
    logger.wandb.tags=[gatv4,parkinsons,hpsearch] \
    hydra.run.dir=${paths.log_dir}/${task_name}/runs/hpsearch_6216 \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    paths.root_dir=/scratch/lcornelis/bgbench \
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
    logger.wandb.tags=[gatv4,parkinsons,hpsearch] \
    hydra.run.dir=${paths.log_dir}/${task_name}/runs/hpsearch_6227 \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    paths.root_dir=/scratch/lcornelis/bgbench \
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
    logger.wandb.tags=[gatv4,parkinsons,hpsearch] \
    hydra.run.dir=${paths.log_dir}/${task_name}/runs/hpsearch_6229 \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    paths.root_dir=/scratch/lcornelis/bgbench \
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
    logger.wandb.tags=[gatv4,parkinsons,hpsearch] \
    hydra.run.dir=${paths.log_dir}/${task_name}/runs/hpsearch_6233 \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    paths.root_dir=/scratch/lcornelis/bgbench \
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
    logger.wandb.tags=[gatv4,parkinsons,hpsearch] \
    hydra.run.dir=${paths.log_dir}/${task_name}/runs/hpsearch_6235 \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    paths.root_dir=/scratch/lcornelis/bgbench \
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
    logger.wandb.tags=[gatv4,parkinsons,hpsearch] \
    hydra.run.dir=${paths.log_dir}/${task_name}/runs/hpsearch_6236 \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    paths.root_dir=/scratch/lcornelis/bgbench \
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
    logger.wandb.tags=[gatv4,parkinsons,hpsearch] \
    hydra.run.dir=${paths.log_dir}/${task_name}/runs/hpsearch_6237 \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    paths.root_dir=/scratch/lcornelis/bgbench \
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
    logger.wandb.tags=[gatv4,parkinsons,hpsearch] \
    hydra.run.dir=${paths.log_dir}/${task_name}/runs/hpsearch_6239 \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    paths.root_dir=/scratch/lcornelis/bgbench \
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
    logger.wandb.tags=[gatv4,parkinsons,hpsearch] \
    hydra.run.dir=${paths.log_dir}/${task_name}/runs/hpsearch_6240 \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    paths.root_dir=/scratch/lcornelis/bgbench \
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
    logger.wandb.tags=[gatv2,motrpac,hpsearch] \
    hydra.run.dir=${paths.log_dir}/${task_name}/runs/hpsearch_4680 \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    paths.root_dir=/scratch/lcornelis/bgbench \
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
    logger.wandb.tags=[gatv4,parkinsons,hpsearch] \
    hydra.run.dir=${paths.log_dir}/${task_name}/runs/hpsearch_6595 \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    paths.root_dir=/scratch/lcornelis/bgbench \
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
    logger.wandb.tags=[gatv4,parkinsons,hpsearch] \
    hydra.run.dir=${paths.log_dir}/${task_name}/runs/hpsearch_6596 \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    paths.root_dir=/scratch/lcornelis/bgbench \
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
    logger.wandb.tags=[gatv4,parkinsons,hpsearch] \
    hydra.run.dir=${paths.log_dir}/${task_name}/runs/hpsearch_6597 \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    paths.root_dir=/scratch/lcornelis/bgbench \
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
    logger.wandb.tags=[gatv4,parkinsons,hpsearch] \
    hydra.run.dir=${paths.log_dir}/${task_name}/runs/hpsearch_6598 \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    paths.root_dir=/scratch/lcornelis/bgbench \
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
    logger.wandb.tags=[gatv4,parkinsons,hpsearch] \
    hydra.run.dir=${paths.log_dir}/${task_name}/runs/hpsearch_6600 \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    paths.root_dir=/scratch/lcornelis/bgbench \
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
    logger.wandb.tags=[gatv4,parkinsons,hpsearch] \
    hydra.run.dir=${paths.log_dir}/${task_name}/runs/hpsearch_6621 \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    paths.root_dir=/scratch/lcornelis/bgbench \
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
    logger.wandb.tags=[gatv4,parkinsons,hpsearch] \
    hydra.run.dir=${paths.log_dir}/${task_name}/runs/hpsearch_6622 \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    paths.root_dir=/scratch/lcornelis/bgbench \
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
    logger.wandb.tags=[gatv4,parkinsons,hpsearch] \
    hydra.run.dir=${paths.log_dir}/${task_name}/runs/hpsearch_6623 \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    paths.root_dir=/scratch/lcornelis/bgbench \
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
    logger.wandb.tags=[gatv2,motrpac,hpsearch] \
    hydra.run.dir=${paths.log_dir}/${task_name}/runs/hpsearch_4700 \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    paths.root_dir=/scratch/lcornelis/bgbench \
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
    logger.wandb.tags=[gatv4,addneuromed,hpsearch] \
    hydra.run.dir=${paths.log_dir}/${task_name}/runs/hpsearch_6933 \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    paths.root_dir=/scratch/lcornelis/bgbench \
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
    logger.wandb.tags=[gatv4,addneuromed,hpsearch] \
    hydra.run.dir=${paths.log_dir}/${task_name}/runs/hpsearch_6935 \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    paths.root_dir=/scratch/lcornelis/bgbench \
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
    logger.wandb.tags=[gatv4,addneuromed,hpsearch] \
    hydra.run.dir=${paths.log_dir}/${task_name}/runs/hpsearch_6958 \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    paths.root_dir=/scratch/lcornelis/bgbench \
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
    logger.wandb.tags=[gatv4,addneuromed,hpsearch] \
    hydra.run.dir=${paths.log_dir}/${task_name}/runs/hpsearch_6959 \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    paths.root_dir=/scratch/lcornelis/bgbench \
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
    logger.wandb.tags=[gatv4,addneuromed,hpsearch] \
    hydra.run.dir=${paths.log_dir}/${task_name}/runs/hpsearch_6972 \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    paths.root_dir=/scratch/lcornelis/bgbench \
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
    logger.wandb.tags=[gatv4,addneuromed,hpsearch] \
    hydra.run.dir=${paths.log_dir}/${task_name}/runs/hpsearch_6978 \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    paths.root_dir=/scratch/lcornelis/bgbench \
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
    logger.wandb.tags=[gatv4,addneuromed,hpsearch] \
    hydra.run.dir=${paths.log_dir}/${task_name}/runs/hpsearch_6980 \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    paths.root_dir=/scratch/lcornelis/bgbench \
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
    logger.wandb.tags=[gatv4,addneuromed,hpsearch] \
    hydra.run.dir=${paths.log_dir}/${task_name}/runs/hpsearch_6981 \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    paths.root_dir=/scratch/lcornelis/bgbench \
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
    logger.wandb.tags=[gatv4,addneuromed,hpsearch] \
    hydra.run.dir=${paths.log_dir}/${task_name}/runs/hpsearch_6982 \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    paths.root_dir=/scratch/lcornelis/bgbench \
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
    logger.wandb.tags=[gatv4,addneuromed,hpsearch] \
    hydra.run.dir=${paths.log_dir}/${task_name}/runs/hpsearch_6983 \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    paths.root_dir=/scratch/lcornelis/bgbench \
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
    logger.wandb.tags=[gatv4,addneuromed,hpsearch] \
    hydra.run.dir=${paths.log_dir}/${task_name}/runs/hpsearch_6984 \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    paths.root_dir=/scratch/lcornelis/bgbench \
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
    logger.wandb.tags=[gatv4,addneuromed,hpsearch] \
    hydra.run.dir=${paths.log_dir}/${task_name}/runs/hpsearch_6992 \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    paths.root_dir=/scratch/lcornelis/bgbench \
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
    logger.wandb.tags=[gatv4,addneuromed,hpsearch] \
    hydra.run.dir=${paths.log_dir}/${task_name}/runs/hpsearch_7004 \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    paths.root_dir=/scratch/lcornelis/bgbench \
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
    logger.wandb.tags=[gatv4,addneuromed,hpsearch] \
    hydra.run.dir=${paths.log_dir}/${task_name}/runs/hpsearch_7005 \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    paths.root_dir=/scratch/lcornelis/bgbench \
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
    logger.wandb.tags=[gatv4,addneuromed,hpsearch] \
    hydra.run.dir=${paths.log_dir}/${task_name}/runs/hpsearch_7006 \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    paths.root_dir=/scratch/lcornelis/bgbench \
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
    logger.wandb.tags=[gatv4,addneuromed,hpsearch] \
    hydra.run.dir=${paths.log_dir}/${task_name}/runs/hpsearch_7007 \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    paths.root_dir=/scratch/lcornelis/bgbench \
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
    logger.wandb.tags=[gatv4,addneuromed,hpsearch] \
    hydra.run.dir=${paths.log_dir}/${task_name}/runs/hpsearch_7008 \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    paths.root_dir=/scratch/lcornelis/bgbench \
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
    logger.wandb.tags=[gatv4,addneuromed,hpsearch] \
    hydra.run.dir=${paths.log_dir}/${task_name}/runs/hpsearch_7342 \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    paths.root_dir=/scratch/lcornelis/bgbench \
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
    logger.wandb.tags=[gatv4,addneuromed,hpsearch] \
    hydra.run.dir=${paths.log_dir}/${task_name}/runs/hpsearch_7360 \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    paths.root_dir=/scratch/lcornelis/bgbench \
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
    logger.wandb.tags=[gatv4,addneuromed,hpsearch] \
    hydra.run.dir=${paths.log_dir}/${task_name}/runs/hpsearch_7363 \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    paths.root_dir=/scratch/lcornelis/bgbench \
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
    logger.wandb.tags=[gatv4,addneuromed,hpsearch] \
    hydra.run.dir=${paths.log_dir}/${task_name}/runs/hpsearch_7364 \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    paths.root_dir=/scratch/lcornelis/bgbench \
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
    logger.wandb.tags=[gatv4,addneuromed,hpsearch] \
    hydra.run.dir=${paths.log_dir}/${task_name}/runs/hpsearch_7365 \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    paths.root_dir=/scratch/lcornelis/bgbench \
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
    logger.wandb.tags=[gatv4,addneuromed,hpsearch] \
    hydra.run.dir=${paths.log_dir}/${task_name}/runs/hpsearch_7366 \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    paths.root_dir=/scratch/lcornelis/bgbench \
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
    logger.wandb.tags=[gatv2,motrpac,hpsearch] \
    hydra.run.dir=${paths.log_dir}/${task_name}/runs/hpsearch_5149 \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    paths.root_dir=/scratch/lcornelis/bgbench \
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
    logger.wandb.tags=[gatv2,motrpac,hpsearch] \
    hydra.run.dir=${paths.log_dir}/${task_name}/runs/hpsearch_5150 \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    paths.root_dir=/scratch/lcornelis/bgbench \
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
    logger.wandb.tags=[gatv4,addneuromed,hpsearch] \
    hydra.run.dir=${paths.log_dir}/${task_name}/runs/hpsearch_7367 \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    paths.root_dir=/scratch/lcornelis/bgbench \
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
    logger.wandb.tags=[gatv2,motrpac,hpsearch] \
    hydra.run.dir=${paths.log_dir}/${task_name}/runs/hpsearch_5151 \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    paths.root_dir=/scratch/lcornelis/bgbench \
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
    logger.wandb.tags=[gatv4,addneuromed,hpsearch] \
    hydra.run.dir=${paths.log_dir}/${task_name}/runs/hpsearch_7368 \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    paths.root_dir=/scratch/lcornelis/bgbench \
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
    logger.wandb.tags=[gatv2,motrpac,hpsearch] \
    hydra.run.dir=${paths.log_dir}/${task_name}/runs/hpsearch_5152 \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    paths.root_dir=/scratch/lcornelis/bgbench \
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
    logger.wandb.tags=[gatv2,motrpac,hpsearch] \
    hydra.run.dir=${paths.log_dir}/${task_name}/runs/hpsearch_5153 \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    paths.root_dir=/scratch/lcornelis/bgbench \
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
    logger.wandb.tags=[gatv2,motrpac,hpsearch] \
    hydra.run.dir=${paths.log_dir}/${task_name}/runs/hpsearch_5154 \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    paths.root_dir=/scratch/lcornelis/bgbench \
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
    logger.wandb.tags=[gatv2,motrpac,hpsearch] \
    hydra.run.dir=${paths.log_dir}/${task_name}/runs/hpsearch_5155 \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    paths.root_dir=/scratch/lcornelis/bgbench \
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
    logger.wandb.tags=[gatv2,motrpac,hpsearch] \
    hydra.run.dir=${paths.log_dir}/${task_name}/runs/hpsearch_5156 \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    paths.root_dir=/scratch/lcornelis/bgbench \
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
    logger.wandb.tags=[gatv4,addneuromed,hpsearch] \
    hydra.run.dir=${paths.log_dir}/${task_name}/runs/hpsearch_7375 \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    paths.root_dir=/scratch/lcornelis/bgbench \
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
    logger.wandb.tags=[gatv4,addneuromed,hpsearch] \
    hydra.run.dir=${paths.log_dir}/${task_name}/runs/hpsearch_7376 \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    paths.root_dir=/scratch/lcornelis/bgbench \
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
    logger.wandb.tags=[gatv4,addneuromed,hpsearch] \
    hydra.run.dir=${paths.log_dir}/${task_name}/runs/hpsearch_7377 \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    paths.root_dir=/scratch/lcornelis/bgbench \
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
    logger.wandb.tags=[gatv4,addneuromed,hpsearch] \
    hydra.run.dir=${paths.log_dir}/${task_name}/runs/hpsearch_7378 \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    paths.root_dir=/scratch/lcornelis/bgbench \
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
    logger.wandb.tags=[gatv4,addneuromed,hpsearch] \
    hydra.run.dir=${paths.log_dir}/${task_name}/runs/hpsearch_7379 \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    paths.root_dir=/scratch/lcornelis/bgbench \
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
    logger.wandb.tags=[gatv4,addneuromed,hpsearch] \
    hydra.run.dir=${paths.log_dir}/${task_name}/runs/hpsearch_7380 \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    paths.root_dir=/scratch/lcornelis/bgbench \
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
    logger.wandb.tags=[gatv4,addneuromed,hpsearch] \
    hydra.run.dir=${paths.log_dir}/${task_name}/runs/hpsearch_7381 \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    paths.root_dir=/scratch/lcornelis/bgbench \
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
    logger.wandb.tags=[gatv4,addneuromed,hpsearch] \
    hydra.run.dir=${paths.log_dir}/${task_name}/runs/hpsearch_7382 \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    paths.root_dir=/scratch/lcornelis/bgbench \
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
    logger.wandb.tags=[gatv4,addneuromed,hpsearch] \
    hydra.run.dir=${paths.log_dir}/${task_name}/runs/hpsearch_7383 \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    paths.root_dir=/scratch/lcornelis/bgbench \
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
    logger.wandb.tags=[gatv4,addneuromed,hpsearch] \
    hydra.run.dir=${paths.log_dir}/${task_name}/runs/hpsearch_7384 \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    paths.root_dir=/scratch/lcornelis/bgbench \
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
    logger.wandb.tags=[gatv4,addneuromed,hpsearch] \
    hydra.run.dir=${paths.log_dir}/${task_name}/runs/hpsearch_7385 \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    paths.root_dir=/scratch/lcornelis/bgbench \
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
    logger.wandb.tags=[gatv4,addneuromed,hpsearch] \
    hydra.run.dir=${paths.log_dir}/${task_name}/runs/hpsearch_7387 \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    paths.root_dir=/scratch/lcornelis/bgbench \
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
    logger.wandb.tags=[gatv4,addneuromed,hpsearch] \
    hydra.run.dir=${paths.log_dir}/${task_name}/runs/hpsearch_7388 \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    paths.root_dir=/scratch/lcornelis/bgbench \
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
    logger.wandb.tags=[gatv4,addneuromed,hpsearch] \
    hydra.run.dir=${paths.log_dir}/${task_name}/runs/hpsearch_7389 \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    paths.root_dir=/scratch/lcornelis/bgbench \
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
    logger.wandb.tags=[gatv4,addneuromed,hpsearch] \
    hydra.run.dir=${paths.log_dir}/${task_name}/runs/hpsearch_7391 \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    paths.root_dir=/scratch/lcornelis/bgbench \
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
    logger.wandb.tags=[gatv4,addneuromed,hpsearch] \
    hydra.run.dir=${paths.log_dir}/${task_name}/runs/hpsearch_7392 \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    paths.root_dir=/scratch/lcornelis/bgbench \
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
    logger.wandb.tags=[gatv4,addneuromed,hpsearch] \
    hydra.run.dir=${paths.log_dir}/${task_name}/runs/hpsearch_7412 \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    paths.root_dir=/scratch/lcornelis/bgbench \
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
    logger.wandb.tags=[gatv4,addneuromed,hpsearch] \
    hydra.run.dir=${paths.log_dir}/${task_name}/runs/hpsearch_7413 \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    paths.root_dir=/scratch/lcornelis/bgbench \
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
    logger.wandb.tags=[gatv4,motrpac,hpsearch] \
    hydra.run.dir=${paths.log_dir}/${task_name}/runs/hpsearch_8111 \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    paths.root_dir=/scratch/lcornelis/bgbench \
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
    logger.wandb.tags=[gatv2,brca,hpsearch] \
    hydra.run.dir=${paths.log_dir}/${task_name}/runs/hpsearch_5806 \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    paths.root_dir=/scratch/lcornelis/bgbench \
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
    logger.wandb.tags=[gatv4,brca,hpsearch] \
    hydra.run.dir=${paths.log_dir}/${task_name}/runs/hpsearch_8543 \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    paths.root_dir=/scratch/lcornelis/bgbench \
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
    logger.wandb.tags=[gatv4,brca,hpsearch] \
    hydra.run.dir=${paths.log_dir}/${task_name}/runs/hpsearch_8567 \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    paths.root_dir=/scratch/lcornelis/bgbench \
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
    logger.wandb.tags=[gatv4,brca,hpsearch] \
    hydra.run.dir=${paths.log_dir}/${task_name}/runs/hpsearch_8590 \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    paths.root_dir=/scratch/lcornelis/bgbench \
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
    logger.wandb.tags=[gatv4,brca,hpsearch] \
    hydra.run.dir=${paths.log_dir}/${task_name}/runs/hpsearch_8591 \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    paths.root_dir=/scratch/lcornelis/bgbench \
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
    logger.wandb.tags=[gatv4,brca,hpsearch] \
    hydra.run.dir=${paths.log_dir}/${task_name}/runs/hpsearch_8853 \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    paths.root_dir=/scratch/lcornelis/bgbench \
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
    logger.wandb.tags=[gatv4,brca,hpsearch] \
    hydra.run.dir=${paths.log_dir}/${task_name}/runs/hpsearch_8854 \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    paths.root_dir=/scratch/lcornelis/bgbench \
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
    logger.wandb.tags=[gatv4,brca,hpsearch] \
    hydra.run.dir=${paths.log_dir}/${task_name}/runs/hpsearch_8855 \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    paths.root_dir=/scratch/lcornelis/bgbench \
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
    logger.wandb.tags=[gatv2,brca,hpsearch] \
    hydra.run.dir=${paths.log_dir}/${task_name}/runs/hpsearch_5928 \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    paths.root_dir=/scratch/lcornelis/bgbench \
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
    logger.wandb.tags=[gatv4,brca,hpsearch] \
    hydra.run.dir=${paths.log_dir}/${task_name}/runs/hpsearch_8901 \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    paths.root_dir=/scratch/lcornelis/bgbench \
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
    logger.wandb.tags=[gatv4,brca,hpsearch] \
    hydra.run.dir=${paths.log_dir}/${task_name}/runs/hpsearch_8904 \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    paths.root_dir=/scratch/lcornelis/bgbench \
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
    logger.wandb.tags=[gatv4,brca,hpsearch] \
    hydra.run.dir=${paths.log_dir}/${task_name}/runs/hpsearch_8949 \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    paths.root_dir=/scratch/lcornelis/bgbench \
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
    logger.wandb.tags=[gatv4,brca,hpsearch] \
    hydra.run.dir=${paths.log_dir}/${task_name}/runs/hpsearch_8950 \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    paths.root_dir=/scratch/lcornelis/bgbench \
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
    logger.wandb.tags=[gatv4,brca,hpsearch] \
    hydra.run.dir=${paths.log_dir}/${task_name}/runs/hpsearch_8951 \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    paths.root_dir=/scratch/lcornelis/bgbench \
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
    logger.wandb.tags=[gatv4,brca,hpsearch] \
    hydra.run.dir=${paths.log_dir}/${task_name}/runs/hpsearch_8952 \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    paths.root_dir=/scratch/lcornelis/bgbench \
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
    logger.wandb.tags=[gatv4,brca,hpsearch] \
    hydra.run.dir=${paths.log_dir}/${task_name}/runs/hpsearch_8973 \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    paths.root_dir=/scratch/lcornelis/bgbench \
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
    logger.wandb.tags=[gatv4,brca,hpsearch] \
    hydra.run.dir=${paths.log_dir}/${task_name}/runs/hpsearch_8974 \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    paths.root_dir=/scratch/lcornelis/bgbench \
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
    logger.wandb.tags=[gps,addneuromed,hpsearch] \
    hydra.run.dir=${paths.log_dir}/${task_name}/runs/hpsearch_10067 \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    paths.root_dir=/scratch/lcornelis/bgbench \
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
    logger.wandb.tags=[gps,addneuromed,hpsearch] \
    hydra.run.dir=${paths.log_dir}/${task_name}/runs/hpsearch_10172 \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    paths.root_dir=/scratch/lcornelis/bgbench \
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
    logger.wandb.tags=[gps,addneuromed,hpsearch] \
    hydra.run.dir=${paths.log_dir}/${task_name}/runs/hpsearch_10210 \
    logger.wandb.project=bgbench_dataset_grid_search_final \
    logger.wandb.entity=bioshape-lab \
    paths.root_dir=/scratch/lcornelis/bgbench \
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
