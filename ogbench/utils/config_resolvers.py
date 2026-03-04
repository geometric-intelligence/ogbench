"""Configuration resolvers for the ogbench package."""

import json
import os

import omegaconf
import torch
from omegaconf import OmegaConf


def register_all_resolvers() -> None:
    """Register all custom OmegaConf resolvers.

    This centralizes resolver registration to avoid duplication across modules. Should be called
    before Hydra initialization in any script that uses configs.
    """
    OmegaConf.register_new_resolver('calculate_num_nodes', calculate_num_nodes, replace=True)
    OmegaConf.register_new_resolver('get_default_metrics', get_default_metrics, replace=True)
    OmegaConf.register_new_resolver('get_default_trainer', get_default_trainer, replace=True)
    OmegaConf.register_new_resolver('get_default_transform', get_default_transform, replace=True)
    OmegaConf.register_new_resolver('get_flattened_channels', get_flattened_channels, replace=True)
    OmegaConf.register_new_resolver('get_required_lifting', get_required_lifting, replace=True)
    OmegaConf.register_new_resolver('get_monitor_metric', get_monitor_metric, replace=True)
    OmegaConf.register_new_resolver('get_monitor_mode', get_monitor_mode, replace=True)
    OmegaConf.register_new_resolver('get_gatv4_output_dim', get_gatv4_output_dim, replace=True)
    OmegaConf.register_new_resolver(
        'get_non_relational_out_channels', get_non_relational_out_channels, replace=True
    )
    OmegaConf.register_new_resolver('infer_in_channels', infer_in_channels, replace=True)
    OmegaConf.register_new_resolver(
        'infer_num_cell_dimensions', infer_num_cell_dimensions, replace=True
    )
    OmegaConf.register_new_resolver(
        'parameter_multiplication', lambda x, y: int(int(x) * int(y)), replace=True
    )
    OmegaConf.register_new_resolver(
        'get_target_normalizer_stats', get_target_normalizer_stats, replace=True
    )


def get_gatv4_output_dim(num_nodes, num_layers=3):
    r"""Get the output dimension for GATv4 based on the number of nodes and layers.

    Parameters
    ----------
    num_nodes : int
        Hidden dimension for the first layer.
    num_layers : int
        Number of layers in the GATv4 model.

    Returns
    -------
    list
        List of hidden dimensions for each layer.
    """
    return num_nodes * num_layers


def calculate_num_nodes(num_samples, train_val_test_split, node_sample_ratio, full_num_nodes):
    r"""Calculate the number of nodes for a given dataset.

    Parameters
    ----------
    num_samples : int
        Total number of samples in the dataset.
    train_val_test_split : list[float]
        Train/validation/test split ratios.
    node_sample_ratio : float or int
        Ratio of nodes to sample.

    Returns
    -------
    int
        Number of nodes.
    """
    n_training_samples = int(num_samples * train_val_test_split[0])
    if node_sample_ratio == 'full':
        return full_num_nodes
    n_nodes = int(n_training_samples / node_sample_ratio)
    if n_nodes > full_num_nodes:
        return full_num_nodes
    return n_nodes


def get_flattened_channels(num_nodes, channels):
    r"""Get the output dimension of flattening a feature matrix.

    Parameters
    ----------
    num_nodes : int
        Hidden dimension for the first layer.
    channels : int
        Channel dimension.

    Returns
    -------
    int
        Flattened channels dimension.
    """
    return num_nodes * channels


def get_non_relational_out_channels(num_nodes, channels, task_level):
    r"""Get the output dimension for a non-relational model.

    Parameters
    ----------
    num_nodes : int
        Number of nodes in the input graph.
    channels : int
        Channel dimension.
    task_level : int
        Task level for the model.

    Returns
    -------
    int
        Output dimension.
    """
    if task_level == 'node':  # node-level task
        return num_nodes * channels
    elif task_level == 'graph':  # graph-level task
        return channels
    else:
        raise ValueError(f'Invalid task level {task_level}')


def get_default_trainer():
    r"""Get default trainer configuration.

    Returns
    -------
    str
        Default trainer configuration file name.
    """
    return 'gpu' if torch.cuda.is_available() else 'cpu'


def get_default_transform(dataset, model):
    r"""Get default transform for a given data domain and model.

    Parameters
    ----------
    dataset : str
        Dataset name. Should be in the format "data_domain/name".
    model : str
        Model name. Should be in the format "model_domain/name".

    Returns
    -------
    str
        Default transform.
    """
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    model_configs_dir = os.path.join(base_dir, 'configs', 'transforms', 'model_defaults')
    dataset_configs_dir = os.path.join(base_dir, 'configs', 'transforms', 'dataset_defaults')
    datasets_with_defaults = [f.split('.')[0] for f in os.listdir(dataset_configs_dir)]
    model_with_defaults = [f.split('.')[0] for f in os.listdir(model_configs_dir)]
    if dataset in datasets_with_defaults:
        return f'dataset_defaults/{dataset}'
    elif model in model_with_defaults:
        return f'model_defaults/{model}'
    else:
        return 'no_transform'


def get_required_lifting(data_domain, model):
    r"""Get required transform for a given data domain and model.

    Parameters
    ----------
    data_domain : str
        Dataset domain.
    model : str
        Model name. Should be in the format "model_domain/name".

    Returns
    -------
    str
        Required transform.
    """
    data_domain = data_domain
    model_domain = model.split('/')[0]
    if data_domain == model_domain:
        return 'no_lifting'
    else:
        return f'{data_domain}2{model_domain}_default'


def get_monitor_metric(task, metric):
    r"""Get monitor metric for a given task.

    Parameters
    ----------
    task : str
        Task, either "classification" or "regression".
    metric : str
        Name of the metric function.

    Returns
    -------
    str
        Monitor metric.

    Raises
    ------
    ValueError
        If the task is invalid.
    """
    if task == 'classification' or task == 'regression' or task == 'multilabel classification':
        return f'val/{metric}'
    else:
        raise ValueError(f'Invalid task {task}')


def get_monitor_mode(task):
    r"""Get monitor mode for a given task.

    Parameters
    ----------
    task : str
        Task, either "classification" or "regression".

    Returns
    -------
    str
        Monitor mode, either "max" or "min".

    Raises
    ------
    ValueError
        If the task is invalid.
    """
    if task == 'classification' or task == 'multilabel classification':
        return 'max'

    elif task == 'regression':
        return 'min'

    else:
        raise ValueError(f'Invalid task {task}')


def infer_in_channels(dataset, transforms):
    r"""Infer the number of input channels for a given dataset.

    Parameters
    ----------
    dataset : DictConfig
        Configuration parameters for the dataset.
    transforms : DictConfig
        Configuration parameters for the transforms.

    Returns
    -------
    list
        List with dimensions of the input channels.
    """

    # Make it possible to pass lifting configuration as file path
    if transforms is not None and transforms.keys() == {'liftings'}:
        transforms = transforms.liftings

    def find_complex_lifting(transforms):
        r"""Find if there is a complex lifting in the complex_transforms.

        Parameters
        ----------
        transforms : List[str]
            List of transforms.

        Returns
        -------
        bool
            True if there is a complex lifting, False otherwise.
        str
            Name of the complex lifting, if it exists.
        """

        if transforms is None:
            return False, None
        complex_transforms = [
            # Default liftig configurations
            'graph2cell_lifting',
            'graph2simplicial_lifting',
            'graph2combinatorial_lifting',
            'graph2hypergraph_lifting',
            'pointcloud2graph_lifting',
            'pointcloud2simplicial_lifting',
            'pointcloud2combinatorial_lifting',
            'pointcloud2hypergraph_lifting',
            'pointcloud2cell_lifting',
            'hypergraph2combinatorial_lifting',
            # Make it possible to run directly from the folder
            'graph2cell',
            'graph2simplicial',
            'graph2combinatorial',
            'graph2hypergraph',
            'pointcloud2graph',
            'pointcloud2simplicial',
            'pointcloud2combinatorial',
            'pointcloud2hypergraph',
            'pointcloud2cell',
            'hypergraph2combinatorial',
        ]
        for t in complex_transforms:
            if t in transforms:
                return True, t
        return False, None

    def check_for_type_feature_lifting(transforms, lifting):
        r"""Check the type of feature lifting in the dataset.

        Parameters
        ----------
        transforms : DictConfig
            Configuration parameters for the transforms.
        lifting : str
            Name of the complex lifting.

        Returns
        -------
        str
            Type of feature lifting.
        """
        lifting_params_keys = transforms[lifting].keys()
        if 'feature_lifting' in lifting_params_keys:
            feature_lifting = transforms[lifting]['feature_lifting']
        else:
            feature_lifting = 'ProjectionSum'

        return feature_lifting

    there_is_complex_lifting, lifting = find_complex_lifting(transforms)
    if there_is_complex_lifting:
        # Get type of feature lifting
        feature_lifting = check_for_type_feature_lifting(transforms, lifting)

        # Check if the dataset.parameters.num_features defines a single value or a list
        if isinstance(dataset.parameters.num_features, int):
            # Case when the dataset has no edge attributes
            if feature_lifting == 'Concatenation':
                return_value = [dataset.parameters.num_features]
                for i in range(2, transforms[lifting].complex_dim + 1):
                    return_value += [int(return_value[-1]) * i]

                return return_value

            else:
                # ProjectionSum feature lifting by default
                return [dataset.parameters.num_features] * transforms[lifting].complex_dim
        # Case when the dataset has edge attributes (cells attributes)
        else:
            assert (
                type(dataset.parameters.num_features) is omegaconf.listconfig.ListConfig
            ), f'num_features should be a list of integers, not {type(dataset.parameters.num_features)}'
            # If preserve_edge_attr == False
            if not transforms[lifting].preserve_edge_attr:
                if feature_lifting == 'Concatenation':
                    return_value = [dataset.parameters.num_features[0]]
                    for i in range(2, transforms[lifting].complex_dim + 1):
                        return_value += [int(return_value[-1]) * i]

                    return return_value

                else:
                    # ProjectionSum feature lifting by default
                    return [dataset.parameters.num_features[0]] * transforms[lifting].complex_dim
            # If preserve_edge_attr == True
            else:
                return list(dataset.parameters.num_features) + [
                    dataset.parameters.num_features[1]
                ] * (transforms[lifting].complex_dim - len(dataset.parameters.num_features))

    # Case when there is no lifting
    else:
        if isinstance(dataset.parameters.num_features, int):
            return [dataset.parameters.num_features]
        else:
            return [dataset.parameters.num_features[0]]


def infer_num_cell_dimensions(selected_dimensions, in_channels):
    r"""Infer the length of a list.

    Parameters
    ----------
    selected_dimensions : list
        List of selected dimensions. If not None it will be used to infer the length.
    in_channels : list
        List of input channels. If selected_dimensions is None, this list will be used to infer the length.

    Returns
    -------
    int
        Length of the input list.
    """
    if selected_dimensions is not None:
        return len(selected_dimensions)
    else:
        return len(in_channels)


def get_default_metrics(task, metrics=None):
    r"""Get default metrics for a given task.

    Parameters
    ----------
    task : str
        Task, either "classification" or "regression".
    metrics : list, optional
        List of metrics to be used. If None, the default metrics will be used.

    Returns
    -------
    list
        List of default metrics.

    Raises
    ------
    ValueError
        If the task is invalid.
    """
    if metrics is not None:
        return metrics
    else:
        if 'classification' in task:
            return ['accuracy', 'precision', 'recall', 'auroc']
        elif 'regression' in task:
            return ['mse', 'mae']
        else:
            raise ValueError(f'Invalid task {task}')


def get_target_normalizer_stats(
    data_dir, data_name, adjacency_threshold, method, node_sample_ratio, train_val_test_split
):
    r"""Get target normalizer statistics from processing stats file.

    Parameters
    ----------
    data_dir : str
        Data directory path.
    data_name : str
        Name of the dataset.
    adjacency_threshold : float
        Adjacency threshold used.
    method : str
        Node selection method used.
    node_sample_ratio : float
        Node sample ratio used.
    train_val_test_split : list[float]
        Train/validation/test split ratios.

    Returns
    -------
    tuple[float, float]
        Target mean and standard deviation.
    """
    # Construct the path to the processing stats file
    stats_path = os.path.join(
        data_dir,
        data_name,
        f'adj_thresh_{adjacency_threshold}',
        method,
        f'p_{node_sample_ratio}',
        f'train_split_{train_val_test_split[0]}',
        'processed',
        'processing_stats.json',
    )

    try:
        with open(stats_path) as f:
            stats = json.load(f)

        target_stats = stats['target_normalizer']
        return target_stats['mean'], target_stats['std']
    except (FileNotFoundError, KeyError) as e:
        # Return default values if file not found or key missing
        print(f'Warning: Could not load target normalizer stats from {stats_path}: {e}')
        return 0.0, 1.0
