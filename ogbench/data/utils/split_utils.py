"""Split utilities."""

import os

import numpy as np
import torch
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from sklearn.utils import shuffle as sklearn_shuffle

from ogbench.dataloader import DataloadDataset


def compute_omics_split_indices(
    labels: np.ndarray,
    *,
    split_type: str = 'fixed',
    k: int = 5,
    fold: int = 0,
    train_val_test_split: list[float] | tuple[float, ...] | None = None,
    random_state: int = 42,
    groups: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """Compute train/valid/test indices for omics graphs (sample-level).

    Indices refer to the **original sample order** (before any reordering).

    Parameters
    ----------
    labels:
        1-D label array of length n_samples.
    split_type:
        ``fixed`` — shuffle with ``random_state``, then cut by
        ``train_val_test_split`` proportions (matches historical HFOmics).
        ``k-fold`` — stratified CV with a 3/1/1 rotation: test is fold
        ``fold``, validation is fold ``(fold + 1) % k``, train is the rest.
        For ``k=5`` this is approximately 60/20/20. Each sample is in the
        test set of exactly one fold and the validation set of exactly one fold.
    k:
        Number of CV folds (k-fold only). Must be ``>= 3``.
    fold:
        Fold index in ``[0, k)`` used as the test fold (k-fold only).
        Validation is the next fold. Usually ``dataset.split_params.data_seed``.
    train_val_test_split:
        Proportions for the fixed split. Defaults to ``[0.7, 0.15, 0.15]``.
    random_state:
        RNG seed for shuffle / StratifiedKFold.
    groups:
        Optional group id per sample (k-fold only). Whole groups are assigned
        to a single fold so train/val/test do not mix groups.

    Returns
    -------
    dict[str, np.ndarray]
        Keys ``train``, ``valid``, ``test`` with disjoint index arrays
        covering ``0 .. n_samples-1``.
    """
    labels = np.asarray(labels)
    if labels.ndim != 1:
        labels = labels.reshape(-1)
    n_samples = len(labels)
    if n_samples == 0:
        raise ValueError('labels must be non-empty')

    ratios = list(train_val_test_split or [0.7, 0.15, 0.15])
    if len(ratios) != 3:
        raise ValueError(f'train_val_test_split must have length 3, got {ratios}')

    if split_type == 'fixed':
        if groups is not None:
            raise ValueError('group-aware splits are only supported for split_type=k-fold')
        order = sklearn_shuffle(np.arange(n_samples), random_state=random_state)
        train_end = int(n_samples * ratios[0])
        val_end = int(n_samples * (ratios[0] + ratios[1]))
        split_idx = {
            'train': np.asarray(order[:train_end], dtype=np.int64),
            'valid': np.asarray(order[train_end:val_end], dtype=np.int64),
            'test': np.asarray(order[val_end:], dtype=np.int64),
        }
    elif split_type == 'k-fold':
        if k < 3:
            raise ValueError(f'k must be >= 3 for k-fold, got {k}')
        if not (0 <= fold < k):
            raise ValueError(f'fold must satisfy 0 <= fold < k, got fold={fold}, k={k}')
        fold_id = _assign_omics_kfold_ids(labels, k=k, random_state=random_state, groups=groups)
        test_fold = int(fold)
        val_fold = (int(fold) + 1) % int(k)
        split_idx = {
            'train': np.flatnonzero((fold_id != test_fold) & (fold_id != val_fold)).astype(
                np.int64
            ),
            'valid': np.flatnonzero(fold_id == val_fold).astype(np.int64),
            'test': np.flatnonzero(fold_id == test_fold).astype(np.int64),
        }
        if any(len(split_idx[key]) == 0 for key in ('train', 'valid', 'test')):
            raise ValueError(f'k-fold rotation produced an empty split for fold={fold}, k={k}')
    else:
        raise ValueError(f"split_type must be 'fixed' or 'k-fold', got {split_type!r}")

    _assert_partition(split_idx, n_samples)
    return split_idx


def _assign_omics_kfold_ids(
    labels: np.ndarray,
    *,
    k: int,
    random_state: int,
    groups: np.ndarray | None,
) -> np.ndarray:
    """Assign each sample to a fold id in ``[0, k)``."""
    n_samples = len(labels)
    dummy = np.zeros((n_samples, 1))
    fold_id = np.empty(n_samples, dtype=np.int64)
    if groups is None:
        splitter = StratifiedKFold(n_splits=k, shuffle=True, random_state=random_state)
        splits = splitter.split(dummy, labels)
    else:
        groups = np.asarray(groups)
        if len(groups) != n_samples:
            raise ValueError(f'groups length {len(groups)} does not match n_samples={n_samples}')
        splitter = StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=random_state)
        splits = splitter.split(dummy, labels, groups)
    for fold_n, (_, test_idx) in enumerate(splits):
        fold_id[test_idx] = fold_n
    return fold_id


def group_kfold_is_feasible(
    groups: np.ndarray,
    labels: np.ndarray,
    k: int,
    *,
    random_state: int = 42,
) -> tuple[bool, str]:
    """Return whether group-aware k-fold 3/1/1 rotation can be built.

    Requires at least ``k`` groups and that every train/val/test split for
    every fold contains at least one sample of each globally present class.
    """
    groups = np.asarray(groups)
    labels = np.asarray(labels).reshape(-1)
    n_groups = len(np.unique(groups))
    if n_groups < k:
        return False, f'need at least k={k} groups, got {n_groups}'
    n_classes = len(np.unique(labels))
    try:
        for fold in range(k):
            split = compute_omics_split_indices(
                labels,
                split_type='k-fold',
                k=k,
                fold=fold,
                random_state=random_state,
                groups=groups,
            )
            for name, idx in split.items():
                part_classes = len(np.unique(labels[idx]))
                if n_classes >= 2 and part_classes < 2:
                    return (
                        False,
                        f'fold {fold} {name} has {part_classes} class(es); need >= 2',
                    )
    except ValueError as exc:
        return False, str(exc)
    return True, f'{n_groups} groups support k={k} group-aware rotation'


def print_group_split_inventory(
    groups: np.ndarray,
    labels: np.ndarray,
    *,
    k: int = 5,
    group_name: str = 'batch',
) -> tuple[bool, str]:
    """Print group sizes/class rates and whether k-fold 3/1/1 grouping is feasible."""
    groups = np.asarray(groups)
    labels = np.asarray(labels).reshape(-1)
    print(f'{group_name} inventory ({len(np.unique(groups))} groups, {len(labels)} samples):')
    for group in np.unique(groups):
        mask = groups == group
        n = int(mask.sum())
        counts = {str(cls): int((labels[mask] == cls).sum()) for cls in np.unique(labels)}
        print(f'  {group_name}={group!r}: n={n}, classes={counts}')
    ok, reason = group_kfold_is_feasible(groups, labels, k)
    if ok:
        print(f'Group-aware k={k} rotation: FEASIBLE ({reason})')
    else:
        print(
            f'Group-aware k={k} rotation: NOT FEASIBLE ({reason}). '
            'Keep sample-stratified splits; do not set grouping=batch.'
        )
    return ok, reason


def _assert_partition(split_idx: dict[str, np.ndarray], n_samples: int) -> None:
    """Ensure train/valid/test form a partition of 0..n_samples-1."""
    parts = [np.asarray(split_idx[k], dtype=np.int64) for k in ('train', 'valid', 'test')]
    concat = np.concatenate(parts)
    if len(concat) != n_samples:
        raise AssertionError(f'Split sizes sum to {len(concat)}, expected {n_samples}')
    if len(np.unique(concat)) != n_samples:
        raise AssertionError('Split indices are not a disjoint partition')


def omics_cache_split_suffix(
    split_type: str = 'fixed',
    k: int = 5,
    fold: int = 0,
    grouping: str | None = None,
) -> str | None:
    """Return an optional cache-path suffix for fold-aware omics artifacts.

    Fixed splits keep the historical cache path (no suffix) so existing
    artifacts remain valid. K-fold caches are isolated per fold, and per
    ``grouping`` because group-aware folds contain different samples.
    """
    if split_type == 'fixed':
        return None
    if split_type == 'k-fold':
        suffix = f'split_k-fold_k_{k}_fold_{fold}'
        if grouping is not None:
            suffix += f'_group_{grouping}'
        return suffix
    raise ValueError(f"split_type must be 'fixed' or 'k-fold', got {split_type!r}")


def build_omics_cache_relative_name(
    data_name: str,
    adjacency_threshold: float,
    adjacency_method: str,
    method: str,
    node_sample_ratio: float | str,
    train_split: float,
    split_type: str = 'fixed',
    k: int = 5,
    fold: int = 0,
    corrections: list[str] | None = None,
    grouping: str | None = None,
    adjacency_target_connectivity: float | None = None,
) -> str:
    """Build the relative HFOmics cache directory name (under data_dir)."""
    adjacency_setting = (
        f'target_connectivity_{adjacency_target_connectivity}'
        if adjacency_method == 'wgcna' and adjacency_target_connectivity is not None
        else f'adj_thresh_{adjacency_threshold}'
    )
    parts = [
        f'{data_name}',
        adjacency_setting,
        f'adj_method_{adjacency_method}',
        f'{method}',
        f'p_{node_sample_ratio}',
        f'train_split_{train_split}',
    ]
    suffix = omics_cache_split_suffix(split_type, k=k, fold=fold, grouping=grouping)
    if suffix is not None:
        parts.append(suffix)
    if corrections:
        parts.append('corr_' + '_'.join(str(name) for name in corrections))
    return os.path.join(*parts)


# Generate splits in different fasions
def k_fold_split(labels, parameters):
    """Return train and valid indices as in K-Fold Cross-Validation.

    If the split already exists it loads it automatically, otherwise it creates the
    split file for the subsequent runs.

    Parameters
    ----------
    labels : torch.Tensor
        Label tensor.
    parameters : DictConfig
        Configuration parameters.

    Returns
    -------
    dict
        Dictionary containing the train, validation and test indices, with keys "train", "valid", and "test".
    """

    data_dir = parameters.data_split_dir
    k = parameters.k
    fold = parameters.data_seed
    assert fold < k, 'data_seed needs to be less than k'

    torch.manual_seed(0)
    np.random.seed(0)

    split_dir = os.path.join(data_dir, f'{k}-fold')

    if not os.path.isdir(split_dir):
        os.makedirs(split_dir)

    split_path = os.path.join(split_dir, f'{fold}.npz')
    if not os.path.isfile(split_path):
        n = labels.shape[0]
        x_idx = np.arange(n)
        x_idx = np.random.permutation(x_idx)
        permuted_labels = labels[x_idx]

        skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)

        # Collect all fold indices first so we can assign test = fold_n, valid = next fold
        # skf.split returns positional indices into x_idx, so we must map them
        # back to original sample indices via x_idx[positional_idx].
        all_folds = list(skf.split(x_idx, permuted_labels))

        for fold_n in range(k):
            # Map positional indices back to original sample indices
            test_idx = x_idx[all_folds[fold_n][1]]
            valid_idx = x_idx[all_folds[(fold_n + 1) % k][1]]
            # Train = everything not in test or valid
            held_out = np.union1d(test_idx, valid_idx)
            train_idx = np.setdiff1d(np.arange(n), held_out)

            split_idx = {
                'train': train_idx,
                'valid': valid_idx,
                'test': test_idx,
            }

            # Check that all nodes/graph have been assigned to some split
            assert np.unique(np.concatenate([train_idx, valid_idx, test_idx])).shape[0] == len(
                labels
            ), 'Not every sample has been loaded.'
            split_path = os.path.join(split_dir, f'{fold_n}.npz')

            np.savez(split_path, **split_idx)

    split_path = os.path.join(split_dir, f'{fold}.npz')
    split_idx = np.load(split_path)

    # Check that all nodes/graph have been assigned to some split
    assert (
        np.unique(
            np.array(
                split_idx['train'].tolist()
                + split_idx['valid'].tolist()
                + split_idx['test'].tolist()
            )
        ).shape[0]
        == labels.shape[0]
    ), 'Not all nodes within splits'

    return split_idx


def random_splitting(labels, parameters, global_data_seed=42):
    r"""Randomly splits label into train/valid/test splits.

    Adapted from https://github.com/CUAI/Non-Homophily-Benchmarks.

    Parameters
    ----------
    labels : torch.Tensor
        Label tensor.
    parameters : DictConfig
        Configuration parameter.
    global_data_seed : int
        Seed for the random number generator.

    Returns
    -------
    dict:
        Dictionary containing the train, validation and test indices with keys "train", "valid", and "test".
    """
    fold = parameters['data_seed']
    data_dir = parameters['data_split_dir']
    train_prop = parameters['train_prop']
    valid_prop = (1 - train_prop) / 2

    # Create split directory if it does not exist
    split_dir = os.path.join(data_dir, f'train_prop={train_prop}_global_seed={global_data_seed}')
    os.makedirs(split_dir, exist_ok=True)

    # Generate splits if the requested fold file does not exist
    split_path = os.path.join(split_dir, f'{fold}.npz')
    if not os.path.isfile(split_path):
        # Set initial seed
        torch.manual_seed(global_data_seed)
        np.random.seed(global_data_seed)
        # Generate a split
        n = labels.shape[0]
        train_num = int(n * train_prop)
        valid_num = int(n * valid_prop)

        # Generate 10 splits
        for fold_n in range(10):
            # Permute indices
            perm = torch.as_tensor(np.random.permutation(n))

            train_indices = perm[:train_num]
            val_indices = perm[train_num : train_num + valid_num]
            test_indices = perm[train_num + valid_num :]
            split_idx = {
                'train': train_indices,
                'valid': val_indices,
                'test': test_indices,
            }

            # Save generated split
            split_path = os.path.join(split_dir, f'{fold_n}.npz')
            np.savez(split_path, **split_idx)

    # Load the split
    split_path = os.path.join(split_dir, f'{fold}.npz')
    split_idx = np.load(split_path)

    # Check that all nodes/graph have been assigned to some split
    assert (
        np.unique(
            np.array(
                split_idx['train'].tolist()
                + split_idx['valid'].tolist()
                + split_idx['test'].tolist()
            )
        ).shape[0]
        == labels.shape[0]
    ), 'Not all nodes within splits'

    return split_idx


def assign_train_val_test_mask_to_graphs(dataset, split_idx):
    """Split the graph dataset into train, validation, and test datasets.

    Parameters
    ----------
    dataset : torch_geometric.data.Dataset
        Considered dataset.
    split_idx : dict
        Dictionary containing the train, validation, and test indices.

    Returns
    -------
    tuple:
        Tuple containing the train, validation, and test datasets.
    """

    data_train_lst, data_val_lst, data_test_lst = [], [], []

    # Assign masks directly by iterating over pre-split indices
    for i in split_idx['train']:
        graph = dataset[i]
        graph.train_mask = torch.tensor([1], dtype=torch.long)
        graph.val_mask = torch.tensor([0], dtype=torch.long)
        graph.test_mask = torch.tensor([0], dtype=torch.long)
        data_train_lst.append(graph)

    for i in split_idx['valid']:
        graph = dataset[i]
        graph.train_mask = torch.tensor([0], dtype=torch.long)
        graph.val_mask = torch.tensor([1], dtype=torch.long)
        graph.test_mask = torch.tensor([0], dtype=torch.long)
        data_val_lst.append(graph)

    for i in split_idx['test']:
        graph = dataset[i]
        graph.train_mask = torch.tensor([0], dtype=torch.long)
        graph.val_mask = torch.tensor([0], dtype=torch.long)
        graph.test_mask = torch.tensor([1], dtype=torch.long)
        data_test_lst.append(graph)

    return (
        DataloadDataset(data_train_lst),
        DataloadDataset(data_val_lst),
        DataloadDataset(data_test_lst),
    )


def load_transductive_splits(dataset, parameters):
    r"""Load the graph dataset with the specified split.

    Parameters
    ----------
    dataset : torch_geometric.data.Dataset
        Graph dataset.
    parameters : DictConfig
        Configuration parameters.

    Returns
    -------
    list:
        List containing the train, validation, and test splits.
    """
    # Extract labels from dataset object
    assert len(dataset) == 1, 'Dataset should have only one graph in a transductive setting.'

    data = dataset.data_list[0]
    labels = data.y.numpy()

    # Ensure labels are one dimensional array
    assert len(labels.shape) == 1, 'Labels should be one dimensional array'

    if parameters.split_type == 'random':
        splits = random_splitting(labels, parameters)

    elif parameters.split_type == 'k-fold':
        splits = k_fold_split(labels, parameters)

    else:
        raise NotImplementedError(
            f"split_type {parameters.split_type} not valid. Choose either 'random' or 'k-fold'"
        )

    # Assign train val test masks to the graph
    data.train_mask = torch.from_numpy(splits['train'])
    data.val_mask = torch.from_numpy(splits['valid'])
    data.test_mask = torch.from_numpy(splits['test'])

    if parameters.get('standardize', False):
        # Standardize the node features respecting train mask
        # Add epsilon to prevent division by zero for constant features
        data.x = (data.x - data.x[data.train_mask].mean(0)) / (
            data.x[data.train_mask].std(0) + 1e-8
        )
        # Only standardize train targets to avoid leaking test label information.
        # Val/test targets are standardized with the same train stats but only
        # when they are actually needed for evaluation.
        y_mean = data.y[data.train_mask].mean(0)
        y_std = data.y[data.train_mask].std(0) + 1e-8
        data.y_mean = y_mean
        data.y_std = y_std
        data.y = (data.y - y_mean) / y_std

    return DataloadDataset([data]), None, None


def load_inductive_splits(dataset, parameters):
    r"""Load multiple-graph datasets with the specified split.

    Parameters
    ----------
    dataset : torch_geometric.data.Dataset
        Graph dataset.
    parameters : DictConfig
        Configuration parameters.

    Returns
    -------
    list:
        List containing the train, validation, and test splits.
    """
    # Extract labels from dataset object
    assert len(dataset) > 1, 'Datasets should have more than one graph in an inductive setting.'
    labels = np.array([data.y.squeeze(0).numpy() for data in dataset.data_list])

    # Omics caches already reorder samples to train|val|test and attach
    # contiguous split_idx. Prefer that over re-splitting (avoids double CV).
    if getattr(dataset, 'uses_precomputed_split', False) and hasattr(dataset, 'split_idx'):
        split_idx = dataset.split_idx

    elif parameters.split_type == 'random':
        split_idx = random_splitting(labels, parameters)

    elif parameters.split_type == 'k-fold':
        split_idx = k_fold_split(labels, parameters)

    elif parameters.split_type == 'fixed' and hasattr(dataset, 'split_idx'):
        split_idx = dataset.split_idx

    else:
        raise NotImplementedError(
            f"split_type {parameters.split_type} not valid. Choose either 'random', 'k-fold' or 'fixed'.\
            If 'fixed' is chosen, the dataset should have the attribute split_idx"
        )

    train_dataset, val_dataset, test_dataset = assign_train_val_test_mask_to_graphs(
        dataset, split_idx
    )

    return train_dataset, val_dataset, test_dataset
