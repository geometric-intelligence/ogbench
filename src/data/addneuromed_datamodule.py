from typing import Any, Dict, Optional, Tuple
import os
import pandas as pd
import requests
import tarfile
import torch
from lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision.transforms import transforms
from tqdm import tqdm
import gzip
import torch_geometric.data
import torch_geometric.transforms as T
import PyWGCNA
from sklearn.metrics import mutual_info_classif
import numpy as np


class AddNeuroMedDataModule(LightningDataModule):
    """`LightningDataModule` for the AddNeuroMed dataset."""

    def __init__(
        self,
        data_dir: str = "data/addneuromed/",
        train_val_test_split: Tuple[float, float, float] = (0.7, 0.15, 0.15),
        batch_size: int = 64,
        num_workers: int = 0,
        pin_memory: bool = False,
    ) -> None:
        """Initialize a `AddNeuroMedDataModule`.

        :param data_dir: The data directory. Defaults to `"data/addneuromed/"`.
        :param train_val_test_split: The train, validation and test split ratios. Defaults to `(0.7, 0.15, 0.15)`.
        :param batch_size: The batch size. Defaults to `64`.
        :param num_workers: The number of workers. Defaults to `0`.
        :param pin_memory: Whether to pin memory. Defaults to `False`.
        """
        super().__init__()

        self.save_hyperparameters(logger=False)

        # data transformations
        self.transforms: transforms.Compose = transforms.Compose([
            transforms.Normalize(mean=[0.0], std=[1.0])  # Standardize features
        ])

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None

        self.batch_size_per_device: int = batch_size

    @property
    def num_classes(self) -> int:
        """Get the number of classes.

        :return: The number of classes in the dataset.
        """
        return 3  # AD, MCI, Control

    def _download_file(self, url: str, output_path: str) -> None:
        """Download a file with progress bar."""
        response: requests.Response = requests.get(url, stream=True)
        total_size: int = int(response.headers.get('content-length', 0))
        
        with open(output_path, 'wb') as f, tqdm(
            desc=os.path.basename(output_path),
            total=total_size,
            unit='iB',
            unit_scale=True,
            unit_divisor=1024,
        ) as pbar:
            for data in response.iter_content(chunk_size=1024):
                size: int = f.write(data)
                pbar.update(size)


    def _read_microarray_data(self, gz_path: str) -> pd.DataFrame:
        """Read microarray data from gzipped file."""
        with gzip.open(gz_path, 'rt') as f:
            return pd.read_csv(f, sep='\t', comment='!')

    def prepare_data(self) -> None:
        """Download data if needed. Lightning ensures that `self.prepare_data()` is called only
        within a single process on CPU, so you can safely add your downloading logic within.
        """
        # Create output directory if it doesn't exist
        if not os.path.exists(self.hparams.data_dir):
            os.makedirs(self.hparams.data_dir)
        
        # GEO dataset URLs
        datasets: Dict[str, str] = {
            'GPL10558': 'https://ftp.ncbi.nlm.nih.gov/geo/series/GSE63nnn/GSE63063/matrix/GSE63063-GPL10558_series_matrix.txt.gz',
            'GPL6947': 'https://ftp.ncbi.nlm.nih.gov/geo/series/GSE63nnn/GSE63063/matrix/GSE63063-GPL6947_series_matrix.txt.gz'
        }
        
        self.raw_data: pd.DataFrame = pd.DataFrame()
        common_genes: set[str] | None = None
        frames: list[pd.DataFrame] = []
        for dataset, url in datasets.items():
            gz_path: str = os.path.join(self.hparams.data_dir, f"{dataset}.txt.gz")
            if not os.path.exists(gz_path):
                print(f"Downloading {dataset}...")
                try:
                    self._download_file(url, gz_path)
                    print(f"Successfully downloaded {dataset}")
                except Exception as e:
                    print(f"Error downloading {dataset}: {str(e)}")
                    raise
            data: pd.DataFrame = self._read_microarray_data(gz_path).transpose()
            frames.append(data)
        # Find common genes between the two datasets
        common_genes = list(set(frames[0].columns).intersection(set(frames[1].columns)))    
        # Filter the two datasets to only include common genes
        frames[0] = frames[0][common_genes]
        frames[1] = frames[1][common_genes]
        # Concatenate the two datasets
        self.raw_data = pd.concat(frames, axis=0)

    def select_nodes(self, node_features, graph_label, n_selected_nodes=100):
        """Select nodes based on graph label."""
        """
        Compute feature importance scores between node features and graph labels using mutual information.
        Since graph labels are discrete (0,1,2), mutual information is more appropriate than correlation.
        
        Args:
            node_features: numpy array of shape (n_samples, n_features)
            graph_label: numpy array of shape (n_samples,) containing discrete labels
            
        Returns:
            selected_features: indices of most informative features
        """
        
        # Compute mutual information between each feature and the graph label
        mi_scores = mutual_info_classif(node_features, graph_label)
        
        # Sort features by mutual information score
        ranked_features = np.argsort(mi_scores)[::-1]
        
        # Select top features (can adjust threshold as needed)
        n_select = min(n_selected_nodes, len(ranked_features))  # Select top 100 or all if less
        selected_features = ranked_features[:n_select]
        
        return selected_features

    def calculate_adjacency_matrix(self, node_features, save_to):
        """Calculate and save adjacency matrix."""
        node_features_df = pd.DataFrame(node_features)
        softThreshold = PyWGCNA.WGCNA.pickSoftThreshold(node_features_df)
        print("Soft threshold:", softThreshold[0])
        adjacency = PyWGCNA.WGCNA.adjacency(
            node_features, power=softThreshold[0], adjacencyType="signed hybrid"
        )

        adjacency_df = pd.DataFrame(adjacency)
        print(f"Saving adjacency matrix to: {save_to}...")
        adjacency_df.to_csv(save_to, header=None, index=False)


    def create_graph_data(self, node_features, graph_label, adj_matrix):
        """Create Data object for each graph.

        Compute attributes x, edge_index, and y for each graph.
        Uses sparse tensor representation for efficiency.
        """
        x = node_features  # what is on the nodes
        adj_tensor = torch.tensor(adj_matrix)  # Transpose the adjacency matrix
        data = torch_geometric.data.Data(x=x, edge_index=None, y=graph_label)
        transform = T.ToSparseTensor()
        data.adj_t = transform(adj_tensor)
        return data   

    def setup(self, stage: Optional[str] = None) -> None:
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.

        This method is called by Lightning before `trainer.fit()`, `trainer.validate()`, `trainer.test()`, and
        `trainer.predict()`, so be careful not to execute things like random split twice!
        """
        # Divide batch size by the number of devices
        print(self.raw_data)
        if self.trainer is not None:
            if self.hparams.batch_size % self.trainer.world_size != 0:
                raise RuntimeError(
                    f"Batch size ({self.hparams.batch_size}) is not divisible by the number of devices ({self.trainer.world_size})."
                )
            self.batch_size_per_device = self.hparams.batch_size // self.trainer.world_size

            print(self.raw_data)

            selected_nodes = self.select_nodes(self.raw_data, self.raw_data["label"]) 
            self.raw_data = self.raw_data[selected_nodes]

            adj_matrix = self.calculate_adjacency_matrix(self.raw_data, "adjacency_matrix.csv")

            graph_data_list = []
            for subject in self.raw_data:
                graph_data_list.append(
                    self.create_graph_data(subject, subject["label"], adj_matrix))

            # FTDDataset(root, "train", config)
            dataset: AddNeuroMedDataset = AddNeuroMedDataset(data, transform=self.transforms)
            
            # Calculate split sizes
            n: int = len(dataset)
            train_size: int = int(n * self.hparams.train_val_test_split[0])
            val_size: int = int(n * self.hparams.train_val_test_split[1])
            test_size: int = n - train_size - val_size
            
            # Split dataset
            self.data_train, self.data_val, self.data_test = random_split(
                dataset=dataset,
                lengths=[train_size, val_size, test_size],
                generator=torch.Generator().manual_seed(42),
            )

    def train_dataloader(self) -> DataLoader[Any]:
        """Create and return the train dataloader.

        :return: The train dataloader.
        """
        return torch_geometric.data.DataLoader(
            dataset=self.data_train,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True,
        )

    def val_dataloader(self) -> DataLoader[Any]:
        """Create and return the validation dataloader.

        :return: The validation dataloader.
        """
        return torch_geometric.data.DataLoader(
            dataset=self.data_val,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def test_dataloader(self) -> DataLoader[Any]:
        """Create and return the test dataloader.

        :return: The test dataloader.
        """
        return torch_geometric.data.DataLoader(
            dataset=self.data_test,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def teardown(self, stage: Optional[str] = None) -> None:
        """Lightning hook for cleaning up after `trainer.fit()`, `trainer.validate()`,
        `trainer.test()`, and `trainer.predict()`.

        :param stage: The stage being torn down. Either `"fit"`, `"validate"`, `"test"`, or `"predict"`.
            Defaults to ``None``.
        """
        pass

    def state_dict(self) -> Dict[Any, Any]:
        """Called when saving a checkpoint. Implement to generate and save the datamodule state.

        :return: A dictionary containing the datamodule state that you want to save.
        """
        return {}

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Called when loading a checkpoint. Implement to reload datamodule state given datamodule
        `state_dict()`.

        :param state_dict: The datamodule state returned by `self.state_dict()`.
        """
        pass

if __name__ == "__main__":
    datamodule = AddNeuroMedDataModule()
    datamodule.prepare_data()
    datamodule.setup()
    train_loader = datamodule.train_dataloader()
    val_loader = datamodule.val_dataloader()
    test_loader = datamodule.test_dataloader()
    
