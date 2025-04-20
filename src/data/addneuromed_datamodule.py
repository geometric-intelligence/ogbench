from typing import Any, Dict, Optional, Tuple
import os
import pandas as pd
import requests
import torch
from lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision.transforms import transforms
from tqdm import tqdm
import gzip
import torch_geometric.data
import torch_geometric.transforms as T
import PyWGCNA
from sklearn.feature_selection import mutual_info_classif
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
        self.train_val_test_split: Tuple[float, float, float] = train_val_test_split

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
            return pd.read_csv(f, sep='\t', comment='!', index_col="ID_REF")
        

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
            # Extract diagnosis labels from the data
            # The labels are in the first row which contains "!Sample_characteristics_ch1"
            labels = None
            with gzip.open(gz_path, 'rt') as f:
                for line in f:
                    if line.startswith('!Sample_characteristics_ch1'):
                        # Extract diagnosis from the line, typically in format "diagnosis: Control/AD/MCI"
                        diagnoses = [x.split(': ')[1].strip().strip('"') for x in line.split('\t')[1:]]
                        # Convert text labels to numeric
                        label_map = {'CTL': 0, 'MCI': 1, 'borderline MCI': 1, 'AD': 2, "OTHER": 3, "CTL to AD": 3, "MCI to CTL": 3}
                        labels = [label_map[d] for d in diagnoses]
                        break
            
            # Add labels to the data frame
            data['label'] = labels
            frames.append(data)

        # Check that labels match for same patients across datasets
        common_patients = set(frames[0].index).intersection(set(frames[1].index))
        assert len(common_patients) == 0, "Common patients found between the two datasets"

        # Find common genes between the two datasets (or common "label" column)
        common_genes = list(set(frames[0].columns).intersection(set(frames[1].columns)))    
        # Filter the two datasets to only include common genes
        frames[0] = frames[0][common_genes]
        frames[1] = frames[1][common_genes]
        # Concatenate the two datasets
        raw_data = pd.concat(frames, axis=0)
        self.labels = raw_data['label']
        self.raw_data = raw_data.drop(columns=['label']).iloc[:, :1000]
        # Select nodes based on graph label
        selected_nodes_path = os.path.join(self.hparams.data_dir, 'selected_nodes.npy')
        adj_matrix_path = os.path.join(self.hparams.data_dir, 'adj_matrix.npy')

        if os.path.exists(selected_nodes_path):
            print("Loading cached selected nodes")
            self._selected_nodes = np.load(selected_nodes_path)
        else:
            print("start selecting nodes")
            self._selected_nodes = self.select_nodes(self.raw_data, self.labels, n_selected=50)
            np.save(selected_nodes_path, self._selected_nodes)
            print("done selecting nodes")
        
        self._selected_data = self.raw_data.iloc[:, self._selected_nodes]
        
        if os.path.exists(adj_matrix_path):
            print("Loading cached adjacency matrix")
            self._adj_matrix = np.load(adj_matrix_path)
        else:
            print("start calculating adjacency matrix")
            self._adj_matrix = self.calculate_adjacency_matrix(self._selected_data)
            np.save(adj_matrix_path, self._adj_matrix)
            print("done calculating adjacency matrix")

    def select_nodes(self, data, labels, n_selected=1000):
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
        mi_scores = mutual_info_classif(data, labels)
        
        # Sort features by mutual information score
        ranked_nodes = np.argsort(mi_scores)[::-1]
        
        # Select top features (can adjust threshold as needed)
        n_select = min(n_selected, len(ranked_nodes))  # Select top 100 or all if less
        selected_nodes = ranked_nodes[:n_select]
        
        return selected_nodes

    def calculate_adjacency_matrix(self, node_features):
        """Calculate and save adjacency matrix."""
        node_features_df = pd.DataFrame(node_features)
        soft_threshold = PyWGCNA.WGCNA.pickSoftThreshold(node_features_df)
        adjacency = PyWGCNA.WGCNA.adjacency(
            node_features,
            power=soft_threshold[0],
            adjacencyType="signed hybrid",
        )

        return adjacency


    def create_graph_data(self, subject_data, subject_label, adj_matrix):
        """Create Data object for each graph.

        Compute attributes x, edge_index, and y for each graph.
        Uses sparse tensor representation for efficiency.
        """ 
        x = torch.tensor(subject_data, dtype=torch.float)
    
        # Convert adjacency matrix to edge index
        edge_index = torch.nonzero(torch.tensor(adj_matrix)).t()
        
        # Create data object
        graph_data = torch_geometric.data.Data(
            x=x,
            edge_index=edge_index,
            y=torch.tensor([subject_label], dtype=torch.long)
        )
        
        # Convert to sparse tensor format
        transform = T.ToSparseTensor()
        graph_data = transform(graph_data)

        
        return graph_data

    def setup(self, stage: Optional[str] = None) -> None:
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.

        This method is called by Lightning before `trainer.fit()`, `trainer.validate()`, `trainer.test()`, and
        `trainer.predict()`, so be careful not to execute things like random split twice!
        """
        graph_data_list = []
        for (_, subject_data), subject_label in zip(self._selected_data.iterrows(), self.labels):

            graph_data_list.append(
                self.create_graph_data(subject_data, subject_label, self._adj_matrix))
   
        self.n_graphs = len(graph_data_list)
        i_train = int(self.n_graphs * self.train_val_test_split[0]) 
        i_val = int(self.n_graphs * (self.train_val_test_split[0] + self.train_val_test_split[1]))

        self.train_graph_data_list = graph_data_list[:i_train]
        self.val_graph_data_list = graph_data_list[i_train:i_val]
        self.test_graph_data_list = graph_data_list[i_val:]

    def train_dataloader(self) -> DataLoader[Any]:
        """Create and return the train dataloader.

        :return: The train dataloader.
        """
        return torch_geometric.data.DataLoader(
            dataset= self.train_graph_data_list,
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
            dataset=self.val_graph_data_list,
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
            dataset=self.test_graph_data_list,
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

    # Print total number of samples in each dataset
    print(f"Number of training samples: {len(train_loader.dataset)}")
    print(f"Number of validation samples: {len(val_loader.dataset)}")
    print(f"Number of test samples: {len(test_loader.dataset)}")
    
    # Print number of batches
    print(f"\nNumber of training batches: {len(train_loader)}")
    print(f"Number of validation batches: {len(val_loader)}")
    print(f"Number of test batches: {len(test_loader)}")
    
