import abc
from typing import Any, Dict, Optional, Tuple
import os
import pandas as pd
import requests
import torch
from lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import transforms
from tqdm import tqdm
import torch_geometric.data
import torch_geometric.transforms as T
import PyWGCNA
from sklearn.feature_selection import mutual_info_regression
from sklearn.impute import SimpleImputer
import numpy as np

def download_file(self, url: str, output_path: str) -> None:
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


class OmicsDataModule(LightningDataModule, abc.ABC):
    """`LightningDataModule` for MortrPac and PanCancer proteomics datasets."""

    def __init__(
        self,
        data_dir: str = "data/proteomics/",
        train_val_test_split: Tuple[float, float, float] = (0.7, 0.15, 0.15),
        batch_size: int = 64,
        num_workers: int = 0,
        pin_memory: bool = False,
        n_selected_nodes: int = 1000,
        imputation_method: str = "mean",
    ) -> None:
        """Initialize a `ProteomicsDataModule`.

        Args:
            data_dir: The data directory
            dataset: Which dataset to use ("mortrpac" or "pancancer")
            train_val_test_split: The train, validation and test split ratios
            batch_size: The batch size
            num_workers: The number of workers
            pin_memory: Whether to pin memory
            n_selected_nodes: Number of nodes to select for graph construction
            imputation_method: Method for handling missing values ("mean", "median", "most_frequent")
        """
        super().__init__()

        self.save_hyperparameters(logger=False)
        self.imputer = SimpleImputer(strategy=imputation_method)

        self.raw_data_path: str = os.path.join(data_dir, f"{self.dataset}_raw_data.parquet")
        self.targets_path: str = os.path.join(data_dir, f"{self.dataset}_targets.npy")

        self.data_train: Dataset | None = None
        self.data_val: Dataset | None = None
        self.data_test: Dataset | None = None

        self.batch_size_per_device: int = batch_size
        self.train_val_test_split: Tuple[float, float, float] = train_val_test_split

    def prepare_data(self) -> None:
        """Download and prepare data."""
        if not os.path.exists(self.hparams.data_dir):
            os.makedirs(self.hparams.data_dir)
        
        if os.path.exists(self.raw_data_path) and os.path.exists(self.targets_path):
            return
        
        print("Preparing dataset...")
        raw_data, targets = self.prepare_dataset()
        raw_data.to_parquet(self.raw_data_path)
        np.save(self.targets_path, targets)

    def _prepare_pancancer(self) -> None:
        """Prepare PanCancer dataset."""
        # Download files
        urls = {
            'proteomics': 'https://ars.els-cdn.com/content/image/1-s2.0-S1535610822002744-mmc3.xlsx',
            'drug_response': 'https://figshare.com/ndownloader/files/34355645'
        }

        for name, url in urls.items():
            file_path = os.path.join(self.hparams.data_dir, f"pancancer_{name}.{'xlsx' if name == 'proteomics' else 'csv.gz'}")
            if not os.path.exists(file_path):
                print(f"Downloading {name}...")
                self._download_file(url, file_path)

        # Load data
        print('Loading data...')
        proteomics_df = pd.read_excel(os.path.join(self.hparams.data_dir, "pancancer_proteomics.xlsx"), sheet_name="Full protein matrix", header=1)
        drug_df = pd.read_csv(os.path.join(self.hparams.data_dir, "pancancer_drug_response.csv.gz"), compression='gzip')

        # Filter for Avagacestat
        drug_df = drug_df[drug_df['drug_name'] == 'Avagacestat']
        drug_df = drug_df.dropna(subset=['ln_IC50'])

        # Split Project_Identifier
        proteomics_df[['model_id', 'cell_line_name']] = proteomics_df['Project_Identifier'].str.split(';', expand=True)

        # Merge datasets
        merged_df = pd.merge(
            proteomics_df,
            drug_df,
            on=['model_id', 'cell_line_name'],
            how='inner'
        )

        # Extract features and target
        protein_cols = [col for col in merged_df.columns if col not in 
                       ['Project_Identifier', 'model_id', 'cell_line_name', 
                        'drug_name', 'ln_IC50']]
        
        # Convert protein expression values to numeric, replacing non-numeric values with NaN
        print('Converting data to numeric...')
        numeric_data = merged_df[protein_cols].apply(pd.to_numeric, errors='coerce')
        
        # Remove columns that are all NaN after conversion
        valid_cols = numeric_data.columns[~numeric_data.isna().all()]
        numeric_data = numeric_data[valid_cols]
        
        print(f'Number of valid protein columns: {len(valid_cols)}')
        
        # Store original column names and index
        original_columns = numeric_data.columns
        original_index = numeric_data.index
        
        # Impute missing values in features
        print('Imputing missing values...')
        imputed_data = self.imputer.fit_transform(numeric_data)
        
        # Create DataFrame with imputed data, preserving original structure
        self.raw_data = pd.DataFrame(
            imputed_data,
            columns=original_columns,
            index=original_index
        )
        
        self.targets = merged_df['ln_IC50'].values

        # Select nodes and create adjacency matrix
        self._prepare_graph_data()

    def _prepare_graph_data(self) -> None:
        """Prepare graph data by selecting nodes and creating adjacency matrix."""
        # Select nodes based on feature importance
        selected_nodes_path = os.path.join(self.hparams.data_dir, f'{self.dataset}_selected_nodes.npy')
        adj_matrix_path = os.path.join(self.hparams.data_dir, f'{self.dataset}_adj_matrix.npy')

        if os.path.exists(selected_nodes_path):
            print("Loading cached selected nodes")
            self._selected_nodes = np.load(selected_nodes_path)
        else:
            print("Selecting nodes...")
            self._selected_nodes = self.select_nodes(
                self.raw_data.values,
                self.targets,
                n_selected=self.hparams.n_selected_nodes
            )
            np.save(selected_nodes_path, self._selected_nodes)

        self._selected_data = self.raw_data.iloc[:, self._selected_nodes]

        if os.path.exists(adj_matrix_path):
            print("Loading cached adjacency matrix")
            self._adj_matrix = np.load(adj_matrix_path)
        else:
            print("Calculating adjacency matrix...")
            self._adj_matrix = self.calculate_adjacency_matrix(self._selected_data)
            np.save(adj_matrix_path, self._adj_matrix)

    def select_nodes(self, data: np.ndarray, targets: np.ndarray, n_selected: int = 1000) -> np.ndarray:
        """Select nodes based on feature importance using mutual information."""
        mi_scores = mutual_info_regression(data, targets)
        ranked_nodes = np.argsort(mi_scores)[::-1]
        return ranked_nodes[:n_selected]

    def calculate_adjacency_matrix(self, node_features: pd.DataFrame) -> np.ndarray:
        """Calculate adjacency matrix using WGCNA."""
        soft_threshold = PyWGCNA.WGCNA.pickSoftThreshold(node_features)
        adjacency = PyWGCNA.WGCNA.adjacency(
            node_features,
            power=soft_threshold[0],
            adjacencyType="signed hybrid",
        )
        return adjacency

    def create_graph_data(self, subject_data: pd.Series, subject_target: float, adj_matrix: np.ndarray) -> torch_geometric.data.Data:
        """Create graph data object."""
        x = torch.tensor(subject_data.values, dtype=torch.float)
        edge_index = torch.nonzero(torch.tensor(adj_matrix)).t()
        
        graph_data = torch_geometric.data.Data(
            x=x,
            edge_index=edge_index,
            y=torch.tensor([subject_target], dtype=torch.float)
        )
        
        transform = T.ToSparseTensor()
        return transform(graph_data)

    def setup(self, stage: Optional[str] = None) -> None:
        """Load data and create train/val/test splits."""
        graph_data_list = []
        for (_, subject_data), subject_target in zip(self._selected_data.iterrows(), self.targets):
            graph_data_list.append(
                self.create_graph_data(subject_data, subject_target, self._adj_matrix)
            )

        self.n_graphs = len(graph_data_list)
        i_train = int(self.n_graphs * self.train_val_test_split[0])
        i_val = int(self.n_graphs * (self.train_val_test_split[0] + self.train_val_test_split[1]))

        self.train_graph_data_list = graph_data_list[:i_train]
        self.val_graph_data_list = graph_data_list[i_train:i_val]
        self.test_graph_data_list = graph_data_list[i_val:]

    def train_dataloader(self) -> DataLoader[Any]:
        """Create and return the train dataloader."""
        return torch_geometric.data.DataLoader(
            dataset=self.train_graph_data_list,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True,
        )

    def val_dataloader(self) -> DataLoader[Any]:
        """Create and return the validation dataloader."""
        return torch_geometric.data.DataLoader(
            dataset=self.val_graph_data_list,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def test_dataloader(self) -> DataLoader[Any]:
        """Create and return the test dataloader."""
        return torch_geometric.data.DataLoader(
            dataset=self.test_graph_data_list,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def teardown(self, stage: Optional[str] = None) -> None:
        """Clean up after training."""
        pass

    def state_dict(self) -> Dict[Any, Any]:
        """Return datamodule state."""
        return {}

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Load datamodule state."""
        pass


class MortrPacDataModule(OmicsDataModule):
    """`LightningDataModule` for MortrPac proteomics datasets."""

    def __init__(self, data_dir: str = "data/proteomics/", *args, **kwargs) -> None:
        super().__init__(data_dir=data_dir, *args, **kwargs)
        self.dataset = "mortrpac"

    def prepare_dataset(self) -> Tuple[pd.DataFrame, np.ndarray]:
        """Prepare MortrPac dataset."""
        # Download files
        urls = {
            'proteomics': 'https://d1yw74buhe0ts0.cloudfront.net/static/motrpac-data-hub/publications/data/related-studies/heritage-proteomics/HERITAGE_proteomics_somalogic.xlsx',
            'analytes': 'https://d1yw74buhe0ts0.cloudfront.net/static/motrpac-data-hub/publications/data/related-studies/heritage-proteomics/HERITAGE_somalogic_analytes.xlsx'
        }

        for name, url in urls.items():
            file_path = os.path.join(self.hparams.data_dir, f"mortrpac_{name}.xlsx")
            if not os.path.exists(file_path):
                print(f"Downloading {name}...")
                download_file(url, file_path)

        # Load data
        proteomics_df = pd.read_excel(os.path.join(self.hparams.data_dir, "mortrpac_proteomics.xlsx"), header=3)
        _ = pd.read_excel(os.path.join(self.hparams.data_dir, "mortrpac_analytes.xlsx"))

        # Extract features and target
        raw_data = proteomics_df.iloc[:, 9:]  # Protein expression values
        targets = proteomics_df['Delta VO2MX (ml/min)'].values

        # Remove rows with missing targets
        mask = ~pd.isna(targets)
        raw_data = raw_data[mask]
        targets = targets[mask]

        # Impute missing values in features
        raw_data = pd.DataFrame(
            self.imputer.fit_transform(raw_data),
            columns=raw_data.columns,
            index=raw_data.index
        )

        return raw_data, targets


class PanCancerDataModule(OmicsDataModule):
    """`LightningDataModule` for PanCancer proteomics datasets."""

    def __init__(self, data_dir: str = "data/proteomics/", *args, **kwargs) -> None:
        super().__init__(data_dir=data_dir, *args, **kwargs)
        self.dataset = "pancancer"   

    def prepare_dataset(self) -> Tuple[pd.DataFrame, np.ndarray]:
        """Prepare PanCancer dataset."""
        # Download files
        urls = {
            'proteomics': 'https://ars.els-cdn.com/content/image/1-s2.0-S1535610822002744-mmc3.xlsx',
            'drug_response': 'https://figshare.com/ndownloader/files/34355645'
        }

        for name, url in urls.items():
            file_path = os.path.join(self.hparams.data_dir, f"pancancer_{name}.{'xlsx' if name == 'proteomics' else 'csv.gz'}")
            if not os.path.exists(file_path):
                print(f"Downloading {name}...")
                self._download_file(url, file_path)

        # Load data
        print('Loading data...')
        proteomics_df = pd.read_excel(os.path.join(self.hparams.data_dir, "pancancer_proteomics.xlsx"), sheet_name="Full protein matrix", header=1)
        drug_df = pd.read_csv(os.path.join(self.hparams.data_dir, "pancancer_drug_response.csv.gz"), compression='gzip')

        # Filter for Avagacestat
        drug_df = drug_df[drug_df['drug_name'] == 'Avagacestat']
        drug_df = drug_df.dropna(subset=['ln_IC50'])

        # Split Project_Identifier
        proteomics_df[['model_id', 'cell_line_name']] = proteomics_df['Project_Identifier'].str.split(';', expand=True)

        # Merge datasets
        merged_df = pd.merge(
            proteomics_df,
            drug_df,
            on=['model_id', 'cell_line_name'],
            how='inner'
        )

        # Extract features and target
        protein_cols = [col for col in merged_df.columns if col not in 
                       ['Project_Identifier', 'model_id', 'cell_line_name', 
                        'drug_name', 'ln_IC50']]
        
        # Convert protein expression values to numeric, replacing non-numeric values with NaN
        print('Converting data to numeric...')
        numeric_data = merged_df[protein_cols].apply(pd.to_numeric, errors='coerce')
        
        # Remove columns that are all NaN after conversion
        valid_cols = numeric_data.columns[~numeric_data.isna().all()]
        numeric_data = numeric_data[valid_cols]
        
        print(f'Number of valid protein columns: {len(valid_cols)}')
        
        # Store original column names and index
        original_columns = numeric_data.columns
        original_index = numeric_data.index
        
        # Impute missing values in features
        print('Imputing missing values...')
        imputed_data = self.imputer.fit_transform(numeric_data)
        
        # Create DataFrame with imputed data, preserving original structure
        raw_data = pd.DataFrame(
            imputed_data,
            columns=original_columns,
            index=original_index
        )
        
        targets = merged_df['ln_IC50'].values

        return raw_data, targets


if __name__ == "__main__":
    # Example usage
    pancancer_datamodule = PanCancerDataModule()
    pancancer_datamodule.prepare_data()
    pancancer_datamodule.setup()
    
    train_loader = pancancer_datamodule.train_dataloader()
    val_loader = pancancer_datamodule.val_dataloader()
    test_loader = pancancer_datamodule.test_dataloader()

    print(f"Number of training samples: {len(train_loader.dataset)}")
    print(f"Number of validation samples: {len(val_loader.dataset)}")
    print(f"Number of test samples: {len(test_loader.dataset)}")
    
    print(f"\nNumber of training batches: {len(train_loader)}")
    print(f"Number of validation batches: {len(val_loader)}")
    print(f"Number of test batches: {len(test_loader)}")

    mortrpac_datamodule = MortrPacDataModule()
    mortrpac_datamodule.prepare_data()
    mortrpac_datamodule.setup()
    
    train_loader = mortrpac_datamodule.train_dataloader()
    val_loader = mortrpac_datamodule.val_dataloader()
    test_loader = mortrpac_datamodule.test_dataloader()

    print(f"Number of training samples: {len(train_loader.dataset)}")
    print(f"Number of validation samples: {len(val_loader.dataset)}")
    print(f"Number of test samples: {len(test_loader.dataset)}")

    print(f"\nNumber of training batches: {len(train_loader)}")
    print(f"Number of validation batches: {len(val_loader)}")
    print(f"Number of test batches: {len(test_loader)}")