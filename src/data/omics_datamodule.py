import abc
from typing import Any, Dict, Final, Optional, Tuple
import os
import pandas as pd
import requests
import torch
from lightning import LightningDataModule
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import torch_geometric.data
import torch_geometric.transforms as T
import PyWGCNA
from sklearn.impute import SimpleImputer
import numpy as np
import gzip
from src.data import transforms
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def download_file(url: str, output_path: str) -> None:
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
    """`LightningDataModule` for MotrPac and PanCancer proteomics datasets."""

    def __init__(
        self,
        data_dir: str = "data/proteomics/",
        train_val_test_split: Tuple[float, float, float] = (0.7, 0.15, 0.15),
        batch_size: int = 64,
        num_workers: int = 0,
        pin_memory: bool = False,
        method: str = "variance",
        imputation_method: str = "mean",
        node_sample_ratio: float = 1.0,
    ) -> None:
        """Initialize a `ProteomicsDataModule`.

        Args:
            data_dir: The data directory
            dataset: Which dataset to use ("motrpac" or "pancancer")
            train_val_test_split: The train, validation and test split ratios
            batch_size: The batch size
            num_workers: The number of workers
            pin_memory: Whether to pin memory
            n_selected_nodes: Number of nodes to select for graph construction
            imputation_method: Method for handling missing values ("mean", "median", "most_frequent")
        """
        super().__init__()
        self.node_sample_ratio = node_sample_ratio
        self.method = method
        self.save_hyperparameters(logger=False)
        self.imputer = SimpleImputer(strategy=imputation_method)
        self.feature_normalizer = transforms.MeanStdNormalizer()
        self.target_normalizer = transforms.MeanStdNormalizer()

        self.selected_data_path: str = os.path.join(data_dir, f"{self.dataset_name}_sr{node_sample_ratio}_{method}_nodes_selected_data.parquet")
        self.targets_path: str = os.path.join(data_dir, f"{self.dataset_name}_sr{node_sample_ratio}_{method}_nodes_targets.npy")
        self.adj_matrix_path: str = os.path.join(data_dir, f"{self.dataset_name}_sr{node_sample_ratio}_{method}_adj_matrix.npy")

        self.data_train: Dataset | None = None
        self.data_val: Dataset | None = None
        self.data_test: Dataset | None = None

        self.batch_size_per_device: int = batch_size
        self.train_val_test_split: Tuple[float, float, float] = train_val_test_split

    def prepare_data(self) -> None:
        """Download and prepare data."""
        if not os.path.exists(self.hparams.data_dir):
            os.makedirs(self.hparams.data_dir)
        
        if os.path.exists(self.selected_data_path) and os.path.exists(self.targets_path) and os.path.exists(self.adj_matrix_path):
            return
        
        print("Preparing dataset...")
        raw_data, targets = self.prepare_dataset()
        np.save(self.targets_path, targets)
        
        n_training_samples = int(raw_data.shape[0] * self.train_val_test_split[0])
        n_nodes = int(n_training_samples / self.node_sample_ratio)
        print(f"Training sample: raw_data.shape: {n_training_samples}, node_sample_ratio: {self.node_sample_ratio}, n_nodes: {n_nodes}")
        print("Selecting nodes...")
        selected_nodes = self.select_nodes(
            raw_data.values,
            targets,
            n_selected=n_nodes,
            method=self.method
        )
        selected_data = raw_data.iloc[:, selected_nodes]
        selected_data.to_parquet(self.selected_data_path)

        print("Calculating adjacency matrix...")
        adj_matrix = self.calculate_adjacency_matrix(selected_data)
        np.save(self.adj_matrix_path, adj_matrix)

        # Calculate and log node degrees
        node_degrees = np.sum(adj_matrix, axis=1)
        print("Node degrees statistics:")
        print(f"Mean degree: {np.mean(node_degrees):.2f}")
        print(f"Median degree: {np.median(node_degrees):.2f}")
        print(f"Min degree: {np.min(node_degrees):.2f}")
        print(f"Max degree: {np.max(node_degrees):.2f}")
        print(f"Total edges: {np.sum(node_degrees)/2:.0f}")

    def select_nodes(self, data: np.ndarray, targets: np.ndarray, n_selected: int = 10, method: str = "variance") -> np.ndarray:
        """Select nodes based on feature importance.
        
        Args:
            data: Feature matrix
            targets: Target values
            n_selected: Number of features to select
            
        Returns:
            Indices of selected features
        """
        if method == "variance":
            # Variance-based filtering
            variances = np.std(data, axis=0)
            ranked_nodes = np.argsort(variances)[::-1]
        elif method == "correlation":
            # Correlation-based filtering
            correlations = np.abs(np.array([np.corrcoef(data[:, i], targets)[0, 1] for i in range(data.shape[1])]))
            ranked_nodes = np.argsort(correlations)[::-1]
        else:
            raise ValueError(f"Invalid method: {method}")
            
        return ranked_nodes[:n_selected]

    def calculate_adjacency_matrix(self, node_features: pd.DataFrame) -> np.ndarray:
        """Calculate adjacency matrix using WGCNA with soft-thresholding and binarization.
        
        Args:
            node_features: DataFrame containing selected node features
            
        Returns:
            Binary adjacency matrix
        """
        # Use WGCNA to find optimal power for scale-free topology
        soft_threshold = PyWGCNA.WGCNA.pickSoftThreshold(node_features)
        power = soft_threshold[0]
        
        # Apply soft-thresholding
        adjacency = PyWGCNA.WGCNA.adjacency(
            node_features,
            power=power,
            adjacencyType="signed hybrid",
        )
        print(adjacency)
        # Binarize adjacency matrix
        adj_matrix = np.where(adjacency > 0.01, 1, 0)
        print(adj_matrix)
        return adj_matrix

    def create_graph_data(self, subject_data: np.ndarray, subject_target: float, adj_matrix: np.ndarray) -> torch_geometric.data.Data:
        """Create graph data object."""
        node_features_raw = subject_data
        node_features_normalized = torch.from_numpy(self.feature_normalizer.transform(node_features_raw)).to(torch.float32)
        edge_index = torch.nonzero(torch.tensor(adj_matrix)).t().contiguous()
        
        y_tensor = np.array([subject_target])
        y_normalized = torch.from_numpy(self.target_normalizer.transform(y_tensor)).to(torch.float32)
        
        graph = torch_geometric.data.Data(
            x=node_features_normalized.unsqueeze(1),
            edge_index=edge_index,
            y=y_normalized
        )
        transform = T.ToSparseTensor()
        return transform(graph)

    def setup(self, stage: Optional[str] = None) -> None:
        """Load data and create train/val/test splits."""
        print("Loading data...")
        selected_data = pd.read_parquet(self.selected_data_path)
        targets = np.load(self.targets_path)
        adj_matrix = np.load(self.adj_matrix_path)

        # Fit normalizers on training data
        train_idx = int(len(selected_data) * self.train_val_test_split[0])
        train_data = selected_data.iloc[:train_idx]
        train_targets = targets[:train_idx]
        print('fitting normalizers')
        self.feature_normalizer.fit(train_data.values)
        self.target_normalizer.fit(train_targets)

        print('normalizers fitted')
        graph_data_list = []
        for (_, subject_data), subject_target in zip(selected_data.iterrows(), targets):
            graph_data_list.append(
                self.create_graph_data(subject_data.values, subject_target, adj_matrix)
            )
        print(f'graph data list length: {len(graph_data_list)}')
        print('graph data list created, shape: ', graph_data_list[0].x.shape)
        self.n_graphs = len(graph_data_list)
        i_train = int(self.n_graphs * self.train_val_test_split[0])
        i_val = int(self.n_graphs * (self.train_val_test_split[0] + self.train_val_test_split[1]))

        self.train_graph_data_list = graph_data_list[:i_train]
        self.val_graph_data_list = graph_data_list[i_train:i_val]
        self.test_graph_data_list = graph_data_list[i_val:]
        print("data loaded")

    def train_dataloader(self) -> DataLoader[Any]:
        """Create and return the train dataloader."""
        return torch_geometric.loader.DataLoader(
            dataset=self.train_graph_data_list,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True,
        )

    def val_dataloader(self) -> DataLoader[Any]:
        """Create and return the validation dataloader."""
        return torch_geometric.loader.DataLoader(
            dataset=self.val_graph_data_list,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def test_dataloader(self) -> DataLoader[Any]:
        """Create and return the test dataloader."""
        return torch_geometric.loader.DataLoader(
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


class MotrPacDataModule(OmicsDataModule):
    """`LightningDataModule` for MotrPac proteomics datasets."""
    dataset_name: Final[str] = "motrpac"

    def __init__(self, data_dir: str = "data/proteomics/", *args, **kwargs) -> None:
        super().__init__(data_dir=data_dir, *args, **kwargs)


    def prepare_dataset(self) -> Tuple[pd.DataFrame, np.ndarray]:
        """Prepare MotrPac dataset."""
        # Download files
        urls = {
            'proteomics': 'https://d1yw74buhe0ts0.cloudfront.net/static/motrpac-data-hub/publications/data/related-studies/heritage-proteomics/HERITAGE_proteomics_somalogic.xlsx',
            'analytes': 'https://d1yw74buhe0ts0.cloudfront.net/static/motrpac-data-hub/publications/data/related-studies/heritage-proteomics/HERITAGE_somalogic_analytes.xlsx'
        }

        for name, url in urls.items():
            file_path = os.path.join(self.hparams.data_dir, f"motrpac_{name}.xlsx")
            if not os.path.exists(file_path):
                print(f"Downloading {name}...")
                download_file(url, file_path)

        # Load data
        proteomics_df = pd.read_excel(os.path.join(self.hparams.data_dir, "motrpac_proteomics.xlsx"), header=3)
        _ = pd.read_excel(os.path.join(self.hparams.data_dir, "motrpac_analytes.xlsx"))

        # Extract features and target
        raw_data = proteomics_df.iloc[:, 9:]  # Protein expression values
        targets = proteomics_df['Delta VO2MX (ml/min)'].values

        # Remove rows with nan values
        mask = ~pd.isna(raw_data)
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
    dataset_name: Final[str] = "pancancer"

    def __init__(self, data_dir: str = "data/proteomics/", *args, **kwargs) -> None:
        super().__init__(data_dir=data_dir, *args, **kwargs)
  

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
                download_file(url, file_path)

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
                       ['Project_Identifier', 'model_id', 'cell_line_name', 'drug_id',
                        'drug_name', 'ln_IC50', 'COSMIC_ID', 'master_cell_id', 'model_name']]
        
        # Convert protein expression values to numeric, replacing non-numeric values with NaN
        print('Converting data to numeric...')
        numeric_data = merged_df[protein_cols].apply(pd.to_numeric, errors='coerce')
        
        # Remove columns that are all NaN after conversion
        valid_cols = numeric_data.columns[~numeric_data.isna().all()]
        raw_data = numeric_data[valid_cols]
        
        print(f'Number of valid protein columns: {len(valid_cols)}')

        targets = merged_df['ln_IC50'].values
         
        print(f'Number of valid protein columns: {len(valid_cols)}')
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


class AddNeuroMedOmicsDataModule(OmicsDataModule):
    """`LightningDataModule` for AddNeuroMed-style omics datasets."""
    dataset_name: Final[str] = "addneuromed_omics"

    def __init__(self, data_dir: str = "data/proteomics/", *args, **kwargs) -> None:
        super().__init__(data_dir=data_dir, *args, **kwargs)

    def prepare_dataset(self) -> Tuple[pd.DataFrame, np.ndarray]:
        """Prepare dataset in AddNeuroMed style.
        
        Returns:
            Tuple of (raw_data, targets) where targets are cognitive scores
        """
        # Create output directory if it doesn't exist
        if not os.path.exists(self.hparams.data_dir):
            os.makedirs(self.hparams.data_dir)

        # GEO dataset URLs
        datasets: Dict[str, str] = {
            'GPL10558': 'https://ftp.ncbi.nlm.nih.gov/geo/series/GSE63nnn/GSE63063/matrix/GSE63063-GPL10558_series_matrix.txt.gz',
            'GPL6947': 'https://ftp.ncbi.nlm.nih.gov/geo/series/GSE63nnn/GSE63063/matrix/GSE63063-GPL6947_series_matrix.txt.gz'
        }
        
        raw_data: pd.DataFrame = pd.DataFrame()
        frames: list[pd.DataFrame] = []
        ages: list[int] = []
        
        for dataset, url in datasets.items():
            gz_path: str = os.path.join(self.hparams.data_dir, f"{dataset}.txt.gz")
            if not os.path.exists(gz_path):
                print(f"Downloading {dataset}...")
                try:
                    download_file(url, gz_path)
                    print(f"Successfully downloaded {dataset}")
                except Exception as e:
                    print(f"Error downloading {dataset}: {str(e)}")
                    raise
                    
            # Read microarray data
            with gzip.open(gz_path, 'rt') as f:
                data = pd.read_csv(f, sep='\t', comment='!', index_col="ID_REF").transpose()
            
            # Extract ages from the data
            with gzip.open(gz_path, 'rt') as f:
                for line in f:
                    if line.startswith('!Sample_characteristics_ch1') and 'age:' in line:
                        ages.extend([int(x.split(': ')[1].strip().strip('"')) for x in line.split('\t')[1:]])
                        break
            
            frames.append(data)

        # Verify no common patients between datasets
        common_patients = set(frames[0].index).intersection(set(frames[1].index))
        assert len(common_patients) == 0, "Common patients found between the two datasets"

        # Find and use common genes
        common_genes = list(set(frames[0].columns).intersection(set(frames[1].columns)))
        frames[0] = frames[0][common_genes]
        frames[1] = frames[1][common_genes]

        # Combine datasets
        raw_data = pd.concat(frames, axis=0)
        targets = np.array(ages)

        # Impute missing values
        raw_data = pd.DataFrame(
            self.imputer.fit_transform(raw_data),
            columns=raw_data.columns,
            index=raw_data.index
        )

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
    
    motrpac_datamodule = MotrPacDataModule()
    motrpac_datamodule.prepare_data()
    motrpac_datamodule.setup()
    
    train_loader = motrpac_datamodule.train_dataloader()
    val_loader = motrpac_datamodule.val_dataloader()
    test_loader = motrpac_datamodule.test_dataloader()

    print(f"Number of training samples: {len(train_loader.dataset)}")
    print(f"Number of validation samples: {len(val_loader.dataset)}")
    print(f"Number of test samples: {len(test_loader.dataset)}")

    print(f"\nNumber of training batches: {len(train_loader)}")
    print(f"Number of validation batches: {len(val_loader)}")
    print(f"Number of test batches: {len(test_loader)}")

    addneuromed_omics_datamodule = AddNeuroMedOmicsDataModule()
    addneuromed_omics_datamodule.prepare_data()
    addneuromed_omics_datamodule.setup()

    train_loader = addneuromed_omics_datamodule.train_dataloader()
    val_loader = addneuromed_omics_datamodule.val_dataloader()
    test_loader = addneuromed_omics_datamodule.test_dataloader()

    print(f"Number of training samples: {len(train_loader.dataset)}")
    print(f"Number of validation samples: {len(val_loader.dataset)}")
    print(f"Number of test samples: {len(test_loader.dataset)}")

    print(f"\nNumber of training batches: {len(train_loader)}")
    print(f"Number of validation batches: {len(val_loader)}")
    print(f"Number of test batches: {len(test_loader)}")
    