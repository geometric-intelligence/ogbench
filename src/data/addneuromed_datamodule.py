from typing import Any, Dict, Optional, Tuple

import os
import pandas as pd
import requests
import torch
from lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision.transforms import transforms
from tqdm import tqdm


class AddNeuroMedDataset(Dataset):
    """Dataset class for AddNeuroMed data."""
    
    def __init__(self, data: pd.DataFrame, transform=None):
        self.data = data
        self.transform = transform
        
    def __len__(self) -> int:
        return len(self.data)
        
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # Get features and target
        features = torch.tensor(self.data.iloc[idx, :-1].values, dtype=torch.float32)
        target = torch.tensor(self.data.iloc[idx, -1], dtype=torch.long)
        
        if self.transform:
            features = self.transform(features)
            
        return features, target


class AddNeuroMedDataModule(LightningDataModule):
    """`LightningDataModule` for the AddNeuroMed dataset."""

    def __init__(
        self,
        data_dir: str = "addneuromed_data/",
        train_val_test_split: Tuple[float, float, float] = (0.7, 0.15, 0.15),
        batch_size: int = 64,
        num_workers: int = 0,
        pin_memory: bool = False,
    ) -> None:
        """Initialize a `AddNeuroMedDataModule`.

        :param data_dir: The data directory. Defaults to `"addneuromed_data/"`.
        :param train_val_test_split: The train, validation and test split ratios. Defaults to `(0.7, 0.15, 0.15)`.
        :param batch_size: The batch size. Defaults to `64`.
        :param num_workers: The number of workers. Defaults to `0`.
        :param pin_memory: Whether to pin memory. Defaults to `False`.
        """
        super().__init__()

        self.save_hyperparameters(logger=False)

        # data transformations
        self.transforms = transforms.Compose([
            transforms.Normalize(mean=[0.0], std=[1.0])  # Standardize features
        ])

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None

        self.batch_size_per_device = batch_size

    @property
    def num_classes(self) -> int:
        """Get the number of classes.

        :return: The number of classes in the dataset.
        """
        return 3  # AD, MCI, Control

    def _download_file(self, url: str, output_path: str) -> None:
        """Download a file with progress bar."""
        response = requests.get(url, stream=True)
        total_size = int(response.headers.get('content-length', 0))
        
        with open(output_path, 'wb') as f, tqdm(
            desc=os.path.basename(output_path),
            total=total_size,
            unit='iB',
            unit_scale=True,
            unit_divisor=1024,
        ) as pbar:
            for data in response.iter_content(chunk_size=1024):
                size = f.write(data)
                pbar.update(size)

    def prepare_data(self) -> None:
        """Download data if needed. Lightning ensures that `self.prepare_data()` is called only
        within a single process on CPU, so you can safely add your downloading logic within.
        """
        # Create output directory if it doesn't exist
        if not os.path.exists(self.hparams.data_dir):
            os.makedirs(self.hparams.data_dir)
        
        # GEO dataset URLs
        datasets = {
            'GSE63060': 'https://www.ncbi.nlm.nih.gov/geo/download/?acc=GSE63060&format=file',
            'GSE63061': 'https://www.ncbi.nlm.nih.gov/geo/download/?acc=GSE63061&format=file'
        }
        
        for dataset, url in datasets.items():
            output_file = os.path.join(self.hparams.data_dir, f"{dataset}.txt.gz")
            if not os.path.exists(output_file):
                print(f"Downloading {dataset}...")
                try:
                    self._download_file(url, output_file)
                    print(f"Successfully downloaded {dataset}")
                except Exception as e:
                    print(f"Error downloading {dataset}: {str(e)}")
                    raise

    def setup(self, stage: Optional[str] = None) -> None:
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.

        This method is called by Lightning before `trainer.fit()`, `trainer.validate()`, `trainer.test()`, and
        `trainer.predict()`, so be careful not to execute things like random split twice!
        """
        # Divide batch size by the number of devices
        if self.trainer is not None:
            if self.hparams.batch_size % self.trainer.world_size != 0:
                raise RuntimeError(
                    f"Batch size ({self.hparams.batch_size}) is not divisible by the number of devices ({self.trainer.world_size})."
                )
            self.batch_size_per_device = self.hparams.batch_size // self.trainer.world_size

        # Load and split datasets only if not loaded already
        if not self.data_train and not self.data_val and not self.data_test:
            # Load the data from GSE63060 and GSE63061
            gse63060 = pd.read_csv(f"{self.hparams.data_dir}/GSE63060.txt.gz", sep='\t')
            gse63061 = pd.read_csv(f"{self.hparams.data_dir}/GSE63061.txt.gz", sep='\t')
            
            # Combine datasets and preprocess
            data = pd.concat([gse63060, gse63061], axis=0)
            
            # Create dataset
            dataset = AddNeuroMedDataset(data, transform=self.transforms)
            
            # Calculate split sizes
            n = len(dataset)
            train_size = int(n * self.hparams.train_val_test_split[0])
            val_size = int(n * self.hparams.train_val_test_split[1])
            test_size = n - train_size - val_size
            
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
        return DataLoader(
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
        return DataLoader(
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
        return DataLoader(
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
    _ = AddNeuroMedDataModule()
