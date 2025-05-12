from typing import Optional
import torch
from torch import nn

class MeanStdNormalizer(nn.Module):
    """Normalize features using mean and standard deviation."""
    
    def __init__(self, mean: Optional[torch.Tensor] = None, std: Optional[torch.Tensor] = None) -> None:
        super().__init__()
        self.mean = mean
        self.std = std
        
    def fit(self, data: torch.Tensor) -> None:
        """Compute mean and standard deviation from data."""
        self.mean = data.mean(dim=0, keepdim=True).T  # shape [3000, 1]
        self.std = data.std(dim=0, keepdim=True).T    # shape [3000, 1]
        # Avoid division by zero
        self.std[self.std == 0] = 1.0
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize input tensor."""
        if self.mean is None or self.std is None:
            raise ValueError("Normalizer must be fitted before use")
        return (x - self.mean) / self.std
    
    def inverse_transform(self, x: torch.Tensor) -> torch.Tensor:
        """Convert normalized data back to original scale."""
        if self.mean is None or self.std is None:
            raise ValueError("Normalizer must be fitted before use")
        return x * self.std + self.mean


class MinMaxNormalizer(nn.Module):
    """Normalize target values using min-max scaling."""
    
    def __init__(self, min_val: Optional[float] = None, max_val: Optional[float] = None) -> None:
        super().__init__()
        self.min_val = min_val
        self.max_val = max_val
        
    def fit(self, data: torch.Tensor) -> None:
        """Compute min and max values from data."""
        self.min_val = data.min().item()
        self.max_val = data.max().item()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize input tensor to [0, 1] range."""
        if self.min_val is None or self.max_val is None:
            raise ValueError("Normalizer must be fitted before use")
        return (x - self.min_val) / (self.max_val - self.min_val)
    
    def inverse_transform(self, x: torch.Tensor) -> torch.Tensor:
        """Convert normalized data back to original scale."""
        if self.min_val is None or self.max_val is None:
            raise ValueError("Normalizer must be fitted before use")
        return x * (self.max_val - self.min_val) + self.min_val