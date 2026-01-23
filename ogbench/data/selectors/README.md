# Node Selectors Module

This module provides an interchangeable node selection system with a standardized interface. Each selector takes training features and labels as input and returns selected node indices as output.

## Architecture

- **AbstractNodeSelector**: Abstract base class defining the interface
- **Concrete Selectors**: Individual implementations (VarianceSelector, CorrelationSelector, etc.)
- **Registry Pattern**: `SELECTOR_REGISTRY` maps string names to selector classes
- **Factory Function**: `get_selector()` instantiates selectors by name

## Adding a New Selector

To add a new node selection strategy:

1. Create a new file in `ogbench/data/selectors/` (e.g., `mutual_information.py`)
2. Inherit from `AbstractNodeSelector` and implement the `select()` method:

```python
import numpy as np
from sklearn.feature_selection import mutual_info_regression
from ogbench.data.selectors.base import AbstractNodeSelector

class MutualInformationSelector(AbstractNodeSelector):
    def select(self, data: np.ndarray, targets: np.ndarray, n_selected: int) -> np.ndarray:
        mi_scores = mutual_info_regression(data, targets, random_state=42)
        ranked_nodes = np.argsort(mi_scores)[::-1]
        return ranked_nodes[:n_selected]
```

3. Register it in `ogbench/data/selectors/__init__.py`:

```python
from ogbench.data.selectors.mutual_information import MutualInformationSelector

SELECTOR_REGISTRY['mutual_information'] = MutualInformationSelector
```

4. Use it by name: `method='mutual_information'`
