"""Data manipulations module."""

from ogbench.transforms.data_manipulations.calculate_simplicial_curvature import (
    CalculateSimplicialCurvature,
)
from ogbench.transforms.data_manipulations.combined_positional_and_structural_encodings import (
    CombinedPSEs,
)
from ogbench.transforms.data_manipulations.electrostatic_encodings import ElectrostaticPE
from ogbench.transforms.data_manipulations.equal_gaus_features import EqualGausFeatures
from ogbench.transforms.data_manipulations.group_homophily import (
    GroupCombinatorialHomophily,
)
from ogbench.transforms.data_manipulations.hk_feature_encodings import HKFE
from ogbench.transforms.data_manipulations.identity_transform import IdentityTransform
from ogbench.transforms.data_manipulations.infere_knn_connectivity import (
    InfereKNNConnectivity,
)
from ogbench.transforms.data_manipulations.infere_radius_connectivity import (
    InfereRadiusConnectivity,
)
from ogbench.transforms.data_manipulations.keep_only_connected_component import (
    KeepOnlyConnectedComponent,
)
from ogbench.transforms.data_manipulations.keep_selected_data_fields import (
    KeepSelectedDataFields,
)
from ogbench.transforms.data_manipulations.khop_precompute import KHopPrecompute
from ogbench.transforms.data_manipulations.laplacian_encodings import LapPE
from ogbench.transforms.data_manipulations.mp_homophily import MessagePassingHomophily
from ogbench.transforms.data_manipulations.node_degrees import NodeDegrees
from ogbench.transforms.data_manipulations.node_features_to_float import (
    NodeFeaturesToFloat,
)
from ogbench.transforms.data_manipulations.one_hot_degree_features import (
    OneHotDegreeFeatures,
)
from ogbench.transforms.data_manipulations.random_walk_encodings import RWSE
from ogbench.transforms.data_manipulations.redefine_simplicial_neighbourhoods import (
    RedefineSimplicialNeighbourhoods,
)

# Create dictionary of all data manipulations
DATA_MANIPULATIONS: dict[str, type] = {
    'CalculateSimplicialCurvature': CalculateSimplicialCurvature,
    'EqualGausFeatures': EqualGausFeatures,
    'GroupCombinatorialHomophily': GroupCombinatorialHomophily,
    'IdentityTransform': IdentityTransform,
    'InfereKNNConnectivity': InfereKNNConnectivity,
    'InfereRadiusConnectivity': InfereRadiusConnectivity,
    'KeepOnlyConnectedComponent': KeepOnlyConnectedComponent,
    'KeepSelectedDataFields': KeepSelectedDataFields,
    'KHopPrecompute': KHopPrecompute,
    'MessagePassingHomophily': MessagePassingHomophily,
    'NodeDegrees': NodeDegrees,
    'NodeFeaturesToFloat': NodeFeaturesToFloat,
    'OneHotDegreeFeatures': OneHotDegreeFeatures,
    'RedefineSimplicialNeighbourhoods': RedefineSimplicialNeighbourhoods,
    'HKFE': HKFE,
    'ElectrostaticPE': ElectrostaticPE,
    'LapPE': LapPE,
    'RWSE': RWSE,
    'CombinedPSEs': CombinedPSEs,
}

# Generate __all__
__all__ = [
    'DATA_MANIPULATIONS',
    'CalculateSimplicialCurvature',
    'EqualGausFeatures',
    'GroupCombinatorialHomophily',
    'IdentityTransform',
    'InfereKNNConnectivity',
    'InfereRadiusConnectivity',
    'KeepOnlyConnectedComponent',
    'KeepSelectedDataFields',
    'KHopPrecompute',
    'MessagePassingHomophily',
    'NodeDegrees',
    'NodeFeaturesToFloat',
    'OneHotDegreeFeatures',
    'RedefineSimplicialNeighbourhoods',
    'HKFE',
    'ElectrostaticPE',
    'LapPE',
    'RWSE',
    'CombinedPSEs',
]
