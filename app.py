#!/usr/bin/env python3
"""Interactive Dash app for exploring BGBench omics datasets.

This app visualizes graph statistics for the three omics datasets
(MotrPac, AddNeuroMed, Parkinsons) with interactive controls.
"""

import json
import os
from pathlib import Path
from typing import Any

import dash
import networkx as nx
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import dcc, html
from dash.dependencies import Input, Output
from huggingface_hub import hf_hub_download
from plotly.subplots import make_subplots
from sklearn.impute import SimpleImputer
from sklearn.utils import shuffle

# Dataset configurations
DATASETS = {
    'motrpac': {
        'data_name': 'motrpac',
        'revision': '9f052d330ce130408a2c7c347b2ed197154da7c8',
        'train_val_test_split': [0.7, 0.2, 0.1],
        'full_name': 'MotrPac',
        'description': 'Exercise response proteomics',
        'color': '#3b82f6',  # Blue
    },
    'addneuromed': {
        'data_name': 'addneuromed',
        'revision': '9f052d330ce130408a2c7c347b2ed197154da7c8',
        'train_val_test_split': [0.7, 0.2, 0.1],
        'full_name': 'AddNeuroMed',
        'description': 'Alzheimer\'s gene expression',
        'color': '#f97316',  # Orange
    },
    'parkinsons': {
        'data_name': 'parkinsons',
        'revision': '9f052d330ce130408a2c7c347b2ed197154da7c8',
        'train_val_test_split': [0.7, 0.15, 0.15],
        'full_name': 'Parkinson\'s',
        'description': 'Parkinson\'s gene expression',
        'color': '#22c55e',  # Green
    },
}

HF_REPO_ID = 'geometric-intelligence/bgbench'

# Cache for loaded datasets and computed graph stats
_data_cache = {}
_stats_cache = {}  # Cache for computed graph statistics keyed by parameters

# Path to precomputed stats file
PRECOMPUTED_STATS_FILE = Path(__file__).parent / 'precomputed_stats.json'


def load_precomputed_stats():
    """Load precomputed stats from JSON file into cache."""
    global _stats_cache
    if PRECOMPUTED_STATS_FILE.exists():
        print(f'Loading precomputed stats from {PRECOMPUTED_STATS_FILE}...')
        with open(PRECOMPUTED_STATS_FILE, 'r') as f:
            precomputed = json.load(f)
        
        # Convert JSON keys back to tuple format for cache
        for key_str, stats in precomputed.items():
            # Key format: "dataset|ratio|method|threshold"
            parts = key_str.split('|')
            dataset = parts[0]
            ratio = float(parts[1])
            method = parts[2]
            threshold = float(parts[3])
            cache_key = (dataset, ratio, method, threshold)
            _stats_cache[cache_key] = stats
        
        print(f'  Loaded {len(_stats_cache)} precomputed combinations')
    else:
        print(f'No precomputed stats found at {PRECOMPUTED_STATS_FILE}')
        print('  Run: python precompute_stats.py to generate them')


def load_raw_data(dataset_name: str) -> tuple[pd.DataFrame, np.ndarray]:
    """Load raw dataset from HuggingFace (cached)."""
    if dataset_name in _data_cache:
        return _data_cache[dataset_name]
    
    print(f'Loading {dataset_name} from HuggingFace...')
    config = DATASETS[dataset_name]
    
    # Download data
    data_file = hf_hub_download(
        repo_id=HF_REPO_ID,
        repo_type='dataset',
        revision=config['revision'],
        filename=f'{dataset_name}_data.parquet',
    )
    targets_file = hf_hub_download(
        repo_id=HF_REPO_ID,
        repo_type='dataset',
        revision=config['revision'],
        filename=f'{dataset_name}_targets.parquet',
    )
    
    # Load data
    raw_data = pd.read_parquet(data_file)
    targets_df = pd.read_parquet(targets_file)
    
    if 'target' in raw_data.columns:
        raw_data = raw_data.drop('target', axis=1)
    
    targets = targets_df['target'].values
    
    # Shuffle and split
    raw_data, targets = shuffle(raw_data, targets, random_state=42)
    
    train_val_test_split = config['train_val_test_split']
    train_idx = int(len(targets) * train_val_test_split[0])
    
    # Get training data only
    train_data = raw_data.iloc[:train_idx]
    train_targets = targets[:train_idx]
    
    # Impute missing values
    imputer = SimpleImputer(strategy='mean')
    train_data_imputed = imputer.fit_transform(train_data)
    train_data = pd.DataFrame(
        train_data_imputed, columns=train_data.columns, index=train_data.index
    )
    
    print(f'  {dataset_name}: {train_data.shape[0]} samples, {train_data.shape[1]} features')
    
    _data_cache[dataset_name] = (train_data, train_targets)
    return train_data, train_targets


def select_nodes(
    data: np.ndarray, targets: np.ndarray, n_selected: int, method: str
) -> np.ndarray:
    """Select nodes based on feature importance."""
    np.random.seed(42)  # For reproducibility in random selection
    
    if method == 'variance':
        variances = np.std(data, axis=0)
        ranked_nodes = np.argsort(variances)[::-1]
    elif method == 'correlation':
        correlations = np.abs(
            np.array([np.corrcoef(data[:, i], targets)[0, 1] for i in range(data.shape[1])])
        )
        # Handle NaN correlations
        correlations = np.nan_to_num(correlations, nan=0.0)
        ranked_nodes = np.argsort(correlations)[::-1]
    elif method == 'random':
        ranked_nodes = np.random.permutation(data.shape[1])
    else:
        raise ValueError(f'Invalid method: {method}')
    
    return ranked_nodes[:n_selected]


def calculate_adjacency_matrix(
    node_features: pd.DataFrame, adjacency_threshold: float
) -> np.ndarray:
    """Calculate adjacency matrix using correlation-based approach."""
    # Use correlation-based adjacency with soft thresholding (power=6)
    corr_matrix = node_features.corr().values
    adjacency = np.abs(corr_matrix) ** 6
    
    # Binarize
    adjacency = np.nan_to_num(adjacency, nan=0.0)
    adj_matrix = np.where(adjacency > adjacency_threshold, 1, 0)
    np.fill_diagonal(adj_matrix, 1)
    
    return adj_matrix


def compute_graph_metrics(adj_matrix: np.ndarray) -> dict[str, float]:
    """Compute various graph metrics from adjacency matrix."""
    n_nodes = adj_matrix.shape[0]
    
    # Remove self-loops for metric computation
    adj_no_diag = adj_matrix.copy()
    np.fill_diagonal(adj_no_diag, 0)
    
    # Create NetworkX graph for advanced metrics
    graph = nx.from_numpy_array(adj_no_diag)
    
    # Node degrees
    node_degrees = np.sum(adj_no_diag, axis=1)
    
    # Number of edges (undirected)
    n_edges = np.sum(adj_no_diag) / 2
    
    # Maximum possible edges
    max_edges = n_nodes * (n_nodes - 1) / 2
    
    # Density
    density = n_edges / max_edges if max_edges > 0 else 0
    
    # Degree statistics
    mean_degree = np.mean(node_degrees) if n_nodes > 0 else 0
    std_degree = np.std(node_degrees) if n_nodes > 0 else 0
    
    # Connected components
    n_components = nx.number_connected_components(graph)
    
    # Largest connected component ratio
    if n_nodes > 0 and n_components > 0:
        largest_cc = max(nx.connected_components(graph), key=len)
        largest_cc_ratio = len(largest_cc) / n_nodes * 100
    else:
        largest_cc_ratio = 0
    
    # Average clustering coefficient
    try:
        avg_clustering = nx.average_clustering(graph)
    except Exception:
        avg_clustering = 0
    
    # Average shortest path length (sample-based for performance)
    try:
        if n_components > 0 and len(largest_cc) > 1:
            subgraph = graph.subgraph(largest_cc)
            # For large graphs, sample nodes for path length estimation
            if len(largest_cc) > 100:
                # Sample 50 random pairs
                sample_nodes = list(largest_cc)[:min(50, len(largest_cc))]
                path_lengths = []
                for source in sample_nodes[:25]:
                    lengths = nx.single_source_shortest_path_length(subgraph, source)
                    path_lengths.extend(lengths.values())
                avg_path_length = np.mean(path_lengths) if path_lengths else 0
            else:
                avg_path_length = nx.average_shortest_path_length(subgraph)
        else:
            avg_path_length = 0
    except Exception:
        avg_path_length = 0
    
    return {
        'n_nodes': n_nodes,
        'n_edges': int(n_edges),
        'density': density * 100,  # as percentage
        'mean_degree': mean_degree,
        'std_degree': std_degree,
        'n_components': n_components,
        'largest_cc_ratio': largest_cc_ratio,
        'avg_clustering': avg_clustering,
        'avg_path_length': avg_path_length,
    }


def get_graph_stats_for_params(
    dataset_name: str,
    node_sample_ratio: float,
    method: str,
    adjacency_threshold: float
) -> dict[str, float]:
    """Compute graph statistics for given parameters (with caching)."""
    # Create cache key from parameters
    cache_key = (dataset_name, node_sample_ratio, method, adjacency_threshold)
    
    # Return cached result if available
    if cache_key in _stats_cache:
        return _stats_cache[cache_key].copy()
    
    train_data, train_targets = load_raw_data(dataset_name)
    
    # Calculate number of nodes based on p ratio
    n_training_samples = len(train_targets)
    if node_sample_ratio >= 1.0:
        # Use all features but cap at 1000 for performance
        n_nodes = min(train_data.shape[1], 1000)
    else:
        n_nodes = int(n_training_samples / node_sample_ratio)
        if n_nodes > train_data.shape[1]:
            n_nodes = train_data.shape[1]
        # Cap for performance
        n_nodes = min(n_nodes, 1000)
    
    # Select nodes
    selected_nodes = select_nodes(
        train_data.values, train_targets, n_selected=n_nodes, method=method
    )
    train_selected = train_data.iloc[:, selected_nodes]
    
    # Calculate adjacency matrix
    adj_matrix = calculate_adjacency_matrix(train_selected, adjacency_threshold)
    
    # Compute metrics
    metrics = compute_graph_metrics(adj_matrix)
    metrics['dataset'] = dataset_name
    
    # Cache the result
    _stats_cache[cache_key] = metrics.copy()
    
    return metrics


# Initialize Dash app
app = dash.Dash(
    __name__,
    suppress_callback_exceptions=True,
    title='BGBench Dataset Explorer'
)

# Custom CSS for modern styling
app.index_string = '''
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>{%title%}</title>
        {%favicon%}
        {%css%}
        <link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
        <style>
            * { box-sizing: border-box; }
            
            body {
                font-family: 'Avenir', 'Avenir Next', -apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, sans-serif;
                background: #f0f4f8;
                margin: 0;
                padding: 0;
                min-height: 100vh;
                color: #1e293b;
            }
            
            .main-container {
                max-width: 1500px;
                margin: 0 auto;
                padding: 20px;
            }
            
            .header {
                background: linear-gradient(135deg, #0f172a 0%, #1e293b 50%, #334155 100%);
                color: white;
                padding: 28px 36px;
                border-radius: 16px;
                margin-bottom: 20px;
                box-shadow: 0 8px 32px rgba(15, 23, 42, 0.25);
                position: relative;
                overflow: hidden;
            }
            
            .header::before {
                content: '';
                position: absolute;
                top: 0;
                right: 0;
                width: 300px;
                height: 100%;
                background: linear-gradient(90deg, transparent, rgba(59, 130, 246, 0.1));
                pointer-events: none;
            }
            
            .header h1 {
                margin: 0 0 6px 0;
                font-size: 2rem;
                font-weight: 700;
                letter-spacing: -0.03em;
            }
            
            .header p {
                margin: 0;
                opacity: 0.75;
                font-size: 1rem;
                font-weight: 400;
            }
            
            .controls-panel {
                background: white;
                border-radius: 14px;
                padding: 24px;
                margin-bottom: 20px;
                box-shadow: 0 2px 12px rgba(0, 0, 0, 0.04);
                border: 1px solid #e2e8f0;
            }
            
            .controls-grid {
                display: grid;
                grid-template-columns: repeat(4, 1fr);
                gap: 24px;
            }
            
            @media (max-width: 1200px) {
                .controls-grid {
                    grid-template-columns: repeat(2, 1fr);
                }
            }
            
            .control-group {
                display: flex;
                flex-direction: column;
            }
            
            .control-label {
                font-size: 0.72rem;
                font-weight: 600;
                color: #64748b;
                margin-bottom: 8px;
                text-transform: uppercase;
                letter-spacing: 0.08em;
            }
            
            .control-value {
                font-family: 'JetBrains Mono', monospace;
                font-size: 1.25rem;
                color: #0f172a;
                font-weight: 600;
                margin-bottom: 10px;
            }
            
            .stats-grid {
                display: grid;
                grid-template-columns: repeat(6, 1fr);
                gap: 12px;
                margin-bottom: 20px;
            }
            
            @media (max-width: 1200px) {
                .stats-grid {
                    grid-template-columns: repeat(3, 1fr);
                }
            }
            
            .stat-card {
                background: white;
                border-radius: 12px;
                padding: 16px;
                text-align: center;
                box-shadow: 0 2px 8px rgba(0, 0, 0, 0.03);
                border: 1px solid #e2e8f0;
                transition: all 0.2s ease;
            }
            
            .stat-card:hover {
                transform: translateY(-2px);
                box-shadow: 0 6px 20px rgba(0, 0, 0, 0.08);
                border-color: #3b82f6;
            }
            
            .stat-value {
                font-family: 'JetBrains Mono', monospace;
                font-size: 1.5rem;
                font-weight: 700;
                color: #0f172a;
                margin-bottom: 4px;
            }
            
            .stat-label {
                font-size: 0.7rem;
                color: #64748b;
                text-transform: uppercase;
                letter-spacing: 0.05em;
                font-weight: 500;
            }
            
            .charts-container {
                background: white;
                border-radius: 14px;
                padding: 20px;
                box-shadow: 0 2px 12px rgba(0, 0, 0, 0.04);
                border: 1px solid #e2e8f0;
            }
            
            .Select-control {
                border-radius: 8px !important;
                border: 2px solid #e2e8f0 !important;
                min-height: 42px !important;
                font-family: 'Avenir', 'Avenir Next', -apple-system, BlinkMacSystemFont, sans-serif !important;
            }
            
            .Select-control:hover {
                border-color: #3b82f6 !important;
            }
            
            .rc-slider {
                margin-top: 8px;
            }
            
            .rc-slider-track {
                background: linear-gradient(90deg, #3b82f6, #6366f1) !important;
                height: 6px !important;
            }
            
            .rc-slider-rail {
                background: #e2e8f0 !important;
                height: 6px !important;
            }
            
            .rc-slider-handle {
                border: 3px solid #3b82f6 !important;
                background: white !important;
                width: 20px !important;
                height: 20px !important;
                margin-top: -7px !important;
                box-shadow: 0 2px 6px rgba(59, 130, 246, 0.3) !important;
            }
            
            .rc-slider-handle:hover {
                border-color: #2563eb !important;
            }
            
            .rc-slider-handle:active {
                box-shadow: 0 0 0 5px rgba(59, 130, 246, 0.2) !important;
            }
            
            .rc-slider-mark-text {
                font-size: 0.7rem !important;
                color: #94a3b8 !important;
            }
            
            input[type="checkbox"] {
                width: 18px;
                height: 18px;
                accent-color: #3b82f6;
                cursor: pointer;
            }
            
            .dataset-checkbox {
                display: flex;
                align-items: center;
                gap: 8px;
                padding: 8px 12px;
                background: #f8fafc;
                border-radius: 8px;
                transition: background 0.2s;
                cursor: pointer;
            }
            
            .dataset-checkbox:hover {
                background: #f1f5f9;
            }
            
            ._dash-loading {
                margin: 40px auto;
            }
        </style>
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
'''

# App layout
app.layout = html.Div([
    html.Div([
        # Header
        html.Div([
            html.H1('BGBench Dataset Explorer'),
            html.P('Interactive visualization of omics dataset graph properties')
        ], className='header'),
        
        # Controls Panel
        html.Div([
            html.Div([
                # Node Sample Ratio
                html.Div([
                    html.Div('Node Sample Ratio', className='control-label'),
                    html.Div(id='ratio-value', className='control-value'),
                    dcc.Slider(
                        id='node-sample-ratio',
                        min=0.5,
                        max=1.0,
                        step=0.1,
                        value=1.0,
                        marks={0.5: '0.5', 0.6: '0.6', 0.7: '0.7', 0.8: '0.8', 0.9: '0.9', 1.0: '1.0'},
                    ),
                ], className='control-group'),
                
                # Node Selection Method
                html.Div([
                    html.Div('Node Selection Method', className='control-label'),
                    dcc.Dropdown(
                        id='node-selection-method',
                        options=[
                            {'label': '📊 Variance', 'value': 'variance'},
                            {'label': '🔗 Correlation', 'value': 'correlation'},
                            {'label': '🎲 Random', 'value': 'random'},
                        ],
                        value='variance',
                        clearable=False,
                    ),
                ], className='control-group'),
                
                # Adjacency Threshold
                html.Div([
                    html.Div('Adjacency Threshold', className='control-label'),
                    html.Div(id='threshold-value', className='control-value'),
                    dcc.Slider(
                        id='adjacency-threshold',
                        min=0.0,
                        max=0.5,
                        step=0.02,
                        value=0.02,
                        marks={0: '0', 0.1: '0.1', 0.2: '0.2', 0.3: '0.3', 0.4: '0.4', 0.5: '0.5'},
                    ),
                ], className='control-group'),
                
                # Dataset Selector
                html.Div([
                    html.Div('Datasets', className='control-label'),
                    dcc.Checklist(
                        id='dataset-selector',
                        options=[
                            {'label': html.Span([' MotrPac'], style={'fontWeight': '500'}), 'value': 'motrpac'},
                            {'label': html.Span([' AddNeuroMed'], style={'fontWeight': '500'}), 'value': 'addneuromed'},
                            {'label': html.Span([' Parkinson\'s'], style={'fontWeight': '500'}), 'value': 'parkinsons'},
                        ],
                        value=['motrpac', 'addneuromed', 'parkinsons'],
                        inline=True,
                        style={'display': 'flex', 'gap': '12px', 'flexWrap': 'wrap', 'marginTop': '4px'}
                    ),
                ], className='control-group'),
            ], className='controls-grid'),
        ], className='controls-panel'),
        
        # Loading indicator
        dcc.Loading(
            id='loading',
            type='circle',
            color='#3b82f6',
            children=[
                # Charts
                html.Div([
                    dcc.Graph(id='main-chart', config={'displayModeBar': True, 'responsive': True}),
                ], className='charts-container'),
            ]
        ),
        
    ], className='main-container'),
], style={'minHeight': '100vh'})


# Callbacks
@app.callback(
    Output('ratio-value', 'children'),
    Input('node-sample-ratio', 'value')
)
def update_ratio_display(value):
    return f'p = {value}'


@app.callback(
    Output('threshold-value', 'children'),
    Input('adjacency-threshold', 'value')
)
def update_threshold_display(value):
    return f'τ = {value:.2f}'


@app.callback(
    Output('main-chart', 'figure'),
    [Input('node-sample-ratio', 'value'),
     Input('node-selection-method', 'value'),
     Input('adjacency-threshold', 'value'),
     Input('dataset-selector', 'value')]
)
def update_visualization(node_sample_ratio, method, adj_threshold, selected_datasets):
    if not selected_datasets:
        selected_datasets = ['motrpac']
    
    # Compute stats for all selected datasets
    all_stats = []
    for dataset_name in selected_datasets:
        try:
            stats = get_graph_stats_for_params(
                dataset_name, node_sample_ratio, method, adj_threshold
            )
            stats['dataset_label'] = DATASETS[dataset_name]['full_name']
            stats['color'] = DATASETS[dataset_name]['color']
            all_stats.append(stats)
        except Exception as e:
            print(f'Error computing stats for {dataset_name}: {e}')
    
    if not all_stats:
        return go.Figure()
    
    # Create comparison chart
    metrics = [
        ('n_nodes', 'Graph Size (Nodes)'),
        ('n_edges', 'Graph Connectivity (Edges)'),
        ('mean_degree', 'Average Node Degree'),
        ('density', 'Graph Density (%)'),
        ('avg_clustering', 'Avg Clustering Coefficient'),
        ('largest_cc_ratio', 'Largest CC / Total Nodes (%)'),
        ('avg_path_length', 'Avg Shortest Path Length'),
        ('n_components', 'Connected Components'),
        ('std_degree', 'Degree Distribution Std Dev'),
    ]
    
    fig = make_subplots(
        rows=3, cols=3,
        subplot_titles=[m[1] for m in metrics],
        vertical_spacing=0.16,
        horizontal_spacing=0.08,
    )
    
    for idx, (metric, title) in enumerate(metrics):
        row = idx // 3 + 1
        col = idx % 3 + 1
        
        x_labels = [s['dataset_label'] for s in all_stats]
        y_values = [s[metric] for s in all_stats]
        colors = [s['color'] for s in all_stats]
        
        # Format text
        if metric in ['density', 'largest_cc_ratio']:
            text_values = [f'{v:.1f}%' for v in y_values]
        elif metric in ['avg_clustering', 'avg_path_length']:
            text_values = [f'{v:.2f}' for v in y_values]
        elif metric in ['n_nodes', 'n_edges', 'n_components']:
            text_values = [f'{int(v):,}' for v in y_values]
        else:
            text_values = [f'{v:.1f}' for v in y_values]
        
        fig.add_trace(
            go.Bar(
                x=x_labels,
                y=y_values,
                marker_color=colors,
                text=text_values,
                textposition='outside',
                textfont=dict(size=10, family='JetBrains Mono'),
                showlegend=False,
                cliponaxis=False,
            ),
            row=row, col=col
        )
        
        # Add padding to y-axis range to prevent text clipping
        if y_values:
            max_val = max(y_values)
            min_val = min(y_values)
            padding = (max_val - min_val) * 0.25 if max_val != min_val else max_val * 0.25
            fig.update_yaxes(range=[0, max_val + padding], row=row, col=col)
    
    fig.update_layout(
        height=850,
        font=dict(family='Avenir, Avenir Next, -apple-system, BlinkMacSystemFont, Segoe UI, Helvetica, sans-serif', size=12),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(248,250,252,1)',
        margin=dict(l=50, r=30, t=50, b=50),
    )
    
    # Update subplot title annotations to have proper spacing
    for annotation in fig['layout']['annotations']:
        annotation['font'] = dict(size=12, color='#374151', family='Avenir, Avenir Next, sans-serif')
        annotation['y'] = annotation['y'] + 0.02  # Move subplot titles up slightly
    
    # Update axes for all subplots
    fig.update_xaxes(
        showgrid=False,
        tickangle=0,
        tickfont=dict(size=10),
    )
    fig.update_yaxes(
        showgrid=True,
        gridcolor='rgba(226,232,240,0.6)',
        zeroline=True,
        zerolinecolor='rgba(226,232,240,0.8)',
        tickfont=dict(size=10),
    )
    
    return fig


# Preload data at startup
print('\n' + '='*60)
print('BGBench Dataset Explorer')
print('='*60)

# Load precomputed stats first (fast)
print('\nLoading precomputed statistics...')
load_precomputed_stats()

# Only load raw data if precomputed stats are missing
if len(_stats_cache) == 0:
    print('\nNo precomputed stats found. Loading datasets from HuggingFace...')
    for dataset_name in DATASETS.keys():
        try:
            load_raw_data(dataset_name)
        except Exception as e:
            print(f'Error preloading {dataset_name}: {e}')
    print('\nDatasets loaded successfully!')
else:
    print(f'\nUsing {len(_stats_cache)} precomputed statistics for fast startup!')

print('='*60 + '\n')


# Run the app
if __name__ == '__main__':
    print('Starting Dash server...')
    print('Open http://127.0.0.1:8050 in your browser\n')
    
    app.run(debug=False, host='0.0.0.0', port=8050)
