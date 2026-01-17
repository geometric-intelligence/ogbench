#!/usr/bin/env python3
"""OgBench - Omics Graph Benchmark Dashboard.

A single-page application with multi-page navigation:
- Landing page: Leaderboard (ML benchmark results)
- Second page: Dataset Explorer (graph statistics visualization)

Run: python app.py
Open: http://127.0.0.1:8050
"""

import json
import base64
from pathlib import Path

import dash
import pandas as pd
import plotly.graph_objects as go
from dash import dcc, html, dash_table
from dash.dependencies import Input, Output
from plotly.subplots import make_subplots

# =============================================================================
# Configuration
# =============================================================================

# Path to data files
RESULTS_FILE = Path(__file__).parent / 'precomputed_results_placeholder.json'
STATS_FILE = Path(__file__).parent / 'precomputed_stats.json'

# Dataset configurations
DATASETS = {
    'motrpac': {'full_name': 'MotrPac', 'color': '#3b82f6', 'emoji': '🧬'},
    'addneuromed': {'full_name': 'AddNeuroMed', 'color': '#f97316', 'emoji': '🧠'},
    'parkinsons': {'full_name': "Parkinson's", 'color': '#22c55e', 'emoji': '🔬'},
}

# Model categories for styling
MODEL_CATEGORIES = {
    'GATv4': 'gnn', 'GATv2': 'gnn', 'GCN': 'gnn', 'GIN': 'gnn',
    'GraphSAGE': 'gnn', 'SAGN': 'gnn', 'MLP': 'neural',
    'Random': 'baseline', 'ElasticNet': 'baseline', 'SVM': 'baseline',
}

MODEL_ORDER = ['SVM', 'ElasticNet', 'MLP', 'GATv4', 'GATv2', 'GIN', 'GCN', 'GraphSAGE', 'SAGN', 'Random']

# Valid parameter values for dataset explorer
VALID_RATIOS = [0.5, 0.6, 0.7, 0.8, 0.9]
VALID_METHODS = ['variance', 'correlation', 'random']
VALID_THRESHOLDS = [0.02, 0.1, 0.2, 0.3, 0.4, 0.5]

# Global data stores
RESULTS_DF = None
PRECOMPUTED_STATS = {}
METRIC_MAX_VALUES = {}

# Logo - base64 encoded SVG
_LOGO_SVG_RAW = '''<svg version="1.1" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 400 386.6">
<style type="text/css">.st0{fill:#75B4E3;}.st1{fill:#F4951E;}.st2{fill:#FBFBFB;}</style>
<path class="st0" d="M305.6,338.3c-24.9,19.5-55.2,31.9-87.9,34.1c-4.6,0.3-9.3,0.5-14,0.5c-32.3,0-62.5-9.5-87.8-25.8l31.1-53.9 c16.1,11.9,35.9,18.9,57.3,18.9c3.8,0,7.5-0.2,11.2-0.7c25.5-2.7,48.2-14.7,64.8-32.6L305.6,338.3z"/>
<path class="st1" d="M339.1,90.9c19.5,24.9,31.9,55.2,34.1,87.9c0.3,4.6,0.5,9.3,0.5,14c0,32.3-9.5,62.5-25.8,87.8l-53.9-31.1 c11.9-16.1,18.9-35.9,18.9-57.3c0-3.8-0.2-7.5-0.7-11.2c-2.7-25.5-14.7-48.2-32.6-64.8L339.1,90.9z"/>
<path class="st0" d="M94.4,48.3c24.9-19.5,55.2-31.9,87.9-34.1c4.6-0.3,9.3-0.5,14-0.5c32.3,0,62.5,9.5,87.8,25.8l-31.1,53.9 c-16.1-11.9-35.9-18.9-57.3-18.9c-3.8,0-7.5,0.2-11.2,0.7c-25.5,2.7-48.2,14.7-64.8,32.6L94.4,48.3z"/>
<path class="st1" d="M60.9,295.7c-19.5-24.9-31.9-55.2-34.1-87.9c-0.3-4.6-0.5-9.3-0.5-14c0-32.3,9.5-62.5,25.8-87.8l53.9,31.1 c-11.9,16.1-18.9,35.9-18.9,57.3c0,3.8,0.2,7.5,0.7,11.2c2.7,25.5,14.7,48.2,32.6,64.8L60.9,295.7z"/>
<path class="st2" d="M280.3,278.8c-3.6,3.6-7.4,6.9-11.5,9.9c-16.1,11.9-35.9,18.9-57.3,18.9c-3.8,0-7.5-0.2-11.2-0.7 c-25.5-2.7-48.2-14.7-64.8-32.6c-3.6-3.9-6.9-8.1-9.9-12.5c-11.6-17-18.4-37.5-18.4-59.5c0-3.8,0.2-7.5,0.7-11.2 c2.7-25.5,14.7-48.2,32.6-64.8c3.9-3.6,8.1-6.9,12.5-9.9c17-11.6,37.5-18.4,59.5-18.4c3.8,0,7.5,0.2,11.2,0.7 c25.5,2.7,48.2,14.7,64.8,32.6c3.6,3.9,6.9,8.1,9.9,12.5c11.6,17,18.4,37.5,18.4,59.5c0,3.8-0.2,7.5-0.7,11.2 C313.5,240.1,301.4,262.7,280.3,278.8z"/>
</svg>'''
LOGO_BASE64 = 'data:image/svg+xml;base64,' + base64.b64encode(_LOGO_SVG_RAW.encode()).decode()


# =============================================================================
# Data Loading Functions
# =============================================================================

def load_results_data():
    """Load leaderboard results data."""
    global RESULTS_DF
    
    if RESULTS_FILE.exists():
        with open(RESULTS_FILE, 'r') as f:
            results = json.load(f)
        RESULTS_DF = pd.DataFrame(list(results.values()))
        print(f'  Loaded {len(RESULTS_DF)} result entries')
    else:
        print(f'  WARNING: Results file not found at {RESULTS_FILE}')
        RESULTS_DF = pd.DataFrame()


def load_stats_data():
    """Load precomputed stats for dataset explorer."""
    global PRECOMPUTED_STATS, METRIC_MAX_VALUES
    
    if STATS_FILE.exists():
        with open(STATS_FILE, 'r') as f:
            PRECOMPUTED_STATS = json.load(f)
        
        # Compute max values for fixed y-axis ranges
        metrics = ['n_nodes', 'n_edges', 'mean_degree', 'density', 'avg_clustering', 
                   'largest_cc_ratio', 'avg_path_length', 'n_components', 'std_degree']
        for m in metrics:
            vals = [v[m] for v in PRECOMPUTED_STATS.values()]
            METRIC_MAX_VALUES[m] = max(vals) * 1.20 if vals else 1
        
        print(f'  Loaded {len(PRECOMPUTED_STATS)} graph configurations')
    else:
        print(f'  WARNING: Stats file not found at {STATS_FILE}')


def get_stats(dataset_name: str, ratio: float, method: str, threshold: float) -> dict | None:
    """Get precomputed statistics for given parameters."""
    key = f'{dataset_name}|{float(ratio)}|{method}|{float(threshold)}'
    return PRECOMPUTED_STATS.get(key)


def compute_aggregate_leaderboard(df, metric='test_accuracy'):
    """Compute aggregate leaderboard across all graphs."""
    if df.empty:
        return pd.DataFrame()
    
    agg = df.groupby('model').agg({
        'test_accuracy': ['mean', 'std'],
        'f1_macro': ['mean', 'std'],
        'runtime_seconds': ['mean', 'sum'],
    }).reset_index()
    
    agg.columns = ['Model', 'Accuracy', 'Acc_Std', 'F1 Macro', 'F1_Std', 'Avg Runtime (s)', 'Total Runtime (s)']
    sort_col = 'Accuracy' if metric == 'test_accuracy' else 'F1 Macro'
    agg = agg.sort_values(sort_col, ascending=False).reset_index(drop=True)
    agg.insert(0, 'Rank', range(1, len(agg) + 1))
    
    return agg


# =============================================================================
# Initialize Dash App
# =============================================================================

app = dash.Dash(
    __name__,
    suppress_callback_exceptions=True,
    title='OgBench'
)

# Professional LIGHT theme CSS
app.index_string = '''
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>{%title%}</title>
        {%favicon%}
        {%css%}
        <link href="https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap" rel="stylesheet">
        <style>
            :root {
                --bg-primary: #f8fafc;
                --bg-secondary: #ffffff;
                --bg-card: #ffffff;
                --bg-hover: #f1f5f9;
                --border-color: #e2e8f0;
                --text-primary: #0f172a;
                --text-secondary: #475569;
                --text-muted: #94a3b8;
                --accent-gold: #d97706;
                --accent-blue: #3b82f6;
                --accent-orange: #f97316;
                --accent-green: #22c55e;
                --gnn-color: #8b5cf6;
                --neural-color: #ec4899;
                --baseline-color: #0ea5e9;
            }
            
            * { box-sizing: border-box; margin: 0; padding: 0; }
            
            body {
                font-family: 'DM Sans', -apple-system, BlinkMacSystemFont, sans-serif;
                background: var(--bg-primary);
                color: var(--text-primary);
                min-height: 100vh;
                line-height: 1.6;
            }
            
            .main-container {
                max-width: 1600px;
                margin: 0 auto;
                padding: 32px 24px;
            }
            
            .header {
                text-align: center;
                margin-bottom: 40px;
                padding: 40px 24px;
                background: linear-gradient(135deg, #ffffff 0%, #f1f5f9 100%);
                border-bottom: 1px solid var(--border-color);
                border-radius: 24px;
                box-shadow: 0 4px 20px rgba(0, 0, 0, 0.04);
            }
            
            .header-logo {
                width: 120px;
                height: 120px;
                margin: 0 auto 20px;
                border-radius: 50%;
                box-shadow: 0 8px 24px rgba(0, 0, 0, 0.15);
                display: block;
            }
            
            .header-badge {
                display: inline-block;
                background: linear-gradient(135deg, #75B4E3, #F4951E);
                color: white;
                padding: 6px 16px;
                border-radius: 20px;
                font-size: 0.75rem;
                font-weight: 600;
                letter-spacing: 0.1em;
                text-transform: uppercase;
                margin-bottom: 12px;
            }
            
            .header h1 {
                font-size: 3rem;
                font-weight: 700;
                letter-spacing: -0.03em;
                margin-bottom: 8px;
                color: var(--text-primary);
            }
            
            .header-subtitle {
                font-size: 1.05rem;
                color: var(--text-secondary);
                max-width: 650px;
                margin: 0 auto;
            }
            
            .header-stats {
                display: flex;
                justify-content: center;
                gap: 48px;
                margin-top: 28px;
            }
            
            .header-stat {
                text-align: center;
            }
            
            .header-stat-value {
                font-family: 'JetBrains Mono', monospace;
                font-size: 2rem;
                font-weight: 600;
                color: var(--accent-blue);
            }
            
            .header-stat-label {
                font-size: 0.8rem;
                color: var(--text-muted);
                text-transform: uppercase;
                letter-spacing: 0.08em;
            }
            
            .nav-link {
                display: inline-flex;
                align-items: center;
                gap: 8px;
                margin-top: 20px;
                padding: 12px 24px;
                background: var(--accent-blue);
                color: white;
                text-decoration: none;
                border-radius: 12px;
                font-weight: 600;
                font-size: 0.9rem;
                transition: all 0.2s ease;
                box-shadow: 0 2px 8px rgba(59, 130, 246, 0.3);
                cursor: pointer;
            }
            
            .nav-link:hover {
                background: #2563eb;
                transform: translateY(-1px);
                box-shadow: 0 4px 12px rgba(59, 130, 246, 0.4);
            }
            
            .controls-panel {
                background: var(--bg-card);
                border: 1px solid var(--border-color);
                border-radius: 16px;
                padding: 24px;
                margin-bottom: 24px;
                box-shadow: 0 2px 8px rgba(0, 0, 0, 0.04);
            }
            
            .controls-grid {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                gap: 24px;
            }
            
            .control-group {
                display: flex;
                flex-direction: column;
            }
            
            .control-label {
                font-size: 0.7rem;
                font-weight: 600;
                color: var(--text-muted);
                text-transform: uppercase;
                letter-spacing: 0.1em;
                margin-bottom: 8px;
            }
            
            .control-value {
                font-family: 'JetBrains Mono', monospace;
                font-size: 1.2rem;
                color: var(--text-primary);
                font-weight: 600;
                margin-bottom: 10px;
            }
            
            .leaderboard-section {
                background: var(--bg-card);
                border: 1px solid var(--border-color);
                border-radius: 16px;
                overflow: hidden;
                margin-bottom: 24px;
                box-shadow: 0 2px 8px rgba(0, 0, 0, 0.04);
            }
            
            .section-header {
                display: flex;
                justify-content: space-between;
                align-items: center;
                padding: 20px 24px;
                border-bottom: 1px solid var(--border-color);
                background: linear-gradient(135deg, #f8fafc 0%, #f1f5f9 100%);
            }
            
            .section-title {
                font-size: 1.1rem;
                font-weight: 600;
                display: flex;
                align-items: center;
                gap: 10px;
                color: var(--text-primary);
            }
            
            .charts-grid {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(500px, 1fr));
                gap: 24px;
            }
            
            .chart-card {
                background: var(--bg-card);
                border: 1px solid var(--border-color);
                border-radius: 16px;
                padding: 20px;
                box-shadow: 0 2px 8px rgba(0, 0, 0, 0.04);
            }
            
            .chart-title {
                font-size: 0.9rem;
                font-weight: 600;
                color: var(--text-secondary);
                margin-bottom: 16px;
                padding-bottom: 12px;
                border-bottom: 1px solid var(--border-color);
            }
            
            .charts-container {
                background: var(--bg-card);
                border-radius: 16px;
                padding: 20px;
                box-shadow: 0 2px 12px rgba(0, 0, 0, 0.04);
                border: 1px solid var(--border-color);
            }
            
            .footer {
                text-align: center;
                padding: 32px;
                color: var(--text-muted);
                font-size: 0.85rem;
                border-top: 1px solid var(--border-color);
                margin-top: 40px;
                background: var(--bg-secondary);
                border-radius: 16px;
            }
            
            .footer-logo {
                width: 40px;
                height: 40px;
                margin-bottom: 12px;
            }
            
            .Select-control {
                background: var(--bg-secondary) !important;
                border: 1px solid var(--border-color) !important;
                border-radius: 10px !important;
            }
            
            .Select-control:hover {
                border-color: var(--accent-blue) !important;
            }
            
            .Select-menu-outer {
                background: var(--bg-card) !important;
                border: 1px solid var(--border-color) !important;
                border-radius: 10px !important;
            }
            
            .Select-option {
                background: var(--bg-card) !important;
                color: var(--text-primary) !important;
            }
            
            .Select-option:hover {
                background: var(--bg-hover) !important;
            }
            
            .rc-slider-track {
                background: linear-gradient(90deg, var(--accent-blue), #6366f1) !important;
                height: 6px !important;
            }
            
            .rc-slider-rail {
                background: var(--border-color) !important;
                height: 6px !important;
            }
            
            .rc-slider-handle {
                border: 3px solid var(--accent-blue) !important;
                background: white !important;
                width: 20px !important;
                height: 20px !important;
                margin-top: -7px !important;
            }
            
            .rc-slider-mark-text {
                font-size: 0.7rem !important;
                color: var(--text-muted) !important;
            }
            
            input[type="checkbox"] {
                width: 18px;
                height: 18px;
                accent-color: var(--accent-blue);
            }
            
            @keyframes fadeIn {
                from { opacity: 0; transform: translateY(10px); }
                to { opacity: 1; transform: translateY(0); }
            }
            
            .leaderboard-section, .chart-card, .charts-container {
                animation: fadeIn 0.4s ease-out;
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


# =============================================================================
# Page Layouts
# =============================================================================

def create_leaderboard_page():
    """Create the leaderboard page layout."""
    return html.Div([
        # Header
        html.Div([
            html.Img(src=LOGO_BASE64, className='header-logo'),
            html.Div('OMICS GRAPH BENCHMARK', className='header-badge'),
            html.H1('OgBench Leaderboard'),
            html.P('Benchmarking GNN, Neural Network, and Classical ML Methods on Graph-Structured Omics Datasets', 
                   className='header-subtitle'),
            html.Div([
                html.Div([
                    html.Div(id='stat-models', className='header-stat-value'),
                    html.Div('Models', className='header-stat-label'),
                ], className='header-stat'),
                html.Div([
                    html.Div(id='stat-graphs', className='header-stat-value'),
                    html.Div('Graph Configs', className='header-stat-label'),
                ], className='header-stat'),
                html.Div([
                    html.Div(id='stat-datasets', className='header-stat-value'),
                    html.Div('Datasets', className='header-stat-label'),
                ], className='header-stat'),
            ], className='header-stats'),
            dcc.Link([
                html.Span('📊'),
                html.Span('Explore Graph Dataset Characteristics'),
            ], href='/explorer', className='nav-link'),
        ], className='header'),
        
        # Controls
        html.Div([
            html.Div([
                html.Div([
                    html.Div('Dataset Filter', className='control-label'),
                    dcc.Dropdown(
                        id='dataset-filter',
                        options=[
                            {'label': '📊 All Datasets (Aggregate)', 'value': 'all'},
                            {'label': f'{DATASETS["motrpac"]["emoji"]} {DATASETS["motrpac"]["full_name"]}', 'value': 'motrpac'},
                            {'label': f'{DATASETS["addneuromed"]["emoji"]} {DATASETS["addneuromed"]["full_name"]}', 'value': 'addneuromed'},
                            {'label': f'{DATASETS["parkinsons"]["emoji"]} {DATASETS["parkinsons"]["full_name"]}', 'value': 'parkinsons'},
                        ],
                        value='all',
                        clearable=False,
                    ),
                ], className='control-group'),
                html.Div([
                    html.Div('Sort By', className='control-label'),
                    dcc.Dropdown(
                        id='sort-metric',
                        options=[
                            {'label': '🎯 Test Accuracy', 'value': 'test_accuracy'},
                            {'label': '📈 F1 Macro', 'value': 'f1_macro'},
                            {'label': '⏱️ Runtime', 'value': 'runtime'},
                        ],
                        value='test_accuracy',
                        clearable=False,
                    ),
                ], className='control-group'),
                html.Div([
                    html.Div('Model Category', className='control-label'),
                    dcc.Dropdown(
                        id='model-category',
                        options=[
                            {'label': '🔷 All Models', 'value': 'all'},
                            {'label': '🌐 GNN Models Only', 'value': 'gnn'},
                            {'label': '🧠 Neural Networks Only', 'value': 'neural'},
                            {'label': '📉 Baselines Only', 'value': 'baseline'},
                        ],
                        value='all',
                        clearable=False,
                    ),
                ], className='control-group'),
            ], className='controls-grid'),
        ], className='controls-panel'),
        
        # Leaderboard table
        html.Div([
            html.Div([
                html.Div([
                    html.Span('🏆', style={'fontSize': '1.2rem'}),
                    html.Span('Leaderboard Rankings'),
                ], className='section-title'),
                html.Div(id='leaderboard-subtitle', style={'color': '#64748b', 'fontSize': '0.85rem'}),
            ], className='section-header'),
            html.Div(id='leaderboard-table-container', style={'padding': '0'}),
        ], className='leaderboard-section'),
        
        # Charts
        html.Div([
            html.Div([
                html.Div('Performance by Model', className='chart-title'),
                dcc.Graph(id='performance-chart', config={'displayModeBar': False}, style={'height': '400px'}),
            ], className='chart-card'),
            html.Div([
                html.Div('Accuracy vs Runtime Trade-off', className='chart-title'),
                dcc.Graph(id='tradeoff-chart', config={'displayModeBar': False}, style={'height': '400px'}),
            ], className='chart-card'),
        ], className='charts-grid'),
        
        html.Div([
            html.Div([
                html.Div('Performance Across Datasets', className='chart-title'),
                dcc.Graph(id='dataset-comparison-chart', config={'displayModeBar': False}, style={'height': '350px'}),
            ], className='chart-card', style={'marginTop': '24px'}),
        ]),
        
        # Footer
        html.Div([
            html.Img(src=LOGO_BASE64, className='footer-logo'),
            html.P('OgBench Leaderboard'),
            html.P('Results aggregated across multiple graph configurations per dataset'),
        ], className='footer'),
    ], className='main-container')


def create_explorer_page():
    """Create the dataset explorer page layout."""
    return html.Div([
        # Header
        html.Div([
            html.Img(src=LOGO_BASE64, className='header-logo'),
            html.Div('DATASET EXPLORER', className='header-badge'),
            html.H1('OgBench Dataset Explorer'),
            html.P('Interactive visualization of omics dataset graph properties', className='header-subtitle'),
            dcc.Link([
                html.Span('🏆'),
                html.Span('Back to Leaderboard'),
            ], href='/', className='nav-link'),
        ], className='header'),
        
        # Controls
        html.Div([
            html.Div([
                html.Div([
                    html.Div('Node Sample Ratio', className='control-label'),
                    html.Div(id='ratio-value', className='control-value'),
                    dcc.Slider(
                        id='node-sample-ratio',
                        min=0.5, max=0.9, step=None, value=0.5,
                        marks={v: f'{v}' for v in VALID_RATIOS},
                        included=False,
                    ),
                ], className='control-group'),
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
                html.Div([
                    html.Div('Adjacency Threshold', className='control-label'),
                    html.Div(id='threshold-value', className='control-value'),
                    dcc.Slider(
                        id='adjacency-threshold',
                        min=0.02, max=0.5, step=None, value=0.02,
                        marks={v: f'{v}' for v in VALID_THRESHOLDS},
                        included=False,
                    ),
                ], className='control-group'),
                html.Div([
                    html.Div('Datasets', className='control-label'),
                    dcc.Checklist(
                        id='dataset-selector',
                        options=[
                            {'label': html.Span([f' {DATASETS["motrpac"]["emoji"]} {DATASETS["motrpac"]["full_name"]}'], 
                                              style={'fontWeight': '600', 'color': DATASETS['motrpac']['color']}), 
                             'value': 'motrpac'},
                            {'label': html.Span([f' {DATASETS["addneuromed"]["emoji"]} {DATASETS["addneuromed"]["full_name"]}'], 
                                              style={'fontWeight': '600', 'color': DATASETS['addneuromed']['color']}), 
                             'value': 'addneuromed'},
                            {'label': html.Span([f' {DATASETS["parkinsons"]["emoji"]} {DATASETS["parkinsons"]["full_name"]}'], 
                                              style={'fontWeight': '600', 'color': DATASETS['parkinsons']['color']}), 
                             'value': 'parkinsons'},
                        ],
                        value=['motrpac', 'addneuromed', 'parkinsons'],
                        inline=True,
                        style={'display': 'flex', 'gap': '16px', 'flexWrap': 'wrap', 'marginTop': '4px'}
                    ),
                ], className='control-group'),
            ], className='controls-grid'),
        ], className='controls-panel'),
        
        # Charts
        html.Div([
            dcc.Graph(id='explorer-chart', config={'displayModeBar': True}, style={'height': '850px'}),
        ], className='charts-container'),
        
        # Footer
        html.Div([
            html.Img(src=LOGO_BASE64, className='footer-logo'),
            html.P('OgBench Dataset Explorer'),
        ], className='footer'),
    ], className='main-container')


# =============================================================================
# App Layout with Routing
# =============================================================================

app.layout = html.Div([
    dcc.Location(id='url', refresh=False),
    html.Div(id='page-content')
], style={'minHeight': '100vh', 'background': '#f8fafc'})


# =============================================================================
# Callbacks
# =============================================================================

# Page routing
@app.callback(Output('page-content', 'children'), Input('url', 'pathname'))
def display_page(pathname):
    if pathname == '/explorer':
        return create_explorer_page()
    return create_leaderboard_page()


# --- Leaderboard callbacks ---

@app.callback(
    [Output('stat-models', 'children'),
     Output('stat-graphs', 'children'),
     Output('stat-datasets', 'children')],
    Input('dataset-filter', 'value')
)
def update_header_stats(_):
    if RESULTS_DF is None or RESULTS_DF.empty:
        return '0', '0', '0'
    return str(RESULTS_DF['model'].nunique()), str(RESULTS_DF['graph_config'].nunique()), str(RESULTS_DF['dataset'].nunique())


@app.callback(
    [Output('leaderboard-table-container', 'children'),
     Output('leaderboard-subtitle', 'children')],
    [Input('dataset-filter', 'value'),
     Input('sort-metric', 'value'),
     Input('model-category', 'value')]
)
def update_leaderboard(dataset_filter, sort_metric, model_category):
    if RESULTS_DF is None or RESULTS_DF.empty:
        return html.Div('No data available', style={'padding': '40px', 'textAlign': 'center'}), ''
    
    df = RESULTS_DF.copy() if dataset_filter == 'all' else RESULTS_DF[RESULTS_DF['dataset'] == dataset_filter].copy()
    subtitle = 'Aggregated across all datasets and graph configurations' if dataset_filter == 'all' else f'Results for {DATASETS.get(dataset_filter, {}).get("full_name", dataset_filter)} dataset'
    
    if model_category != 'all':
        models_in_category = [m for m, cat in MODEL_CATEGORIES.items() if cat == model_category]
        df = df[df['model'].isin(models_in_category)]
    
    leaderboard = compute_aggregate_leaderboard(df, sort_metric)
    if leaderboard.empty:
        return html.Div('No results match the current filters', style={'padding': '40px', 'textAlign': 'center'}), subtitle
    
    leaderboard['Accuracy_Display'] = leaderboard.apply(lambda r: f"{r['Accuracy']:.1%} ± {r['Acc_Std']:.1%}", axis=1)
    leaderboard['F1_Display'] = leaderboard.apply(lambda r: f"{r['F1 Macro']:.1%} ± {r['F1_Std']:.1%}", axis=1)
    leaderboard['Runtime_Display'] = leaderboard['Avg Runtime (s)'].apply(lambda x: f"{x:.1f}s")
    leaderboard['Category'] = leaderboard['Model'].map(MODEL_CATEGORIES)
    
    table = dash_table.DataTable(
        data=leaderboard.to_dict('records'),
        columns=[
            {'name': 'Rank', 'id': 'Rank'}, {'name': 'Model', 'id': 'Model'},
            {'name': 'Category', 'id': 'Category'}, {'name': 'Test Accuracy', 'id': 'Accuracy_Display'},
            {'name': 'F1 Macro', 'id': 'F1_Display'}, {'name': 'Avg Runtime', 'id': 'Runtime_Display'},
        ],
        style_table={'overflowX': 'auto'},
        style_header={'backgroundColor': '#f8fafc', 'color': '#64748b', 'fontWeight': '600', 'fontSize': '0.75rem',
                      'textTransform': 'uppercase', 'letterSpacing': '0.08em', 'padding': '16px 20px', 'borderBottom': '2px solid #e2e8f0'},
        style_cell={'backgroundColor': '#ffffff', 'color': '#0f172a', 'padding': '16px 20px', 'fontFamily': "'DM Sans', sans-serif",
                    'fontSize': '0.95rem', 'borderBottom': '1px solid #e2e8f0', 'textAlign': 'left'},
        style_data_conditional=[
            {'if': {'column_id': 'Rank'}, 'fontFamily': "'JetBrains Mono', monospace", 'fontWeight': '600', 'width': '80px'},
            {'if': {'filter_query': '{Rank} = 1', 'column_id': 'Rank'}, 'color': '#d97706'},
            {'if': {'filter_query': '{Rank} = 2', 'column_id': 'Rank'}, 'color': '#64748b'},
            {'if': {'filter_query': '{Rank} = 3', 'column_id': 'Rank'}, 'color': '#a16207'},
            {'if': {'column_id': 'Model'}, 'fontWeight': '600'},
            {'if': {'filter_query': '{Category} = "gnn"', 'column_id': 'Category'}, 'color': '#8b5cf6'},
            {'if': {'filter_query': '{Category} = "neural"', 'column_id': 'Category'}, 'color': '#ec4899'},
            {'if': {'filter_query': '{Category} = "baseline"', 'column_id': 'Category'}, 'color': '#0ea5e9'},
            {'if': {'column_id': 'Accuracy_Display'}, 'fontFamily': "'JetBrains Mono', monospace"},
            {'if': {'column_id': 'F1_Display'}, 'fontFamily': "'JetBrains Mono', monospace"},
            {'if': {'column_id': 'Runtime_Display'}, 'fontFamily': "'JetBrains Mono', monospace"},
            {'if': {'filter_query': '{Rank} = 1'}, 'backgroundColor': 'rgba(217, 119, 6, 0.06)'},
        ],
        style_as_list_view=True,
    )
    return table, subtitle


@app.callback(Output('performance-chart', 'figure'), [Input('dataset-filter', 'value'), Input('model-category', 'value')])
def update_performance_chart(dataset_filter, model_category):
    if RESULTS_DF is None or RESULTS_DF.empty:
        return go.Figure()
    
    df = RESULTS_DF.copy() if dataset_filter == 'all' else RESULTS_DF[RESULTS_DF['dataset'] == dataset_filter].copy()
    if model_category != 'all':
        df = df[df['model'].isin([m for m, cat in MODEL_CATEGORIES.items() if cat == model_category])]
    
    agg = df.groupby('model').agg({'test_accuracy': 'mean'}).reset_index().sort_values('test_accuracy', ascending=True)
    colors = ['#8b5cf6' if MODEL_CATEGORIES.get(m) == 'gnn' else '#ec4899' if MODEL_CATEGORIES.get(m) == 'neural' else '#0ea5e9' for m in agg['model']]
    
    fig = go.Figure(go.Bar(x=agg['test_accuracy'], y=agg['model'], orientation='h', marker_color=colors,
                           text=[f"{v:.1%}" for v in agg['test_accuracy']], textposition='outside',
                           textfont=dict(family='JetBrains Mono', size=12, color='#0f172a')))
    fig.update_layout(xaxis=dict(title='Test Accuracy', tickformat='.0%', gridcolor='#e2e8f0', range=[0, 1]),
                      yaxis=dict(title='', gridcolor='#e2e8f0'), plot_bgcolor='#ffffff', paper_bgcolor='rgba(0,0,0,0)',
                      font=dict(family='DM Sans', color='#0f172a'), margin=dict(l=100, r=80, t=20, b=40), height=400)
    return fig


@app.callback(Output('tradeoff-chart', 'figure'), [Input('dataset-filter', 'value'), Input('model-category', 'value')])
def update_tradeoff_chart(dataset_filter, model_category):
    if RESULTS_DF is None or RESULTS_DF.empty:
        return go.Figure()
    
    df = RESULTS_DF.copy() if dataset_filter == 'all' else RESULTS_DF[RESULTS_DF['dataset'] == dataset_filter].copy()
    if model_category != 'all':
        df = df[df['model'].isin([m for m, cat in MODEL_CATEGORIES.items() if cat == model_category])]
    
    agg = df.groupby('model').agg({'test_accuracy': 'mean', 'runtime_seconds': 'mean'}).reset_index()
    colors = ['#8b5cf6' if MODEL_CATEGORIES.get(m) == 'gnn' else '#ec4899' if MODEL_CATEGORIES.get(m) == 'neural' else '#0ea5e9' for m in agg['model']]
    
    fig = go.Figure(go.Scatter(x=agg['runtime_seconds'], y=agg['test_accuracy'], mode='markers+text',
                               marker=dict(size=20, color=colors, line=dict(width=2, color='#ffffff')),
                               text=agg['model'], textposition='top center', textfont=dict(family='DM Sans', size=11, color='#0f172a')))
    fig.update_layout(xaxis=dict(title='Average Runtime (seconds)', gridcolor='#e2e8f0', type='log'),
                      yaxis=dict(title='Test Accuracy', tickformat='.0%', gridcolor='#e2e8f0'),
                      plot_bgcolor='#ffffff', paper_bgcolor='rgba(0,0,0,0)', font=dict(family='DM Sans', color='#0f172a'),
                      margin=dict(l=60, r=40, t=40, b=60), height=400, showlegend=False)
    return fig


@app.callback(Output('dataset-comparison-chart', 'figure'), Input('model-category', 'value'))
def update_dataset_comparison(model_category):
    if RESULTS_DF is None or RESULTS_DF.empty:
        return go.Figure()
    
    df = RESULTS_DF.copy()
    if model_category != 'all':
        df = df[df['model'].isin([m for m, cat in MODEL_CATEGORIES.items() if cat == model_category])]
    
    agg = df.groupby(['model', 'dataset']).agg({'test_accuracy': 'mean'}).reset_index()
    model_order_filtered = [m for m in MODEL_ORDER if m in agg['model'].unique()]
    
    fig = go.Figure()
    for dataset_key, dataset_info in DATASETS.items():
        subset = agg[agg['dataset'] == dataset_key].set_index('model').reindex(model_order_filtered).reset_index()
        fig.add_trace(go.Bar(name=f"{dataset_info['emoji']} {dataset_info['full_name']}", x=subset['model'], y=subset['test_accuracy'],
                             marker_color=dataset_info['color'], text=[f"{v:.1%}" if pd.notna(v) else '' for v in subset['test_accuracy']],
                             textposition='outside', textfont=dict(family='JetBrains Mono', size=10)))
    
    fig.update_layout(barmode='group', xaxis=dict(title='', gridcolor='#e2e8f0', categoryorder='array', categoryarray=model_order_filtered),
                      yaxis=dict(title='Test Accuracy', tickformat='.0%', gridcolor='#e2e8f0', range=[0, 1]),
                      plot_bgcolor='#ffffff', paper_bgcolor='rgba(0,0,0,0)', font=dict(family='DM Sans', color='#0f172a'),
                      margin=dict(l=60, r=40, t=20, b=60), height=350, legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1, bgcolor='rgba(0,0,0,0)'))
    return fig


# --- Explorer callbacks ---

@app.callback(Output('ratio-value', 'children'), Input('node-sample-ratio', 'value'))
def update_ratio_display(value):
    return f'p = {value}'


@app.callback(Output('threshold-value', 'children'), Input('adjacency-threshold', 'value'))
def update_threshold_display(value):
    return f'τ = {value:.2f}'


@app.callback(
    Output('explorer-chart', 'figure'),
    [Input('node-sample-ratio', 'value'), Input('node-selection-method', 'value'),
     Input('adjacency-threshold', 'value'), Input('dataset-selector', 'value')]
)
def update_explorer_visualization(node_sample_ratio, method, adj_threshold, selected_datasets):
    if not selected_datasets:
        selected_datasets = ['motrpac']
    
    all_stats = []
    for ds_name in selected_datasets:
        stats = get_stats(ds_name, node_sample_ratio, method, adj_threshold)
        if stats:
            all_stats.append({**stats, 'dataset': ds_name})
    
    metrics = [
        ('n_nodes', 'Graph Size (Nodes)'), ('n_edges', 'Graph Connectivity (Edges)'),
        ('mean_degree', 'Average Node Degree'), ('density', 'Graph Density (%)'),
        ('avg_clustering', 'Avg Clustering Coefficient'), ('largest_cc_ratio', 'Largest CC / Total Nodes (%)'),
        ('avg_path_length', 'Avg Shortest Path Length'), ('n_components', 'Connected Components'),
        ('std_degree', 'Degree Distribution Std Dev'),
    ]
    
    fig = make_subplots(rows=3, cols=3, subplot_titles=[m[1] for m in metrics], vertical_spacing=0.16, horizontal_spacing=0.08)
    x_labels = [f"{DATASETS[d]['emoji']} {DATASETS[d]['full_name']}" for d in ['motrpac', 'addneuromed', 'parkinsons']]
    
    for idx, (metric, _) in enumerate(metrics):
        row, col = idx // 3 + 1, idx % 3 + 1
        y_values, colors, text_values = [], [], []
        
        for ds_name in ['motrpac', 'addneuromed', 'parkinsons']:
            stat = next((s for s in all_stats if s['dataset'] == ds_name), None)
            if stat:
                v = stat[metric]
                y_values.append(v)
                colors.append(DATASETS[ds_name]['color'])
                if metric in ['density', 'largest_cc_ratio']:
                    text_values.append(f'{v:.1f}%')
                elif metric in ['avg_clustering', 'avg_path_length']:
                    text_values.append(f'{v:.2f}')
                elif metric in ['n_nodes', 'n_edges', 'n_components']:
                    text_values.append(f'{int(v):,}')
                else:
                    text_values.append(f'{v:.1f}')
            else:
                y_values.append(0)
                colors.append('rgba(200,200,200,0.3)')
                text_values.append('')
        
        fig.add_trace(go.Bar(x=x_labels, y=y_values, marker_color=colors, text=text_values, textposition='outside',
                             textfont=dict(size=14, family='JetBrains Mono', color='#0f172a'), showlegend=False, cliponaxis=False), row=row, col=col)
        
        if metric in METRIC_MAX_VALUES:
            fig.update_yaxes(range=[0, METRIC_MAX_VALUES[metric]], row=row, col=col)
    
    fig.update_layout(height=850, font=dict(family='DM Sans', size=14, color='#0f172a'), paper_bgcolor='rgba(0,0,0,0)',
                      plot_bgcolor='#ffffff', margin=dict(l=50, r=30, t=60, b=70))
    
    for annotation in fig['layout']['annotations']:
        annotation['font'] = dict(size=14, color='#475569', family='DM Sans')
        annotation['y'] = annotation['y'] + 0.02
    
    fig.update_xaxes(showgrid=False, tickangle=-35, tickfont=dict(size=11, color='#475569'), fixedrange=True, categoryorder='array', categoryarray=x_labels)
    fig.update_yaxes(showgrid=True, gridcolor='rgba(226,232,240,0.8)', zeroline=True, zerolinecolor='rgba(226,232,240,1)', tickfont=dict(size=12, color='#475569'), fixedrange=True)
    
    return fig


# =============================================================================
# Startup
# =============================================================================

print('\n' + '='*60)
print('OgBench - Omics Graph Benchmark')
print('='*60)

print('\nLoading data...')
load_results_data()
load_stats_data()

if RESULTS_DF is not None and not RESULTS_DF.empty:
    print(f'\n✓ Leaderboard: {len(RESULTS_DF)} results, {RESULTS_DF["model"].nunique()} models')

if PRECOMPUTED_STATS:
    print(f'✓ Explorer: {len(PRECOMPUTED_STATS)} graph configurations')

print('='*60 + '\n')


if __name__ == '__main__':
    print('Starting OgBench...')
    print('Open http://127.0.0.1:8050 in your browser\n')
    print('Pages:')
    print('  - Leaderboard: http://127.0.0.1:8050/')
    print('  - Dataset Explorer: http://127.0.0.1:8050/explorer\n')
    
    app.run(debug=False, host='0.0.0.0', port=8050)
