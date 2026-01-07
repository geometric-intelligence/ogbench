#!/usr/bin/env python3
"""Interactive Dash app for exploring BGBench omics datasets.

This app visualizes precomputed graph statistics for the three omics datasets
(MotrPac, AddNeuroMed, Parkinsons) with interactive controls.

Note: This app requires precomputed stats. Run `python precompute_stats.py` first.
"""

import json
from pathlib import Path

import dash
import plotly.graph_objects as go
from dash import dcc, html
from dash.dependencies import Input, Output
from plotly.subplots import make_subplots

# Dataset configurations
DATASETS = {
    'motrpac': {
        'full_name': 'MotrPac',
        'color': '#3b82f6',  # Blue
    },
    'addneuromed': {
        'full_name': 'AddNeuroMed',
        'color': '#f97316',  # Orange
    },
    'parkinsons': {
        'full_name': 'Parkinson\'s',
        'color': '#22c55e',  # Green
    },
}

# Valid parameter values (must match precomputed stats)
# Note: p=1.0 excluded because it was capped at 1000 nodes during precomputation
VALID_RATIOS = [0.5, 0.6, 0.7, 0.8, 0.9]
VALID_METHODS = ['variance', 'correlation', 'random']
VALID_THRESHOLDS = [0.02, 0.1, 0.2, 0.3, 0.4, 0.5]

# Precomputed stats loaded at startup
PRECOMPUTED_STATS = {}
METRIC_MAX_VALUES = {}

# Path to precomputed stats file
PRECOMPUTED_STATS_FILE = Path(__file__).parent / 'precomputed_stats.json'


def load_precomputed_stats():
    """Load precomputed stats from JSON file."""
    global PRECOMPUTED_STATS, METRIC_MAX_VALUES
    
    if not PRECOMPUTED_STATS_FILE.exists():
        print(f'ERROR: No precomputed stats found at {PRECOMPUTED_STATS_FILE}')
        print('  Run: python precompute_stats.py to generate them')
        return
    
    print(f'Loading precomputed stats from {PRECOMPUTED_STATS_FILE}...')
    with open(PRECOMPUTED_STATS_FILE, 'r') as f:
        PRECOMPUTED_STATS = json.load(f)
    
    # Compute max values for each metric (for fixed y-axis ranges)
    metrics = ['n_nodes', 'n_edges', 'mean_degree', 'density', 'avg_clustering', 
               'largest_cc_ratio', 'avg_path_length', 'n_components', 'std_degree']
    for m in metrics:
        vals = [v[m] for v in PRECOMPUTED_STATS.values()]
        METRIC_MAX_VALUES[m] = max(vals) * 1.20 if vals else 1  # 20% padding for text labels
    
    print(f'  Loaded {len(PRECOMPUTED_STATS)} precomputed combinations')


def get_stats(dataset_name: str, ratio: float, method: str, threshold: float) -> dict | None:
    """Get precomputed statistics for given parameters."""
    # Build key with consistent float formatting to match JSON keys
    ratio_str = f'{float(ratio)}'  # Ensure "1" becomes "1.0"
    threshold_str = f'{float(threshold)}'  # Ensure "0.02" format
    key = f'{dataset_name}|{ratio_str}|{method}|{threshold_str}'
    return PRECOMPUTED_STATS.get(key)


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
                        max=0.9,
                        step=None,
                        value=0.5,  # Start with lowest p (most nodes)
                        marks={v: f'{v}' for v in VALID_RATIOS},
                        included=False,
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
                        min=0.02,
                        max=0.5,
                        step=None,
                        value=0.02,
                        marks={v: f'{v}' for v in VALID_THRESHOLDS},
                        included=False,
                    ),
                ], className='control-group'),
                
                # Dataset Selector (colored labels)
                html.Div([
                    html.Div('Datasets', className='control-label'),
                    dcc.Checklist(
                        id='dataset-selector',
                        options=[
                            {'label': html.Span([' MotrPac'], style={'fontWeight': '600', 'color': DATASETS['motrpac']['color']}), 'value': 'motrpac'},
                            {'label': html.Span([' AddNeuroMed'], style={'fontWeight': '600', 'color': DATASETS['addneuromed']['color']}), 'value': 'addneuromed'},
                            {'label': html.Span([' Parkinson\'s'], style={'fontWeight': '600', 'color': DATASETS['parkinsons']['color']}), 'value': 'parkinsons'},
                        ],
                        value=['motrpac', 'addneuromed', 'parkinsons'],
                        inline=True,
                        style={'display': 'flex', 'gap': '16px', 'flexWrap': 'wrap', 'marginTop': '4px'}
                    ),
                ], className='control-group'),
            ], className='controls-grid'),
        ], className='controls-panel'),
        
        # Charts container (no loading indicator for smoother transitions)
        html.Div([
            dcc.Graph(
                id='main-chart',
                config={'displayModeBar': True, 'responsive': True},
                # Disable default loading state for smoother animations
                style={'height': '850px'}
            ),
        ], className='charts-container'),
        
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
    
    # Get stats for selected datasets
    all_stats = []
    for ds_name in selected_datasets:
        stats = get_stats(ds_name, node_sample_ratio, method, adj_threshold)
        if stats:
            all_stats.append({**stats, 'dataset': ds_name})
    
    # Define metrics to display
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
    
    # Create subplot figure
    fig = make_subplots(
        rows=3, cols=3,
        subplot_titles=[m[1] for m in metrics],
        vertical_spacing=0.16,
        horizontal_spacing=0.08,
    )
    
    # Fixed x-axis labels (always all 3 datasets)
    x_labels = [DATASETS[d]['full_name'] for d in ['motrpac', 'addneuromed', 'parkinsons']]
    
    for idx, (metric, _) in enumerate(metrics):
        row = idx // 3 + 1
        col = idx % 3 + 1
        
        # Build y-values for all 3 datasets
        y_values = []
        colors = []
        text_values = []
        
        for ds_name in ['motrpac', 'addneuromed', 'parkinsons']:
            stat = next((s for s in all_stats if s['dataset'] == ds_name), None)
            if stat:
                v = stat[metric]
                y_values.append(v)
                colors.append(DATASETS[ds_name]['color'])
                # Format text label
                if metric in ['density', 'largest_cc_ratio']:
                    text_values.append(f'{v:.1f}%')
                elif metric in ['avg_clustering', 'avg_path_length']:
                    text_values.append(f'{v:.2f}')
                elif metric in ['n_nodes', 'n_edges', 'n_components']:
                    text_values.append(f'{int(v):,}')
                else:
                    text_values.append(f'{v:.1f}')
            else:
                # Dataset not selected - show zero height bar
                y_values.append(0)
                colors.append('rgba(200,200,200,0.3)')
                text_values.append('')
        
        fig.add_trace(
            go.Bar(
                x=x_labels,
                y=y_values,
                marker_color=colors,
                text=text_values,
                textposition='outside',
                textfont=dict(size=14, family='JetBrains Mono', weight='bold'),
                showlegend=False,
                cliponaxis=False,
            ),
            row=row, col=col
        )
        
        # Fixed y-axis range (enables smooth bar height transitions)
        if metric in METRIC_MAX_VALUES:
            fig.update_yaxes(range=[0, METRIC_MAX_VALUES[metric]], row=row, col=col)
    
    # Layout settings
    fig.update_layout(
        height=850,
        font=dict(family='Avenir, Avenir Next, -apple-system, BlinkMacSystemFont, Segoe UI, Helvetica, sans-serif', size=14),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(248,250,252,1)',
        margin=dict(l=50, r=30, t=60, b=70),  # Extra bottom margin for diagonal labels
    )
    
    # Style subplot titles
    for annotation in fig['layout']['annotations']:
        annotation['font'] = dict(size=14, color='#374151', family='Avenir, Avenir Next, sans-serif', weight='bold')
        annotation['y'] = annotation['y'] + 0.02
    
    # Style axes (fixed ranges for smooth transitions)
    fig.update_xaxes(
        showgrid=False,
        tickangle=-35,  # Diagonal labels to avoid overlap
        tickfont=dict(size=11, family='Avenir, Avenir Next, sans-serif'),
        fixedrange=True,
        categoryorder='array',
        categoryarray=x_labels,  # Lock x-axis order
    )
    fig.update_yaxes(
        showgrid=True,
        gridcolor='rgba(226,232,240,0.6)',
        zeroline=True,
        zerolinecolor='rgba(226,232,240,0.8)',
        tickfont=dict(size=12, family='Avenir, Avenir Next, sans-serif'),
        fixedrange=True,
    )
    
    return fig


# Startup
print('\n' + '='*60)
print('BGBench Dataset Explorer')
print('='*60)

print('\nLoading precomputed statistics...')
load_precomputed_stats()

if not PRECOMPUTED_STATS:
    print('\n⚠️  ERROR: No precomputed stats found!')
    print('   Run: python precompute_stats.py to generate them')
else:
    print(f'\n✓ Loaded {len(PRECOMPUTED_STATS)} precomputed combinations')
    print(f'✓ Valid ratios: {VALID_RATIOS}')
    print(f'✓ Valid methods: {VALID_METHODS}')
    print(f'✓ Valid thresholds: {VALID_THRESHOLDS}')

print('='*60 + '\n')


# Run the app
if __name__ == '__main__':
    print('Starting Dash server...')
    print('Open http://127.0.0.1:8050 in your browser\n')
    
    app.run(debug=False, host='0.0.0.0', port=8050)
