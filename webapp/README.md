# OgBench Webapp

Static website for the OgBench (Omics Graph Benchmark) leaderboard and dataset explorer. Built with Astro, React, and Plotly.js.

## Architecture

```
webapp/
├── public/data/           # JSON data files served at /data/*
│   ├── results.json       # Benchmark results for all models
│   └── stats.json         # Graph statistics for all configurations
├── src/
│   ├── pages/             # Astro pages (routes)
│   │   ├── index.astro    # Leaderboard (/)
│   │   └── explorer.astro # Dataset Explorer (/explorer)
│   ├── components/        # UI components
│   │   ├── Leaderboard.tsx   # React: filters, table, charts
│   │   ├── Explorer.tsx      # React: sliders, 9-chart grid
│   │   ├── Logo.tsx          # React: SVG logo
│   │   ├── Header.astro      # Astro: page header
│   │   └── Footer.astro      # Astro: page footer
│   ├── lib/               # Utilities
│   │   ├── types.ts       # TypeScript interfaces
│   │   ├── constants.ts   # Datasets, models, colors
│   │   └── data.ts        # Data processing functions
│   ├── layouts/
│   │   └── Layout.astro   # Base HTML layout
│   └── styles/
│       └── global.css     # Tailwind + custom styles
└── dist/                  # Build output (static files)
```

### How It Works

1. **Build time**: Astro generates static HTML for each page
2. **Runtime**: React components hydrate client-side with `client:only="react"`
3. **Data**: JSON files in `public/data/` are fetched by React components on mount
4. **Charts**: Plotly.js renders interactive visualizations

The site is fully static — no server required. All interactivity runs in the browser.

## Development

```bash
# Install dependencies
curl -fsSL https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.7/install.sh | bash
source ~/.bashrc
source ~/.profile
nvm install --lts
make install

# Start dev server (http://localhost:4321)
make dev

# Type check
make lint

# Build for production
make build

# Preview production build
make preview
```

## Deployment

Deploy to Cloudflare Pages:

```bash
make deploy
```

This runs `wrangler pages deploy dist/ --project-name ogbench`.

### Manual Deployment

The `dist/` directory contains all static files. Deploy to any static host:

- **Cloudflare Pages**: `npx wrangler pages deploy dist/`
- **Vercel**: `vercel deploy dist/`
- **Netlify**: drag & drop `dist/` folder
- **GitHub Pages**: copy `dist/` contents to repo
- **Any web server**: serve `dist/` as static files

## Updating Data

### Regenerating Graph Statistics

The `tutorials/dataset_stats_analysis.py` script computes graph statistics for all parameter combinations and outputs `stats.json` for the webapp.

**Prerequisites:**

The script requires the full `ogbench` environment with PyTorch:

```bash
# Install PyTorch (see https://pytorch.org for platform-specific instructions)
pip install torch

# Install other dependencies
pip install torch-geometric networkx numpy pandas scikit-learn huggingface_hub joblib omegaconf

# Install ogbench package (from repo root)
pip install -e .
```

**Run the script:**

```bash
cd tutorials
python dataset_stats_analysis.py
```

**Note:** If the environment is not properly set up, some selection methods (like `distance_correlation`) may not have data available in the webapp.

This will:
1. Load datasets using the HFOmicsDataset loader
2. Compute graph statistics for all parameter combinations
3. Save results to both CSV files (in `./stats/`) and JSON for the webapp (`webapp/public/data/stats.json`)

**Parameters computed:**

| Parameter | Values |
|-----------|--------|
| Datasets | `motrpac`, `addneuromed`, `parkinsons`, `covidaki` |
| Node sample ratios | `full`, `1.0`, `0.5`, `0.3` |
| Selection methods | `variance`, `correlation`, `distance_correlation`, `random` |
| Adjacency thresholds | 10 values from 0.0 to 1.0 |

**Metrics computed per graph:**

- `num_nodes` — Number of nodes (features)
- `num_edges` — Number of edges
- `density_pct` — Graph density (%)
- `avg_degree` — Average node degree
- `degree_std` — Standard deviation of degrees
- `num_connected_components` — Number of connected components
- `largest_cc_ratio_pct` — Largest connected component / total nodes (%)
- `avg_clustering_coeff` — Average clustering coefficient
- `avg_shortest_path_length` — Average shortest path length

**Expected runtime:** Varies based on number of parallel jobs and dataset sizes.

### JSON File Structure

**`public/data/results.json`** — Benchmark results:

```json
{
  "motrpac|0.5|variance|0.02|SVM": {
    "graph_config": "motrpac|0.5|variance|0.02",
    "model": "SVM",
    "dataset": "motrpac",
    "test_accuracy": 0.8549,
    "f1_macro": 0.8159,
    "f1_weighted": 0.8198,
    "precision_macro": 0.8121,
    "recall_macro": 0.7921,
    "runtime_seconds": 5.46,
    "epochs": 1
  }
}
```

**`public/data/stats.json`** — Graph statistics:

```json
{
  "motrpac|0.5|variance|0.02": {
    "num_nodes": 914,
    "num_edges": 135087,
    "density_pct": 32.37,
    "avg_degree": 295.59,
    "degree_std": 240.65,
    "num_connected_components": 172,
    "largest_cc_ratio_pct": 77.68,
    "avg_clustering_coeff": 0.72,
    "avg_shortest_path_length": 2.59,
    "dataset": "motrpac"
  }
}
```

### How to Update

1. **Replace the JSON files** in `public/data/`:
   ```bash
   cp /path/to/new/results.json public/data/results.json
   cp /path/to/new/stats.json public/data/stats.json
   ```

2. **Rebuild and deploy**:
   ```bash
   make deploy
   ```

### Key Format

Results key: `{dataset}|{ratio}|{method}|{threshold}|{model}`
Stats key: `{dataset}|{ratio}|{method}|{threshold}`

Where:
- `dataset`: `motrpac`, `addneuromed`, `parkinsons`, or `covidaki`
- `ratio`: node sample ratio (`full`, `1.0`, `0.5`, `0.3`)
- `method`: `variance`, `correlation`, `distance_correlation`, or `random`
- `threshold`: adjacency threshold (0.0–1.0)
- `model`: `SVM`, `ElasticNet`, `MLP`, `GATv4`, `GATv2`, `GIN`, `GCN`, `GraphSAGE`, `SAGN`, `Random`

## Tech Stack

- **[Astro](https://astro.build/)** — Static site generator
- **[React](https://react.dev/)** — Interactive components (islands)
- **[Plotly.js](https://plotly.com/javascript/)** — Charts
- **[Tailwind CSS](https://tailwindcss.com/)** — Styling
- **[Cloudflare Pages](https://pages.cloudflare.com/)** — Hosting
