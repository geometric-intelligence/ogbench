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

The `precompute_stats.py` script computes graph statistics for all parameter combinations and outputs `stats.json`.

**Prerequisites:**

```bash
pip install networkx numpy pandas scikit-learn huggingface_hub
```

**Run the script:**

```bash
cd webapp
python precompute_stats.py
```

This will:
1. Download datasets from HuggingFace (`geometric-intelligence/bgbench`)
2. Compute graph statistics for all 324 combinations:
   - 3 datasets × 6 ratios × 3 methods × 6 thresholds
3. Save results to `public/data/stats.json`

**Parameters computed:**

| Parameter | Values |
|-----------|--------|
| Datasets | `motrpac`, `addneuromed`, `parkinsons` |
| Node sample ratios | 0.5, 0.6, 0.7, 0.8, 0.9, 1.0 |
| Selection methods | `variance`, `correlation`, `random` |
| Adjacency thresholds | 0.02, 0.1, 0.2, 0.3, 0.4, 0.5 |

**Metrics computed per graph:**

- `n_nodes` — Number of nodes (features)
- `n_edges` — Number of edges
- `density` — Graph density (%)
- `mean_degree` — Average node degree
- `std_degree` — Standard deviation of degrees
- `n_components` — Number of connected components
- `largest_cc_ratio` — Largest connected component / total nodes (%)
- `avg_clustering` — Average clustering coefficient
- `avg_path_length` — Average shortest path length (sampled for large graphs)

**Expected runtime:** ~2-5 minutes (depends on network speed for initial download).

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
    "n_nodes": 914,
    "n_edges": 135087,
    "density": 32.37,
    "mean_degree": 295.59,
    "std_degree": 240.65,
    "n_components": 172,
    "largest_cc_ratio": 77.68,
    "avg_clustering": 0.72,
    "avg_path_length": 2.59,
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
- `dataset`: `motrpac`, `addneuromed`, or `parkinsons`
- `ratio`: node sample ratio (0.5–0.9)
- `method`: `variance`, `correlation`, or `random`
- `threshold`: adjacency threshold (0.02–0.5)
- `model`: `SVM`, `ElasticNet`, `MLP`, `GATv4`, `GATv2`, `GIN`, `GCN`, `GraphSAGE`, `SAGN`, `Random`

## Tech Stack

- **[Astro](https://astro.build/)** — Static site generator
- **[React](https://react.dev/)** — Interactive components (islands)
- **[Plotly.js](https://plotly.com/javascript/)** — Charts
- **[Tailwind CSS](https://tailwindcss.com/)** — Styling
- **[Cloudflare Pages](https://pages.cloudflare.com/)** — Hosting
