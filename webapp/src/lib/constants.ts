import type {
  DatasetInfo,
  DatasetName,
  LeaderboardDatasetName,
  ModelCategory,
  RankingMetric,
  DisplayMetric,
  AdjacencyMethod,
  StatsMetric,
} from './types';

export const DATASETS: Record<DatasetName, DatasetInfo> = {
  motrpac: { fullName: 'Heritage', color: '#3b82f6', emoji: '🧬' },
  addneuromed: { fullName: 'AddNeuroMed', color: '#f97316', emoji: '🧠' },
  parkinsons: { fullName: "Parkinson's", color: '#22c55e', emoji: '🔬' },
  brca: { fullName: 'BRCA', color: '#a855f7', emoji: '🎗️' },
  tuberculosis: { fullName: 'Tuberculosis', color: '#14b8a6', emoji: '🦠' },
  smoking: { fullName: 'Smoking', color: '#ef4444', emoji: '🚬' },
};

/** Datasets shown in the Dataset Explorer (all datasets with graph statistics). */
export const EXPLORER_DATASETS: readonly DatasetName[] = [
  'motrpac',
  'addneuromed',
  'parkinsons',
  'brca',
  'tuberculosis',
  'smoking',
] as const;

/** Datasets with benchmark results on the Leaderboard. */
export const LEADERBOARD_DATASETS: readonly LeaderboardDatasetName[] = [
  'motrpac',
  'addneuromed',
  'parkinsons',
  'brca',
] as const;

export const MODEL_CATEGORIES: Record<string, ModelCategory> = {
  'MLA-GNN': 'gnn',
  GATv2: 'gnn',
  GCN: 'gnn',
  GIN: 'gnn',
  GraphSAGE: 'gnn',
  SAGN: 'gnn',
  ChebNet: 'gnn',
  GPS: 'gnn',
  MLP: 'neural',
  ElasticNet: 'baseline',
  SVM: 'baseline',
};

export const MODEL_ORDER = [
  'MLP',
  'GIN',
  'GCN',
  'GATv2',
  'GraphSAGE',
  'SAGN',
  'ChebNet',
  'GPS',
  'MLA-GNN',
];

export const BASELINE_MODELS = ['ElasticNet', 'SVM'];

export const MODEL_COLORS: Record<string, string> = {
  MLP: '#6baed6',
  GIN: '#e41a1c',
  GCN: '#ff7f00',
  GATv2: '#ffab40',
  GraphSAGE: '#ffd54f',
  SAGN: '#984ea3',
  ChebNet: '#4d9221',
  GPS: '#17becf',
  'MLA-GNN': '#a6d96a',
  ElasticNet: '#000000',
  SVM: '#000000',
};

export const VALID_RATIOS = [0.3, 0.5, 0.8, 1.0] as const;
export const VALID_METHODS = ['variance', 'correlation', 'distance_correlation', 'random'] as const;
export const VALID_READOUTS = ['NoReadOut', 'OmicsReadOut'] as const;
export const VALID_ADJACENCY_METHODS: readonly AdjacencyMethod[] = ['string', 'wgcna'] as const;

export const ADJACENCY_METHOD_LABELS: Record<string, string> = {
  string: 'PPI Network',
  wgcna: 'Co-expression (WGCNA)',
};

export const METHOD_LABELS: Record<string, string> = {
  variance: 'Variance',
  correlation: 'Correlation',
  distance_correlation: 'Distance Correlation',
  random: 'Random',
};

export const RATIO_LABELS: Record<number, string> = {
  0.3: '0.3',
  0.5: '0.5',
  0.8: '0.8',
  1.0: '1.0',
};

export const VALID_THRESHOLDS = [0.02, 0.1, 0.2, 0.3, 0.4, 0.5];

export const CATEGORY_COLORS: Record<ModelCategory, string> = {
  gnn: '#8b5cf6',
  neural: '#ec4899',
  baseline: '#0ea5e9',
};

export const METRIC_LABELS: Record<StatsMetric, string> = {
  num_nodes: 'Number of Nodes',
  num_edges: 'Number of Edges',
  avg_degree: 'Average Node Degree',
  density_pct: 'Graph Density (%)',
  degree_std: 'Degree Standard Deviation',
  largest_cc_ratio_pct: 'Largest Connected Component (%)',
  num_connected_components: 'Connected Components',
  clustering_coefficient: 'Clustering Coefficient (LCC)',
  diameter: 'Diameter (LCC)',
  modularity: 'Modularity (LCC)',
  homophily: 'Feature Homophily',
};

/** Metrics shown in the Explorer grid, 3 per row. */
export const EXPLORER_METRICS: readonly StatsMetric[] = [
  'num_nodes',
  'num_edges',
  'avg_degree',
  'density_pct',
  'degree_std',
  'largest_cc_ratio_pct',
  'clustering_coefficient',
  'homophily',
  'modularity',
] as const;

export const RANKING_METRICS: Record<RankingMetric, string> = {
  val_f1_macro: 'Val F1 Macro',
  test_f1_macro: 'Test F1 Macro',
};

export const DISPLAY_METRICS: Record<DisplayMetric, string> = {
  test_f1_macro: 'Test F1 Macro',
  train_f1_macro: 'Train F1 Macro',
  test_f1_weighted: 'Test F1 Weighted',
  test_accuracy: 'Test Accuracy',
  test_auroc: 'Test AUROC',
};
