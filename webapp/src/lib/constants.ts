import type { DatasetInfo, DatasetName, ModelCategory } from './types';

export const DATASETS: Record<DatasetName, DatasetInfo> = {
  motrpac: { fullName: 'MotrPac', color: '#3b82f6', emoji: '🧬' },
  addneuromed: { fullName: 'AddNeuroMed', color: '#f97316', emoji: '🧠' },
  parkinsons: { fullName: "Parkinson's", color: '#22c55e', emoji: '🔬' },
};

export const MODEL_CATEGORIES: Record<string, ModelCategory> = {
  GATv4: 'gnn',
  GATv2: 'gnn',
  GCN: 'gnn',
  GIN: 'gnn',
  GraphSAGE: 'gnn',
  SAGN: 'gnn',
  MLP: 'neural',
  Random: 'baseline',
  ElasticNet: 'baseline',
  SVM: 'baseline',
};

export const MODEL_ORDER = [
  'SVM',
  'ElasticNet',
  'MLP',
  'GATv4',
  'GATv2',
  'GIN',
  'GCN',
  'GraphSAGE',
  'SAGN',
  'Random',
];

export const VALID_RATIOS = [0.5, 0.6, 0.7, 0.8, 0.9] as const;
export const VALID_METHODS = ['variance', 'correlation', 'distance_correlation', 'random'] as const;
export const VALID_THRESHOLDS = [0.02, 0.1, 0.2, 0.3, 0.4, 0.5];

export const CATEGORY_COLORS: Record<ModelCategory, string> = {
  gnn: '#8b5cf6',
  neural: '#ec4899',
  baseline: '#0ea5e9',
};

export const METRIC_LABELS: Record<string, string> = {
  num_nodes: 'Number of Nodes',
  num_edges: 'Number of Edges',
  avg_degree: 'Average Node Degree',
  density_pct: 'Graph Density (%)',
  avg_clustering_coeff: 'Average Clustering Coefficient',
  largest_cc_ratio_pct: 'Largest Connected Component (%)',
  avg_shortest_path_length: 'Average Shortest Path Length',
  num_connected_components: 'Connected Components',
  degree_std: 'Degree Standard Deviation',
};
