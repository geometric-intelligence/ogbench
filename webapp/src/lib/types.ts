export interface ResultEntry {
  graph_config: string;
  model: string;
  dataset: string;
  test_accuracy: number;
  f1_macro: number;
  f1_weighted: number;
  precision_macro: number;
  recall_macro: number;
  runtime_seconds: number;
  epochs: number;
}

export interface GraphStats {
  num_nodes: number;
  num_edges: number;
  density_pct: number;
  avg_degree: number;
  degree_std: number;
  num_connected_components: number;
  largest_cc_ratio_pct: number;
  avg_clustering_coeff: number;
  avg_shortest_path_length: number;
  dataset: string;
}

export interface LeaderboardEntry {
  rank: number;
  model: string;
  category: ModelCategory;
  accuracy: number;
  accStd: number;
  f1Macro: number;
  f1Std: number;
  avgRuntime: number;
  totalRuntime: number;
}

export type ModelCategory = 'gnn' | 'neural' | 'baseline';

export interface DatasetInfo {
  fullName: string;
  color: string;
  emoji: string;
}

export type DatasetName = 'motrpac' | 'addneuromed' | 'parkinsons';

export type NodeSelectionMethod = 'variance' | 'correlation' | 'distance_correlation' | 'random';
