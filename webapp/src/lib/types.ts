export interface ResultEntry {
  graph_config: string;
  model: string;
  dataset: string;
  readout: string; // 'NoReadOut' | 'OmicsReadOut' | 'baseline'
  node_sample_ratio: number;
  method: string;
  // Validation metrics (for ranking/selection)
  val_accuracy: number;
  val_accuracy_std: number;
  val_f1_macro: number;
  val_f1_macro_std: number;
  // Test metrics (for display/evaluation)
  test_accuracy: number;
  test_accuracy_std: number;
  test_f1_macro: number;
  test_f1_macro_std: number;
  test_f1_weighted: number;
  test_f1_weighted_std: number;
  auroc: number;
  auroc_std: number;
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
  // Ranking metric values (what determines order)
  rankValue: number;
  rankStd: number;
  // Display metric values (what's shown)
  displayValue: number;
  displayStd: number;
  // Additional metrics for charts
  testAccuracy: number;
  testAccuracyStd: number;
  testF1Macro: number;
  testF1MacroStd: number;
  auroc: number;
  aurocStd: number;
  isBaseline: boolean;
}

export type ModelCategory = 'gnn' | 'neural' | 'baseline';

export interface DatasetInfo {
  fullName: string;
  color: string;
  emoji: string;
}

export type DatasetName = 'motrpac' | 'addneuromed' | 'parkinsons';

export type NodeSelectionMethod = 'variance' | 'correlation' | 'distance_correlation' | 'random';

// Metrics for ranking models (validation metrics)
export type RankingMetric = 'val_accuracy' | 'val_f1_macro';

// Metrics for displaying model performance (test metrics)
export type DisplayMetric = 'test_accuracy' | 'test_f1_macro' | 'auroc';
