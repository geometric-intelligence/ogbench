export type AdjacencyMethod = 'string' | 'wgcna';

export interface ResultEntry {
  graph_config: string;
  model: string;
  dataset: string;
  readout: string; // 'NoReadOut' | 'OmicsReadOut' | 'baseline'
  node_sample_ratio: number;
  method: string;
  adjacency_method?: string;
  val_f1_macro: number;
  val_f1_macro_std: number;
  test_f1_macro: number;
  test_f1_macro_std: number;
  train_f1_macro: number;
  train_f1_macro_std: number;
  test_f1_weighted: number;
  test_f1_weighted_std: number;
  test_accuracy: number;
  test_accuracy_std: number;
  test_auroc: number;
  test_auroc_std: number;
}

export interface GraphStats {
  num_nodes: number;
  num_edges: number;
  density_pct: number;
  avg_degree: number;
  degree_std: number;
  num_connected_components: number;
  largest_cc_ratio_pct: number;
  /** Approximate average clustering coefficient of the largest connected component. */
  clustering_coefficient: number | null;
  /** Approximate (2-sweep) diameter of the largest connected component. */
  diameter: number | null;
  /** Louvain modularity of the largest connected component. */
  modularity: number | null;
  /** Mean cosine similarity of train-averaged node features along edges. */
  homophily: number | null;
  dataset: string;
  adjacency_method?: string;
}

export type StatsMetric = Exclude<keyof GraphStats, 'dataset' | 'adjacency_method'>;

/** Sample-to-node ratio; 'full' keeps every feature as a node. */
export type StatsRatio = number | 'full';

export interface LeaderboardEntry {
  rank: number;
  model: string;
  category: ModelCategory;
  rankValue: number;
  rankStd: number;
  displayValue: number;
  displayStd: number;
  testF1Macro: number;
  testF1MacroStd: number;
  isBaseline: boolean;
}

export type ModelCategory = 'gnn' | 'neural' | 'baseline';

export interface DatasetInfo {
  fullName: string;
  color: string;
  emoji: string;
}

export type DatasetName = 'motrpac' | 'addneuromed' | 'parkinsons' | 'brca' | 'tuberculosis' | 'smoking';

/** Datasets with benchmark results in results.json (shown on the Leaderboard). */
export type LeaderboardDatasetName = 'motrpac' | 'addneuromed' | 'parkinsons' | 'brca';

export type NodeSelectionMethod = 'variance' | 'correlation' | 'distance_correlation' | 'random';

export type SampleNodeRatio = 0.3 | 0.5 | 0.8 | 1.0;

export type RankingMetric = 'val_f1_macro' | 'test_f1_macro';

export type DisplayMetric = 'test_f1_macro' | 'train_f1_macro' | 'test_f1_weighted' | 'test_accuracy' | 'test_auroc';
