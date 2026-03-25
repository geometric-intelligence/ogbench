import type { ResultEntry, GraphStats, LeaderboardEntry, DatasetName, RankingMetric, DisplayMetric } from './types';
import { MODEL_CATEGORIES } from './constants';

// Helper to compute mean of array
function mean(arr: number[]): number {
  if (arr.length === 0) return 0;
  return arr.reduce((a, b) => a + b, 0) / arr.length;
}

// Get the value and std fields for a ranking metric (can be val or test metrics)
function getRankingMetricValue(entry: ResultEntry, metric: RankingMetric): { value: number; std: number } {
  switch (metric) {
    case 'val_accuracy':
      return { value: entry.val_accuracy, std: entry.val_accuracy_std };
    case 'val_f1_macro':
      return { value: entry.val_f1_macro, std: entry.val_f1_macro_std };
    case 'test_accuracy':
      return { value: entry.test_accuracy, std: entry.test_accuracy_std };
    case 'test_f1_macro':
      return { value: entry.test_f1_macro, std: entry.test_f1_macro_std };
    case 'auroc':
      return { value: entry.auroc, std: entry.auroc_std };
    default:
      return { value: entry.val_accuracy, std: entry.val_accuracy_std };
  }
}

// Get the value and std fields for a display metric
export function getDisplayMetricValue(entry: ResultEntry, metric: DisplayMetric): { value: number; std: number } {
  switch (metric) {
    case 'test_accuracy':
      return { value: entry.test_accuracy, std: entry.test_accuracy_std };
    case 'test_f1_macro':
      return { value: entry.test_f1_macro, std: entry.test_f1_macro_std };
    case 'auroc':
      return { value: entry.auroc, std: entry.auroc_std };
    default:
      return { value: entry.test_accuracy, std: entry.test_accuracy_std };
  }
}

/**
 * Compute leaderboard with dual-metric support
 * For each model, selects the BEST configuration (highest ranking metric value)
 * rather than averaging across all configurations.
 * 
 * @param results - Array of result entries
 * @param rankBy - Metric used to rank/order models and select best config
 * @param displayMetric - Metric displayed in the leaderboard (from the best config)
 */
export function computeLeaderboard(
  results: ResultEntry[],
  rankBy: RankingMetric = 'val_accuracy',
  displayMetric: DisplayMetric = 'test_accuracy'
): LeaderboardEntry[] {
  // Group by model
  const byModel: Record<string, ResultEntry[]> = {};
  for (const r of results) {
    if (!byModel[r.model]) byModel[r.model] = [];
    byModel[r.model].push(r);
  }

  // For each model, find the BEST configuration (highest ranking metric)
  const aggregates: LeaderboardEntry[] = Object.entries(byModel).map(([model, entries]) => {
    const isBaseline = entries.some((e) => e.readout === 'baseline');
    
    // Find the entry with the highest ranking metric value
    let bestEntry = entries[0];
    let bestRankValue = getRankingMetricValue(bestEntry, rankBy).value;
    
    for (const entry of entries) {
      const rankValue = getRankingMetricValue(entry, rankBy).value;
      if (rankValue > bestRankValue) {
        bestEntry = entry;
        bestRankValue = rankValue;
      }
    }
    
    // Use the best entry's values for both ranking and display
    const rankMetric = getRankingMetricValue(bestEntry, rankBy);
    const displayMetricVal = getDisplayMetricValue(bestEntry, displayMetric);

    return {
      rank: 0,
      model,
      category: MODEL_CATEGORIES[model] || 'baseline',
      rankValue: rankMetric.value,
      rankStd: rankMetric.std,
      displayValue: displayMetricVal.value,
      displayStd: displayMetricVal.std,
      testAccuracy: bestEntry.test_accuracy,
      testAccuracyStd: bestEntry.test_accuracy_std,
      testF1Macro: bestEntry.test_f1_macro,
      testF1MacroStd: bestEntry.test_f1_macro_std,
      auroc: bestEntry.auroc,
      aurocStd: bestEntry.auroc_std,
      isBaseline,
    };
  });

  // Sort by ranking metric (descending - higher is better)
  // Baselines go to the end since they don't have validation metrics
  aggregates.sort((a, b) => {
    // Baselines sort by display metric since they don't have rank metrics
    if (a.isBaseline && b.isBaseline) {
      return b.displayValue - a.displayValue;
    }
    if (a.isBaseline) return 1; // a goes after b
    if (b.isBaseline) return -1; // b goes after a
    return b.rankValue - a.rankValue;
  });

  // Assign ranks
  aggregates.forEach((entry, idx) => {
    entry.rank = idx + 1;
  });

  return aggregates;
}

export function filterResults(
  results: ResultEntry[],
  dataset: DatasetName | 'all',
  method: string | 'all' = 'all',
  ratio: number | 'all' = 'all'
): ResultEntry[] {
  let filtered = results;

  if (dataset !== 'all') {
    filtered = filtered.filter((r) => r.dataset === dataset);
  }

  if (method !== 'all') {
    // Filter by method, but keep baselines (they don't have a method)
    filtered = filtered.filter((r) => r.method === method || r.method === 'baseline');
  }

  if (ratio !== 'all') {
    // Filter by ratio, but keep baselines (they don't have a ratio)
    filtered = filtered.filter((r) => r.node_sample_ratio === ratio || r.node_sample_ratio === 0.0);
  }

  return filtered;
}

export function getStatsKey(
  dataset: string,
  ratio: number,
  method: string,
  threshold: number
): string {
  return `${dataset}|${ratio}|${method}|${threshold}`;
}

export function getStats(
  allStats: Record<string, GraphStats>,
  dataset: string,
  ratio: number,
  method: string,
  threshold: number
): GraphStats | null {
  const key = getStatsKey(dataset, ratio, method, threshold);
  return allStats[key] || null;
}

export function computeMetricMaxValues(stats: Record<string, GraphStats>): Record<string, number> {
  const metrics = [
    'num_nodes',
    'num_edges',
    'avg_degree',
    'density_pct',
    'avg_clustering_coeff',
    'largest_cc_ratio_pct',
    'avg_shortest_path_length',
    'num_connected_components',
    'degree_std',
  ];

  const maxValues: Record<string, number> = {};

  for (const metric of metrics) {
    const values = Object.values(stats).map((s) => s[metric as keyof GraphStats] as number);
    maxValues[metric] = values.length > 0 ? Math.max(...values) * 1.2 : 1;
  }

  return maxValues;
}

export interface ModelDataByDataset {
  value: number;
  std: number;
}

/**
 * Get model performance data by dataset for charts.
 * For each model+dataset combination, selects the BEST configuration
 * (highest ranking metric value) rather than averaging.
 * 
 * @param results - Array of result entries
 * @param modelOrder - Order of models to include
 * @param displayMetric - Metric to display in the chart
 * @param rankBy - Metric used to select the best configuration (defaults to displayMetric)
 */
export function getModelsByDataset(
  results: ResultEntry[],
  modelOrder: string[],
  displayMetric: DisplayMetric = 'test_accuracy',
  rankBy?: RankingMetric
): Record<DatasetName, Record<string, ModelDataByDataset>> {
  // Use displayMetric as rankBy if not specified (for consistency)
  const rankMetric = rankBy || displayMetric;
  
  // Group by dataset and model
  const byDatasetModel: Record<DatasetName, Record<string, ResultEntry[]>> = {
    motrpac: {},
    addneuromed: {},
    parkinsons: {},
  };

  for (const r of results) {
    const ds = r.dataset as DatasetName;
    if (!byDatasetModel[ds][r.model]) byDatasetModel[ds][r.model] = [];
    byDatasetModel[ds][r.model].push(r);
  }

  const result: Record<DatasetName, Record<string, ModelDataByDataset>> = {
    motrpac: {},
    addneuromed: {},
    parkinsons: {},
  };

  for (const ds of ['motrpac', 'addneuromed', 'parkinsons'] as DatasetName[]) {
    for (const model of modelOrder) {
      const entries = byDatasetModel[ds][model];
      if (entries && entries.length > 0) {
        // Find the best entry based on ranking metric
        let bestEntry = entries[0];
        let bestRankValue = getRankingMetricValue(bestEntry, rankMetric).value;
        
        for (const entry of entries) {
          const rankValue = getRankingMetricValue(entry, rankMetric).value;
          if (rankValue > bestRankValue) {
            bestEntry = entry;
            bestRankValue = rankValue;
          }
        }
        
        // Use the best entry's display metric values
        const { value, std } = getDisplayMetricValue(bestEntry, displayMetric);
        result[ds][model] = { value, std };
      }
    }
  }

  return result;
}
