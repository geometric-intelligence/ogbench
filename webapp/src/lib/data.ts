import type { ResultEntry, GraphStats, LeaderboardEntry, ModelCategory, DatasetName, RankingMetric, DisplayMetric } from './types';
import { MODEL_CATEGORIES } from './constants';

// Helper to compute mean of array
function mean(arr: number[]): number {
  if (arr.length === 0) return 0;
  return arr.reduce((a, b) => a + b, 0) / arr.length;
}

// Helper to compute pooled standard deviation from individual stds
function pooledStd(stds: number[]): number {
  if (stds.length === 0) return 0;
  // For simplicity, use mean of stds (approximation)
  return mean(stds);
}

// Get the value and std fields for a ranking metric
function getRankingMetricValue(entry: ResultEntry, metric: RankingMetric): { value: number; std: number } {
  switch (metric) {
    case 'val_accuracy':
      return { value: entry.val_accuracy, std: entry.val_accuracy_std };
    case 'val_f1_macro':
      return { value: entry.val_f1_macro, std: entry.val_f1_macro_std };
    default:
      return { value: entry.val_accuracy, std: entry.val_accuracy_std };
  }
}

// Get the value and std fields for a display metric
function getDisplayMetricValue(entry: ResultEntry, metric: DisplayMetric): { value: number; std: number } {
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
 * @param results - Array of result entries
 * @param rankBy - Metric used to rank/order models (validation metrics)
 * @param displayMetric - Metric displayed in the leaderboard (test metrics)
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

  // Compute aggregates for each model
  const aggregates: LeaderboardEntry[] = Object.entries(byModel).map(([model, entries]) => {
    const isBaseline = entries.some((e) => e.readout === 'baseline');
    
    // Extract ranking metric values
    const rankValues = entries.map((e) => getRankingMetricValue(e, rankBy));
    const rankMean = mean(rankValues.map((v) => v.value));
    const rankStdPooled = pooledStd(rankValues.map((v) => v.std));
    
    // Extract display metric values
    const displayValues = entries.map((e) => getDisplayMetricValue(e, displayMetric));
    const displayMean = mean(displayValues.map((v) => v.value));
    const displayStdPooled = pooledStd(displayValues.map((v) => v.std));
    
    // Extract all test metrics for charts
    const testAccuracies = entries.map((e) => e.test_accuracy);
    const testAccuracyStds = entries.map((e) => e.test_accuracy_std);
    const testF1Macros = entries.map((e) => e.test_f1_macro);
    const testF1MacroStds = entries.map((e) => e.test_f1_macro_std);
    const aurocs = entries.map((e) => e.auroc);
    const aurocStds = entries.map((e) => e.auroc_std);

    return {
      rank: 0,
      model,
      category: MODEL_CATEGORIES[model] || 'baseline',
      rankValue: rankMean,
      rankStd: rankStdPooled,
      displayValue: displayMean,
      displayStd: displayStdPooled,
      testAccuracy: mean(testAccuracies),
      testAccuracyStd: pooledStd(testAccuracyStds),
      testF1Macro: mean(testF1Macros),
      testF1MacroStd: pooledStd(testF1MacroStds),
      auroc: mean(aurocs),
      aurocStd: pooledStd(aurocStds),
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
  category: ModelCategory | 'all'
): ResultEntry[] {
  let filtered = results;

  if (dataset !== 'all') {
    filtered = filtered.filter((r) => r.dataset === dataset);
  }

  if (category !== 'all') {
    const modelsInCategory = Object.entries(MODEL_CATEGORIES)
      .filter(([_, cat]) => cat === category)
      .map(([model]) => model);
    filtered = filtered.filter((r) => modelsInCategory.includes(r.model));
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

export function getModelsByDataset(
  results: ResultEntry[],
  modelOrder: string[],
  displayMetric: DisplayMetric = 'test_accuracy'
): Record<DatasetName, Record<string, number>> {
  const byDataset: Record<DatasetName, Record<string, number[]>> = {
    motrpac: {},
    addneuromed: {},
    parkinsons: {},
  };

  for (const r of results) {
    const ds = r.dataset as DatasetName;
    if (!byDataset[ds][r.model]) byDataset[ds][r.model] = [];
    const { value } = getDisplayMetricValue(r, displayMetric);
    byDataset[ds][r.model].push(value);
  }

  const result: Record<DatasetName, Record<string, number>> = {
    motrpac: {},
    addneuromed: {},
    parkinsons: {},
  };

  for (const ds of ['motrpac', 'addneuromed', 'parkinsons'] as DatasetName[]) {
    for (const model of modelOrder) {
      const vals = byDataset[ds][model];
      if (vals && vals.length > 0) {
        result[ds][model] = mean(vals);
      }
    }
  }

  return result;
}
