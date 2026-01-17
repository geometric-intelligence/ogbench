import type { ResultEntry, GraphStats, LeaderboardEntry, ModelCategory, DatasetName } from './types';
import { MODEL_CATEGORIES } from './constants';

// Standard deviation helper
function std(arr: number[]): number {
  if (arr.length === 0) return 0;
  const mean = arr.reduce((a, b) => a + b, 0) / arr.length;
  const variance = arr.reduce((sum, val) => sum + (val - mean) ** 2, 0) / arr.length;
  return Math.sqrt(variance);
}

export function computeLeaderboard(
  results: ResultEntry[],
  sortMetric: 'test_accuracy' | 'f1_macro' | 'runtime' = 'test_accuracy'
): LeaderboardEntry[] {
  // Group by model
  const byModel: Record<string, ResultEntry[]> = {};
  for (const r of results) {
    if (!byModel[r.model]) byModel[r.model] = [];
    byModel[r.model].push(r);
  }

  // Compute aggregates
  const aggregates: LeaderboardEntry[] = Object.entries(byModel).map(([model, entries]) => {
    const accuracies = entries.map((e) => e.test_accuracy);
    const f1s = entries.map((e) => e.f1_macro);
    const runtimes = entries.map((e) => e.runtime_seconds);

    return {
      rank: 0,
      model,
      category: MODEL_CATEGORIES[model] || 'baseline',
      accuracy: accuracies.reduce((a, b) => a + b, 0) / accuracies.length,
      accStd: std(accuracies),
      f1Macro: f1s.reduce((a, b) => a + b, 0) / f1s.length,
      f1Std: std(f1s),
      avgRuntime: runtimes.reduce((a, b) => a + b, 0) / runtimes.length,
      totalRuntime: runtimes.reduce((a, b) => a + b, 0),
    };
  });

  // Sort
  if (sortMetric === 'test_accuracy') {
    aggregates.sort((a, b) => b.accuracy - a.accuracy);
  } else if (sortMetric === 'f1_macro') {
    aggregates.sort((a, b) => b.f1Macro - a.f1Macro);
  } else {
    aggregates.sort((a, b) => a.avgRuntime - b.avgRuntime);
  }

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
    'n_nodes',
    'n_edges',
    'mean_degree',
    'density',
    'avg_clustering',
    'largest_cc_ratio',
    'avg_path_length',
    'n_components',
    'std_degree',
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
  modelOrder: string[]
): Record<DatasetName, Record<string, number>> {
  const byDataset: Record<DatasetName, Record<string, number[]>> = {
    motrpac: {},
    addneuromed: {},
    parkinsons: {},
  };

  for (const r of results) {
    const ds = r.dataset as DatasetName;
    if (!byDataset[ds][r.model]) byDataset[ds][r.model] = [];
    byDataset[ds][r.model].push(r.test_accuracy);
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
        result[ds][model] = vals.reduce((a, b) => a + b, 0) / vals.length;
      }
    }
  }

  return result;
}
