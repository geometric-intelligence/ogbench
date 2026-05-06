import type { ResultEntry, GraphStats, LeaderboardEntry, DatasetName, RankingMetric, DisplayMetric } from './types';
import { MODEL_CATEGORIES } from './constants';

function getRankingMetricValue(entry: ResultEntry, metric: RankingMetric): { value: number; std: number } {
  switch (metric) {
    case 'val_f1_macro':
      return { value: entry.val_f1_macro, std: entry.val_f1_macro_std };
    case 'test_f1_macro':
      return { value: entry.test_f1_macro, std: entry.test_f1_macro_std };
    default:
      return { value: entry.val_f1_macro, std: entry.val_f1_macro_std };
  }
}

export function getDisplayMetricValue(entry: ResultEntry, metric: DisplayMetric): { value: number; std: number } {
  switch (metric) {
    case 'test_f1_macro':
      return { value: entry.test_f1_macro, std: entry.test_f1_macro_std };
    case 'train_f1_macro':
      return { value: entry.train_f1_macro, std: entry.train_f1_macro_std };
    default:
      return { value: entry.test_f1_macro, std: entry.test_f1_macro_std };
  }
}

/**
 * For each model, selects the BEST configuration (highest ranking metric value)
 * rather than averaging across all configurations.
 */
export function computeLeaderboard(
  results: ResultEntry[],
  rankBy: RankingMetric = 'val_f1_macro',
  displayMetric: DisplayMetric = 'test_f1_macro'
): LeaderboardEntry[] {
  const byModel: Record<string, ResultEntry[]> = {};
  for (const r of results) {
    if (!byModel[r.model]) byModel[r.model] = [];
    byModel[r.model].push(r);
  }

  const aggregates: LeaderboardEntry[] = Object.entries(byModel).map(([model, entries]) => {
    const isBaseline = entries.some((e) => e.readout === 'baseline');

    let bestEntry = entries[0];
    let bestRankValue = getRankingMetricValue(bestEntry, rankBy).value;

    for (const entry of entries) {
      const rankValue = getRankingMetricValue(entry, rankBy).value;
      if (rankValue > bestRankValue) {
        bestEntry = entry;
        bestRankValue = rankValue;
      }
    }

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
      testF1Macro: bestEntry.test_f1_macro,
      testF1MacroStd: bestEntry.test_f1_macro_std,
      isBaseline,
    };
  });

  aggregates.sort((a, b) => {
    if (a.isBaseline && b.isBaseline) return b.displayValue - a.displayValue;
    if (a.isBaseline) return 1;
    if (b.isBaseline) return -1;
    return b.rankValue - a.rankValue;
  });

  aggregates.forEach((entry, idx) => {
    entry.rank = idx + 1;
  });

  return aggregates;
}

export function filterResults(
  results: ResultEntry[],
  dataset: DatasetName | 'all',
  method: string | 'all' = 'all',
  ratio: number | 'all' = 'all',
  adjacencyMethod: string | 'all' = 'all'
): ResultEntry[] {
  let filtered = results;

  if (dataset !== 'all') {
    filtered = filtered.filter((r) => r.dataset === dataset);
  }

  if (method !== 'all') {
    filtered = filtered.filter((r) => r.method === method || r.method === 'baseline');
  }

  if (ratio !== 'all') {
    filtered = filtered.filter((r) => r.node_sample_ratio === ratio || r.node_sample_ratio === 0.0);
  }

  if (adjacencyMethod !== 'all') {
    filtered = filtered.filter(
      (r) => r.adjacency_method === adjacencyMethod || r.method === 'baseline'
    );
  }

  return filtered;
}

export function getStatsKey(
  dataset: string,
  ratio: number,
  method: string,
  threshold: number,
  adjacencyMethod?: string
): string {
  if (adjacencyMethod) {
    return `${dataset}|${ratio}|${method}|${threshold}|${adjacencyMethod}`;
  }
  return `${dataset}|${ratio}|${method}|${threshold}`;
}

export function getStats(
  allStats: Record<string, GraphStats>,
  dataset: string,
  ratio: number,
  method: string,
  threshold: number,
  adjacencyMethod?: string
): GraphStats | null {
  const key = getStatsKey(dataset, ratio, method, threshold, adjacencyMethod);
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
 * For each model+dataset combination, selects the BEST configuration
 * (highest ranking metric value) rather than averaging.
 */
export function getModelsByDataset(
  results: ResultEntry[],
  modelOrder: string[],
  displayMetric: DisplayMetric = 'test_f1_macro',
  rankBy?: RankingMetric
): Record<DatasetName, Record<string, ModelDataByDataset>> {
  const rankMetric = rankBy || (displayMetric as unknown as RankingMetric) || 'val_f1_macro';

  const allDatasets: DatasetName[] = ['motrpac', 'addneuromed', 'parkinsons', 'brca'];

  const byDatasetModel: Record<DatasetName, Record<string, ResultEntry[]>> = {
    motrpac: {},
    addneuromed: {},
    parkinsons: {},
    brca: {},
  };

  for (const r of results) {
    const ds = r.dataset as DatasetName;
    if (!byDatasetModel[ds]) continue;
    if (!byDatasetModel[ds][r.model]) byDatasetModel[ds][r.model] = [];
    byDatasetModel[ds][r.model].push(r);
  }

  const result: Record<DatasetName, Record<string, ModelDataByDataset>> = {
    motrpac: {},
    addneuromed: {},
    parkinsons: {},
    brca: {},
  };

  for (const ds of allDatasets) {
    for (const model of modelOrder) {
      const entries = byDatasetModel[ds][model];
      if (entries && entries.length > 0) {
        let bestEntry = entries[0];
        let bestRankValue = getRankingMetricValue(bestEntry, rankMetric).value;

        for (const entry of entries) {
          const rankValue = getRankingMetricValue(entry, rankMetric).value;
          if (rankValue > bestRankValue) {
            bestEntry = entry;
            bestRankValue = rankValue;
          }
        }

        const { value, std } = getDisplayMetricValue(bestEntry, displayMetric);
        result[ds][model] = { value, std };
      }
    }
  }

  return result;
}
