import type {
  ResultEntry,
  GraphStats,
  LeaderboardEntry,
  DatasetName,
  LeaderboardDatasetName,
  RankingMetric,
  DisplayMetric,
  StatsMetric,
  StatsRatio,
} from './types';
import { MODEL_CATEGORIES, LEADERBOARD_DATASETS, METRIC_LABELS } from './constants';

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
    case 'test_f1_weighted':
      return { value: entry.test_f1_weighted, std: entry.test_f1_weighted_std };
    case 'test_accuracy':
      return { value: entry.test_accuracy, std: entry.test_accuracy_std };
    case 'test_auroc':
      return { value: entry.test_auroc, std: entry.test_auroc_std };
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
    filtered = filtered.filter((r) => r.method === method);
  }

  if (ratio !== 'all') {
    filtered = filtered.filter((r) => r.node_sample_ratio === ratio);
  }

  if (adjacencyMethod !== 'all') {
    filtered = filtered.filter(
      (r) => r.adjacency_method === adjacencyMethod || r.adjacency_method === 'baseline'
    );
  }

  return filtered;
}

/** Parse a ratio segment of a stats key: 'full' stays a string, everything else is numeric. */
export function parseStatsRatio(raw: string): StatsRatio | null {
  if (raw === 'full') return 'full';
  const n = parseFloat(raw);
  return Number.isNaN(n) ? null : n;
}

/** Sort ratios numerically with 'full' (all features) last. */
export function compareStatsRatios(a: StatsRatio, b: StatsRatio): number {
  if (a === b) return 0;
  if (a === 'full') return 1;
  if (b === 'full') return -1;
  return a - b;
}

export function getStatsKey(
  dataset: string,
  ratio: StatsRatio,
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
  ratio: StatsRatio,
  method: string,
  threshold: number,
  adjacencyMethod?: string
): GraphStats | null {
  const key = getStatsKey(dataset, ratio, method, threshold, adjacencyMethod);
  return allStats[key] || null;
}

/** Per-metric y-axis ceiling (1.2 × max) over the given stats; null/NaN values are ignored. */
export function computeMetricMaxValues(stats: Iterable<GraphStats>): Record<StatsMetric, number> {
  const maxValues = {} as Record<StatsMetric, number>;
  const entries = [...stats];
  for (const metric of Object.keys(METRIC_LABELS) as StatsMetric[]) {
    let max = 0;
    for (const s of entries) {
      const v = s[metric];
      if (typeof v === 'number' && Number.isFinite(v) && v > max) max = v;
    }
    maxValues[metric] = max > 0 ? max * 1.2 : 1;
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
): Record<LeaderboardDatasetName, Record<string, ModelDataByDataset>> {
  const rankMetric = rankBy || (displayMetric as unknown as RankingMetric) || 'val_f1_macro';

  const byDatasetModel = Object.fromEntries(LEADERBOARD_DATASETS.map((ds) => [ds, {}])) as Record<
    LeaderboardDatasetName,
    Record<string, ResultEntry[]>
  >;

  for (const r of results) {
    const ds = r.dataset as LeaderboardDatasetName;
    if (!byDatasetModel[ds]) continue;
    if (!byDatasetModel[ds][r.model]) byDatasetModel[ds][r.model] = [];
    byDatasetModel[ds][r.model].push(r);
  }

  const result = Object.fromEntries(LEADERBOARD_DATASETS.map((ds) => [ds, {}])) as Record<
    LeaderboardDatasetName,
    Record<string, ModelDataByDataset>
  >;

  for (const ds of LEADERBOARD_DATASETS) {
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
