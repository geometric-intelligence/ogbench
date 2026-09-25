import { useState, useEffect, useMemo, lazy } from 'react';

// Dynamically import Plotly to avoid SSR issues
const Plot = lazy(() => import('react-plotly.js'));
import type { GraphStats, DatasetName, NodeSelectionMethod, StatsMetric, StatsRatio } from '../lib/types';
import {
  DATASETS,
  EXPLORER_DATASETS,
  EXPLORER_METRICS,
  VALID_ADJACENCY_METHODS,
  ADJACENCY_METHOD_LABELS,
  METRIC_LABELS,
} from '../lib/constants';
import { getStats, computeMetricMaxValues, parseStatsRatio, compareStatsRatios } from '../lib/data';

const GRID_COLUMNS = 3;
const ROW_HEIGHT_PX = 320;

const PERCENT_METRICS: ReadonlySet<StatsMetric> = new Set(['density_pct', 'largest_cc_ratio_pct']);
const INTEGER_METRICS: ReadonlySet<StatsMetric> = new Set([
  'num_nodes',
  'num_edges',
  'num_connected_components',
  'diameter',
]);
const UNIT_INTERVAL_METRICS: ReadonlySet<StatsMetric> = new Set([
  'clustering_coefficient',
  'modularity',
  'homophily',
]);

function formatMetricValue(metric: StatsMetric, value: number): string {
  if (PERCENT_METRICS.has(metric)) return `${value.toFixed(1)}%`;
  if (INTEGER_METRICS.has(metric)) return Math.round(value).toLocaleString();
  if (UNIT_INTERVAL_METRICS.has(metric)) return value.toFixed(3);
  return value.toFixed(1);
}

/** Pick the option closest to `target` (numeric options only). */
function nearestThreshold(options: number[], target: number): number {
  return options.reduce((best, t) => (Math.abs(t - target) < Math.abs(best - target) ? t : best), options[0]);
}

/** Parse stats keys "dataset|ratio|method|threshold|adjacency_method" to derive unique options. */
function deriveOptionsFromStats(stats: Record<string, GraphStats>): {
  ratios: StatsRatio[];
  thresholds: number[];
  adjacencyMethods: string[];
} {
  const ratioSet = new Set<StatsRatio>();
  const thresholdSet = new Set<number>();
  const adjMethodSet = new Set<string>();
  for (const key of Object.keys(stats)) {
    const parts = key.split('|');
    if (parts.length < 4) continue;
    const r = parseStatsRatio(parts[1]);
    const t = parseFloat(parts[3]);
    if (r !== null) ratioSet.add(r);
    if (!Number.isNaN(t)) thresholdSet.add(t);
    if (parts.length >= 5 && parts[4]) adjMethodSet.add(parts[4]);
  }
  return {
    ratios: [...ratioSet].sort(compareStatsRatios),
    thresholds: [...thresholdSet].sort((a, b) => a - b),
    adjacencyMethods: [...adjMethodSet].sort(),
  };
}

export default function Explorer() {
  const [allStats, setAllStats] = useState<Record<string, GraphStats>>({});
  const [loading, setLoading] = useState(true);
  const [nodeSampleRatio, setNodeSampleRatio] = useState<StatsRatio>(0.5);
  const [nodeSelectionMethod, setNodeSelectionMethod] = useState<NodeSelectionMethod>('variance');
  const [adjacencyThreshold, setAdjacencyThreshold] = useState(0.5);
  const [adjacencyMethod, setAdjacencyMethod] = useState<string | undefined>(undefined);

  // All datasets are always shown
  const datasetOrder: readonly DatasetName[] = EXPLORER_DATASETS;

  const { ratios: validRatios, thresholds: validThresholds, adjacencyMethods: validAdjacencyMethods } = useMemo(() => {
    const derived = deriveOptionsFromStats(allStats);
    return {
      ratios: derived.ratios.length > 0 ? derived.ratios : ([0.3, 0.5, 1, 'full'] as StatsRatio[]),
      thresholds: derived.thresholds.length > 0 ? derived.thresholds : [0, 0.5, 1],
      adjacencyMethods: derived.adjacencyMethods.length > 0 ? derived.adjacencyMethods : [...VALID_ADJACENCY_METHODS],
    };
  }, [allStats]);

  // When data loads, ensure selected values exist in the data; otherwise pick the closest available
  useEffect(() => {
    if (loading || Object.keys(allStats).length === 0) return;
    const { ratios, thresholds, adjacencyMethods } = deriveOptionsFromStats(allStats);
    if (ratios.length === 0 || thresholds.length === 0) return;
    setNodeSampleRatio((prev) => (ratios.includes(prev) ? prev : ratios[0]));
    setAdjacencyThreshold((prev) => (thresholds.includes(prev) ? prev : nearestThreshold(thresholds, prev)));
    if (adjacencyMethods.length > 0) {
      setAdjacencyMethod((prev) => (prev && adjacencyMethods.includes(prev) ? prev : adjacencyMethods[0]));
    }
  }, [loading, allStats]);

  // Load data on mount
  useEffect(() => {
    fetch('/data/stats.json')
      .then((res) => res.json())
      .then((data) => {
        setAllStats(data);
        setLoading(false);
      })
      .catch((err) => {
        console.error('Failed to load stats:', err);
        setLoading(false);
      });
  }, []);

  // Get current stats for selected parameters
  const currentStats = useMemo(() => {
    const stats: (GraphStats & { dataset: DatasetName })[] = [];
    for (const ds of datasetOrder) {
      const s = getStats(allStats, ds, nodeSampleRatio, nodeSelectionMethod, adjacencyThreshold, adjacencyMethod);
      if (s) {
        stats.push({ ...s, dataset: ds });
      }
    }
    return stats;
  }, [allStats, nodeSampleRatio, nodeSelectionMethod, adjacencyThreshold, adjacencyMethod]);

  // Y-axis ceilings scaled to the bars currently on screen: edge counts span ~100× across τ and
  // node counts ~50× between 'full' and subsampled graphs, so a global max flattens everything.
  const metricMaxValues = useMemo(() => computeMetricMaxValues(currentStats), [currentStats]);

  const metrics = EXPLORER_METRICS.map((key) => [key, METRIC_LABELS[key]] as [StatsMetric, string]);
  const gridRows = Math.ceil(metrics.length / GRID_COLUMNS);
  const plotHeight = gridRows * ROW_HEIGHT_PX;

  if (loading) {
    return (
      <div className="flex items-center justify-center p-16">
        <div className="text-text-muted">Loading data...</div>
      </div>
    );
  }

  // Build subplot data
  const subplotData: Plotly.Data[] = [];
  const annotations: Partial<Plotly.Annotations>[] = [];

  const xLabels = datasetOrder.map((ds) => `${DATASETS[ds].emoji} ${DATASETS[ds].fullName}`);
  // Explicit subplot domains (paper coordinates) so each row reserves room for its title above
  // and its angled tick labels below, instead of relying on Plotly's grid gap heuristics.
  const colGap = 0.035;
  const rowTitlePad = 0.035;
  const rowTickPad = 0.09;
  const xDomain = (col: number): [number, number] => [
    col / GRID_COLUMNS + (col > 0 ? colGap : 0),
    (col + 1) / GRID_COLUMNS - (col < GRID_COLUMNS - 1 ? colGap : 0),
  ];
  const yDomain = (row: number): [number, number] => [
    1 - (row + 1) / gridRows + (row < gridRows - 1 ? rowTickPad : 0),
    1 - row / gridRows - rowTitlePad,
  ];

  for (let idx = 0; idx < metrics.length; idx++) {
    const [metric, label] = metrics[idx];
    const row = Math.floor(idx / GRID_COLUMNS);
    const col = idx % GRID_COLUMNS;
    const xAxisId = idx === 0 ? 'x' : `x${idx + 1}`;
    const yAxisId = idx === 0 ? 'y' : `y${idx + 1}`;

    const yValues: (number | null)[] = [];
    const colors: string[] = [];
    const textValues: string[] = [];

    for (const ds of datasetOrder) {
      const stat = currentStats.find((s) => s.dataset === ds);
      const v = stat ? stat[metric] : null;
      if (typeof v === 'number' && Number.isFinite(v)) {
        yValues.push(v);
        colors.push(DATASETS[ds].color);
        textValues.push(formatMetricValue(metric, v));
      } else {
        yValues.push(null);
        colors.push('rgba(200,200,200,0.3)');
        textValues.push('');
      }
    }

    subplotData.push({
      type: 'bar',
      x: xLabels,
      y: yValues,
      marker: { color: colors },
      text: textValues,
      textposition: 'outside',
      textfont: { size: 12, family: 'JetBrains Mono', color: '#0f172a' },
      showlegend: false,
      cliponaxis: false,
      xaxis: xAxisId,
      yaxis: yAxisId,
    } as Plotly.Data);

    // Subplot title as an annotation centered above each subplot
    const [x0, x1] = xDomain(col);
    annotations.push({
      text: `<b>${label}</b>`,
      xref: 'paper',
      yref: 'paper',
      x: (x0 + x1) / 2,
      y: yDomain(row)[1] + 0.005,
      showarrow: false,
      font: { size: 12, color: '#1e293b', family: 'DM Sans' },
      xanchor: 'center',
      yanchor: 'bottom',
    });
  }

  // Build layout with a rows × 3 subplot grid
  const layout: Partial<Plotly.Layout> = {
    height: plotHeight,
    font: { family: 'DM Sans', size: 14, color: '#0f172a' },
    paper_bgcolor: 'rgba(0,0,0,0)',
    plot_bgcolor: '#ffffff',
    margin: { l: 60, r: 40, t: 30, b: 80 },
    annotations,
  };

  // Configure axes for each subplot
  for (let idx = 0; idx < metrics.length; idx++) {
    const [metric] = metrics[idx];
    const axisNum = idx === 0 ? '' : `${idx + 1}`;
    const row = Math.floor(idx / GRID_COLUMNS);
    const col = idx % GRID_COLUMNS;

    (layout as Record<string, unknown>)[`xaxis${axisNum}`] = {
      domain: xDomain(col),
      anchor: idx === 0 ? 'y' : `y${idx + 1}`,
      showgrid: false,
      tickangle: -30,
      tickfont: { size: 10, color: '#475569' },
      fixedrange: true,
    };

    (layout as Record<string, unknown>)[`yaxis${axisNum}`] = {
      domain: yDomain(row),
      anchor: idx === 0 ? 'x' : `x${idx + 1}`,
      showgrid: true,
      gridcolor: 'rgba(226,232,240,0.8)',
      zeroline: true,
      zerolinecolor: 'rgba(226,232,240,1)',
      tickfont: { size: 12, color: '#475569' },
      fixedrange: true,
      range: [0, metricMaxValues[metric] || 1],
    };
  }

  return (
    <div>
      {/* Controls */}
      <div className="controls-panel">
        <div className="controls-grid">
          <div className="control-group">
            <div className="control-label">Sample-to-Node Ratio</div>
            <div className="control-value">p = {nodeSampleRatio === 'full' ? 'full (all features)' : nodeSampleRatio}</div>
            <input
              type="range"
              min={0}
              max={validRatios.length - 1}
              step={1}
              value={Math.max(0, validRatios.indexOf(nodeSampleRatio))}
              onChange={(e) => setNodeSampleRatio(validRatios[parseInt(e.target.value)] ?? validRatios[0])}
            />
            <div className="flex justify-between text-xs text-text-muted mt-1">
              {validRatios.map((r) => (
                <span key={String(r)}>{r}</span>
              ))}
            </div>
          </div>

          <div className="control-group">
            <div className="control-label">Node Selection Method</div>
            <select
              value={nodeSelectionMethod}
              onChange={(e) => setNodeSelectionMethod(e.target.value as NodeSelectionMethod)}
            >
              <option value="variance">📊 Variance</option>
              <option value="correlation">🔗 Correlation</option>
              <option value="distance_correlation">📐 Distance Correlation</option>
              <option value="random">🎲 Random</option>
            </select>
          </div>

          <div className="control-group">
            <div className="control-label">Graph Construction</div>
            <select
              value={adjacencyMethod ?? validAdjacencyMethods[0] ?? 'string'}
              onChange={(e) => setAdjacencyMethod(e.target.value)}
            >
              {validAdjacencyMethods.map((m) => (
                <option key={m} value={m}>
                  {ADJACENCY_METHOD_LABELS[m] ?? m}
                </option>
              ))}
            </select>
          </div>

          <div className="control-group">
            <div className="control-label">Adjacency Threshold</div>
            <div className="control-value">τ = {adjacencyThreshold.toFixed(2)}</div>
            <input
              type="range"
              min={0}
              max={validThresholds.length - 1}
              step={1}
              value={Math.max(0, validThresholds.indexOf(adjacencyThreshold))}
              onChange={(e) => setAdjacencyThreshold(validThresholds[parseInt(e.target.value)] ?? validThresholds[0])}
            />
            <div className="flex justify-between text-xs text-text-muted mt-1">
              {validThresholds.map((t) => (
                <span key={t}>{t}</span>
              ))}
            </div>
          </div>

        </div>
      </div>

      {/* Charts */}
      <div className="charts-container">
        <Plot
          data={subplotData}
          layout={layout}
          config={{ displayModeBar: true, responsive: true }}
          style={{ width: '100%', height: `${plotHeight}px` }}
        />
      </div>
    </div>
  );
}
