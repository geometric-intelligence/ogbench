import { useState, useEffect, useMemo, lazy } from 'react';

// Dynamically import Plotly to avoid SSR issues
const Plot = lazy(() => import('react-plotly.js'));
import type { GraphStats, DatasetName, NodeSelectionMethod } from '../lib/types';
import { DATASETS, VALID_RATIOS, VALID_THRESHOLDS, METRIC_LABELS } from '../lib/constants';
import { getStats, computeMetricMaxValues } from '../lib/data';

export default function Explorer() {
  const [allStats, setAllStats] = useState<Record<string, GraphStats>>({});
  const [loading, setLoading] = useState(true);
  const [nodeSampleRatio, setNodeSampleRatio] = useState<number>(0.5);
  const [nodeSelectionMethod, setNodeSelectionMethod] = useState<NodeSelectionMethod>('variance');
  const [adjacencyThreshold, setAdjacencyThreshold] = useState(0.02);

  // All datasets are always shown
  const datasetOrder: DatasetName[] = ['motrpac', 'addneuromed', 'parkinsons'];

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

  // Compute max values for y-axis ranges
  const metricMaxValues = useMemo(() => computeMetricMaxValues(allStats), [allStats]);

  // Get current stats for selected parameters
  const currentStats = useMemo(() => {
    const stats: (GraphStats & { dataset: DatasetName })[] = [];
    for (const ds of datasetOrder) {
      const s = getStats(allStats, ds, nodeSampleRatio, nodeSelectionMethod, adjacencyThreshold);
      if (s) {
        stats.push({ ...s, dataset: ds });
      }
    }
    return stats;
  }, [allStats, nodeSampleRatio, nodeSelectionMethod, adjacencyThreshold]);

  const metrics = Object.entries(METRIC_LABELS) as [keyof GraphStats, string][];

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

  for (let idx = 0; idx < metrics.length; idx++) {
    const [metric, label] = metrics[idx];
    const row = Math.floor(idx / 3);
    const col = idx % 3;
    const xAxisId = idx === 0 ? 'x' : `x${idx + 1}`;
    const yAxisId = idx === 0 ? 'y' : `y${idx + 1}`;

    const xLabels = datasetOrder.map((ds) => `${DATASETS[ds].emoji} ${DATASETS[ds].fullName}`);
    const yValues: (number | null)[] = [];
    const colors: string[] = [];
    const textValues: string[] = [];

    for (const ds of datasetOrder) {
      const stat = currentStats.find((s) => s.dataset === ds);
      if (stat) {
        const v = stat[metric] as number;
        yValues.push(v);
        colors.push(DATASETS[ds].color);

        if (metric === 'density_pct' || metric === 'largest_cc_ratio_pct') {
          textValues.push(`${v.toFixed(1)}%`);
        } else if (metric === 'avg_clustering_coeff' || metric === 'avg_shortest_path_length') {
          textValues.push(v.toFixed(2));
        } else if (metric === 'num_nodes' || metric === 'num_edges' || metric === 'num_connected_components') {
          textValues.push(Math.round(v).toLocaleString());
        } else {
          textValues.push(v.toFixed(1));
        }
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
      textfont: { size: 14, family: 'JetBrains Mono', color: '#0f172a' },
      showlegend: false,
      cliponaxis: false,
      xaxis: xAxisId,
      yaxis: yAxisId,
    } as Plotly.Data);

    // Add subplot title as annotation - centered above each subplot
    // Calculate positions based on grid layout with gaps
    const colCenters = [0.14, 0.5, 0.86]; // Center positions for 3 columns
    const rowTops = [1.0, 0.63, 0.27]; // Top positions for 3 rows (above each row)
    
    annotations.push({
      text: `<b>${label}</b>`,
      xref: 'paper',
      yref: 'paper',
      x: colCenters[col],
      y: rowTops[row],
      showarrow: false,
      font: { size: 12, color: '#1e293b', family: 'DM Sans' },
      xanchor: 'center',
      yanchor: 'bottom',
    });
  }

  // Build layout with 3x3 subplots - professional spacing
  const layout: Partial<Plotly.Layout> = {
    height: 1000,
    font: { family: 'DM Sans', size: 14, color: '#0f172a' },
    paper_bgcolor: 'rgba(0,0,0,0)',
    plot_bgcolor: '#ffffff',
    margin: { l: 60, r: 40, t: 50, b: 50 },
    annotations,
    grid: { 
      rows: 3, 
      columns: 3, 
      pattern: 'independent',
      xgap: 0.1,
      ygap: 0.18,
    },
  };

  // Configure axes for each subplot
  for (let idx = 0; idx < metrics.length; idx++) {
    const [metric] = metrics[idx];
    const axisNum = idx === 0 ? '' : `${idx + 1}`;

    (layout as Record<string, unknown>)[`xaxis${axisNum}`] = {
      showgrid: false,
      tickangle: 0,
      tickfont: { size: 11, color: '#475569' },
      fixedrange: true,
    };

    (layout as Record<string, unknown>)[`yaxis${axisNum}`] = {
      showgrid: true,
      gridcolor: 'rgba(226,232,240,0.8)',
      zeroline: true,
      zerolinecolor: 'rgba(226,232,240,1)',
      tickfont: { size: 12, color: '#475569' },
      fixedrange: true,
      range: [0, metricMaxValues[metric as string] || 1],
    };
  }

  return (
    <div>
      {/* Controls */}
      <div className="controls-panel">
        <div className="controls-grid">
          <div className="control-group">
            <div className="control-label">Node Sample Ratio</div>
            <div className="control-value">p = {nodeSampleRatio}</div>
            <input
              type="range"
              min={0}
              max={VALID_RATIOS.length - 1}
              step={1}
              value={VALID_RATIOS.indexOf(nodeSampleRatio)}
              onChange={(e) => setNodeSampleRatio(VALID_RATIOS[parseInt(e.target.value)])}
            />
            <div className="flex justify-between text-xs text-text-muted mt-1">
              {VALID_RATIOS.map((r) => (
                <span key={r}>{r}</span>
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
            <div className="control-label">Adjacency Threshold</div>
            <div className="control-value">τ = {adjacencyThreshold.toFixed(2)}</div>
            <input
              type="range"
              min={0}
              max={VALID_THRESHOLDS.length - 1}
              step={1}
              value={VALID_THRESHOLDS.indexOf(adjacencyThreshold)}
              onChange={(e) => setAdjacencyThreshold(VALID_THRESHOLDS[parseInt(e.target.value)])}
            />
            <div className="flex justify-between text-xs text-text-muted mt-1">
              {VALID_THRESHOLDS.map((t) => (
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
          style={{ width: '100%', height: '1000px' }}
        />
      </div>
    </div>
  );
}
