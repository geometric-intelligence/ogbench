import { useState, useEffect, useMemo, lazy } from 'react';

const Plot = lazy(() => import('react-plotly.js'));
import type { ResultEntry, DatasetName, RankingMetric, DisplayMetric } from '../lib/types';
import { DATASETS, MODEL_ORDER, BASELINE_MODELS, MODEL_COLORS, RANKING_METRICS, DISPLAY_METRICS, VALID_METHODS, VALID_RATIOS, METHOD_LABELS, RATIO_LABELS, ADJACENCY_METHOD_LABELS } from '../lib/constants';
import { computeLeaderboard, filterResults, getModelsByDataset, getDisplayMetricValue } from '../lib/data';

const ALL_DATASETS: DatasetName[] = ['motrpac', 'addneuromed', 'parkinsons', 'brca'];

export default function Leaderboard() {
  const [results, setResults] = useState<ResultEntry[]>([]);
  const [loading, setLoading] = useState(true);
  const [datasetFilter, setDatasetFilter] = useState<DatasetName | 'all'>('all');
  const [methodFilter, setMethodFilter] = useState<string | 'all'>('all');
  const [ratioFilter, setRatioFilter] = useState<number | 'all'>('all');
  const [adjacencyMethodFilter, setAdjacencyMethodFilter] = useState<string | 'all'>('all');
  const [rankBy, setRankBy] = useState<RankingMetric>('val_f1_macro');
  const [displayMetric, setDisplayMetric] = useState<DisplayMetric>('test_f1_macro');

  useEffect(() => {
    fetch('/data/results.json')
      .then((res) => res.json())
      .then((data) => {
        const entries = Object.values(data) as ResultEntry[];
        setResults(entries);
        setLoading(false);
      })
      .catch((err) => {
        console.error('Failed to load results:', err);
        setLoading(false);
      });
  }, []);

  const availableAdjacencyMethods = useMemo(() => {
    const methods = new Set<string>();
    for (const r of results) {
      if (r.adjacency_method) methods.add(r.adjacency_method);
    }
    return [...methods].sort();
  }, [results]);

  const filteredResults = useMemo(
    () => filterResults(results, datasetFilter, methodFilter, ratioFilter, adjacencyMethodFilter),
    [results, datasetFilter, methodFilter, ratioFilter, adjacencyMethodFilter]
  );

  const chartFilteredResults = useMemo(
    () => filterResults(results, 'all', methodFilter, ratioFilter, adjacencyMethodFilter),
    [results, methodFilter, ratioFilter, adjacencyMethodFilter]
  );

  const leaderboard = useMemo(
    () => computeLeaderboard(filteredResults, rankBy, displayMetric),
    [filteredResults, rankBy, displayMetric]
  );

  const subtitle = useMemo(() => {
    const parts: string[] = [];
    if (datasetFilter === 'all') {
      parts.push('All datasets');
    } else {
      parts.push(DATASETS[datasetFilter].fullName);
    }
    if (methodFilter !== 'all') {
      parts.push(METHOD_LABELS[methodFilter]);
    }
    if (ratioFilter !== 'all') {
      parts.push(`${RATIO_LABELS[ratioFilter]} nodes`);
    }
    if (adjacencyMethodFilter !== 'all') {
      parts.push(ADJACENCY_METHOD_LABELS[adjacencyMethodFilter] ?? adjacencyMethodFilter);
    }
    return parts.join(' • ');
  }, [datasetFilter, methodFilter, ratioFilter, adjacencyMethodFilter]);

  const allModelsData = useMemo(() => {
    const allModels = [...MODEL_ORDER, ...BASELINE_MODELS];
    return getModelsByDataset(chartFilteredResults, allModels, displayMetric, rankBy);
  }, [chartFilteredResults, displayMetric, rankBy]);

  const baselinesByDataset = useMemo(() => {
    const baselines: Record<DatasetName, Record<string, { value: number; std: number }>> = {
      motrpac: {},
      addneuromed: {},
      parkinsons: {},
      brca: {},
    };

    for (const ds of ALL_DATASETS) {
      for (const model of BASELINE_MODELS) {
        const entries = chartFilteredResults.filter((r) => r.model === model && r.dataset === ds);
        if (entries.length > 0) {
          const metricValues = entries.map((e) => getDisplayMetricValue(e, displayMetric));
          const value = metricValues.reduce((sum, m) => sum + m.value, 0) / metricValues.length;
          const std = metricValues.reduce((sum, m) => sum + m.std, 0) / metricValues.length;
          baselines[ds][model] = { value, std };
        }
      }
    }
    return baselines;
  }, [chartFilteredResults, displayMetric]);

  const filteredModels = MODEL_ORDER;

  const displayMetricLabel = DISPLAY_METRICS[displayMetric];
  const rankMetricLabel = RANKING_METRICS[rankBy];

  const chartTitle = useMemo(() => {
    let title = `${displayMetricLabel} of Best Models`;
    title += ` (Ranked by ${rankMetricLabel})`;

    if (methodFilter !== 'all') {
      title += ` using ${METHOD_LABELS[methodFilter]} Selection`;
    } else {
      title += ` across All Selection Methods`;
    }

    if (ratioFilter !== 'all') {
      title += ` at ${RATIO_LABELS[ratioFilter]} Sample-to-Node Ratio`;
    }

    if (adjacencyMethodFilter !== 'all') {
      title += ` — ${ADJACENCY_METHOD_LABELS[adjacencyMethodFilter] ?? adjacencyMethodFilter}`;
    }

    return title;
  }, [displayMetricLabel, rankMetricLabel, methodFilter, ratioFilter, adjacencyMethodFilter]);

  if (loading) {
    return (
      <div className="flex items-center justify-center p-16">
        <div className="text-text-muted">Loading data...</div>
      </div>
    );
  }

  const facetedChartData: Plotly.Data[] = [];
  const facetedAnnotations: Partial<Plotly.Annotations>[] = [];

  ALL_DATASETS.forEach((ds, dsIdx) => {
    const xAxisId = dsIdx === 0 ? 'x' : `x${dsIdx + 1}`;
    const yAxisId = dsIdx === 0 ? 'y' : `y${dsIdx + 1}`;

    const values: number[] = [];
    const errors: number[] = [];
    const colors: string[] = [];
    const textLabels: string[] = [];

    for (const model of filteredModels) {
      const data = allModelsData[ds][model];
      if (data) {
        values.push(data.value);
        errors.push(data.std);
        colors.push(MODEL_COLORS[model] || '#888888');
        textLabels.push(`${(data.value * 100).toFixed(1)}%`);
      } else {
        values.push(0);
        errors.push(0);
        colors.push('#888888');
        textLabels.push('');
      }
    }

    facetedChartData.push({
      type: 'bar',
      name: DATASETS[ds].fullName,
      x: filteredModels,
      y: values,
      error_y: {
        type: 'data',
        array: errors,
        visible: true,
        color: '#333333',
        thickness: 1.5,
        width: 4,
      },
      marker: { color: colors },
      text: textLabels,
      textposition: 'outside',
      textfont: { family: 'JetBrains Mono', size: 10, color: '#0f172a' },
      showlegend: false,
      xaxis: xAxisId,
      yaxis: yAxisId,
    } as Plotly.Data);

    const baselineData = baselinesByDataset[ds];

    if (baselineData.ElasticNet && filteredModels.length > 0) {
      facetedChartData.push({
        type: 'scatter',
        mode: 'lines',
        name: dsIdx === 0 ? 'Elastic Net' : undefined,
        x: filteredModels,
        y: Array(filteredModels.length).fill(baselineData.ElasticNet.value),
        line: { color: '#000000', width: 2, dash: 'dash' },
        showlegend: dsIdx === 0,
        legendgroup: 'ElasticNet',
        xaxis: xAxisId,
        yaxis: yAxisId,
        hoverinfo: 'y+name',
        connectgaps: true,
      } as Plotly.Data);
    }

    if (baselineData.SVM && filteredModels.length > 0) {
      facetedChartData.push({
        type: 'scatter',
        mode: 'lines',
        name: dsIdx === 0 ? 'SVM' : undefined,
        x: filteredModels,
        y: Array(filteredModels.length).fill(baselineData.SVM.value),
        line: { color: '#000000', width: 2, dash: 'dot' },
        showlegend: dsIdx === 0,
        legendgroup: 'SVM',
        xaxis: xAxisId,
        yaxis: yAxisId,
        hoverinfo: 'y+name',
        connectgaps: true,
      } as Plotly.Data);
    }

    const colCenters = [0.125, 0.375, 0.625, 0.875];
    facetedAnnotations.push({
      text: `<b>${DATASETS[ds].emoji} ${DATASETS[ds].fullName}</b>`,
      xref: 'paper',
      yref: 'paper',
      x: colCenters[dsIdx],
      y: 1.08,
      showarrow: false,
      font: { size: 14, color: '#1e293b', family: 'DM Sans' },
      xanchor: 'center',
      yanchor: 'bottom',
    });
  });

  const allValuesWithError = ALL_DATASETS.flatMap((ds) =>
    filteredModels.map((m) => {
      const data = allModelsData[ds][m];
      return data ? data.value + data.std : 0;
    })
  );
  const allValuesOnly = ALL_DATASETS.flatMap((ds) =>
    filteredModels.map((m) => {
      const data = allModelsData[ds][m];
      return data ? data.value : 0;
    }).filter((v) => v > 0)
  );
  const baselineVals = ALL_DATASETS.flatMap((ds) =>
    Object.values(baselinesByDataset[ds]).map((b) => b.value)
  ).filter((v) => v > 0);
  const maxValue = Math.max(...allValuesWithError, ...baselineVals, 0);
  const minValue = Math.min(...allValuesOnly, ...baselineVals, 1);
  const yRange = [Math.max(0, minValue - 0.08), Math.min(1, maxValue + 0.12)];

  const facetedLayout: Partial<Plotly.Layout> = {
    height: 420,
    font: { family: 'DM Sans', size: 12, color: '#0f172a' },
    paper_bgcolor: 'rgba(0,0,0,0)',
    plot_bgcolor: '#ffffff',
    margin: { l: 60, r: 20, t: 60, b: 80 },
    annotations: facetedAnnotations,
    legend: {
      orientation: 'h',
      yanchor: 'bottom',
      y: 1.15,
      xanchor: 'center',
      x: 0.5,
      bgcolor: 'rgba(0,0,0,0)',
    },
    grid: {
      rows: 1,
      columns: 4,
      pattern: 'independent',
      xgap: 0.06,
    },
    xaxis: {
      showgrid: false,
      tickangle: -45,
      tickfont: { size: 9, color: '#475569' },
      fixedrange: true,
    },
    xaxis2: {
      showgrid: false,
      tickangle: -45,
      tickfont: { size: 9, color: '#475569' },
      fixedrange: true,
    },
    xaxis3: {
      showgrid: false,
      tickangle: -45,
      tickfont: { size: 9, color: '#475569' },
      fixedrange: true,
    },
    xaxis4: {
      showgrid: false,
      tickangle: -45,
      tickfont: { size: 9, color: '#475569' },
      fixedrange: true,
    },
    yaxis: {
      title: { text: displayMetricLabel, font: { size: 12 } },
      showgrid: true,
      gridcolor: 'rgba(226,232,240,0.8)',
      tickformat: '.0%',
      tickfont: { size: 11, color: '#475569' },
      fixedrange: true,
      range: yRange,
    },
    yaxis2: {
      showgrid: true,
      gridcolor: 'rgba(226,232,240,0.8)',
      tickformat: '.0%',
      tickfont: { size: 11, color: '#475569' },
      fixedrange: true,
      range: yRange,
    },
    yaxis3: {
      showgrid: true,
      gridcolor: 'rgba(226,232,240,0.8)',
      tickformat: '.0%',
      tickfont: { size: 11, color: '#475569' },
      fixedrange: true,
      range: yRange,
    },
    yaxis4: {
      showgrid: true,
      gridcolor: 'rgba(226,232,240,0.8)',
      tickformat: '.0%',
      tickfont: { size: 11, color: '#475569' },
      fixedrange: true,
      range: yRange,
    },
  };

  return (
    <div>
      {/* Controls */}
      <div className="controls-panel">
        <div className="controls-grid">
          <div className="control-group">
            <div className="control-label">Dataset Filter</div>
            <select
              value={datasetFilter}
              onChange={(e) => setDatasetFilter(e.target.value as DatasetName | 'all')}
            >
              <option value="all">📊 All Datasets (Aggregate)</option>
              {Object.entries(DATASETS).map(([key, ds]) => (
                <option key={key} value={key}>
                  {ds.emoji} {ds.fullName}
                </option>
              ))}
            </select>
          </div>

          <div className="control-group">
            <div className="control-label">Node Selection Method</div>
            <select
              value={methodFilter}
              onChange={(e) => setMethodFilter(e.target.value as string | 'all')}
            >
              <option value="all">🔀 All Methods</option>
              {VALID_METHODS.map((method) => (
                <option key={method} value={method}>
                  🔹 {METHOD_LABELS[method]}
                </option>
              ))}
            </select>
          </div>

          <div className="control-group">
            <div className="control-label">Sample-Node Ratio</div>
            <select
              value={ratioFilter}
              onChange={(e) => {
                const val = e.target.value;
                setRatioFilter(val === 'all' ? 'all' : parseFloat(val));
              }}
            >
              <option value="all">📏 All Ratios</option>
              {VALID_RATIOS.map((ratio) => (
                <option key={ratio} value={ratio}>
                  📐 {RATIO_LABELS[ratio]}
                </option>
              ))}
            </select>
          </div>

          {availableAdjacencyMethods.length > 0 && (
            <div className="control-group">
              <div className="control-label">Graph Construction</div>
              <select
                value={adjacencyMethodFilter}
                onChange={(e) => setAdjacencyMethodFilter(e.target.value as string | 'all')}
              >
                <option value="all">🔀 All Methods</option>
                {availableAdjacencyMethods.map((m) => (
                  <option key={m} value={m}>
                    🔹 {ADJACENCY_METHOD_LABELS[m] ?? m}
                  </option>
                ))}
              </select>
            </div>
          )}

          <div className="control-group">
            <div className="control-label">Rank Models By</div>
            <select
              value={rankBy}
              onChange={(e) => setRankBy(e.target.value as RankingMetric)}
            >
              {Object.entries(RANKING_METRICS).map(([key, label]) => (
                <option key={key} value={key}>
                  📊 {label}
                </option>
              ))}
            </select>
          </div>

          <div className="control-group">
            <div className="control-label">Display Metric</div>
            <select
              value={displayMetric}
              onChange={(e) => setDisplayMetric(e.target.value as DisplayMetric)}
            >
              {Object.entries(DISPLAY_METRICS).map(([key, label]) => (
                <option key={key} value={key}>
                  🎯 {label}
                </option>
              ))}
            </select>
          </div>
        </div>
      </div>

      {/* Leaderboard Table */}
      <div className="leaderboard-section">
        <div className="section-header">
          <div className="section-title">
            <span style={{ fontSize: '1.2rem' }}>🏆</span>
            <span>Leaderboard Rankings</span>
          </div>
          <div style={{ color: '#64748b', fontSize: '0.85rem' }}>
            {subtitle} — Ranked by {rankMetricLabel}, displaying {displayMetricLabel}
          </div>
        </div>

        {leaderboard.length === 0 ? (
          <div className="p-10 text-center text-text-muted">
            No results match the current filters
          </div>
        ) : (
          <table className="leaderboard-table">
            <thead>
              <tr>
                <th>Rank</th>
                <th>Model</th>
                <th>Category</th>
                <th>{rankMetricLabel} (rank)</th>
                <th>{displayMetricLabel} (display)</th>
              </tr>
            </thead>
            <tbody>
              {leaderboard.map((entry) => (
                <tr
                  key={entry.model}
                  className={entry.rank === 1 ? 'highlight-row' : ''}
                >
                  <td
                    className={`rank-cell ${entry.rank <= 3 ? `rank-${entry.rank}` : ''}`}
                  >
                    {entry.rank}
                  </td>
                  <td className="model-cell">
                    <span
                      style={{
                        display: 'inline-block',
                        width: '12px',
                        height: '12px',
                        borderRadius: '2px',
                        backgroundColor: MODEL_COLORS[entry.model] || '#888',
                        marginRight: '8px',
                      }}
                    />
                    {entry.model}
                  </td>
                  <td className={`category-${entry.category}`}>{entry.category}</td>
                  <td className="mono-cell">
                    {entry.isBaseline ? (
                      <span style={{ color: '#94a3b8' }}>N/A</span>
                    ) : (
                      <>
                        {(entry.rankValue * 100).toFixed(1)}%
                        {entry.rankStd > 0 && (
                          <span style={{ color: '#94a3b8' }}> ± {(entry.rankStd * 100).toFixed(1)}%</span>
                        )}
                      </>
                    )}
                  </td>
                  <td className="mono-cell">
                    {(entry.displayValue * 100).toFixed(1)}%
                    {entry.displayStd > 0 && (
                      <span style={{ color: '#94a3b8' }}> ± {(entry.displayStd * 100).toFixed(1)}%</span>
                    )}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>

      {/* Faceted Bar Chart */}
      <div className="chart-card" style={{ marginTop: '24px' }}>
        <div className="chart-title">{chartTitle}</div>
        <Plot
          data={facetedChartData}
          layout={facetedLayout}
          config={{ displayModeBar: false, responsive: true }}
          style={{ width: '100%', height: '420px' }}
        />
      </div>
    </div>
  );
}
