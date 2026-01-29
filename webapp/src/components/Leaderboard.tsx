import { useState, useEffect, useMemo, lazy } from 'react';

// Dynamically import Plotly to avoid SSR issues
const Plot = lazy(() => import('react-plotly.js'));
import type { ResultEntry, ModelCategory, DatasetName, RankingMetric, DisplayMetric } from '../lib/types';
import { DATASETS, MODEL_CATEGORIES, MODEL_ORDER, BASELINE_MODELS, MODEL_COLORS, RANKING_METRICS, DISPLAY_METRICS } from '../lib/constants';
import { computeLeaderboard, filterResults, getModelsByDataset } from '../lib/data';

export default function Leaderboard() {
  const [results, setResults] = useState<ResultEntry[]>([]);
  const [loading, setLoading] = useState(true);
  const [datasetFilter, setDatasetFilter] = useState<DatasetName | 'all'>('all');
  const [modelCategory, setModelCategory] = useState<ModelCategory | 'all'>('all');
  // New dual-metric selection
  const [rankBy, setRankBy] = useState<RankingMetric>('val_f1_macro');
  const [displayMetric, setDisplayMetric] = useState<DisplayMetric>('test_f1_macro');

  // Load data on mount
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

  // Compute filtered results and leaderboard
  const filteredResults = useMemo(
    () => filterResults(results, datasetFilter, modelCategory),
    [results, datasetFilter, modelCategory]
  );

  const leaderboard = useMemo(
    () => computeLeaderboard(filteredResults, rankBy, displayMetric),
    [filteredResults, rankBy, displayMetric]
  );

  const subtitle =
    datasetFilter === 'all'
      ? 'Aggregated across all datasets and graph configurations'
      : `Results for ${DATASETS[datasetFilter].fullName} dataset`;

  // Get all models data by dataset for the faceted chart
  const allModelsData = useMemo(() => {
    // Include both MODEL_ORDER and BASELINE_MODELS
    const allModels = [...MODEL_ORDER, ...BASELINE_MODELS];
    return getModelsByDataset(results, allModels, displayMetric);
  }, [results, displayMetric]);

  // Get baseline values for horizontal lines (averaged across all data)
  const baselineValues = useMemo(() => {
    const baselines: Record<string, { value: number; std: number }> = {};
    for (const model of BASELINE_MODELS) {
      const entries = results.filter((r) => r.model === model);
      if (entries.length > 0) {
        // For baselines, get the display metric value
        let value = 0;
        let std = 0;
        if (displayMetric === 'test_accuracy') {
          value = entries.reduce((sum, e) => sum + e.test_accuracy, 0) / entries.length;
          std = entries.reduce((sum, e) => sum + e.test_accuracy_std, 0) / entries.length;
        } else if (displayMetric === 'test_f1_macro') {
          value = entries.reduce((sum, e) => sum + e.test_f1_macro, 0) / entries.length;
          std = entries.reduce((sum, e) => sum + e.test_f1_macro_std, 0) / entries.length;
        } else if (displayMetric === 'auroc') {
          value = entries.reduce((sum, e) => sum + e.auroc, 0) / entries.length;
          std = entries.reduce((sum, e) => sum + e.auroc_std, 0) / entries.length;
        }
        baselines[model] = { value, std };
      }
    }
    return baselines;
  }, [results, displayMetric]);

  // Get baseline values per dataset for faceted chart
  const baselinesByDataset = useMemo(() => {
    const baselines: Record<DatasetName, Record<string, { value: number; std: number }>> = {
      motrpac: {},
      addneuromed: {},
      parkinsons: {},
    };
    
    for (const ds of ['motrpac', 'addneuromed', 'parkinsons'] as DatasetName[]) {
      for (const model of BASELINE_MODELS) {
        const entries = results.filter((r) => r.model === model && r.dataset === ds);
        if (entries.length > 0) {
          let value = 0;
          let std = 0;
          if (displayMetric === 'test_accuracy') {
            value = entries.reduce((sum, e) => sum + e.test_accuracy, 0) / entries.length;
            std = entries.reduce((sum, e) => sum + e.test_accuracy_std, 0) / entries.length;
          } else if (displayMetric === 'test_f1_macro') {
            value = entries.reduce((sum, e) => sum + e.test_f1_macro, 0) / entries.length;
            std = entries.reduce((sum, e) => sum + e.test_f1_macro_std, 0) / entries.length;
          } else if (displayMetric === 'auroc') {
            value = entries.reduce((sum, e) => sum + e.auroc, 0) / entries.length;
            std = entries.reduce((sum, e) => sum + e.auroc_std, 0) / entries.length;
          }
          baselines[ds][model] = { value, std };
        }
      }
    }
    return baselines;
  }, [results, displayMetric]);

  // Filter models based on category selection
  const filteredModels = useMemo(() => {
    if (modelCategory === 'all') return MODEL_ORDER;
    return MODEL_ORDER.filter((m) => MODEL_CATEGORIES[m] === modelCategory);
  }, [modelCategory]);

  // Get label for the display metric
  const displayMetricLabel = DISPLAY_METRICS[displayMetric];
  const rankMetricLabel = RANKING_METRICS[rankBy];

  if (loading) {
    return (
      <div className="flex items-center justify-center p-16">
        <div className="text-text-muted">Loading data...</div>
      </div>
    );
  }

  // Build faceted chart data (one subplot per dataset)
  const datasets: DatasetName[] = ['motrpac', 'addneuromed', 'parkinsons'];
  const facetedChartData: Plotly.Data[] = [];
  const facetedAnnotations: Partial<Plotly.Annotations>[] = [];

  // Create bar traces for each dataset
  datasets.forEach((ds, dsIdx) => {
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

    // Add horizontal lines for baselines
    const baselineData = baselinesByDataset[ds];
    
    // ElasticNet line (dashed)
    if (baselineData.ElasticNet) {
      facetedChartData.push({
        type: 'scatter',
        mode: 'lines',
        name: dsIdx === 0 ? 'Elastic Net' : undefined,
        x: [filteredModels[0], filteredModels[filteredModels.length - 1]],
        y: [baselineData.ElasticNet.value, baselineData.ElasticNet.value],
        line: { color: '#000000', width: 2, dash: 'dash' },
        showlegend: dsIdx === 0,
        legendgroup: 'ElasticNet',
        xaxis: xAxisId,
        yaxis: yAxisId,
        hoverinfo: 'name+y',
      } as Plotly.Data);
    }

    // SVM line (dotted)
    if (baselineData.SVM) {
      facetedChartData.push({
        type: 'scatter',
        mode: 'lines',
        name: dsIdx === 0 ? 'SVM' : undefined,
        x: [filteredModels[0], filteredModels[filteredModels.length - 1]],
        y: [baselineData.SVM.value, baselineData.SVM.value],
        line: { color: '#000000', width: 2, dash: 'dot' },
        showlegend: dsIdx === 0,
        legendgroup: 'SVM',
        xaxis: xAxisId,
        yaxis: yAxisId,
        hoverinfo: 'name+y',
      } as Plotly.Data);
    }

    // Add subplot title annotation with emoji
    const colCenters = [0.16, 0.5, 0.84];
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

  // Compute y-axis range for faceted chart - tighter range to show differences
  const allValuesWithError = datasets.flatMap((ds) =>
    filteredModels.map((m) => {
      const data = allModelsData[ds][m];
      return data ? data.value + data.std : 0;
    })
  );
  const allValuesOnly = datasets.flatMap((ds) =>
    filteredModels.map((m) => {
      const data = allModelsData[ds][m];
      return data ? data.value : 0;
    }).filter((v) => v > 0)
  );
  const baselineVals = Object.values(baselineValues).map((b) => b.value).filter((v) => v > 0);
  const maxValue = Math.max(...allValuesWithError, ...baselineVals);
  const minValue = Math.min(...allValuesOnly, ...baselineVals);
  // Tight range: start 5% below minimum, end 10% above max (for text labels)
  const yRange = [Math.max(0, minValue - 0.08), Math.min(1, maxValue + 0.12)];

  // Build faceted chart layout
  const facetedLayout: Partial<Plotly.Layout> = {
    height: 400,
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
      columns: 3,
      pattern: 'independent',
      xgap: 0.08,
    },
    xaxis: {
      showgrid: false,
      tickangle: -45,
      tickfont: { size: 10, color: '#475569' },
      fixedrange: true,
    },
    xaxis2: {
      showgrid: false,
      tickangle: -45,
      tickfont: { size: 10, color: '#475569' },
      fixedrange: true,
    },
    xaxis3: {
      showgrid: false,
      tickangle: -45,
      tickfont: { size: 10, color: '#475569' },
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
  };

  // Build dataset comparison chart data (grouped bars with error bars and baseline lines)
  const datasetComparisonData: Plotly.Data[] = [];
  
  // Add bar traces for each dataset
  datasets.forEach((ds) => {
    const values: (number | null)[] = [];
    const errors: number[] = [];
    const textLabels: string[] = [];
    
    for (const model of filteredModels) {
      const data = allModelsData[ds][model];
      if (data && data.value > 0) {
        values.push(data.value);
        errors.push(data.std);
        textLabels.push(`${(data.value * 100).toFixed(1)}%`);
      } else {
        values.push(null);
        errors.push(0);
        textLabels.push('');
      }
    }

    datasetComparisonData.push({
      type: 'bar',
      name: `${DATASETS[ds].emoji} ${DATASETS[ds].fullName}`,
      x: filteredModels,
      y: values,
      error_y: {
        type: 'data',
        array: errors,
        visible: true,
        color: '#333333',
        thickness: 1.5,
        width: 3,
      },
      marker: { color: DATASETS[ds].color },
      text: textLabels,
      textposition: 'outside',
      textfont: { family: 'JetBrains Mono', size: 9 },
    } as Plotly.Data);
  });

  // Add horizontal lines for baselines (ElasticNet - dashed, SVM - dotted)
  if (baselineValues.ElasticNet) {
    datasetComparisonData.push({
      type: 'scatter',
      mode: 'lines',
      name: 'Elastic Net',
      x: [filteredModels[0], filteredModels[filteredModels.length - 1]],
      y: [baselineValues.ElasticNet.value, baselineValues.ElasticNet.value],
      line: { color: '#000000', width: 2, dash: 'dash' },
      hoverinfo: 'name+y',
    } as Plotly.Data);
  }

  if (baselineValues.SVM) {
    datasetComparisonData.push({
      type: 'scatter',
      mode: 'lines',
      name: 'SVM',
      x: [filteredModels[0], filteredModels[filteredModels.length - 1]],
      y: [baselineValues.SVM.value, baselineValues.SVM.value],
      line: { color: '#000000', width: 2, dash: 'dot' },
      hoverinfo: 'name+y',
    } as Plotly.Data);
  }

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
            <div className="control-label">Model Category</div>
            <select
              value={modelCategory}
              onChange={(e) => setModelCategory(e.target.value as ModelCategory | 'all')}
            >
              <option value="all">🔷 All Models</option>
              <option value="gnn">🌐 GNN Models Only</option>
              <option value="neural">🧠 Neural Networks Only</option>
              <option value="baseline">📉 Baselines Only</option>
            </select>
          </div>

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

      {/* Faceted Bar Chart - Performance by Dataset */}
      <div className="chart-card" style={{ marginTop: '24px' }}>
        <div className="chart-title">Best Overall Model Performance by {displayMetricLabel}</div>
        <Plot
          data={facetedChartData}
          layout={facetedLayout}
          config={{ displayModeBar: false, responsive: true }}
          style={{ width: '100%', height: '400px' }}
        />
      </div>

      {/* Dataset Comparison Chart */}
      <div className="chart-card" style={{ marginTop: '24px' }}>
        <div className="chart-title">{displayMetricLabel} Across Datasets</div>
        <Plot
          data={datasetComparisonData}
          layout={{
            barmode: 'group',
            xaxis: {
              gridcolor: '#e2e8f0',
              categoryorder: 'array',
              categoryarray: filteredModels,
              tickangle: -45,
            },
            yaxis: {
              title: { text: displayMetricLabel },
              tickformat: '.0%',
              gridcolor: '#e2e8f0',
              range: yRange, // Use same tight range as faceted chart
            },
            plot_bgcolor: '#ffffff',
            paper_bgcolor: 'rgba(0,0,0,0)',
            font: { family: 'DM Sans', color: '#0f172a' },
            margin: { l: 60, r: 40, t: 20, b: 80 },
            height: 400,
            legend: {
              orientation: 'h',
              yanchor: 'bottom',
              y: 1.02,
              xanchor: 'right',
              x: 1,
              bgcolor: 'rgba(0,0,0,0)',
            },
          }}
          config={{ displayModeBar: false }}
          style={{ width: '100%', height: '400px' }}
        />
      </div>
    </div>
  );
}
