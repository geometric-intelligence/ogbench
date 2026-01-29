import { useState, useEffect, useMemo, lazy } from 'react';

// Dynamically import Plotly to avoid SSR issues
const Plot = lazy(() => import('react-plotly.js'));
import type { ResultEntry, ModelCategory, DatasetName, RankingMetric, DisplayMetric } from '../lib/types';
import { DATASETS, MODEL_CATEGORIES, MODEL_ORDER, CATEGORY_COLORS, RANKING_METRICS, DISPLAY_METRICS } from '../lib/constants';
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

  // Performance chart data - uses display metric
  const performanceChartData = useMemo(() => {
    const sorted = [...leaderboard].sort((a, b) => a.displayValue - b.displayValue);
    return {
      x: sorted.map((e) => e.displayValue),
      y: sorted.map((e) => e.model),
      colors: sorted.map((e) => CATEGORY_COLORS[e.category]),
      text: sorted.map((e) => `${(e.displayValue * 100).toFixed(1)}%`),
    };
  }, [leaderboard]);

  // Rank vs Display scatter chart data
  const rankVsDisplayData = useMemo(() => {
    // Only include non-baseline models for this chart
    const nonBaselines = leaderboard.filter((e) => !e.isBaseline);
    return {
      x: nonBaselines.map((e) => e.rankValue),
      y: nonBaselines.map((e) => e.displayValue),
      text: nonBaselines.map((e) => e.model),
      colors: nonBaselines.map((e) => CATEGORY_COLORS[e.category]),
    };
  }, [leaderboard]);

  // Dataset comparison data
  const datasetComparisonData = useMemo(() => {
    const modelData = getModelsByDataset(results, MODEL_ORDER, displayMetric);
    const filteredModels =
      modelCategory === 'all'
        ? MODEL_ORDER
        : MODEL_ORDER.filter((m) => MODEL_CATEGORIES[m] === modelCategory);

    return {
      models: filteredModels,
      datasets: (['motrpac', 'addneuromed', 'parkinsons'] as DatasetName[]).map((ds) => ({
        name: `${DATASETS[ds].emoji} ${DATASETS[ds].fullName}`,
        color: DATASETS[ds].color,
        values: filteredModels.map((m) => modelData[ds][m] || null),
      })),
    };
  }, [results, modelCategory, displayMetric]);

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
                  <td className="model-cell">{entry.model}</td>
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

      {/* Charts Grid */}
      <div className="charts-grid">
        <div className="chart-card">
          <div className="chart-title">{displayMetricLabel} by Model</div>
          <Plot
            data={[
              {
                type: 'bar',
                orientation: 'h',
                x: performanceChartData.x,
                y: performanceChartData.y,
                marker: { color: performanceChartData.colors },
                text: performanceChartData.text,
                textposition: 'outside',
                textfont: { family: 'JetBrains Mono', size: 12, color: '#0f172a' },
                cliponaxis: false,
              },
            ]}
            layout={{
              xaxis: {
                title: { text: displayMetricLabel },
                tickformat: '.0%',
                gridcolor: '#e2e8f0',
                range: [0, 1],
              },
              yaxis: { gridcolor: '#e2e8f0' },
              plot_bgcolor: '#ffffff',
              paper_bgcolor: 'rgba(0,0,0,0)',
              font: { family: 'DM Sans', color: '#0f172a' },
              margin: { l: 100, r: 80, t: 20, b: 40 },
              height: 400,
            }}
            config={{ displayModeBar: false }}
            style={{ width: '100%', height: '400px' }}
          />
        </div>

        <div className="chart-card">
          <div className="chart-title">{rankMetricLabel} vs {displayMetricLabel}</div>
          <Plot
            data={[
              {
                type: 'scatter',
                mode: 'text+markers' as const,
                x: rankVsDisplayData.x,
                y: rankVsDisplayData.y,
                text: rankVsDisplayData.text,
                textposition: 'top center',
                textfont: { family: 'DM Sans', size: 11, color: '#0f172a' },
                marker: {
                  size: 20,
                  color: rankVsDisplayData.colors,
                  line: { width: 2, color: '#ffffff' },
                },
              },
            ]}
            layout={{
              xaxis: {
                title: { text: rankMetricLabel },
                tickformat: '.0%',
                gridcolor: '#e2e8f0',
              },
              yaxis: {
                title: { text: displayMetricLabel },
                tickformat: '.0%',
                gridcolor: '#e2e8f0',
              },
              plot_bgcolor: '#ffffff',
              paper_bgcolor: 'rgba(0,0,0,0)',
              font: { family: 'DM Sans', color: '#0f172a' },
              margin: { l: 60, r: 40, t: 40, b: 60 },
              height: 400,
              showlegend: false,
            }}
            config={{ displayModeBar: false }}
            style={{ width: '100%', height: '400px' }}
          />
        </div>
      </div>

      {/* Dataset Comparison Chart */}
      <div className="chart-card" style={{ marginTop: '24px' }}>
        <div className="chart-title">{displayMetricLabel} Across Datasets</div>
        <Plot
          data={datasetComparisonData.datasets.map((ds) => ({
            type: 'bar' as const,
            name: ds.name,
            x: datasetComparisonData.models,
            y: ds.values,
            marker: { color: ds.color },
            text: ds.values.map((v) => (v !== null ? `${(v * 100).toFixed(1)}%` : '')),
            textposition: 'outside' as const,
            textfont: { family: 'JetBrains Mono', size: 10 },
          }))}
          layout={{
            barmode: 'group',
            xaxis: {
              gridcolor: '#e2e8f0',
              categoryorder: 'array',
              categoryarray: datasetComparisonData.models,
            },
            yaxis: {
              title: { text: displayMetricLabel },
              tickformat: '.0%',
              gridcolor: '#e2e8f0',
              range: [0, 1],
            },
            plot_bgcolor: '#ffffff',
            paper_bgcolor: 'rgba(0,0,0,0)',
            font: { family: 'DM Sans', color: '#0f172a' },
            margin: { l: 60, r: 40, t: 20, b: 60 },
            height: 350,
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
          style={{ width: '100%', height: '350px' }}
        />
      </div>
    </div>
  );
}
