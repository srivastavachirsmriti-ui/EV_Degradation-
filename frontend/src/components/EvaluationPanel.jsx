export default function EvaluationPanel({ evaluation, note }) {
  if (!evaluation) {
    return (
      <div className="card chart-full">
        <h3>Model evaluation</h3>
        <div className="banner warn" style={{ margin: "10px 0 0" }}>
          No valid evaluation data is available, so no accuracy metrics are
          shown. {note}
        </div>
      </div>
    );
  }

  return (
    <div className="card chart-full">
      <h3>Model evaluation</h3>
      <p className="subtitle">
        {evaluation.note || "Metrics loaded from provided evaluation data"}
      </p>
      <div className="row">
        {Object.entries(evaluation.metrics || {}).map(([key, value]) => (
          <div key={key} style={{ minWidth: 140 }}>
            <div className="kpi-label">{key.replace(/_/g, " ")}</div>
            <div className="kpi-value" style={{ fontSize: "1.3rem" }}>
              {typeof value === "number" ? value.toFixed(3) : String(value)}
            </div>
          </div>
        ))}
      </div>
      <div className="table-scroll" style={{ marginTop: 20 }}>
        <table><thead><tr><th>Benchmark</th><th>Baseline</th><th>Current</th></tr></thead>
          <tbody>
            <tr><td>Classification accuracy</td><td>{(evaluation.baseline_classification.accuracy * 100).toFixed(2)}%</td><td>{(evaluation.classification.accuracy * 100).toFixed(2)}%</td></tr>
            <tr><td>Classification macro F1</td><td>{evaluation.baseline_classification.macro_f1.toFixed(3)}</td><td>{evaluation.classification.macro_f1.toFixed(3)}</td></tr>
            <tr><td>Regression MAE (percentage points)</td><td>{evaluation.baseline_regression.mae.toFixed(3)}</td><td>{evaluation.regression.mae.toFixed(3)}</td></tr>
            <tr><td>Regression R²</td><td>{evaluation.baseline_regression.r2.toFixed(3)}</td><td>{evaluation.regression.r2.toFixed(3)}</td></tr>
          </tbody>
        </table>
      </div>
    </div>
  );
}
