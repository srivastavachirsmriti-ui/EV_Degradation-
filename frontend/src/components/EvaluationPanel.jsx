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
    <div id="model-evaluation" className="card chart-full">
      <h3>Performance at a glance</h3>
      <p className="subtitle">
        Results from synthetic data. Similar complaint templates stay together during validation so repeated wording cannot inflate the score.
      </p>
      <div className="validation-summary">
        <div><span className="kpi-label">Grouped validation accuracy</span><strong>{(evaluation.classification.accuracy * 100).toFixed(2)}%</strong><span>Whole complaint families held out</span></div>
        <div><span className="kpi-label">Macro F1</span><strong>{evaluation.classification.macro_f1.toFixed(3)}</strong><span>Equal weight for every category</span></div>
        <div><span className="kpi-label">Degradation MAE</span><strong>{evaluation.regression.mae.toFixed(3)}<small> pp</small></strong><span>Average absolute error</span></div>
      </div>
      <details className="validation-explainer"><summary>How should I read these numbers?</summary><p><strong>Accuracy</strong> is the share of complaints given the correct category. <strong>Macro F1</strong> balances precision and recall across all five categories. <strong>MAE</strong> is the average error in a degradation estimate, measured in percentage points.</p><p>{evaluation.note}</p><p>The feature design used this collection. An independent set of real complaints is still needed before making real-world accuracy claims.</p></details>
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
