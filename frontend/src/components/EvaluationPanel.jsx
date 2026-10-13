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
    </div>
  );
}
