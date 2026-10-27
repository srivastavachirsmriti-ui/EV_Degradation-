import { useState } from "react";
import { predict } from "../api.js";

const SAMPLE =
  "Fast charging on my Nissan Leaf has become much slower recently. " +
  "A session that took about 26 minutes now takes nearly 57 minutes.";

export default function PredictForm() {
  const [text, setText] = useState(SAMPLE);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);

  async function onSubmit(event) {
    event.preventDefault();
    if (!text.trim() || loading) return;
    setLoading(true);
    setError(null);
    setResult(null);
    try {
      const response = await predict(text);
      setResult(response);
    } catch (err) {
      setError(err.message || "Prediction failed.");
    } finally {
      setLoading(false);
    }
  }

  const scores = result?.decision_scores ?? [];
  const topScore = scores.length ? scores[0].score : 0;
  const minScore = scores.length ? Math.min(...scores.map((s) => s.score)) : 0;

  return (
    <div className="card">
      <h3>Classify a complaint</h3>
      <p className="subtitle">
        Classifies the complaint with a fast calibrated model and flags uncertain results for review.
      </p>
      <form onSubmit={onSubmit}>
        <label className="field">
          Complaint text
          <textarea
            value={text}
            onChange={(event) => setText(event.target.value)}
            placeholder="Describe the EV battery problem…"
            maxLength={5000}
          />
        </label>
        <div className="row" style={{ marginTop: 12 }}>
          <button type="submit" disabled={loading || !text.trim()}>
            {loading ? "Predicting…" : "Predict issue"}
          </button>
          <button
            type="button"
            className="secondary"
            onClick={() => setText(SAMPLE)}
            disabled={loading}
          >
            Load sample
          </button>
        </div>
      </form>

      {error && <div className="banner error">{error}</div>}

      {result && (
        <div className="prediction">
          <h4>Prediction</h4>
          {result.needs_review && <div className="banner warn">Low confidence. Please review this category manually.</div>}
          <div className="row">
            <span className="badge">Issue: {result.issue_category}</span>
            <span className="badge">{result.cluster_label}</span>
          </div>

          <p className="kpi-foot" style={{ marginTop: 10 }}>
            Cleaned text: <em>{result.cleaned_text}</em>
          </p>

          {result.probabilities ? (
            <>
            <p className="kpi-foot">{result.probability_note} Path: {result.classification_path}.</p>
            <div className="score-list">
              {Object.entries(result.probabilities).map(([label, value]) => (
                <div className="score-row" key={label}>
                  <span>{label}</span>
                  <div className="score-track">
                    <div className="score-fill" style={{ width: `${value * 100}%` }} />
                  </div>
                  <span>{(value * 100).toFixed(1)}%</span>
                </div>
              ))}
            </div>
            </>
          ) : (
            <>
              <p className="kpi-foot" style={{ marginTop: 12 }}>
                {result.probability_note}
              </p>
              <div className="score-list">
                {scores.map((item) => {
                  const span = Math.abs(topScore - minScore) || 1;
                  const width = Math.max(4, ((item.score - minScore) / span) * 100);
                  return (
                    <div className="score-row" key={item.label}>
                      <span>{item.label}</span>
                      <div className="score-track">
                        <div className="score-fill" style={{ width: `${width}%` }} />
                      </div>
                      <span>{item.score.toFixed(2)}</span>
                    </div>
                  );
                })}
              </div>
            </>
          )}
        </div>
      )}
    </div>
  );
}
