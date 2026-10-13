import { useEffect, useState } from "react";
import { getAnalytics, getHealth } from "./api.js";
import AgeTrendChart from "./components/AgeTrendChart.jsx";
import CategoryChart from "./components/CategoryChart.jsx";
import ClusterChart from "./components/ClusterChart.jsx";
import ComplaintTable from "./components/ComplaintTable.jsx";
import EvaluationPanel from "./components/EvaluationPanel.jsx";
import KpiCards from "./components/KpiCards.jsx";
import PredictForm from "./components/PredictForm.jsx";
import SeverityChart from "./components/SeverityChart.jsx";

const EVALUATION_NOTE =
  "No held-out evaluation file was provided. The classifier would score " +
  "near-perfectly on its own training data, so no accuracy claim is made. " +
  "Add a versioned evaluation file to display real metrics.";

export default function App() {
  const [health, setHealth] = useState(null);
  const [analytics, setAnalytics] = useState(null);
  const [error, setError] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    let active = true;
    (async () => {
      try {
        const [healthData, analyticsData] = await Promise.all([
          getHealth(),
          getAnalytics(),
        ]);
        if (!active) return;
        setHealth(healthData);
        setAnalytics(analyticsData);
      } catch (err) {
        if (active) setError(err.message);
      } finally {
        if (active) setLoading(false);
      }
    })();
    return () => {
      active = false;
    };
  }, []);

  const modelOk = health?.models?.loaded;
  const pillClass = error ? "bad" : modelOk ? "ok" : "warn";
  const pillText = error
    ? "API offline"
    : modelOk
      ? "API connected · models loaded"
      : "API connected · models unavailable";

  return (
    <div className="app">
      <header className="topbar">
        <div>
          <h1>EV Battery Degradation Complaint Mining</h1>
          <p>
            React + Flask dashboard over the existing classifier, K-Means and
            dataset.
          </p>
        </div>
        <span className="status-pill">
          <span className={`dot ${pillClass}`} />
          {loading ? "Connecting…" : pillText}
        </span>
      </header>

      {error && (
        <div className="banner error">
          {error} — start the backend with{" "}
          <code>python backend/app.py</code> and reload.
        </div>
      )}

      {health && !health.sklearn_version_match && (
        <div className="banner warn">
          Model/ scikit-learn version mismatch: models were trained with{" "}
          {health.expected_sklearn_version}, this environment has{" "}
          {health.sklearn_version}. Predictions should still run but may differ;
          install the pinned version for best fidelity.
        </div>
      )}

      {analytics && (
        <>
          <KpiCards kpis={analytics.kpis} dataset={analytics.dataset} />

          <section className="grid charts">
            <CategoryChart data={analytics.category_distribution} />
            <ClusterChart data={analytics.cluster_distribution} />
            <SeverityChart data={analytics.severity_distribution} />
            <AgeTrendChart data={analytics.age_trend} note={analytics.trend_note} />
            <EvaluationPanel
              evaluation={analytics.evaluation}
              note={EVALUATION_NOTE}
            />
            <PredictForm />
            <ComplaintTable
              categories={analytics.category_distribution.map((c) => c.category)}
              clusters={analytics.cluster_distribution.map((c) => c.cluster_id)}
            />
          </section>

          <p className="footer-note">
            This dashboard is built on the trained artifacts in the project root
            and reports statistics computed live from{" "}
            <code>dashboard_data.csv</code>. Figures are calculated from the
            complaint dataset only. No accuracy claim is made.
          </p>
        </>
      )}

      {loading && !error && (
        <div className="spinner">Loading dashboard data…</div>
      )}
    </div>
  );
}
