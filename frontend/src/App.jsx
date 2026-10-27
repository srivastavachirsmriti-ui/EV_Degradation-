import { useEffect, useRef, useState } from "react";
import { getAnalytics, getHealth } from "./api.js";
import AgeTrendChart from "./components/AgeTrendChart.jsx";
import CategoryChart from "./components/CategoryChart.jsx";
import ClusterChart from "./components/ClusterChart.jsx";
import ComplaintTable from "./components/ComplaintTable.jsx";
import EvaluationPanel from "./components/EvaluationPanel.jsx";
import KpiCards from "./components/KpiCards.jsx";
import PredictForm from "./components/PredictForm.jsx";
import SeverityChart from "./components/SeverityChart.jsx";
import DegradationForm from "./components/DegradationForm.jsx";
import Icon from "./components/Icon.jsx";

const VIEWS = {
  overview: { label: "Overview", title: "A clearer view of battery health.", description: "Explore the dataset, spot recurring concerns and choose what to investigate." },
  classify: { label: "Classify a complaint", title: "Make sense of a complaint.", description: "Describe the concern in everyday language. We’ll suggest a category and show how certain the model is." },
  estimate: { label: "Estimate degradation", title: "Understand battery wear.", description: "Enter the vehicle’s operating details to get an estimated capacity loss and a range of uncertainty." },
  results: { label: "Model results", title: "See how the models perform.", description: "Compare the original models with the improved versions, and understand what the results mean." },
  records: { label: "Browse records", title: "Explore the stories behind the data.", description: "Search complaints or narrow the collection by vehicle, issue and severity." },
};
const readView = () => Object.hasOwn(VIEWS, window.location.hash.slice(1)) ? window.location.hash.slice(1) : "overview";

export default function App() {
  const [view, setView] = useState(readView);
  const [health, setHealth] = useState(null);
  const [analytics, setAnalytics] = useState(null);
  const [error, setError] = useState(null);
  const [loading, setLoading] = useState(true);
  const [retry, setRetry] = useState(0);
  const [theme, setTheme] = useState(() => {
    const saved = localStorage.getItem("ev-theme");
    return saved === "dark" || saved === "light" ? saved : "system";
  });
  const heading = useRef(null);
  useEffect(() => {
    const preference = window.matchMedia("(prefers-color-scheme: dark)");
    const apply = () => { document.documentElement.dataset.theme = theme === "system" ? (preference.matches ? "dark" : "light") : theme; };
    apply(); localStorage.setItem("ev-theme", theme);
    preference.addEventListener("change", apply);
    return () => preference.removeEventListener("change", apply);
  }, [theme]);
  useEffect(() => {
    const change = () => { setView(readView()); window.scrollTo({ top: 0 }); };
    window.addEventListener("hashchange", change);
    return () => window.removeEventListener("hashchange", change);
  }, []);
  useEffect(() => { heading.current?.focus({ preventScroll: true }); }, [view]);
  useEffect(() => {
    // Refresh evidence on navigation without unmounting forms or discarding input.
    if (view !== "results") return;
    let active = true;
    getAnalytics().then((data) => { if (active) setAnalytics(data); }).catch(() => {});
    return () => { active = false; };
  }, [view]);
  useEffect(() => {
    let active = true;
    setLoading(true); setError(null);
    Promise.all([getHealth(), getAnalytics()]).then(([h, a]) => {
      if (active) { setHealth(h); setAnalytics(a); }
    }).catch((err) => { if (active) setError(err.message); })
      .finally(() => { if (active) setLoading(false); });
    return () => { active = false; };
  }, [retry]);

  // Task-based navigation keeps a professor's review and a technician's prediction
  // workflow one click away. Keep forms mounted to preserve work between views.
  return <div className="workspace">
    <a className="skip-link" href="#main-content">Skip to content</a>
    <aside className="sidebar">
      <a className="brand" href="#overview"><span className="brand-symbol"><Icon name="battery" size={25} /></span><span>Battery<span className="brand-light">insights</span></span></a>
      <div className="nav-caption">YOUR WORKSPACE</div>
      <nav aria-label="Main navigation">{Object.entries(VIEWS).map(([key, item]) =>
        <a href={`#${key}`} key={key} className={`nav-item ${view === key ? "selected" : ""}`} aria-current={view === key ? "page" : undefined}><Icon name={key} /><span>{item.label}</span></a>)}</nav>
      <div className="sidebar-bottom">
        {analytics?.evaluation && <section className="model-insights" aria-label="Model insights"><h2>Model insights</h2><dl>
          <div><dt>Accuracy</dt><dd>{(analytics.evaluation.classification.accuracy * 100).toFixed(2)}%</dd></div>
          <div><dt>Macro F1</dt><dd>{analytics.evaluation.classification.macro_f1.toFixed(3)}</dd></div>
          <div><dt>Validation loss</dt><dd>{analytics.evaluation.development_holdout.log_loss.toFixed(3)}</dd></div>
          <div><dt>Tests</dt><dd>{analytics.evaluation.checks?.tests_passed ? `${analytics.evaluation.checks.tests_passed} passed` : "Not recorded"}</dd></div>
        </dl><p>Accuracy: grouped validation.<br />Loss: development log loss.</p><a href="#results">View full results <Icon name="arrow" size={14} /></a></section>}
        <label className="theme-control">Appearance<select value={theme} onChange={(e) => setTheme(e.target.value)} aria-label="Appearance"><option value="system">System</option><option value="light">Light</option><option value="dark">Dark</option></select></label>
        <span className={`dot ${health?.models?.loaded ? "ok" : "warn"}`} />{loading ? "Loading project" : error ? "Connection unavailable" : "Project ready"}
      </div>
    </aside>
    <main id="main-content" className="main-content">
      <div className="page-context"><span>EV battery research</span><span className="dataset-label">Synthetic dataset</span></div>
      <header className="page-heading"><p className="eyebrow">{VIEWS[view].label}</p><h1 ref={heading} tabIndex={-1}>{VIEWS[view].title}</h1><p>{VIEWS[view].description}</p></header>
      {error && <div className="banner error" role="alert"><strong>We couldn’t load the project.</strong><p>Check that the local service is running, then try again.</p><button onClick={() => setRetry(retry + 1)}>Try again</button></div>}
      {loading && !error && <div className="loading-state" role="status">Loading battery insights…</div>}
      {analytics && <>
        {view === "overview" && <>
          <KpiCards kpis={analytics.kpis} dataset={analytics.dataset} />
          <div className="section-heading"><h2>Explore the collection</h2><span>2,500 synthetic records, five issue categories</span></div>
          <section className="overview-charts"><CategoryChart data={analytics.category_distribution} /><SeverityChart data={analytics.severity_distribution} /><AgeTrendChart data={analytics.age_trend} note={analytics.trend_note} /></section>
          <div className="section-heading"><h2>What would you like to understand?</h2></div>
          <div className="action-grid"><a className="action-card" href="#classify"><span className="action-icon"><Icon name="classify" size={26} /></span><div><h3>Understand a complaint</h3><p>Turn an owner’s description into a clear issue category.</p></div><Icon name="arrow" /></a><a className="action-card" href="#estimate"><span className="action-icon warm"><Icon name="estimate" size={26} /></span><div><h3>Estimate battery wear</h3><p>Explore how operating conditions relate to capacity loss.</p></div><Icon name="arrow" /></a></div>
          <div className="research-note"><Icon name="info" /><p>This is a research prototype using generated records. The patterns shown here describe the dataset and should not be treated as findings about real vehicles.</p></div>
        </>}
        <section hidden={view !== "classify"} className="task-layout"><PredictForm /><aside className="help-panel"><h3>A useful description includes…</h3><p>What changed, when it happens and what you expected instead.</p><blockquote>“Charging stops at 60%, even when I try a different charger.”</blockquote><h3>Reading the result</h3><p>The suggested category organises the complaint. Confidence shows the model’s certainty, and a review message means a person should check the suggestion.</p><p className="small-note">A complaint category is not a battery diagnosis.</p></aside></section>
        <section hidden={view !== "estimate"} className="task-layout"><DegradationForm /><aside className="help-panel"><h3>Before you begin</h3><p>Use the vehicle’s average operating values. A fast-charge share of 0.30 means 30% of charging uses fast charging.</p><h3>What does degradation mean?</h3><p>A prediction of 4% means an estimated loss of four percentage points of the original battery capacity.</p><h3>Why show a range?</h3><p>The model cannot know the exact amount of wear. The prediction interval shows uncertainty on comparable synthetic data.</p></aside></section>
        {view === "results" && <div className="results-layout"><EvaluationPanel evaluation={analytics.evaluation} note={analytics.evaluation_note} /><details className="details-card"><summary>Explore how complaints are grouped</summary><p>These groups collect similar wording. They are exploratory and do not represent verified faults or the five classification categories.</p><ClusterChart data={analytics.cluster_distribution} /></details></div>}
        <section hidden={view !== "records"}><ComplaintTable categories={analytics.category_distribution.map((c) => c.category)} clusters={analytics.cluster_distribution.map((c) => c.cluster_id)} /></section>
        <footer className="footer-note">EV Battery Degradation Complaint Mining<span>Built for learning. Synthetic data, transparent results.</span></footer>
      </>}
    </main>
  </div>;
}
