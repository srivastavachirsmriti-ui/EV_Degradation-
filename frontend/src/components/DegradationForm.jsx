import { useState } from "react";
import { predictDegradation } from "../api.js";

const FIELDS = [
  ["vehicle_age_months", "Vehicle age (months)", 0, 360, 1, 48],
  ["total_charging_cycles", "Charging cycles", 0, 10000, 1, 600],
  ["avg_temperature_c", "Average temperature (°C)", -50, 85, 0.1, 28],
  ["fast_charge_ratio", "Fast charge ratio (0 to 1)", 0, 1, 0.01, 0.3],
  ["avg_discharge_rate_c", "Average discharge rate (C)", 0, 10, 0.01, 1.2],
];

export default function DegradationForm() {
  const [values, setValues] = useState(Object.fromEntries(FIELDS.map(([key, , , , , value]) => [key, value])));
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [loading, setLoading] = useState(false);
  async function submit(event) {
    event.preventDefault(); setLoading(true); setError(null); setResult(null);
    try { setResult(await predictDegradation(values)); }
    catch (err) { setError(err.message); }
    finally { setLoading(false); }
  }
  return <div className="card">
    <h3>Estimate battery degradation</h3>
    <p className="subtitle">A numerical estimate from five operating inputs, trained on synthetic records.</p>
    <form onSubmit={submit}>
      <div className="regression-fields">{FIELDS.map(([key, label, min, max, step]) =>
        <label className="field" key={key}>{label}<input type="number" required min={min} max={max} step={step}
          value={values[key]} onChange={(e) => setValues({ ...values, [key]: e.target.value === "" ? "" : Number(e.target.value) })} /></label>)}</div>
      <button type="submit" disabled={loading}>{loading ? "Estimating…" : "Estimate degradation"}</button>
    </form>
    {error && <div className="banner error" role="alert">{error}</div>}
    {result && <div className="prediction" aria-live="polite">
      <h4>{result.degradation_pct.toFixed(2)}% estimated degradation</h4>
      <p>90% prediction interval: {result.interval_90.lower.toFixed(2)}% to {result.interval_90.upper.toFixed(2)}%</p>
      {result.needs_review && <div className="banner warn">Outside the training range: {result.out_of_training_range.join(", ")}. Review this estimate.</div>}
      <p className="kpi-foot">{result.note}</p>
    </div>}
  </div>;
}
