import { useEffect, useState } from "react";
import { getComplaints, getFilters } from "../api.js";

const PAGE_SIZE = 15;

// Display-only: strip the "syn_" prefix so IDs read as "0001" in the UI.
// The underlying dataset values are left untouched.
function displayId(recordId) {
  if (recordId === null || recordId === undefined) return "—";
  return String(recordId).replace(/^syn[_-]/i, "");
}

const EMPTY_FILTERS = {
  search: "",
  category: "all",
  severity: "all",
  battery_type: "all",
  cluster_id: "all",
};

export default function ComplaintTable({ categories = [], clusters = [] }) {
  const [options, setOptions] = useState({ categories, severities: [], battery_types: [], clusters });
  const [filters, setFilters] = useState(EMPTY_FILTERS);
  const [debouncedSearch, setDebouncedSearch] = useState("");
  const [page, setPage] = useState(0);
  const [data, setData] = useState({ rows: [], total: 0, count: 0 });
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  useEffect(() => {
    getFilters()
      .then(setOptions)
      .catch(() => {
        /* fall back to props */
      });
  }, []);

  useEffect(() => {
    const timer = setTimeout(() => setDebouncedSearch(filters.search), 350);
    return () => clearTimeout(timer);
  }, [filters.search]);

  useEffect(() => {
    setPage(0);
  }, [debouncedSearch, filters.category, filters.severity, filters.battery_type, filters.cluster_id]);

  useEffect(() => {
    let active = true;
    setLoading(true);
    setError(null);
    getComplaints({
      search: debouncedSearch,
      category: filters.category,
      severity: filters.severity,
      battery_type: filters.battery_type,
      cluster_id: filters.cluster_id,
      limit: PAGE_SIZE,
      offset: page * PAGE_SIZE,
    })
      .then((payload) => {
        if (active) setData(payload);
      })
      .catch((err) => {
        if (active) setError(err.message);
      })
      .finally(() => {
        if (active) setLoading(false);
      });
    return () => {
      active = false;
    };
  }, [debouncedSearch, filters.category, filters.severity, filters.battery_type, filters.cluster_id, page]);

  const update = (key) => (event) =>
    setFilters((prev) => ({ ...prev, [key]: event.target.value }));

  const totalPages = Math.max(1, Math.ceil(data.total / PAGE_SIZE));

  return (
    <div className="card chart-full">
      <h3>Complaint explorer</h3>
      <p className="subtitle">
        Search and filter the raw dataset served by GET /api/complaints
      </p>

      <div className="row" style={{ marginBottom: 14 }}>
        <label className="field" style={{ flex: 2, minWidth: 220 }}>
          Search
          <input
            value={filters.search}
            onChange={update("search")}
            placeholder="Search complaint text, model, category…"
          />
        </label>
        <label className="field" style={{ flex: 1, minWidth: 150 }}>
          Category
          <select value={filters.category} onChange={update("category")}>
            <option value="all">All</option>
            {options.categories.map((c) => (
              <option key={c} value={c}>{c}</option>
            ))}
          </select>
        </label>
        <label className="field" style={{ flex: 1, minWidth: 130 }}>
          Severity
          <select value={filters.severity} onChange={update("severity")}>
            <option value="all">All</option>
            {options.severities.map((s) => (
              <option key={s} value={s}>{s}</option>
            ))}
          </select>
        </label>
        <label className="field" style={{ flex: 1, minWidth: 130 }}>
          Battery
          <select value={filters.battery_type} onChange={update("battery_type")}>
            <option value="all">All</option>
            {options.battery_types.map((b) => (
              <option key={b} value={b}>{b}</option>
            ))}
          </select>
        </label>
        <label className="field" style={{ flex: 1, minWidth: 120 }}>
          Cluster
          <select value={filters.cluster_id} onChange={update("cluster_id")}>
            <option value="all">All</option>
            {options.clusters.map((c) => (
              <option key={c} value={String(c)}>{`Cluster ${c}`}</option>
            ))}
          </select>
        </label>
      </div>

      {error && <div className="banner error">{error}</div>}

      <div className="table-wrap">
        <table>
          <thead>
            <tr>
              <th>ID</th>
              <th>Category</th>
              <th>Severity</th>
              <th>Model</th>
              <th>Age (mo)</th>
              <th>Cycles</th>
              <th>Health %</th>
              <th>Degr. %</th>
              <th>Cluster</th>
              <th>Complaint</th>
            </tr>
          </thead>
          <tbody>
            {data.rows.map((row) => (
              <tr key={row.record_id}>
                <td>{displayId(row.record_id)}</td>
                <td>{row.issue_label}</td>
                <td>
                  <span className={`tag ${row.severity}`}>{row.severity}</span>
                </td>
                <td>{row.car_model}</td>
                <td>{row.vehicle_age_months}</td>
                <td>{row.total_charging_cycles}</td>
                <td>{row.battery_health_pct}</td>
                <td>{row.degradation_pct}</td>
                <td>{row.cluster_id}</td>
                <td style={{ minWidth: 280 }}>{row.post_text}</td>
              </tr>
            ))}
            {!loading && data.rows.length === 0 && (
              <tr>
                <td colSpan={10} className="spinner">
                  No complaints match the current filters.
                </td>
              </tr>
            )}
          </tbody>
        </table>
      </div>

      <div className="pagination">
        <span>
          {loading
            ? "Loading…"
            : `${data.total.toLocaleString()} matching · showing ${
                data.rows.length ? page * PAGE_SIZE + 1 : 0
              }–${page * PAGE_SIZE + data.rows.length}`}
        </span>
        <span className="row">
          <button
            className="secondary"
            onClick={() => setPage((p) => Math.max(0, p - 1))}
            disabled={page === 0 || loading}
          >
            Previous
          </button>
          <span style={{ alignSelf: "center" }}>
            Page {page + 1} / {totalPages}
          </span>
          <button
            className="secondary"
            onClick={() => setPage((p) => p + 1)}
            disabled={page + 1 >= totalPages || loading}
          >
            Next
          </button>
        </span>
      </div>
    </div>
  );
}
