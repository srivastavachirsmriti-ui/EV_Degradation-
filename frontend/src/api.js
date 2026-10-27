// Thin wrapper around the Flask API. In dev, VITE_API_BASE is empty and the
// Vite proxy forwards /api to the backend. In production, set VITE_API_BASE.
const API_BASE = (import.meta.env.VITE_API_BASE || "").replace(/\/$/, "");

async function request(path, options = {}) {
  let response;
  try {
    response = await fetch(`${API_BASE}${path}`, options);
  } catch (networkError) {
    throw new Error(
      "Cannot reach the API. Is the Flask backend running on port 5001?"
    );
  }

  const text = await response.text();
  let body = null;
  if (text) {
    try {
      body = JSON.parse(text);
    } catch {
      body = null;
    }
  }

  if (!response.ok) {
    const message =
      (body && (body.message || body.error)) ||
      `Request failed with status ${response.status}`;
    const error = new Error(message);
    error.status = response.status;
    error.details = body && body.details;
    throw error;
  }
  return body;
}

export function getHealth() {
  return request("/api/health");
}

export function getAnalytics() {
  return request("/api/analytics");
}

export function getFilters() {
  return request("/api/filters");
}

export function getComplaints(params = {}) {
  const query = new URLSearchParams();
  Object.entries(params).forEach(([key, value]) => {
    if (value !== undefined && value !== null && value !== "" && value !== "all") {
      query.set(key, value);
    }
  });
  const suffix = query.toString() ? `?${query.toString()}` : "";
  return request(`/api/complaints${suffix}`);
}

export function predict(text) {
  return request("/api/predict", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ text }),
  });
}

export function predictDegradation(values) {
  return request("/api/predict-degradation", {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify(values),
  });
}
