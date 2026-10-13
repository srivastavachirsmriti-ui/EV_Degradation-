# EV Battery Degradation Complaint Mining

A full-stack dashboard for exploring EV battery complaints and classifying new
complaints with the **existing trained artifacts** in this repository.

* **Backend** — Python **Flask** REST API (`backend/`)
* **Frontend** — **React + Vite** dashboard (`frontend/`)

The backend loads the pre-trained models directly from the project root and reads
the dataset at request time. **No model is retrained and no existing file is
modified.**

---

## Project structure

```
ev_dashboard_bundle/
├── final_classifier.joblib            # existing (untouched) trained pipeline
├── cluster_tfidf_vectorizer.joblib    # existing (untouched) TF-IDF vectorizer
├── kmeans_model.joblib                # existing (untouched) K-Means model
├── dashboard_data.csv                 # existing (untouched) dataset
├── app.py                             # existing Streamlit prototype (untouched)
├── requirements.txt                   # existing Streamlit deps (untouched)
├── backend/
│   ├── app.py                # Flask API entrypoint
│   ├── config.py             # paths + CORS + server config (env-overridable)
│   ├── text_cleaning.py      # reproduced training-time cleaning
│   ├── model_service.py      # loads/serves the .joblib models
│   ├── analytics.py          # stats computed from dashboard_data.csv
│   ├── requirements.txt
│   ├── conftest.py
│   └── tests/test_api.py
├── frontend/
│   ├── index.html
│   ├── package.json
│   ├── vite.config.js        # dev proxy: /api -> http://127.0.0.1:5001
│   └── src/
│       ├── main.jsx
│       ├── App.jsx
│       ├── api.js
│       ├── styles.css
│       └── components/       # KPI cards, charts, table, predict form, …
├── .gitignore
└── README.md
```

---

## Prerequisites

* **Python 3.12** (models were pickled with scikit-learn **1.6.1**)
* **Node.js 18+** and npm

---

## 1. Run the backend (Flask API)

From the project root:

```bash
# create + activate a virtual environment
python3 -m venv backend/.venv
source backend/.venv/bin/activate        # Windows: backend\.venv\Scripts\activate

# install dependencies (pins scikit-learn==1.6.1 to match the trained models)
pip install -r backend/requirements.txt

# start the API
python backend/app.py
```

The API runs at **http://127.0.0.1:5001**. Verify:

```bash
curl http://127.0.0.1:5001/api/health
```

### Backend environment overrides (optional)

| Variable | Default | Purpose |
| --- | --- | --- |
| `EV_DATA_PATH` | `./dashboard_data.csv` | dataset location |
| `EV_CLASSIFIER_PATH` | `./final_classifier.joblib` | classifier |
| `EV_CLUSTER_VECTORIZER_PATH` | `./cluster_tfidf_vectorizer.joblib` | cluster TF-IDF |
| `EV_KMEANS_PATH` | `./kmeans_model.joblib` | K-Means |
| `EV_API_HOST` / `EV_API_PORT` | `127.0.0.1` / `5001` | bind address |
| `EV_API_DEBUG` | `0` | Flask debug mode |
| `EV_CORS_ORIGINS` | Vite localhost origins | comma-separated allowed origins |

---

## 2. Run the frontend (React + Vite)

In a **second terminal**:

```bash
cd frontend
npm install
npm run dev
```

Open **http://localhost:5173**. During development Vite proxies `/api` to
`http://127.0.0.1:5001`, so both apps are reached from one origin.

For a production build:

```bash
cd frontend
npm run build      # outputs frontend/dist
npm run preview    # serves the build on http://localhost:4173
```

When serving a production build from a different origin than the API, set
`VITE_API_BASE` (see `frontend/.env.example`) and add that origin to
`EV_CORS_ORIGINS`.

---

## Using it in VS Code

1. Open the project folder (`File → Open Folder…`).
2. Open two integrated terminals (`Terminal → Split Terminal`).
3. **Terminal 1 — backend:** run the four backend commands above.
4. **Terminal 2 — frontend:** `cd frontend && npm install && npm run dev`.
5. Ctrl/Cmd-click the `http://localhost:5173` link Vite prints.

Tip: the repo ignores `.vscode/*` except `extensions.json`. Recommended
extensions: **Python**, **Pylance**, **ES7+ React snippets**, **Prettier**.

---

## API reference

| Method | Endpoint | Description |
| --- | --- | --- |
| `GET` | `/api/health` | Service, model, and dataset status |
| `POST` | `/api/predict` | Classify + cluster a new complaint |
| `GET` | `/api/analytics` | KPIs + distributions computed from the CSV |
| `GET` | `/api/complaints` | Searchable/filterable complaint table (paginated) |
| `GET` | `/api/filters` | Distinct filter values for the UI |

### `POST /api/predict`

Request:

```bash
curl -X POST http://127.0.0.1:5001/api/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "Fast charging on my Nissan Leaf is much slower than before."}'
```

Response (abridged):

```json
{
  "cleaned_text": "fast charging on my nissan leaf is much slower than before.",
  "issue_category": "Charging Issue",
  "cluster_id": 0,
  "cluster_label": "Cluster 0 – Charging Issue",
  "supports_probabilities": false,
  "probabilities": null,
  "decision_scores": [{ "label": "Charging Issue", "score": 0.31 }],
  "probability_note": "…uncalibrated decision-function margins, not probabilities."
}
```

Errors return `{"error": true, "message": "...", "status": <code>}`.

---

## Tests

```bash
cd backend
source .venv/bin/activate
pytest -q
```

The suite covers `/api/health`, `/api/predict` (success + validation errors),
`/api/analytics`, `/api/complaints`, `/api/filters`, 404 handling, and the
reproduced cleaning function.

---

## Important notes (please read)

* **Synthetic data.** Every row has `data_origin = "synthetic"`. Figures describe
  this corpus only and are **not** evidence of real-world accuracy.
* **No accuracy claim.** No held-out evaluation file exists. The models would
  score ~100% on their own training data, which is meaningless, so the dashboard
  shows an "evaluation not available" message instead of a misleading number.
* **No probabilities.** The classifier is `SGDClassifier(loss="hinge")`, which
  has no `predict_proba`. The API returns `probabilities: null` and exposes
  uncalibrated decision-function margins separately, clearly labelled.
* **No timestamps.** The dataset has no date column, so the "time trend" uses
  `vehicle_age_months` buckets as a clearly-labelled proxy.
* **Model version.** `scikit-learn==1.6.1` is pinned because that is the version
  the artifacts were trained with. `/api/health` reports whether the running
  version matches.
* **Assets are untouched.** `app.py`, `requirements.txt`, `dashboard_data.csv`
  and all three `.joblib` files are read in place and were not modified.

### Reference: reproduced cleaning

The training cleaning was reverse-engineered from the stored `clean_text` column
and reproduces it for **all 2,500 rows**:

1. lowercase
2. replace every character except `a-z`, `0-9`, `.`, `%`, whitespace with a space
3. collapse whitespace and strip
