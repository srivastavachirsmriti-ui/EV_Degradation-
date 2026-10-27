# EV Battery Degradation Complaint Mining

A college project that classifies synthetic EV complaints, explores text clusters,
 and estimates battery degradation from operating inputs. The application includes
 a React/Vite dashboard, Flask API, reproducible model selection, and versioned
 evaluation evidence. All 2,500 records are synthetic; scores are internal
 development benchmarks and do not establish real-vehicle performance.

## Run locally

Use **Python 3.12** and Node.js 18+.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r backend/requirements.txt
python backend/app.py
```

In another terminal:

```bash
cd frontend
npm ci
npm run dev
```

Open http://localhost:5173. The API binds to 127.0.0.1:5001. `npm run build`
creates the frontend production build. The Vite development server proxies `/api`
to Flask. Set `VITE_API_BASE` for a separately hosted frontend and configure
`EV_CORS_ORIGINS` on the API. `EV_API_HOST`, `EV_API_PORT`, `EV_API_DEBUG`,
`EV_DATA_PATH`, `EV_CLASSIFIER_PATH`, `EV_REGRESSOR_PATH`,
`EV_CLUSTER_VECTORIZER_PATH`, `EV_KMEANS_PATH`, and `EV_EVALUATION_PATH`
provide optional environment overrides. Restart after changing data or models:
loaded models, dataset analytics, and cluster metadata are process-cached.

The legacy Streamlit app uses the same prediction service. Install
`requirements.txt`, then run `streamlit run app.py`.

## Models and validation

| Measured result | Original / reconstructed baseline | Improved |
| --- | ---: | ---: |
| Classification pooled grouped OOF accuracy | 51.88% | 87.36% |
| Classification pooled grouped OOF macro F1 | 0.500 | 0.858 |
| Regression MAE (percentage points) | 1.125 | 1.064 |
| Regression R² | 0.515 | 0.569 |

The accuracy increase is **35.48 percentage points**. These are synthetic,
post-selection development results, with only 25 independent wording families.
The classifier has no independent external test. Regression uses the fixed
500-row synthetic test. Full protocols and per-class errors follow below.

The dashboard offers five task views, system/light/dark appearance, preserved
form inputs and a bottom-left Model insights summary. See [design rationale](docs/DESIGN.md)
for the audience, palette, navigation and usability decisions.

The original TF-IDF/SGD classifier used only four optimization passes. A clone
retrained on template-separated folds supplies a comparable baseline. Exact
text duplicates and substitutions of vehicle names/numbers must not cross folds.
There are only **25 template families**, five per category.

`ml/train.py` compares eight text candidates using four development folds. It
adds 14 explicit symptom/intent features and context interactions. A calibrated,
compact logistic model handles high-confidence inputs. A richer domain + TF-IDF
model handles the remaining inputs. Sigmoid calibration uses family-separated
folds, not the final evaluation labels. Confidence is only meaningful within the
synthetic domain. Unsupported wording and negation trigger manual-review flags.
Clustering remains exploratory and retains the original K-Means artifacts.

The final classifier retains the 504-row template holdout. That partition was
reused during feature development and is therefore a **development holdout**,
not an independent final test. `ml/validate.py` also evaluates all 25 families in
five outer folds, with calibration in four inner folds. Feature design has seen
this corpus, so even this score is internal post-selection validation. See
`benchmarks/validation.json` for the primary full-corpus result and per-class
scores. Do not present the original artifact's resubstitution score as accuracy.

Regression compares Ridge, quadratic Ridge, Random Forest, Extra Trees and
histogram gradient boosting. Only age, charging cycles, temperature, fast-charge
ratio and discharge rate are inputs. `battery_health_pct + degradation_pct = 100`
exactly, so using health would leak the target. Text, issue labels, severity and
stored predictions are also excluded. The 500-row test partition matches the
reconstructed mean/RF baselines. The fitted regressor uses 1,600 rows and reserves
400 separate rows for a 90% split-conformal prediction interval. The API flags
inputs outside fitted feature ranges. R² is variance explained, not accuracy.

## Reproduce

```bash
# Starting commit artifacts only; do not overwrite a frozen baseline casually.
python -m ml.baseline
# Uses the recorded baseline partition, fits models and writes artifacts/metrics.
python -m ml.train
# Runs full template OOF comparisons and warm inference timings.
python -m ml.validate
python -m pytest backend/tests -q
```

To rerun the original baseline, recover `final_classifier.joblib` from starting
commit `e69fa94` in a separate checkout. The currently deployed file is a model
bundle dictionary, not the original pipeline. The API supports both formats.

Machine-readable evidence stays versioned under `benchmarks/`: data/artifact
SHA-256 hashes, model candidates and fold scores, row IDs, family splits,
confusion matrices, regression predictions, and warm local timings. The failed
generic-only experiment is retained as `exploratory_generic.json`. This makes
improvements inspectable rather than relying on claims in a PDF. Metrics are
hidden by the API if data/model hashes do not match.

## API

| Method | Endpoint | Behavior |
| --- | --- | --- |
| GET | `/api/health` | Model/data status and runtime-version warnings |
| POST | `/api/predict` | `{ "text": "...", "include_cluster": true }` |
| POST | `/api/predict-batch` | `{ "texts": ["..."], "include_cluster": false }`, 1-100 records |
| POST | `/api/predict-degradation` | Exactly the five numeric fields below |
| GET | `/api/analytics` | Dataset distributions and matching evaluation evidence |
| GET | `/api/complaints` | Text/category/severity/battery/cluster filters and pagination |
| GET | `/api/filters` | Available filter values |

```json
{
  "vehicle_age_months": 48,
  "total_charging_cycles": 600,
  "avg_temperature_c": 28,
  "fast_charge_ratio": 0.3,
  "avg_discharge_rate_c": 1.2
}
```

Prediction responses include probabilities, `confidence`, `needs_review`, the
routing path, and an optional cluster. Batch calls transform arrays together.
A bounded cache accelerates repeated single complaints without exposing mutable
cached dictionaries. Inputs reject non-finite numbers, booleans as numbers,
unreadable text, oversized bodies and target/extra fields. File artifacts must
be trusted: joblib deserialization executes Python and is not an upload format.

## Report and presentation

The completed academic report and slide deck are generated locally by
`scripts/build_deliverables.py` after installing `requirements-artifacts.txt`.
They preserve the student team and registration numbers. `output/`, `tmp/`,
reports, PDFs, PPTX files and screenshots are ignored and must not be committed.
Only the reproducible build code and machine-readable benchmark evidence are
tracked. No video or external validation is fabricated.
