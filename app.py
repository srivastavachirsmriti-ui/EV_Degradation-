import os

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st


DATA_PATH = "dashboard_data.csv"
VECTORIZER_PATH = "cluster_tfidf_vectorizer.joblib"
CLUSTER_MODEL_PATH = "kmeans_model.joblib"
CLASSIFIER_PATH = "final_classifier.joblib"


@st.cache_data
def load_dataset():
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)
    numeric_cols = [
        "vehicle_age_months",
        "total_charging_cycles",
        "avg_temperature_c",
        "fast_charge_ratio",
        "avg_discharge_rate_c",
        "battery_health_pct",
        "degradation_pct",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


@st.cache_resource
def load_models():
    models = {}
    for name, path in [
        ("vectorizer", VECTORIZER_PATH),
        ("clusterer", CLUSTER_MODEL_PATH),
        ("classifier", CLASSIFIER_PATH),
    ]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model file not found: {path}")
        models[name] = joblib.load(path)
    return models


def predict_issue_from_text(text: str):
    if not text or not text.strip():
        return {"issue": "Please enter a complaint description.", "cluster": None}

    models = load_models()
    vectorizer = models["vectorizer"]
    clusterer = models["clusterer"]
    classifier = models["classifier"]

    try:
        X = vectorizer.transform([text])
        cluster = int(clusterer.predict(X)[0])
    except Exception:
        cluster = None

    try:
        issue = classifier.predict(X)[0]
    except Exception:
        try:
            issue = classifier.predict([text])[0]
        except Exception:
            issue = "Unable to classify"

    return {"issue": str(issue), "cluster": cluster}


st.set_page_config(page_title="EV Battery Health Dashboard", layout="wide")


@st.cache_data
def summarize_data(df: pd.DataFrame):
    summary = {
        "records": len(df),
        "avg_battery_health": round(df["battery_health_pct"].mean(), 2) if "battery_health_pct" in df else 0,
        "avg_degradation": round(df["degradation_pct"].mean(), 2) if "degradation_pct" in df else 0,
        "issue_count": df["issue_label"].nunique() if "issue_label" in df else 0,
    }
    return summary


st.title("EV Battery Health Insights Dashboard")
st.caption("Synthetic EV owner battery and charging issue data")

try:
    df = load_dataset()
    load_models()
except FileNotFoundError as exc:
    st.error(str(exc))
    st.stop()

with st.sidebar:
    st.header("Filters")
    issue_filter = st.multiselect(
        "Issue label",
        options=sorted(df["issue_label"].dropna().unique().tolist()),
        default=sorted(df["issue_label"].dropna().unique().tolist()),
    )
    severity_filter = st.multiselect(
        "Severity",
        options=sorted(df["severity"].dropna().unique().tolist()),
        default=sorted(df["severity"].dropna().unique().tolist()),
    )
    battery_filter = st.multiselect(
        "Battery type",
        options=sorted(df["battery_type"].dropna().unique().tolist()),
        default=sorted(df["battery_type"].dropna().unique().tolist()),
    )
    model_filter = st.multiselect(
        "Vehicle model",
        options=sorted(df["car_model"].dropna().unique().tolist()),
        default=sorted(df["car_model"].dropna().unique().tolist())[:10],
    )

filtered_df = df.copy()
if issue_filter:
    filtered_df = filtered_df[filtered_df["issue_label"].isin(issue_filter)]
if severity_filter:
    filtered_df = filtered_df[filtered_df["severity"].isin(severity_filter)]
if battery_filter:
    filtered_df = filtered_df[filtered_df["battery_type"].isin(battery_filter)]
if model_filter:
    filtered_df = filtered_df[filtered_df["car_model"].isin(model_filter)]

summary = summarize_data(filtered_df)

col1, col2, col3, col4 = st.columns(4)
col1.metric("Records", f"{summary['records']:,}")
col2.metric("Avg battery health", f"{summary['avg_battery_health']}%")
col3.metric("Avg degradation", f"{summary['avg_degradation']}%")
col4.metric("Issue categories", f"{summary['issue_count']}")

st.subheader("Issue distribution")
issue_counts = filtered_df["issue_label"].value_counts().sort_values(ascending=False)
fig, ax = plt.subplots(figsize=(10, 4.5))
issue_counts.plot(kind="bar", ax=ax, color="#5B8FF9")
ax.set_title("Reported EV battery issues")
ax.set_xlabel("Issue type")
ax.set_ylabel("Count")
ax.tick_params(axis="x", rotation=30)
st.pyplot(fig)

st.subheader("Battery health comparison")
health_by_model = (
    filtered_df.groupby("car_model", as_index=False)["battery_health_pct"].mean().sort_values("battery_health_pct")
)
fig2, ax2 = plt.subplots(figsize=(11, 5))
ax2.barh(health_by_model["car_model"], health_by_model["battery_health_pct"], color="#4CAF50")
ax2.invert_yaxis()
ax2.set_title("Average battery health by vehicle model")
ax2.set_xlabel("Battery health (%)")
ax2.set_ylabel("Model")
st.pyplot(fig2)

st.subheader("Operational risk signals")
num = filtered_df["battery_health_pct"].mean() if not filtered_df.empty else 0
avg_temp = filtered_df["avg_temperature_c"].mean() if not filtered_df.empty else 0
avg_cycles = filtered_df["total_charging_cycles"].mean() if not filtered_df.empty else 0

risk_cols = st.columns(3)
risk_cols[0].metric("Average battery health", f"{num:.1f}%")
risk_cols[1].metric("Average temp", f"{avg_temp:.1f}°C")
risk_cols[2].metric("Average cycles", f"{avg_cycles:.0f}")

st.subheader("Predict issue from a customer complaint")
text_input = st.text_area(
    "Describe the EV battery problem",
    value="Fast charging on my Nissan Leaf has become much slower recently. A session that took about 26 minutes now takes nearly 57 minutes.",
    height=120,
)
if st.button("Predict issue"):
    prediction = predict_issue_from_text(text_input)
    st.success(f"Predicted issue: {prediction['issue']}")
    if prediction["cluster"] is not None:
        st.info(f"Assigned cluster: {prediction['cluster']}")

st.subheader("Filtered records")
if filtered_df.empty:
    st.warning("No records match the selected filters.")
else:
    st.dataframe(filtered_df.head(20), use_container_width=True)
