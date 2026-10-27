"""Genuine dashboard statistics computed from the existing dataset.

Everything here is derived at request time from ``dashboard_data.csv``. No
values are hardcoded and no evaluation metrics are fabricated.

Notes about the dataset:
* ``data_origin`` is ``synthetic`` for every row, so these figures describe the
  synthetic corpus only and must not be read as real-world performance.
* There is no timestamp column, so the "time trend" is approximated with
  ``vehicle_age_months`` buckets and labelled as such.
* No held-out evaluation file exists, therefore ``evaluation`` is ``None``.
"""

from __future__ import annotations

import math
import json
import hashlib
import threading
from functools import lru_cache
from typing import Any

import numpy as np
import pandas as pd

from config import DATA_PATH, EVALUATION_PATH, PROJECT_ROOT, CLASSIFIER_PATH, REGRESSOR_PATH

_LOCK = threading.Lock()
_CACHE: dict[str, Any] = {}

AGE_BINS = [0, 12, 24, 48, 72, 96, float("inf")]
AGE_LABELS = ["0-11", "12-23", "24-47", "48-71", "72-95", "96+"]

NUMERIC_COLUMNS = [
    "vehicle_age_months",
    "total_charging_cycles",
    "avg_temperature_c",
    "fast_charge_ratio",
    "avg_discharge_rate_c",
    "battery_health_pct",
    "degradation_pct",
]


def _json_safe(value: Any) -> Any:
    """Recursively convert numpy/pandas types and NaN to JSON-friendly values."""
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
    if isinstance(value, float):
        return None if (math.isnan(value) or math.isinf(value)) else value
    if value is pd.NA or value is None:
        return None
    return value


def load_dataframe() -> pd.DataFrame:
    with _LOCK:
        if "df" not in _CACHE:
            if not DATA_PATH.exists():
                raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")
            df = pd.read_csv(DATA_PATH)
            for col in NUMERIC_COLUMNS:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors="coerce")
            _CACHE["df"] = df
        return _CACHE["df"]


def _distribution(series: pd.Series, key_name: str) -> list[dict[str, Any]]:
    counts = series.value_counts(dropna=True)
    return [{key_name: str(k), "count": int(v)} for k, v in counts.items()]


@lru_cache(maxsize=1)
def cluster_labels() -> dict[int, dict[str, Any]]:
    """Derive a human label for each K-Means cluster from the dataset itself."""
    df = load_dataframe()
    labels: dict[int, dict[str, Any]] = {}
    if "cluster_id" not in df.columns or "issue_label" not in df.columns:
        return labels
    for cluster_id, group in df.groupby("cluster_id"):
        counts = group["issue_label"].value_counts()
        dominant = str(counts.idxmax())
        labels[int(cluster_id)] = {
            "cluster_id": int(cluster_id),
            "dominant_issue": dominant,
            "dominant_share": round(float(counts.iloc[0]) / len(group), 3),
            "size": int(len(group)),
        }
    return labels


def _age_trend(df: pd.DataFrame) -> list[dict[str, Any]]:
    if "vehicle_age_months" not in df.columns:
        return []
    work = df.copy()
    work["age_bucket"] = pd.cut(
        work["vehicle_age_months"], bins=AGE_BINS, labels=AGE_LABELS, right=False
    )
    rows: list[dict[str, Any]] = []
    for label in AGE_LABELS:
        subset = work[work["age_bucket"] == label]
        if subset.empty:
            continue
        rows.append(
            {
                "age_bucket": label,
                "count": int(len(subset)),
                "avg_battery_health": round(float(subset["battery_health_pct"].mean()), 2),
                "avg_degradation": round(float(subset["degradation_pct"].mean()), 2),
                "avg_cycles": round(float(subset["total_charging_cycles"].mean()), 1),
            }
        )
    return rows


def evaluation_payload():
    if not EVALUATION_PATH.exists():
        return None
    result = json.loads(EVALUATION_PATH.read_text())
    if result["dataset_sha256"] != hashlib.sha256(DATA_PATH.read_bytes()).hexdigest():
        return None
    for name, path in [("final_classifier.joblib", CLASSIFIER_PATH), ("degradation_regressor.joblib", REGRESSOR_PATH)]:
        if not path.exists() or result["artifact_sha256"][name] != hashlib.sha256(path.read_bytes()).hexdigest():
            return None
    baseline = json.loads((PROJECT_ROOT / "benchmarks/baseline.json").read_text())
    return {"note": result["note"], "classification": result["classification"],
            "regression": result["regression"],
            "baseline_classification": baseline["reconstructed_sgd_group_test"],
            "baseline_regression": baseline["regression_reconstructed_rf"],
            "metrics": {"classification_accuracy": result["classification"]["accuracy"],
                        "classification_macro_f1": result["classification"]["macro_f1"],
                        "regression_mae": result["regression"]["mae"], "regression_r2": result["regression"]["r2"]}}


@lru_cache(maxsize=1)
def analytics_payload() -> dict[str, Any]:
    df = load_dataframe()
    total = int(len(df))

    kpis = {
        "total_complaints": total,
        "avg_battery_health": round(float(df["battery_health_pct"].mean()), 2),
        "avg_degradation": round(float(df["degradation_pct"].mean()), 2),
        "avg_vehicle_age_months": round(float(df["vehicle_age_months"].mean()), 1),
        "avg_charging_cycles": round(float(df["total_charging_cycles"].mean()), 1),
        "issue_categories": int(df["issue_label"].nunique()),
        "clusters": int(df["cluster_id"].nunique()),
        "high_severity": int((df["severity"] == "High").sum()),
        "high_severity_share": round(float((df["severity"] == "High").mean()), 3),
    }

    labels = cluster_labels()
    cluster_distribution = []
    if "cluster_id" in df.columns:
        counts = df["cluster_id"].value_counts().sort_index()
        for cluster_id, count in counts.items():
            meta = labels.get(int(cluster_id), {})
            cluster_distribution.append(
                {
                    "cluster_id": int(cluster_id),
                    "count": int(count),
                    "dominant_issue": meta.get("dominant_issue"),
                    "dominant_share": meta.get("dominant_share"),
                }
            )

    car_model = []
    if "car_model" in df.columns:
        grouped = (
            df.groupby("car_model")
            .agg(
                count=("record_id", "count"),
                avg_battery_health=("battery_health_pct", "mean"),
                avg_degradation=("degradation_pct", "mean"),
            )
            .reset_index()
            .sort_values("avg_battery_health")
        )
        car_model = [
            {
                "car_model": str(row["car_model"]),
                "count": int(row["count"]),
                "avg_battery_health": round(float(row["avg_battery_health"]), 2),
                "avg_degradation": round(float(row["avg_degradation"]), 2),
            }
            for _, row in grouped.iterrows()
        ]

    evaluation = evaluation_payload()
    evaluation_note = "Synthetic development evaluation with versioned splits and artifact hashes." if evaluation else "Evaluation is missing or does not match the dataset/model artifacts."

    payload = {
        "dataset": {
            "path": str(DATA_PATH),
            "rows": total,
            "origin": str(df["data_origin"].iloc[0]) if "data_origin" in df.columns else "unknown",
            "synthetic_disclaimer": (
                "All records are synthetic. Results describe this synthetic "
                "corpus only and are not evidence of real-world accuracy."
            ),
        },
        "kpis": kpis,
        "category_distribution": _distribution(df["issue_label"], "category"),
        "cluster_distribution": cluster_distribution,
        "severity_distribution": _distribution(df["severity"], "severity"),
        "battery_type_distribution": _distribution(df["battery_type"], "battery_type"),
        "driving_style_distribution": _distribution(df["driving_style"], "driving_style"),
        "car_model_health": car_model,
        "age_trend": _age_trend(df),
        "trend_note": (
            "Dataset has no timestamp/date column. Trends use vehicle_age_months "
            "buckets as the closest available proxy and are not calendar trends."
        ),
        "evaluation": evaluation,
        "evaluation_note": evaluation_note,
    }
    return _json_safe(payload)


def complaints_payload(
    search: str | None = None,
    category: str | None = None,
    severity: str | None = None,
    cluster_id: int | None = None,
    battery_type: str | None = None,
    limit: int = 25,
    offset: int = 0,
) -> dict[str, Any]:
    df = load_dataframe()
    filtered = df

    if category:
        filtered = filtered[filtered["issue_label"] == category]
    if severity:
        filtered = filtered[filtered["severity"] == severity]
    if battery_type:
        filtered = filtered[filtered["battery_type"] == battery_type]
    if cluster_id is not None:
        filtered = filtered[filtered["cluster_id"] == cluster_id]
    if search:
        needle = search.strip().lower()
        haystack = (
            filtered["post_text"].astype(str).str.lower()
            + " "
            + filtered["issue_label"].astype(str).str.lower()
            + " "
            + filtered["car_model"].astype(str).str.lower()
        )
        filtered = filtered[haystack.str.contains(needle, regex=False)]

    total = int(len(filtered))
    limit = max(1, min(int(limit), 200))
    offset = max(0, int(offset))
    window = filtered.iloc[offset : offset + limit]

    columns = [
        "record_id",
        "issue_label",
        "severity",
        "car_model",
        "battery_type",
        "vehicle_age_months",
        "total_charging_cycles",
        "battery_health_pct",
        "degradation_pct",
        "cluster_id",
        "predicted_issue",
        "post_text",
    ]
    present = [c for c in columns if c in window.columns]
    rows = window[present].to_dict(orient="records")
    rows = _json_safe(rows)

    return {
        "total": total,
        "limit": limit,
        "offset": offset,
        "count": len(rows),
        "rows": rows,
    }


def filter_options() -> dict[str, list[str]]:
    df = load_dataframe()
    return {
        "categories": sorted(df["issue_label"].dropna().unique().tolist()),
        "severities": sorted(df["severity"].dropna().unique().tolist()),
        "battery_types": sorted(df["battery_type"].dropna().unique().tolist()),
        "clusters": sorted(int(c) for c in df["cluster_id"].dropna().unique().tolist()),
    }
