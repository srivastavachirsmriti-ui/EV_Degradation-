"""Evaluation primitives shared by baseline and model-selection runs."""
from __future__ import annotations

import hashlib
import re
import time

import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import StratifiedKFold

FEATURES = ["vehicle_age_months", "total_charging_cycles", "avg_temperature_c",
            "fast_charge_ratio", "avg_discharge_rate_c"]


def template_groups(df):
    """Collapse vehicle substitutions and numbers, without consulting labels."""
    cars = sorted(df.car_model.dropna().str.lower().unique(), key=len, reverse=True)
    pattern = re.compile("|".join(re.escape(car) for car in cars))
    def group(text):
        text = pattern.sub("<vehicle>", str(text).lower())
        text = re.sub(r"\d+(?:\.\d+)?", "<number>", text)
        return hashlib.sha256(text.encode()).hexdigest()[:16]
    return df.post_text.map(group)


def template_folds(y, groups, n_splits=5):
    """Stratify unique families, guaranteeing every class in every fold.

    Row-weighted StratifiedGroupKFold can omit a class with only 25 families.
    This corpus has exactly five pure-label families per class.
    """
    unique = np.unique(groups)
    labels = []
    for group in unique:
        group_labels = np.unique(np.asarray(y)[np.asarray(groups) == group])
        if len(group_labels) != 1:
            raise ValueError("A template family has conflicting labels")
        labels.append(group_labels[0])
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    for train, test in cv.split(unique, labels):
        yield (np.flatnonzero(np.isin(groups, unique[train])),
               np.flatnonzero(np.isin(groups, unique[test])))


def classification_metrics(y, predicted, labels):
    return {"accuracy": float(accuracy_score(y, predicted)),
            "macro_f1": float(f1_score(y, predicted, average="macro", zero_division=0)),
            "per_class": classification_report(y, predicted, labels=labels, output_dict=True, zero_division=0),
            "confusion_matrix": confusion_matrix(y, predicted, labels=labels).tolist(),
            "labels": list(labels), "n": len(y)}


def regression_metrics(y, predicted):
    return {"mae": float(mean_absolute_error(y, predicted)),
            "rmse": float(np.sqrt(mean_squared_error(y, predicted))),
            "r2": float(r2_score(y, predicted)), "n": len(y)}


def latency_ms(function, values, repeats=100):
    """Warm model, then measure independent sequential single-record calls."""
    function(values[:1])
    times = []
    for i in range(repeats):
        start = time.perf_counter()
        function(values[i % len(values):i % len(values) + 1])
        times.append((time.perf_counter() - start) * 1000)
    return {"median_ms": float(np.median(times)), "p95_ms": float(np.percentile(times, 95)),
            "repeats": repeats}
