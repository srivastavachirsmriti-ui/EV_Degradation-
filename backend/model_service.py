"""Cached, batched inference with a calibrated fast stage and fallback model."""
from __future__ import annotations

import copy
import sys
import threading
import warnings
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any

import joblib
import numpy as np
import pandas as pd

from config import CLASSIFIER_PATH, CLUSTER_VECTORIZER_PATH, KMEANS_PATH, REGRESSOR_PATH, PROJECT_ROOT
from text_cleaning import clean_text

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from ml.features import DomainFeatures


@dataclass
class ModelBundle:
    classifier: Any
    cluster_vectorizer: Any
    kmeans: Any
    fast_classifier: Any = None
    fast_threshold: float = .97
    review_threshold: float = .65
    classes: list[str] = field(default_factory=list)
    n_clusters: int = 0
    warnings: list[str] = field(default_factory=list)

    @property
    def supports_probabilities(self):
        return hasattr(self.classifier, "predict_proba")


class ModelService:
    def __init__(self):
        self._bundle = None
        self._regressor = None
        self._lock = threading.RLock()

    def load(self, force=False):
        with self._lock:
            if self._bundle is not None and not force:
                return self._bundle
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                stored = joblib.load(CLASSIFIER_PATH)
                classifier = stored.get("classifier") if isinstance(stored, dict) else stored
                options = stored if isinstance(stored, dict) else {}
                self._bundle = ModelBundle(
                    classifier=classifier,
                    fast_classifier=options.get("fast_classifier"),
                    fast_threshold=options.get("fast_threshold", .97),
                    review_threshold=options.get("review_threshold", .65),
                    cluster_vectorizer=joblib.load(CLUSTER_VECTORIZER_PATH),
                    kmeans=joblib.load(KMEANS_PATH),
                    classes=[str(c) for c in classifier.classes_],
                    warnings=[str(w.message) for w in caught],
                )
            self._bundle.n_clusters = int(self._bundle.kmeans.n_clusters)
            if force:
                self._cached_prediction.cache_clear()
                self._regressor = None
            return self._bundle

    def predict(self, text, include_cluster=True):
        self.load()
        return copy.deepcopy(self._cached_prediction(clean_text(text), include_cluster))

    @lru_cache(maxsize=1024)
    def _cached_prediction(self, cleaned, include_cluster):
        return self.predict_many([cleaned], include_cluster)[0]

    def predict_many(self, texts, include_cluster=True):
        bundle = self.load()
        cleaned = [clean_text(t) for t in texts]
        if any(not any(c.isalnum() for c in t) for t in cleaned):
            raise ValueError("Complaint must contain readable letters or numbers.")
        if not cleaned:
            return []
        paths = ["fallback"] * len(cleaned)
        probabilities = None
        margins = None
        if bundle.supports_probabilities:
            if bundle.fast_classifier is not None:
                probabilities = bundle.fast_classifier.predict_proba(cleaned)
                fallback = probabilities.max(axis=1) < bundle.fast_threshold
                if fallback.any():
                    probabilities[fallback] = bundle.classifier.predict_proba(np.asarray(cleaned)[fallback])
                paths = ["fallback" if flag else "fast" for flag in fallback]
            else:
                probabilities = bundle.classifier.predict_proba(cleaned)
            predicted = np.asarray(bundle.classes)[probabilities.argmax(axis=1)]
        else:
            predicted = bundle.classifier.predict(cleaned)
            if hasattr(bundle.classifier, "decision_function"):
                margins = bundle.classifier.decision_function(cleaned)
        clusters = [None] * len(cleaned)
        if include_cluster:
            clusters = bundle.kmeans.predict(bundle.cluster_vectorizer.transform(cleaned))
        results = []
        semantic = DomainFeatures().transform(cleaned).toarray()
        for i, issue in enumerate(predicted):
            p = probabilities[i] if probabilities is not None else None
            confidence = float(p.max()) if p is not None else None
            decision = None
            if margins is not None:
                row = np.atleast_1d(margins[i])
                if len(row) == 1 and len(bundle.classes) == 2:
                    row = np.array([-row[0], row[0]])
                decision = sorted([{"label": label, "score": float(score)} for label, score in zip(bundle.classes, row)], key=lambda v: -v["score"])
            context_review = not semantic[i, :8].any() or bool(semantic[i, 8])
            results.append({"cleaned_text": cleaned[i], "issue_category": str(issue),
                            "cluster_id": int(clusters[i]) if clusters[i] is not None else None,
                            "supports_probabilities": bundle.supports_probabilities,
                            "probabilities": dict(zip(bundle.classes, map(float, p))) if p is not None else None,
                            "confidence": confidence, "needs_review": context_review or confidence is None or confidence < bundle.review_threshold,
                            "review_reason": "Unsupported wording or negation requires manual review." if context_review else "Low confidence." if confidence is None or confidence < bundle.review_threshold else None,
                            "classification_path": paths[i], "decision_scores": decision,
                            "probability_note": "Group-calibrated probabilities on synthetic templates. Confidence does not establish real-world correctness." if p is not None else "Uncalibrated decision margins, not probabilities."})
        return results

    def load_regressor(self):
        with self._lock:
            if self._regressor is None:
                self._regressor = joblib.load(REGRESSOR_PATH)
            return self._regressor

    def predict_degradation(self, values):
        stored = self.load_regressor()
        x = pd.DataFrame([{key: values[key] for key in stored["features"]}])
        estimate = float(np.clip(stored["model"].predict(x)[0], 0, 100))
        width = stored["interval_half_width"]
        outside = [key for key, (low, high) in stored["bounds"].items() if not low <= values[key] <= high]
        return {"degradation_pct": estimate, "interval_90": {"lower": max(0., estimate - width), "upper": min(100., estimate + width)},
                "out_of_training_range": outside, "needs_review": bool(outside), "model": stored["selected_model"],
                "note": "Synthetic-data estimate in percentage points. The 90% split-conformal interval assumes exchangeable data within the training domain; coverage is not validated for real vehicles or out-of-range inputs."}


model_service = ModelService()
