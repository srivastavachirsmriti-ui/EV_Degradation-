"""Loads and serves the existing trained models.

The models are loaded with ``joblib`` exactly as pickled -- nothing is retrained
or replaced. Two distinct artifacts are used:

* ``final_classifier.joblib`` is a full ``sklearn.pipeline.Pipeline`` whose first
  step is its own ``TfidfVectorizer``. It therefore consumes **raw strings**
  (the cleaned complaint text), not a pre-vectorised matrix.
* ``cluster_tfidf_vectorizer.joblib`` + ``kmeans_model.joblib`` form the
  clustering path; the vectorizer output is fed to K-Means.

The classifier is an ``SGDClassifier(loss="hinge")`` which does **not** implement
``predict_proba``. Probability output is therefore reported as unavailable and
uncalibrated ``decision_function`` margins are exposed separately.
"""

from __future__ import annotations

import threading
import warnings
from dataclasses import dataclass, field
from typing import Any

import joblib
import numpy as np

from config import (
    CLASSIFIER_PATH,
    CLUSTER_VECTORIZER_PATH,
    KMEANS_PATH,
)
from text_cleaning import clean_text


@dataclass
class ModelBundle:
    classifier: Any
    cluster_vectorizer: Any
    kmeans: Any
    classes: list[str] = field(default_factory=list)
    n_clusters: int = 0

    @property
    def supports_probabilities(self) -> bool:
        return hasattr(self.classifier, "predict_proba")


class ModelService:
    """Thread-safe, lazily-loaded singleton wrapper around the models."""

    def __init__(self) -> None:
        self._bundle: ModelBundle | None = None
        self._lock = threading.Lock()

    def load(self, force: bool = False) -> ModelBundle:
        if self._bundle is not None and not force:
            return self._bundle
        with self._lock:
            if self._bundle is not None and not force:
                return self._bundle
            for path in (CLASSIFIER_PATH, CLUSTER_VECTORIZER_PATH, KMEANS_PATH):
                if not path.exists():
                    raise FileNotFoundError(f"Required model file not found: {path}")

            with warnings.catch_warnings():
                # Version mismatch (if any) is surfaced through /api/health.
                warnings.simplefilter("ignore")
                classifier = joblib.load(CLASSIFIER_PATH)
                cluster_vectorizer = joblib.load(CLUSTER_VECTORIZER_PATH)
                kmeans = joblib.load(KMEANS_PATH)

            classes = [str(c) for c in getattr(classifier, "classes_", [])]
            bundle = ModelBundle(
                classifier=classifier,
                cluster_vectorizer=cluster_vectorizer,
                kmeans=kmeans,
                classes=classes,
                n_clusters=int(getattr(kmeans, "n_clusters", 0)),
            )
            self._bundle = bundle
            return bundle

    def predict(self, text: str) -> dict[str, Any]:
        bundle = self.load()
        cleaned = clean_text(text)

        issue = str(bundle.classifier.predict([cleaned])[0])
        cluster_vector = bundle.cluster_vectorizer.transform([cleaned])
        cluster = int(bundle.kmeans.predict(cluster_vector)[0])

        result: dict[str, Any] = {
            "cleaned_text": cleaned,
            "issue_category": issue,
            "cluster_id": cluster,
            "supports_probabilities": bundle.supports_probabilities,
            "probabilities": None,
            "decision_scores": self._decision_scores(bundle, cleaned),
            "probability_note": (
                "Classifier is SGDClassifier(loss='hinge'); it does not expose "
                "predict_proba. Rows below are uncalibrated decision-function "
                "margins, not probabilities."
                if not bundle.supports_probabilities
                else "Probabilities produced by predict_proba."
            ),
        }
        return result

    @staticmethod
    def _decision_scores(bundle: ModelBundle, cleaned: str) -> list[dict[str, Any]] | None:
        if bundle.supports_probabilities or not hasattr(bundle.classifier, "decision_function"):
            return None
        try:
            margins = bundle.classifier.decision_function([cleaned])[0]
        except Exception:
            return None
        scores = [
            {"label": label, "score": round(float(margin), 4)}
            for label, margin in zip(bundle.classes, np.atleast_1d(margins))
        ]
        scores.sort(key=lambda item: item["score"], reverse=True)
        return scores


model_service = ModelService()
