"""Full family-level OOF validation and application timings after selection.

This is an internal synthetic development benchmark, not an independent test.
"""
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import make_pipeline
from threadpoolctl import threadpool_limits

from ml.evaluate import classification_metrics, latency_ms, template_folds, template_groups

ROOT = Path(__file__).resolve().parents[1]


def main():
    df = pd.read_csv(ROOT / "dashboard_data.csv")
    text, y, groups = df.clean_text.to_numpy(), df.issue_label.to_numpy(), template_groups(df).to_numpy()
    stored = joblib.load(ROOT / "final_classifier.joblib")
    candidate = clone(stored["classifier"].estimator)
    baseline = make_pipeline(TfidfVectorizer(max_features=10000, min_df=2, ngram_range=(1, 2)),
                             SGDClassifier(class_weight="balanced", max_iter=4, random_state=42, tol=None))
    predicted = np.empty(len(y), dtype=object)
    old_predicted = np.empty(len(y), dtype=object)
    folds = []
    for i, (train, test) in enumerate(template_folds(y, groups)):
        inner = list(template_folds(y[train], groups[train], 4))
        model = CalibratedClassifierCV(clone(candidate), method="sigmoid", cv=inner, ensemble=False)
        model.fit(text[train], y[train])
        predicted[test] = model.predict(text[test])
        old_predicted[test] = clone(baseline).fit(text[train], y[train]).predict(text[test])
        folds.append({"fold": i + 1, "n": len(test),
                      "current": classification_metrics(y[test], predicted[test], sorted(set(y))),
                      "baseline": classification_metrics(y[test], old_predicted[test], sorted(set(y)))})
    oof = classification_metrics(y, predicted, sorted(set(y)))
    old_oof = classification_metrics(y, old_predicted, sorted(set(y)))
    pd.DataFrame({"record_id": df.record_id, "template_group": groups, "actual": y,
                  "baseline": old_predicted, "current": predicted}).to_csv(ROOT / "benchmarks/group_oof_predictions.csv", index=False)
    # Resample complete template families so duplicates do not create false precision.
    rng = np.random.default_rng(42)
    unique = np.unique(groups)
    boot = []
    for _ in range(1000):
        sampled = rng.choice(unique, len(unique), replace=True)
        indices = np.concatenate([np.flatnonzero(groups == group) for group in sampled])
        boot.append(float(np.mean(y[indices] == predicted[indices])))
    sys.path.insert(0, str(ROOT / "backend"))
    from model_service import ModelService
    service = ModelService()
    service.load()
    def current_uncached(values):
        return service.predict_many(values, include_cluster=True)
    timings = {"current_uncached_classify_and_cluster": latency_ms(current_uncached, text),
               "current_classify_only": latency_ms(lambda v: service.predict_many(v, False), text)}
    for t in text[:100]:
        service.predict(t)
    timings["current_cached_classify_and_cluster"] = latency_ms(lambda v: service.predict(v[0]), text[:100])
    timings["batch_100"] = latency_ms(lambda _: service.predict_many(text[:100], False), text, 30)
    result = {"scope": "Post-selection internal synthetic development validation. 5 outer folds hold one family per category; sigmoid calibration uses 4 inner family folds. Feature design has used this corpus, so this is not an independent external test.",
              "baseline_oof": old_oof, "current_oof": oof, "folds": folds,
              "family_bootstrap_accuracy_95_interval": np.quantile(boot, [.025, .975]).tolist(),
              "timings": timings}
    (ROOT / "benchmarks/validation.json").write_text(json.dumps(result, indent=2))
    print(json.dumps({"baseline": old_oof["accuracy"], "current": oof["accuracy"],
                      "macro_f1": oof["macro_f1"], "accuracy_interval": result["family_bootstrap_accuracy_95_interval"], "timings": timings}, indent=2))


if __name__ == "__main__":
    with threadpool_limits(limits=1):
        main()
