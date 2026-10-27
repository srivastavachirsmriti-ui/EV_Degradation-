"""Run before changing artifacts: python -m ml.baseline."""
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.base import clone
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedGroupKFold, train_test_split
from sklearn.pipeline import make_pipeline
from threadpoolctl import threadpool_limits

from ml.evaluate import FEATURES, classification_metrics, latency_ms, regression_metrics, template_groups

ROOT = Path(__file__).resolve().parents[1]


def main():
    df = pd.read_csv(ROOT / "dashboard_data.csv")
    groups = template_groups(df)
    labels = sorted(df.issue_label.unique())
    text = df.clean_text.to_numpy()
    y = df.issue_label.to_numpy()
    train, test = next(StratifiedGroupKFold(5, shuffle=True, random_state=42).split(text, y, groups))
    original = joblib.load(ROOT / "final_classifier.joblib")
    reconstructed = clone(original).fit(text[train], y[train])
    ri, rt = train_test_split(np.arange(len(df)), test_size=.2, random_state=42)
    x, target = df[FEATURES], df.degradation_pct
    rf = make_pipeline(SimpleImputer(strategy="median"), RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=1))
    dummy = DummyRegressor().fit(x.iloc[ri], target.iloc[ri])
    rf.fit(x.iloc[ri], target.iloc[ri])
    vectorizer = joblib.load(ROOT / "cluster_tfidf_vectorizer.joblib")
    clusterer = joblib.load(ROOT / "kmeans_model.joblib")
    def old_path(values):
        return original.predict(values), clusterer.predict(vectorizer.transform(values))
    result = {
        "source_commit": "e69fa9404047bc8c093f0fb1e4ffa1c0e109d409",
        "dataset_sha256": hashlib.sha256((ROOT / "dashboard_data.csv").read_bytes()).hexdigest(),
        "sklearn_version": sklearn.__version__,
        "audit": {"rows": len(df), "origins": df.data_origin.value_counts().to_dict(),
                  "template_families": int(groups.nunique()), "exact_duplicate_texts": int(df.clean_text.duplicated().sum()),
                  "class_counts": df.issue_label.value_counts().to_dict(),
                  "health_target_sum_max_error": float(abs(df.battery_health_pct + df.degradation_pct - 100).max()),
                  "source_urls_present": int(df.source_url.notna().sum()),
                  "regressor_in_original_repository": False},
        "classification_protocol": "First StratifiedGroupKFold(5,shuffle=True,random_state=42) fold; vehicle/numeric template groups; refit clone only on train",
        "split": {"train_ids": df.record_id.iloc[train].tolist(), "test_ids": df.record_id.iloc[test].tolist(),
                  "train_groups": sorted(set(groups.iloc[train])), "test_groups": sorted(set(groups.iloc[test]))},
        "original_artifact_resubstitution_not_test": classification_metrics(y, original.predict(text), labels),
        "reconstructed_sgd_group_test": classification_metrics(y[test], reconstructed.predict(text[test]), labels),
        "old_classify_and_cluster_latency": latency_ms(old_path, text),
        "regression_protocol": "80/20 random row split, seed=42, five numeric inputs only; reconstructed RF, not original missing artifact",
        "regression_mean": regression_metrics(target.iloc[rt], dummy.predict(x.iloc[rt])),
        "regression_reconstructed_rf": regression_metrics(target.iloc[rt], rf.predict(x.iloc[rt])),
    }
    (ROOT / "benchmarks").mkdir(exist_ok=True)
    (ROOT / "benchmarks/baseline.json").write_text(json.dumps(result, indent=2))
    print(json.dumps({k:v for k,v in result.items() if k != "split"}, indent=2))


if __name__ == "__main__":
    with threadpool_limits(limits=1):
        main()
