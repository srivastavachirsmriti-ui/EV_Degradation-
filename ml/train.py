"""Select on development folds, then evaluate once on frozen test partitions.

Run from the project root: python -m ml.train
The deployed artifacts retain the test holdout and calibration partition.
"""
from __future__ import annotations

import hashlib
import json
import platform
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge, SGDClassifier
from sklearn.metrics import f1_score, log_loss, mean_absolute_error
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.naive_bayes import ComplementNB
from sklearn.pipeline import FeatureUnion, Pipeline, make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.svm import LinearSVC
from threadpoolctl import threadpool_limits

from ml.evaluate import FEATURES, classification_metrics, latency_ms, regression_metrics, template_groups, template_folds
from ml.features import DomainFeatures

ROOT = Path(__file__).resolve().parents[1]


def word_features():
    return TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True, max_features=12000)


def combined_features():
    return FeatureUnion([
        ("word", word_features()),
        ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2,
                                 sublinear_tf=True, max_features=20000))])


def main():
    df = pd.read_csv(ROOT / "dashboard_data.csv")
    baseline = json.loads((ROOT / "benchmarks/baseline.json").read_text())
    digest = hashlib.sha256((ROOT / "dashboard_data.csv").read_bytes()).hexdigest()
    if digest != baseline["dataset_sha256"]:
        raise ValueError("Dataset changed since baseline; create a new versioned benchmark")
    text, y = df.clean_text.to_numpy(), df.issue_label.to_numpy()
    labels = sorted(set(y))
    groups = template_groups(df).to_numpy()
    train = np.flatnonzero(df.record_id.isin(baseline["split"]["train_ids"]))
    test = np.flatnonzero(df.record_id.isin(baseline["split"]["test_ids"]))
    if set(groups[train]) & set(groups[test]):
        raise ValueError("Template leakage")
    cv = list(template_folds(y[train], groups[train], 4))
    candidates = {
        "sgd_converged": make_pipeline(word_features(), SGDClassifier(class_weight="balanced", max_iter=2000, tol=1e-5, random_state=42)),
        "word_logistic": make_pipeline(word_features(), LogisticRegression(C=4, class_weight="balanced", max_iter=1500, random_state=42)),
        "word_char_logistic": make_pipeline(combined_features(), LogisticRegression(C=4, class_weight="balanced", max_iter=1500, random_state=42)),
        "word_svm": make_pipeline(word_features(), LinearSVC(C=1, class_weight="balanced", random_state=42)),
        "word_char_svm": make_pipeline(combined_features(), LinearSVC(C=1, class_weight="balanced", random_state=42)),
        "complement_nb": make_pipeline(word_features(), ComplementNB(alpha=.5)),
        "domain_logistic": make_pipeline(DomainFeatures(), LogisticRegression(C=1, class_weight="balanced", max_iter=1000, random_state=42)),
        "domain_word_logistic": make_pipeline(FeatureUnion([("word", word_features()), ("domain", DomainFeatures())], transformer_weights={"word": 1, "domain": 3}), LogisticRegression(C=1, class_weight="balanced", max_iter=1000, random_state=42)),
    }
    selection = {}
    for name, model in candidates.items():
        scores = cross_val_score(model, text[train], y[train], cv=cv, scoring="f1_macro", n_jobs=1)
        selection[name] = {"mean_macro_f1": float(scores.mean()), "folds": scores.tolist()}
        print(name, selection[name], flush=True)
    best = max(selection, key=lambda name: selection[name]["mean_macro_f1"])
    # Group-disjoint OOF probabilities, followed by one estimator fit on all dev data.
    slow = CalibratedClassifierCV(clone(candidates[best]), method="sigmoid", cv=cv, ensemble=False)
    slow.fit(text[train], y[train])
    fast = CalibratedClassifierCV(clone(candidates["domain_logistic"]), method="sigmoid", cv=cv, ensemble=False)
    fast.fit(text[train], y[train])
    # Fixed conservative routing threshold, specified before looking at test labels.
    threshold = .90
    def cascade_proba(values):
        p = fast.predict_proba(values)
        fallback = p.max(axis=1) < threshold
        if fallback.any():
            p[fallback] = slow.predict_proba(np.asarray(values)[fallback])
        return p
    # This is the single final test evaluation. No subsequent tuning uses it.
    p = cascade_proba(text[test])
    pred = fast.classes_[p.argmax(axis=1)]
    fast_test = fast.predict_proba(text[test])
    classification = classification_metrics(y[test], pred, labels)
    classification["log_loss"] = float(log_loss(y[test], p, labels=labels))
    classification["fast_path_share"] = float((fast_test.max(axis=1) >= threshold).mean())
    classification["review_share"] = float((p.max(axis=1) < .65).mean())
    classification["fast_only"] = classification_metrics(y[test], fast.classes_[fast_test.argmax(axis=1)], labels)
    classification["fallback_only"] = classification_metrics(y[test], slow.predict(text[test]), labels)
    classification["latency"] = latency_ms(cascade_proba, text[test])
    classification["batch_100_latency"] = latency_ms(lambda _: cascade_proba(text[test][:100]), text[test], repeats=20)
    joblib.dump({"classifier": slow, "fast_classifier": fast, "fast_threshold": threshold,
                 "review_threshold": .65, "selected_model": best,
                 "training_scope": "20 synthetic template families; test families excluded"},
                ROOT / "final_classifier.joblib", compress=3)

    # Regression: same 500-row outer test as the baseline. Reserve 400 dev rows
    # for split-conformal residuals, and never refit the estimator on those rows.
    development, rtest = train_test_split(np.arange(len(df)), test_size=.2, random_state=42)
    fit, calibration = train_test_split(development, test_size=.2, random_state=43)
    x, target = df[FEATURES], df.degradation_pct
    def numeric(model, polynomial=False):
        steps = [("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]
        if polynomial:
            steps.append(("polynomial", PolynomialFeatures(2, include_bias=False)))
        return Pipeline(steps + [("regressor", model)])
    regressors = {
        "ridge_linear": numeric(Ridge(alpha=1)),
        "ridge_quadratic": numeric(Ridge(alpha=10), True),
        "random_forest": numeric(RandomForestRegressor(n_estimators=200, min_samples_leaf=3, random_state=42, n_jobs=1)),
        "extra_trees": numeric(ExtraTreesRegressor(n_estimators=250, min_samples_leaf=3, random_state=42, n_jobs=1)),
        "hist_gradient_boosting": numeric(HistGradientBoostingRegressor(max_iter=200, max_leaf_nodes=15, l2_regularization=5, random_state=42)),
    }
    regression_selection = {}
    rcv = KFold(5, shuffle=True, random_state=42)
    for name, model in regressors.items():
        scores = -cross_val_score(model, x.iloc[fit], target.iloc[fit], cv=rcv, scoring="neg_mean_absolute_error", n_jobs=1)
        regression_selection[name] = {"mean_mae": float(scores.mean()), "folds": scores.tolist()}
        print(name, regression_selection[name], flush=True)
    rbest = min(regression_selection, key=lambda name: regression_selection[name]["mean_mae"])
    regressor = regressors[rbest].fit(x.iloc[fit], target.iloc[fit])
    residuals = abs(target.iloc[calibration].to_numpy() - regressor.predict(x.iloc[calibration]))
    level = min(1., np.ceil((len(residuals) + 1) * .90) / len(residuals))
    q = float(np.quantile(residuals, level, method="higher"))
    rpred = regressor.predict(x.iloc[rtest])
    regression = regression_metrics(target.iloc[rtest], rpred)
    regression.update({"selected_model": rbest, "interval_nominal_coverage": .90,
                       "interval_observed_coverage": float((abs(target.iloc[rtest].to_numpy() - rpred) <= q).mean()),
                       "interval_half_width": q, "fit_rows": len(fit), "calibration_rows": len(calibration),
                       "latency": latency_ms(regressor.predict, x.iloc[rtest])})
    bounds = {name: [float(x.iloc[fit][name].min()), float(x.iloc[fit][name].max())] for name in FEATURES}
    joblib.dump({"model": regressor, "features": FEATURES, "interval_half_width": q,
                 "bounds": bounds, "selected_model": rbest}, ROOT / "degradation_regressor.joblib", compress=3)
    predictions = pd.DataFrame({"record_id": df.record_id.iloc[test], "actual": y[test], "predicted": pred,
                                "confidence": p.max(axis=1)})
    predictions.to_csv(ROOT / "benchmarks/classification_predictions.csv", index=False)
    pd.DataFrame({"record_id": df.record_id.iloc[rtest], "actual": target.iloc[rtest], "predicted": rpred,
                  "residual": target.iloc[rtest].to_numpy() - rpred}).to_csv(ROOT / "benchmarks/regression_predictions.csv", index=False)
    result = {"schema_version": 1, "dataset_sha256": digest,
              "environment": {"python": platform.python_version(), "sklearn": sklearn.__version__, "platform": platform.platform()},
              "note": "Synthetic data only. Template-separated development benchmark for classification, reused during domain-feature development. Not an independent final test. Random row test for regression. Candidate selection uses development CV.",
              "classification_selection": selection, "classification_selected": best, "classification": classification,
              "classification_test_ids": df.record_id.iloc[test].tolist(),
              "regression_selection": regression_selection, "regression": regression,
              "regression_split": {"fit_ids": df.record_id.iloc[fit].tolist(), "calibration_ids": df.record_id.iloc[calibration].tolist(),
                                   "test_ids": df.record_id.iloc[rtest].tolist()},
              "artifact_sha256": {name: hashlib.sha256((ROOT/name).read_bytes()).hexdigest()
                                  for name in ["final_classifier.joblib", "degradation_regressor.joblib"]}}
    (ROOT / "benchmarks/latest.json").write_text(json.dumps(result, indent=2))
    print(json.dumps({"classification": classification, "regression": regression}, indent=2))


if __name__ == "__main__":
    with threadpool_limits(limits=1):
        main()
