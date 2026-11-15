"""Tune each model's main settings, then compare before and after on KDDTest+.

Usage:
    python tune.py        (run train.py first: it provides the "before" numbers)

Method:
  - Settings are chosen by 3-fold stratified cross-validation F1 on
    KDDTrain+ only. The test set is never used to choose settings; it is
    used once at the end to measure the tuned models.
  - To keep the run under about 15 minutes on a laptop, the neural network
    is searched on a 30,000-row stratified sample and the SVM on its usual
    20,000-row sample. The chosen settings are then refitted on the same
    rows train.py uses, so "before" and "after" are directly comparable.

Outputs:
    results/tuning.json               grids, CV scores, before/after metrics
    results/tuning_before_after.png   F1, FPR and FNR before vs after
    results/kmeans_sweep.png          CV F1 for every K-Means setting tried
    models/best_model.joblib          the tuned model with the highest test F1
"""

import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import make_scorer
from sklearn.model_selection import GridSearchCV, StratifiedKFold

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from ids.data import TEST_FILE, TRAIN_FILE, load_nsl_kdd, split_features_target  # noqa: E402
from ids.evaluate import (  # noqa: E402
    compute_metrics,
    pick_best_model,
    plot_before_after,
    plot_kmeans_sweep,
)
from ids.models import (  # noqa: E402
    RANDOM_STATE,
    SVM_SAMPLE_SIZE,
    build_all_models,
    stratified_sample,
)

ROOT = Path(__file__).resolve().parent
RESULTS_DIR = ROOT / "results"
BASELINE_FILE = RESULTS_DIR / "metrics.json"
TUNING_FILE = RESULTS_DIR / "tuning.json"
BEST_MODEL_FILE = ROOT / "models" / "best_model.joblib"

MLP_SEARCH_SAMPLE = 30_000
CV_FOLDS = 3

# Small, deliberate grids: each value is a sensible alternative to the default,
# not an exhaustive search. "model__" points the setting at the Pipeline's model step.
PARAM_GRIDS: dict[str, dict[str, list]] = {
    "neural_network": {
        "model__hidden_layer_sizes": [(64, 32), (128, 64)],  # network size
        "model__alpha": [1e-4, 1e-3],                       # L2 penalty against overfitting
        "model__learning_rate_init": [1e-3, 1e-2],           # step size while learning
    },
    "svm": {
        "model__C": [1, 10, 100],              # how hard to avoid training mistakes
        "model__gamma": ["scale", 0.01, 0.1],  # how local each training point's influence is
    },
    "kmeans": {
        "model__n_clusters": [4, 8, 16, 32],
        "model__percentile": [90, 95, 97.5, 99],
    },
}


def false_positive_rate(y_true, y_pred) -> float:
    """FPR for one CV fold. Recorded for the report; not used to choose settings."""
    return compute_metrics(y_true, y_pred)["false_positive_rate"]


# Several scores are recorded per setting, but only F1 decides the winner.
CV_SCORING = {
    "f1": "f1",
    "recall": "recall",
    "false_positive_rate": make_scorer(false_positive_rate),
}


def search_rows(name, X, y):
    """Rows used for the grid search (smaller than the final training rows where needed)."""
    if name == "neural_network":
        return stratified_sample(X, y, MLP_SEARCH_SAMPLE)
    if name == "svm":
        return stratified_sample(X, y, SVM_SAMPLE_SIZE)
    return X, y


def final_training_rows(name, X, y):
    """Same training rows as train.py, so before/after comparisons are fair."""
    if name == "svm":
        return stratified_sample(X, y, SVM_SAMPLE_SIZE)
    return X, y


def grid_search(name, pipeline, X, y) -> GridSearchCV:
    """Run a cross-validated grid search and return the fitted search object."""
    search = GridSearchCV(
        pipeline,
        PARAM_GRIDS[name],
        scoring=CV_SCORING,
        refit=False,  # we refit ourselves on the final training rows
        cv=StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE),
        n_jobs=-1,    # use every CPU core
    )
    X_search, y_search = search_rows(name, X, y)
    search.fit(X_search, y_search)
    return search


def best_params_by_f1(search: GridSearchCV) -> tuple[dict, float]:
    """The settings with the highest mean CV F1, and that F1."""
    best = int(np.argmax(search.cv_results_["mean_test_f1"]))
    return search.cv_results_["params"][best], float(search.cv_results_["mean_test_f1"][best])


def cv_table(search: GridSearchCV) -> list[dict]:
    """Every setting tried, with its mean CV scores, for the JSON report."""
    res = search.cv_results_
    return [
        {
            "params": {k.replace("model__", ""): (list(v) if isinstance(v, tuple) else v)
                       for k, v in params.items()},
            "cv_f1": round(float(res["mean_test_f1"][i]), 4),
            "cv_recall": round(float(res["mean_test_recall"][i]), 4),
            "cv_false_positive_rate": round(float(res["mean_test_false_positive_rate"][i]), 4),
        }
        for i, params in enumerate(res["params"])
    ]


def kmeans_f1_grid(search: GridSearchCV) -> np.ndarray:
    """Rearrange the K-Means CV F1 scores into a clusters x percentile grid."""
    clusters = PARAM_GRIDS["kmeans"]["model__n_clusters"]
    percentiles = PARAM_GRIDS["kmeans"]["model__percentile"]
    grid = np.zeros((len(clusters), len(percentiles)))
    for params, f1 in zip(search.cv_results_["params"], search.cv_results_["mean_test_f1"]):
        grid[clusters.index(params["model__n_clusters"]),
             percentiles.index(params["model__percentile"])] = f1
    return grid


def readable(params: dict) -> dict:
    """Strip the 'model__' prefix and make tuples JSON-friendly."""
    return {k.replace("model__", ""): (list(v) if isinstance(v, tuple) else v)
            for k, v in params.items()}


def main() -> None:
    run_start = time.perf_counter()
    if not BASELINE_FILE.exists():
        sys.exit("results/metrics.json not found - run train.py first.")
    baseline = json.loads(BASELINE_FILE.read_text())["test_set"]

    train_df, test_df = load_nsl_kdd(TRAIN_FILE), load_nsl_kdd(TEST_FILE)
    X_train, y_train = split_features_target(train_df)
    X_test, y_test = split_features_target(test_df)

    report: dict = {"models": {}}
    before, after, fitted = {}, {}, {}

    for name, pipeline in build_all_models().items():
        n_settings = int(np.prod([len(v) for v in PARAM_GRIDS[name].values()]))
        print(f"\n[{name}] searching {n_settings} settings x {CV_FOLDS} folds ...")
        start = time.perf_counter()
        search = grid_search(name, pipeline, X_train, y_train)
        search_seconds = time.perf_counter() - start
        params, cv_f1 = best_params_by_f1(search)
        print(f"  best by CV F1 ({cv_f1:.4f}): {readable(params)}  [{search_seconds:.0f} s]")

        # Refit the chosen settings on the final training rows, then test once.
        X_fit, y_fit = final_training_rows(name, X_train, y_train)
        tuned = pipeline.set_params(**params).fit(X_fit, y_fit)
        tuned_metrics = compute_metrics(y_test, tuned.predict(X_test))
        fitted[name] = tuned
        before[name] = baseline[name]
        after[name] = tuned_metrics
        print(f"  test F1: before {baseline[name]['f1']:.4f} -> after {tuned_metrics['f1']:.4f}")

        if name == "kmeans":
            kmeans_grid = kmeans_f1_grid(search)

        report["models"][name] = {
            "search_rows": len(search_rows(name, X_train, y_train)[0]),
            "training_rows": len(X_fit),
            "grid": {k.replace("model__", ""): [list(v) if isinstance(v, tuple) else v
                                                for v in vals]
                     for k, vals in PARAM_GRIDS[name].items()},
            "best_params": readable(params),
            "best_cv_f1": cv_f1,
            "search_seconds": round(search_seconds, 1),
            "all_settings": cv_table(search),
            "test_before": {k: baseline[name][k] for k in tuned_metrics},
            "test_after": tuned_metrics,
        }

    best = pick_best_model(after, metric="f1")
    joblib.dump({"model_name": best, "tuned": True, "pipeline": fitted[best]}, BEST_MODEL_FILE)

    report.update({
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "cv_folds": CV_FOLDS,
        "random_state": RANDOM_STATE,
        "selection": "settings chosen by mean CV F1 on KDDTrain+; test set used only to report",
        "best_tuned_model": {"name": best, "selected_by": "test-set F1",
                             "file": f"models/{BEST_MODEL_FILE.name}"},
        "total_run_seconds": round(time.perf_counter() - run_start, 1),
    })
    TUNING_FILE.write_text(json.dumps(report, indent=2))
    plot_before_after(before, after, RESULTS_DIR / "tuning_before_after.png")
    plot_kmeans_sweep(kmeans_grid, PARAM_GRIDS["kmeans"]["model__n_clusters"],
                      PARAM_GRIDS["kmeans"]["model__percentile"],
                      RESULTS_DIR / "kmeans_sweep.png")

    print("\nTest-set results before -> after tuning:")
    print(f"  {'model':<16}{'F1':>17}{'FPR':>17}{'FNR':>17}")
    for name in after:
        b, a = before[name], after[name]
        print(f"  {name:<16}" + "".join(
            f"{b[k]:>8.4f}->{a[k]:<7.4f}"
            for k in ("f1", "false_positive_rate", "false_negative_rate")))
    print(f"\nBest tuned model by test F1: {best} -> saved to {BEST_MODEL_FILE}")
    print(f"Total run time: {report['total_run_seconds']} s")


if __name__ == "__main__":
    main()
