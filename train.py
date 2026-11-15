"""Train the three detectors, compare them on KDDTest+, and save the best one.

Usage:
    python train.py            # train on KDDTrain+, evaluate on KDDTest+
    python train.py --cv       # also run 5-fold cross-validation on KDDTrain+

Outputs:
    results/metrics.json        every number reported in the README
    models/best_model.joblib    the pipeline with the highest test F1
"""

import argparse
import json
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline

# Make the src/ package importable without installing it.
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from ids.data import TEST_FILE, TRAIN_FILE, load_nsl_kdd, split_features_target  # noqa: E402
from ids.evaluate import (  # noqa: E402
    compute_metrics,
    detection_rate_by_category,
    detection_rate_seen_vs_unseen,
    pick_best_model,
    plot_confusion_matrix,
    plot_detection_by_category,
    plot_error_rates,
    summarise_folds,
)
from ids.models import (  # noqa: E402
    RANDOM_STATE,
    SVM_SAMPLE_SIZE,
    build_all_models,
    stratified_sample,
)

RESULTS_DIR = Path(__file__).resolve().parent / "results"
MODELS_DIR = Path(__file__).resolve().parent / "models"
METRICS_FILE = RESULTS_DIR / "metrics.json"
BEST_MODEL_FILE = MODELS_DIR / "best_model.joblib"


def training_rows_for(name: str, X: pd.DataFrame, y: pd.Series,
                      svm_sample_size: int) -> tuple[pd.DataFrame, pd.Series]:
    """Return the rows a model trains on: a stratified sample for the SVM, all rows otherwise."""
    if name == "svm":
        return stratified_sample(X, y, svm_sample_size)
    return X, y


def fit_and_evaluate(name: str, pipeline: Pipeline,
                     X_train: pd.DataFrame, y_train: pd.Series,
                     X_test: pd.DataFrame, y_test: pd.Series,
                     svm_sample_size: int) -> tuple[Pipeline, dict, np.ndarray]:
    """Fit one pipeline, score it on the test set, and record timings.

    Returns the fitted pipeline, its metrics, and its test-set predictions
    (kept for the per-category breakdown and charts).
    """
    X_fit, y_fit = training_rows_for(name, X_train, y_train, svm_sample_size)

    start = time.perf_counter()
    pipeline.fit(X_fit, y_fit)
    train_seconds = time.perf_counter() - start

    start = time.perf_counter()
    y_pred = pipeline.predict(X_test)
    predict_seconds = time.perf_counter() - start

    metrics = compute_metrics(y_test, y_pred)
    metrics.update({
        "training_rows": len(X_fit),
        "train_seconds": round(train_seconds, 2),
        "predict_seconds": round(predict_seconds, 2),
    })
    return pipeline, metrics, y_pred


def save_charts(test_results: dict[str, dict], by_category: dict[str, dict]) -> list[Path]:
    """Write the confusion matrices, error-rate chart and category chart to results/."""
    paths = []
    for name, metrics in test_results.items():
        path = RESULTS_DIR / f"confusion_matrix_{name}.png"
        plot_confusion_matrix(metrics, name, path)
        paths.append(path)
    paths.append(RESULTS_DIR / "error_rates.png")
    plot_error_rates(test_results, paths[-1])
    paths.append(RESULTS_DIR / "detection_by_category.png")
    plot_detection_by_category(by_category, paths[-1])
    return paths


def cross_validate(name: str, pipeline: Pipeline, X: pd.DataFrame, y: pd.Series,
                   n_folds: int, svm_sample_size: int) -> dict:
    """Stratified k-fold cross-validation on the training set only.

    Each fold trains a fresh copy of the pipeline on (k-1)/k of KDDTrain+
    and scores it on the remaining 1/k. All attack types in the validation
    fold were also present in training, so this measures how well a model
    recognises *known* attacks.
    """
    folds = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=RANDOM_STATE)
    fold_metrics = []
    for fold_number, (train_idx, val_idx) in enumerate(folds.split(X, y), start=1):
        X_fit, y_fit = training_rows_for(name, X.iloc[train_idx], y.iloc[train_idx],
                                         svm_sample_size)
        model = clone(pipeline).fit(X_fit, y_fit)
        y_pred = model.predict(X.iloc[val_idx])
        fold_metrics.append(compute_metrics(y.iloc[val_idx], y_pred))
        print(f"    fold {fold_number}/{n_folds}: F1 = {fold_metrics[-1]['f1']:.4f}")
    return summarise_folds(fold_metrics)


def print_results_table(results: dict[str, dict]) -> None:
    """Print the headline test-set metrics as an aligned table."""
    header = f"{'model':<16}{'acc':>8}{'prec':>8}{'recall':>8}{'F1':>8}{'FPR':>8}{'FNR':>8}"
    print(header)
    print("-" * len(header))
    for name, m in results.items():
        print(f"{name:<16}{m['accuracy']:>8.4f}{m['precision']:>8.4f}{m['recall']:>8.4f}"
              f"{m['f1']:>8.4f}{m['false_positive_rate']:>8.4f}{m['false_negative_rate']:>8.4f}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cv", action="store_true",
                        help="also run stratified k-fold cross-validation on the training set")
    parser.add_argument("--cv-folds", type=int, default=5, help="number of CV folds (default 5)")
    parser.add_argument("--svm-sample", type=int, default=SVM_SAMPLE_SIZE,
                        help=f"rows used to train the SVM (default {SVM_SAMPLE_SIZE})")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_start = time.perf_counter()

    print("Loading NSL-KDD ...")
    train_df, test_df = load_nsl_kdd(TRAIN_FILE), load_nsl_kdd(TEST_FILE)
    X_train, y_train = split_features_target(train_df)
    X_test, y_test = split_features_target(test_df)
    print(f"  train: {len(X_train)} rows, test: {len(X_test)} rows")

    test_results: dict[str, dict] = {}
    cv_results: dict[str, dict] | None = {} if args.cv else None
    fitted: dict[str, Pipeline] = {}
    by_category: dict[str, dict] = {}
    seen_vs_unseen: dict[str, dict] = {}
    training_attack_names = set(train_df.loc[train_df["is_attack"] == 1, "attack_name"])

    for name, pipeline in build_all_models().items():
        if args.cv:
            print(f"\n[{name}] {args.cv_folds}-fold cross-validation on KDDTrain+")
            cv_results[name] = cross_validate(name, pipeline, X_train, y_train,
                                              args.cv_folds, args.svm_sample)

        print(f"\n[{name}] training on KDDTrain+, evaluating on KDDTest+")
        fitted[name], test_results[name], y_pred = fit_and_evaluate(
            name, pipeline, X_train, y_train, X_test, y_test, args.svm_sample)
        by_category[name] = detection_rate_by_category(test_df["attack_category"], y_pred)
        seen_vs_unseen[name] = detection_rate_seen_vs_unseen(
            test_df["attack_name"], y_pred, training_attack_names)
        m = test_results[name]
        print(f"  F1 = {m['f1']:.4f}  (trained on {m['training_rows']} rows "
              f"in {m['train_seconds']} s)")

    best = pick_best_model(test_results, metric="f1")

    MODELS_DIR.mkdir(exist_ok=True)
    # Save the pipeline together with its name, so detect.py can report which model it uses.
    joblib.dump({"model_name": best, "pipeline": fitted[best]}, BEST_MODEL_FILE)

    report = {
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "dataset": {
            "name": "NSL-KDD",
            "train_file": TRAIN_FILE.name, "train_rows": len(X_train),
            "test_file": TEST_FILE.name, "test_rows": len(X_test),
        },
        "settings": {
            "random_state": RANDOM_STATE,
            "svm_sample_size": args.svm_sample,
            "cv_folds": args.cv_folds if args.cv else None,
        },
        "environment": {
            "python": platform.python_version(),
            "scikit_learn": sklearn.__version__,
        },
        "test_set": test_results,
        "detection_by_category": by_category,
        "detection_seen_vs_unseen_attack_types": seen_vs_unseen,
        "cross_validation": cv_results if args.cv else "not yet measured",
        "best_model": {"name": best, "selected_by": "test-set F1",
                       "file": f"models/{BEST_MODEL_FILE.name}"},
        "total_run_seconds": round(time.perf_counter() - run_start, 1),
    }
    RESULTS_DIR.mkdir(exist_ok=True)
    METRICS_FILE.write_text(json.dumps(report, indent=2))
    chart_paths = save_charts(test_results, by_category)

    print("\nTest-set results (KDDTest+):")
    print_results_table(test_results)
    if args.cv:
        # "+/-" rather than the plus-minus symbol, which the Windows console can garble.
        print("\nCross-validation F1 on KDDTrain+ (mean +/- std):")
        for name, summary in cv_results.items():
            print(f"  {name:<16}{summary['f1']['mean']:.4f} +/- {summary['f1']['std']:.4f}")
    print("\nDetection rate by attack category (KDDTest+):")
    print(f"  {'model':<16}" + "".join(f"{c:>8}" for c in by_category[best])
          + f"{'unseen':>9}")
    for name, cats in by_category.items():
        unseen = seen_vs_unseen[name]["unseen_in_training"]["detection_rate"]
        print(f"  {name:<16}" + "".join(f"{v['detection_rate']:>8.4f}" for v in cats.values())
              + f"{unseen:>9.4f}")
    print(f"\nCharts saved: {', '.join(p.name for p in chart_paths)}")
    print(f"\nBest model by F1: {best}  -> saved to {BEST_MODEL_FILE}")
    print(f"Metrics written to {METRICS_FILE}")
    print(f"Total run time: {report['total_run_seconds']} s")


if __name__ == "__main__":
    main()
