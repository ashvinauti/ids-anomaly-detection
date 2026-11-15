"""Metrics for judging an intrusion detector.

Throughout, "positive" means attack (1) and "negative" means normal (0):

  TP  attack correctly flagged        FN  attack missed (silent breach)
  TN  normal correctly passed         FP  normal wrongly flagged (false alarm)

Accuracy alone is misleading for an IDS: a detector that never alerts would
still score 43% on KDDTest+ and 53% on KDDTrain+. So each model is also
judged on the two error types a SOC cares about:

  false positive rate = FP / (FP + TN)   share of normal traffic that raises an alert
  false negative rate = FN / (FN + TP)   share of attacks that get through
"""

import numpy as np
from sklearn.metrics import confusion_matrix


def _safe_divide(numerator: float, denominator: float) -> float:
    """Return numerator / denominator, or 0.0 when the denominator is 0.

    For example, precision is undefined if a model never flags anything;
    reporting 0.0 is the common convention (and matches scikit-learn's
    zero_division=0 option).
    """
    return float(numerator / denominator) if denominator else 0.0


def confusion_counts(y_true, y_pred) -> dict[str, int]:
    """Count TN, FP, FN and TP for a binary (0 = normal, 1 = attack) task."""
    # labels=[0, 1] fixes the matrix layout even if one class is absent.
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}


def compute_metrics(y_true, y_pred) -> dict[str, float | int]:
    """All headline metrics for one model, computed from the confusion counts.

    Calculated by hand rather than with separate sklearn calls so that each
    formula is visible and easy to explain.
    """
    c = confusion_counts(y_true, y_pred)
    tp, tn, fp, fn = c["tp"], c["tn"], c["fp"], c["fn"]

    precision = _safe_divide(tp, tp + fp)  # of the alerts raised, how many were real?
    recall = _safe_divide(tp, tp + fn)     # of the real attacks, how many were caught?
    f1 = _safe_divide(2 * precision * recall, precision + recall)

    return {
        **c,
        "accuracy": _safe_divide(tp + tn, tp + tn + fp + fn),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "false_positive_rate": _safe_divide(fp, fp + tn),
        "false_negative_rate": _safe_divide(fn, fn + tp),
    }


def summarise_folds(fold_metrics: list[dict[str, float]]) -> dict[str, dict[str, float]]:
    """Mean and standard deviation of each rate metric across CV folds.

    Count fields (tp, fp, ...) are skipped because they depend on fold size.
    """
    rate_keys = ["accuracy", "precision", "recall", "f1",
                 "false_positive_rate", "false_negative_rate"]
    return {
        key: {
            "mean": float(np.mean([m[key] for m in fold_metrics])),
            "std": float(np.std([m[key] for m in fold_metrics])),
        }
        for key in rate_keys
    }


def pick_best_model(results: dict[str, dict[str, float]], metric: str = "f1") -> str:
    """Name of the model with the highest value of `metric` (F1 by default)."""
    return max(results, key=lambda name: results[name][metric])
