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

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # draw straight to files; no window needed
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.metrics import confusion_matrix  # noqa: E402

# One fixed colour per model, used in every chart so a reader can follow a
# model across figures. Colours are from a colour-blind-checked palette.
MODEL_COLOURS = {"neural_network": "#2a78d6", "svm": "#eb6834", "kmeans": "#1baf7a"}
MODEL_LABELS = {"neural_network": "Neural network", "svm": "SVM", "kmeans": "K-Means"}
ATTACK_CATEGORY_ORDER = ["DoS", "Probe", "R2L", "U2R"]
TEXT_COLOUR = "#52514e"


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


# --- Breakdown by attack type ----------------------------------------------

def detection_rate_by_category(categories, y_pred) -> dict[str, dict[str, float | int]]:
    """For each attack category, the share of its records the model flagged.

    This is recall calculated separately for DoS, Probe, R2L and U2R. It shows
    *which kinds* of attack a model misses, which a single F1 score hides.
    """
    categories = np.asarray(categories)
    y_pred = np.asarray(y_pred)
    breakdown = {}
    for category in ATTACK_CATEGORY_ORDER:
        mask = categories == category
        n_records = int(mask.sum())
        n_detected = int(y_pred[mask].sum())
        breakdown[category] = {
            "records": n_records,
            "detected": n_detected,
            "detection_rate": _safe_divide(n_detected, n_records),
        }
    return breakdown


def detection_rate_seen_vs_unseen(attack_names, y_pred,
                                  training_attack_names: set[str]) -> dict[str, dict]:
    """Detection rate for attack types that were / were not in the training data.

    KDDTest+ contains attack types never seen in KDDTrain+. Splitting the
    test attacks this way shows how well each model generalises to new attacks.
    """
    attack_names = np.asarray(attack_names)
    y_pred = np.asarray(y_pred)
    is_attack = attack_names != "normal"
    seen = np.isin(attack_names, list(training_attack_names))
    result = {}
    for label, mask in (("seen_in_training", is_attack & seen),
                        ("unseen_in_training", is_attack & ~seen)):
        n_records = int(mask.sum())
        n_detected = int(y_pred[mask].sum())
        result[label] = {"records": n_records, "detected": n_detected,
                         "detection_rate": _safe_divide(n_detected, n_records)}
    return result


# --- Charts ----------------------------------------------------------------

def _style_axes(ax) -> None:
    """Quiet axes so the data stands out: no top/right border, light grid."""
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#c3c2b7")
    ax.tick_params(colors=TEXT_COLOUR)
    ax.yaxis.grid(True, color="#e6e5e0", linewidth=0.8)
    ax.set_axisbelow(True)


def plot_confusion_matrix(counts: dict[str, int], model_name: str, path: Path) -> None:
    """Save a 2x2 confusion matrix showing counts and row percentages.

    Rows are the true class, columns the prediction. Row percentages make
    the false positive rate (top right) and false negative rate (bottom left)
    visible directly.
    """
    matrix = np.array([[counts["tn"], counts["fp"]], [counts["fn"], counts["tp"]]])
    row_share = matrix / matrix.sum(axis=1, keepdims=True)

    fig, ax = plt.subplots(figsize=(4.6, 4))
    ax.imshow(row_share, cmap="Blues", vmin=0, vmax=1)
    cell_names = [["True negative", "False positive"], ["False negative", "True positive"]]
    for i in range(2):
        for j in range(2):
            # Dark cells get white text so the numbers stay readable.
            colour = "white" if row_share[i, j] > 0.6 else "#0b0b0b"
            ax.text(j, i, f"{cell_names[i][j]}\n{matrix[i, j]:,}\n({row_share[i, j]:.1%})",
                    ha="center", va="center", color=colour, fontsize=9)
    ax.set_xticks([0, 1], labels=["Predicted normal", "Predicted attack"])
    ax.set_yticks([0, 1], labels=["Actual normal", "Actual attack"])
    ax.tick_params(length=0, colors=TEXT_COLOUR)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_title(f"{MODEL_LABELS.get(model_name, model_name)}: confusion matrix (KDDTest+)",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _grouped_bars(ax, group_labels: list[str], values_by_model: dict[str, list[float]],
                  value_format: str) -> None:
    """Draw one bar per model inside each group, with the value printed above each bar."""
    n_models = len(values_by_model)
    width = 0.8 / n_models
    x = np.arange(len(group_labels))
    for i, (model, values) in enumerate(values_by_model.items()):
        positions = x - 0.4 + width * (i + 0.5)
        bars = ax.bar(positions, values, width=width * 0.92,
                      color=MODEL_COLOURS.get(model), label=MODEL_LABELS.get(model, model))
        ax.bar_label(bars, labels=[value_format.format(v) for v in values],
                     fontsize=8, color=TEXT_COLOUR, padding=2)
    ax.set_xticks(x, labels=group_labels)


def plot_error_rates(test_results: dict[str, dict], path: Path) -> None:
    """Save a bar chart comparing false positive and false negative rates per model.

    Both rates share one axis on purpose: it shows how much larger missed
    attacks are than false alarms for every model on KDDTest+.
    """
    values = {
        name: [m["false_positive_rate"], m["false_negative_rate"]]
        for name, m in test_results.items()
    }
    fig, ax = plt.subplots(figsize=(6.5, 4))
    _grouped_bars(ax, ["False positive rate\n(normal traffic flagged)",
                       "False negative rate\n(attacks missed)"], values, "{:.1%}")
    _style_axes(ax)
    ax.set_ylim(0, 1)
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.set_title("Error rates on KDDTest+ (lower is better)", fontsize=10)
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_detection_by_category(by_model: dict[str, dict[str, dict]], path: Path) -> None:
    """Save a bar chart of detection rate for each attack category and model."""
    first = next(iter(by_model.values()))
    group_labels = [f"{c}\n(n = {first[c]['records']:,})" for c in ATTACK_CATEGORY_ORDER]
    values = {name: [cats[c]["detection_rate"] for c in ATTACK_CATEGORY_ORDER]
              for name, cats in by_model.items()}
    fig, ax = plt.subplots(figsize=(7.5, 4))
    _grouped_bars(ax, group_labels, values, "{:.0%}")
    _style_axes(ax)
    ax.set_ylim(0, 1.08)
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.set_ylabel("Share of attacks detected", color=TEXT_COLOUR)
    ax.set_title("Detection rate by attack category (KDDTest+)", fontsize=10)
    ax.legend(frameon=False, fontsize=9, loc="upper right")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
