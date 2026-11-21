"""Metrics, result tables and comparison with the paper's reported numbers."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

# Paper (Amaya-Tejera et al., 2024, Table 9): best test accuracy per dataset.
PAPER_TARGETS = {
    "banknote": dict(distance="sokal_michener", test_size=0.60, C=40, accuracy=0.949),
    "rice":     dict(distance="sokal_michener", test_size=0.60, C=70, accuracy=0.890),
}


def _div(a, b):
    return float(a) / float(b) if b > 0 else 0.0


def classification_metrics(y_true, y_pred):
    """Binary metrics with class 1 as the positive class."""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    precision, recall = _div(tp, tp + fp), _div(tp, tp + fn)
    return dict(
        accuracy=_div(tp + tn, tp + tn + fp + fn), precision=precision, recall=recall,
        specificity=_div(tn, tn + fp), f1=_div(2 * precision * recall, precision + recall),
        tn=int(tn), fp=int(fp), fn=int(fn), tp=int(tp))


def public_columns(df):
    return df[[c for c in df.columns if not c.startswith("_")]]


def save_table(df, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    public_columns(df).to_csv(path, index=False, float_format="%.4f")


def comparison_table(df, value_col):
    """Dataset, distance, test size, C, training accuracy and test/validation accuracy."""
    cols = ["dataset", "method", "distance", "test_size", "C", "train_acc", value_col]
    return df[cols].sort_values(["dataset", "method", "test_size", "C"]).reset_index(drop=True)


def best_per_method(df, value_col):
    """Best configuration of every method (ties: higher training accuracy)."""
    ordered = df.sort_values([value_col, "train_acc"], ascending=False)
    return ordered.groupby("method", sort=False).head(1).reset_index(drop=True)


def best_proposed(df, value_col):
    sub = df[df["kernel"] == "proposed"]
    return sub.sort_values([value_col, "train_acc"], ascending=False).iloc[0]


def paper_comparison(df, dataset, value_col="test_acc"):
    """Plain-text comparison with the paper (nothing is forced or hard-coded in the model)."""
    t = PAPER_TARGETS[dataset]
    m = df[(df["method"] == f"proposed-{t['distance']}") & np.isclose(df["test_size"], t["test_size"])
           & (df["C"] == t["C"])]
    lines = [f"Paper target ({dataset}): {t['distance']}, test size {t['test_size']:.0%}, "
             f"C={t['C']} -> {t['accuracy']:.3f}"]
    if len(m):
        a = float(m[value_col].iloc[0])
        lines.append(f"  This run, same configuration: {a:.3f}  (difference {a - t['accuracy']:+.3f}); "
                     f"mean over all random splits: {float(m['mean_split_acc'].iloc[0]):.3f}")
    b = best_proposed(df, value_col)
    lines.append(f"  This run, best proposed config: {b['distance']}, N={b['test_size']:.0%}, "
                 f"C={b['C']} -> {b[value_col]:.3f}")
    lines.append("  NOTE: Mode A accuracies are maxima over many random splits chosen on the test "
                 "split itself -> optimistic, not a generalization estimate.")
    return "\n".join(lines)
