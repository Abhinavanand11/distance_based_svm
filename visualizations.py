"""All figures (Matplotlib only)."""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from distance_metrics import DISTANCES, pairwise_distance
from kernel import distance_kernel, min_eigenvalue


def _save(fig, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _subsample(Xb, y, n, seed):
    idx = np.random.RandomState(seed).choice(len(y), size=min(n, len(y)), replace=False)
    idx = idx[np.argsort(y[idx], kind="stable")]       # group by class
    return Xb[idx], y[idx]


def plot_class_distribution(y, class_names, path, title):
    counts = np.bincount(y)
    fig, ax = plt.subplots(figsize=(4.5, 3.5))
    bars = ax.bar(class_names, counts, color=["#4c72b0", "#dd8452"])
    for b, c in zip(bars, counts):
        ax.text(b.get_x() + b.get_width() / 2, c, f"{c}\n({c / counts.sum():.1%})", ha="center", va="bottom")
    ax.set(title=title, ylabel="instances", ylim=(0, counts.max() * 1.2))
    _save(fig, path)


def plot_binary_matrix(Xb, y, feature_names, path, max_rows=200, seed=0):
    Xs, ys = _subsample(Xb, y, max_rows, seed)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.imshow(Xs, aspect="auto", cmap="Greys", interpolation="nearest")
    ax.axhline(np.searchsorted(ys, 1) - 0.5, color="red", lw=1)
    ax.set_xticks(range(len(feature_names)))
    ax.set_xticklabels(feature_names, rotation=90, fontsize=6)
    ax.set(title=f"Binary feature matrix ({len(ys)} random rows, sorted by class; red line = class boundary)",
           ylabel="instance")
    _save(fig, path)


def plot_kernel_heatmaps(Xb, y, path, n_samples=300, seed=0):
    """K = 1 - D heatmaps on a random subsample grouped by class. Returns min eigenvalues (PSD check)."""
    Xs, _ = _subsample(Xb, y, n_samples, seed)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.4))
    eig = {}
    for ax, d in zip(axes, DISTANCES):
        K = distance_kernel(Xs, Xs, d)
        eig[d] = min_eigenvalue(K)
        im = ax.imshow(K, cmap="viridis", vmin=0, vmax=1)
        ax.set_title(f"{d}\nK=1-D (min eigenvalue {eig[d]:.2e})", fontsize=9)
        fig.colorbar(im, ax=ax, fraction=0.046)
    fig.suptitle(f"Kernel matrices on {len(Xs)} random instances (rows/cols grouped by class)")
    _save(fig, path)
    return eig


def classical_mds(D, k=2):
    """Classical (Torgerson) MDS. Visualization only."""
    n = D.shape[0]
    J = np.eye(n) - np.ones((n, n)) / n
    w, V = np.linalg.eigh(-0.5 * J @ (D ** 2) @ J)
    order = np.argsort(w)[::-1][:k]
    return V[:, order] * np.sqrt(np.maximum(w[order], 0))


def plot_mds(Xb, y, path, n_samples=400, seed=0):
    """2-D MDS of the distance matrices. This is a picture only: the SVM is trained on the full
    kernel matrix, never on these 2-D coordinates."""
    Xs, ys = _subsample(Xb, y, n_samples, seed)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.4))
    for ax, d in zip(axes, DISTANCES):
        Z = classical_mds(pairwise_distance(Xs, Xs, d))
        for c, col in ((0, "#4c72b0"), (1, "#dd8452")):
            ax.scatter(Z[ys == c, 0], Z[ys == c, 1], s=12, alpha=0.5, c=col, label=f"class {c}")
        ax.set_title(d)
        ax.legend()
    fig.suptitle("Classical MDS of the distance matrices (2-D visualization only, NOT the SVM input)")
    _save(fig, path)


def plot_confusion_matrices(items, class_names, path):
    """items: list of (title, [[tn, fp], [fn, tp]])."""
    fig, axes = plt.subplots(1, len(items), figsize=(4.2 * len(items), 3.8), squeeze=False)
    for ax, (title, cm) in zip(axes[0], items):
        cm = np.asarray(cm)
        ax.imshow(cm, cmap="Blues")
        for i in range(2):
            for j in range(2):
                ax.text(j, i, cm[i, j], ha="center", va="center",
                        color="white" if cm[i, j] > cm.max() / 2 else "black")
        ax.set_xticks([0, 1], [f"pred {c}" for c in class_names])
        ax.set_yticks([0, 1], [f"true {c}" for c in class_names])
        ax.set_title(title, fontsize=8)
    _save(fig, path)


def plot_accuracy_comparison(df, value_col, path, title):
    best = df.groupby("method", sort=False)[value_col].max()
    fig, ax = plt.subplots(figsize=(9, 4))
    colors = ["#c44e52" if m.startswith("proposed") else "#8c8c8c" for m in best.index]
    bars = ax.bar(best.index, best.values, color=colors)
    for b, v in zip(bars, best.values):
        ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.3f}", ha="center", va="bottom", fontsize=8)
    ax.set(title=title, ylabel=value_col, ylim=(max(0, best.min() - 0.08), min(1.02, best.max() + 0.04)))
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    _save(fig, path)


def _style(method):
    return "-" if method.startswith("proposed") else "--"


def plot_accuracy_vs_c(df, value_col, path, title):
    sizes = sorted(df["test_size"].unique())
    fig, axes = plt.subplots(1, len(sizes), figsize=(4.2 * len(sizes), 3.8), sharey=True, squeeze=False)
    for ax, n in zip(axes[0], sizes):
        sub = df[np.isclose(df["test_size"], n)]
        for m in sub["method"].unique():
            s = sub[sub["method"] == m].sort_values("C")
            ax.plot(s["C"], s[value_col], _style(m), marker="o", ms=3, label=m)
        ax.set(title=f"N = {n:.0%}", xlabel="C")
    axes[0][0].set_ylabel(value_col)
    axes[0][-1].legend(fontsize=7)
    fig.suptitle(title)
    _save(fig, path)


def plot_accuracy_vs_test_size(df, value_col, path, title):
    fig, ax = plt.subplots(figsize=(6, 4))
    for m in df["method"].unique():
        s = df[df["method"] == m].groupby("test_size")[value_col].max()
        ax.plot(s.index * 100, s.values, _style(m), marker="o", label=m)
    ax.set(title=title, xlabel="test size N (%)", ylabel=f"{value_col} (best over C)")
    ax.legend(fontsize=7)
    _save(fig, path)
