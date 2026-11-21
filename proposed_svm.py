"""Algorithm 1 of the paper (iterative best-subset search) with precomputed kernels."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from joblib import Parallel, delayed
from sklearn.svm import SVC

from evaluation import classification_metrics


# ---------------------------------------------------------------- splitting
def split_indices(y, test_size, rng, stratify=False):
    """Random (optionally stratified) split into (train_idx, test_idx). test_size is a fraction."""
    y = np.asarray(y)
    if stratify:
        parts = []
        for c in np.unique(y):
            idx = np.flatnonzero(y == c)
            rng.shuffle(idx)
            parts.append(idx[:int(round(test_size * len(idx)))])
        test = np.concatenate(parts)
    else:
        test = rng.permutation(len(y))[:int(round(test_size * len(y)))]
    mask = np.zeros(len(y), dtype=bool)
    mask[test] = True
    return np.flatnonzero(~mask), np.flatnonzero(mask)


# ---------------------------------------------------------------- SVM helpers
def fit_precomputed_svm(K, y, train_idx, C):
    """Fit on K(X_train, X_train)."""
    model = SVC(C=C, kernel="precomputed", cache_size=500)
    model.fit(K[np.ix_(train_idx, train_idx)], y[train_idx])
    return model


def predict_precomputed(model, K, rows, train_idx):
    """Predict with K(X_rows, X_train); K may be the full matrix or a (test x train) block."""
    return model.predict(K[np.ix_(rows, train_idx)])


# ---------------------------------------------------------------- Algorithm 1
@dataclass
class SearchResult:
    best_accuracy: float
    best_train_idx: np.ndarray
    best_test_idx: np.ndarray
    accuracies: list
    n_iterations: int


def search_best_subset(K, y, test_size, C, patience=100, max_iter=1000, seed=0, stratify=False):
    """Repeat random train/test splits; keep the training subset with the best accuracy.

    `wait` is reset to 0 whenever the accuracy strictly improves (as in the paper's text; the
    printed pseudo-code never resets it). Stops after `patience` consecutive non-improving
    iterations or `max_iter` iterations. K is the FULL kernel matrix over the rows of y;
    each iteration slices K(train, train) and K(test, train) from it.
    """
    rng = np.random.RandomState(seed)
    best_acc, best_tr, best_te = 0.0, None, None
    wait, it, accs = 0, 0, []
    while wait < patience and it < max_iter:
        it += 1
        wait += 1
        tr, te = split_indices(y, test_size, rng, stratify)
        if len(np.unique(y[tr])) < 2:
            continue                                    # cannot train a binary SVM
        model = fit_precomputed_svm(K, y, tr, C)
        acc = float(np.mean(predict_precomputed(model, K, te, tr) == y[te]))
        accs.append(acc)
        if acc > best_acc:
            best_acc, best_tr, best_te, wait = acc, tr, te, 0
    if best_tr is None:
        raise RuntimeError("No valid split found")
    return SearchResult(best_acc, best_tr, best_te, accs, it)


def run_config(K, y, meta, test_size, C, patience, max_iter, seed, stratify):
    """One (kernel, N, C) experiment: search, refit on the best subset, compute metrics."""
    res = search_best_subset(K, y, test_size, C, patience, max_iter, seed, stratify)
    tr, te = res.best_train_idx, res.best_test_idx
    model = fit_precomputed_svm(K, y, tr, C)           # step 12: final SVM on the saved subset
    train_acc = float(np.mean(predict_precomputed(model, K, tr, tr) == y[tr]))
    m = classification_metrics(y[te], predict_precomputed(model, K, te, tr))
    return {**meta, "test_size": test_size, "C": C, "n_train": len(tr), "n_eval": len(te),
            "n_iterations": res.n_iterations, "train_acc": train_acc,
            "test_acc": m.pop("accuracy"), "mean_split_acc": float(np.mean(res.accuracies)),
            **m, "_train_idx": tr}


def config_seed(base_seed, test_size, C):
    """Same (N, C) -> same stream of random splits for every kernel (comparable splits)."""
    return (base_seed * 10007 + int(round(test_size * 100)) * 101 + int(C)) % (2 ** 31 - 1)


def run_grid(K, y, meta, test_sizes, c_values, patience, max_iter, base_seed, stratify, n_jobs=1):
    tasks = [(n, c) for n in test_sizes for c in c_values]
    rows = Parallel(n_jobs=n_jobs)(
        delayed(run_config)(K, y, meta, n, c, patience, max_iter, config_seed(base_seed, n, c), stratify)
        for n, c in tasks)
    for r in rows:
        print(f"  [{r['method']:<24}] N={r['test_size']:.0%} C={r['C']:<3} "
              f"best={r['test_acc']:.3f} mean={r['mean_split_acc']:.3f} iters={r['n_iterations']}")
    return rows


def random_subset_baseline(K_train, K_eval, y_train, y_eval, n_train, C, n_draws, seed):
    """DIAGNOSTIC ONLY (Mode B): mean/std accuracy on the final test set of random training subsets of the
    same size as the selected one. Never used for any selection decision."""
    rng = np.random.RandomState(seed)
    accs = []
    for _ in range(n_draws):
        tr, _ = split_indices(y_train, 1 - n_train / len(y_train), rng, stratify=True)
        model = fit_precomputed_svm(K_train, y_train, tr, C)
        accs.append(np.mean(predict_precomputed(model, K_eval, np.arange(len(y_eval)), tr) == y_eval))
    return float(np.mean(accs)), float(np.std(accs))
