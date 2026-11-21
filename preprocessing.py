"""Equal-width binning + one-hot encoding (the paper's binarization)."""
from __future__ import annotations

import numpy as np


class EqualWidthBinarizer:
    """Split each feature's [min, max] range into `n_bins` equal-width intervals and
    one-hot encode the interval -> exactly one active column per original feature.

    * fit() stores min/max per feature (call it on TRAINING data for honest evaluation).
    * Bin i covers [edge_i, edge_{i+1}); a value equal to an inner edge goes to the upper
      bin and the maximum belongs to the last bin.
    * Values outside the fitted range (e.g. unseen test data) fall into the first/last bin.
    * A constant feature collapses into a single (always-active) bin.
    * Missing values raise an error: drop or impute them before binarizing.
    """

    def __init__(self, n_bins: int = 3):
        if n_bins < 2:
            raise ValueError("n_bins must be >= 2")
        self.n_bins = n_bins

    @staticmethod
    def _check(X):
        X = np.asarray(X, dtype=np.float64)
        if X.ndim != 2:
            raise ValueError("X must be 2-D")
        if np.isnan(X).any():
            raise ValueError("NaN values found; remove or impute them before binarizing")
        return X

    def fit(self, X, feature_names=None):
        X = self._check(X)
        self.n_features_in_ = X.shape[1]
        self.min_, self.max_ = X.min(axis=0), X.max(axis=0)
        self.inner_edges_ = [np.linspace(lo, hi, self.n_bins + 1)[1:-1]
                             for lo, hi in zip(self.min_, self.max_)]
        names = feature_names or [f"f{j}" for j in range(self.n_features_in_)]
        self.feature_names_out_ = []
        for j, name in enumerate(names):
            e = np.linspace(self.min_[j], self.max_[j], self.n_bins + 1)
            self.feature_names_out_ += [f"{name}[{e[k]:.4g},{e[k + 1]:.4g})" for k in range(self.n_bins)]
        return self

    def transform(self, X):
        X = self._check(X)
        if X.shape[1] != self.n_features_in_:
            raise ValueError("Wrong number of features")
        n = X.shape[0]
        out = np.zeros((n, self.n_features_in_ * self.n_bins), dtype=np.float64)
        rows = np.arange(n)
        for j, edges in enumerate(self.inner_edges_):
            out[rows, j * self.n_bins + np.digitize(X[:, j], edges)] = 1.0
        assert (out.sum(axis=1) == self.n_features_in_).all()   # one active bin per feature
        return out

    def fit_transform(self, X, feature_names=None):
        return self.fit(X, feature_names).transform(X)
