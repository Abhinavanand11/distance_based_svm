"""Kernel matrices: the proposed K = 1 - D, plus standard kernels for comparison.

Every kernel is returned as a dense block K(A, B) so that SVC(kernel="precomputed")
can be used uniformly: K(X_train, X_train) for fitting, K(X_test, X_train) for predicting.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics.pairwise import (linear_kernel, polynomial_kernel,
                                      rbf_kernel, sigmoid_kernel)

from distance_metrics import DISTANCES, pairwise_distance

PROPOSED_KERNELS = tuple(DISTANCES)
STANDARD_KERNELS = ("linear", "rbf", "poly", "sigmoid")


def distance_kernel(A, B, distance):
    """Proposed kernel K(x, y) = 1 - D(x, y)."""
    return 1.0 - pairwise_distance(A, B, distance)


def scale_gamma(X):
    """sklearn's gamma='scale': 1 / (n_features * X.var()). Fit on training data only."""
    X = np.asarray(X, dtype=np.float64)
    v = X.var()
    return 1.0 / (X.shape[1] * v) if v > 0 else 1.0


def standard_kernel(A, B, name, gamma):
    if name == "linear":
        return linear_kernel(A, B)
    if name == "rbf":
        return rbf_kernel(A, B, gamma=gamma)
    if name == "poly":                      # degree 3, coef0 0 (as in the paper / sklearn default)
        return polynomial_kernel(A, B, degree=3, gamma=gamma, coef0=0.0)
    if name == "sigmoid":
        return sigmoid_kernel(A, B, gamma=gamma, coef0=0.0)
    raise ValueError(f"Unknown kernel '{name}'")


def kernel_matrix(A, B, name, gamma=None):
    """Dispatch on kernel name (a distance name -> proposed kernel)."""
    A = np.asarray(A, dtype=np.float64)
    B = np.asarray(B, dtype=np.float64)
    if name in PROPOSED_KERNELS:
        return distance_kernel(A, B, name)
    return standard_kernel(A, B, name, scale_gamma(A) if gamma is None else gamma)


def min_eigenvalue(K):
    """Smallest eigenvalue of a square kernel matrix (< 0 means not positive semi-definite)."""
    return float(np.linalg.eigvalsh((K + K.T) / 2).min())
