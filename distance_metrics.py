"""Binary similarity coefficients and the corresponding distances.

p = #features where both are 1     q = #features where x=1, y=0
r = #features where x=0, y=1       s = #features where both are 0

SIMILARITY (large = alike)                 DISTANCE = 1 - similarity (small = alike)
Jaccard            p / (p+q+r)             1 - p/(p+q+r)             = (q+r)/(p+q+r)
Rogers-Tanimoto    (p+s) / (p+s+2(q+r))    2(q+r) / (p+s+2(q+r))
Sokal-Michener     (p+s) / (p+q+r+s)       (q+r) / (p+q+r+s)

NOTE (paper, Table 3): the Rogers-Tanimoto and Sokal-Michener rows are labelled
"D" but contain the similarity coefficients (p+s)/(...). The text says the
dissimilarity is 1 - coefficient, which is what is implemented here.

NOTE: scipy/sklearn's "sokalmichener" is 2(q+r)/(n+(q+r)), which equals
Rogers-Tanimoto, NOT (q+r)/n. This project uses the textbook definition above.

Zero denominators: only Jaccard can have one (both vectors all-zero). Two empty
sets are treated as identical (similarity 1, distance 0).
"""
from __future__ import annotations

import numpy as np


def pair_counts(A, B):
    """Return p, q, r, s matrices of shape (len(A), len(B)) for binary rows."""
    A = np.asarray(A, dtype=np.float64)
    B = np.asarray(B, dtype=np.float64)
    if A.shape[1] != B.shape[1]:
        raise ValueError("A and B must have the same number of columns")
    for M in (A, B):
        if not np.isin(M, (0.0, 1.0)).all():
            raise ValueError("Distance measures require binary (0/1) input")
    p = A @ B.T
    q = A.sum(axis=1)[:, None] - p
    r = B.sum(axis=1)[None, :] - p
    s = A.shape[1] - p - q - r
    return p, q, r, s


def _safe_divide(num, den, fill):
    out = np.full(num.shape, fill, dtype=np.float64)
    np.divide(num, den, out=out, where=den > 0)
    return out


def jaccard_similarity(A, B):
    p, q, r, _ = pair_counts(A, B)
    return _safe_divide(p, p + q + r, 1.0)


def rogers_tanimoto_similarity(A, B):
    p, q, r, s = pair_counts(A, B)
    return _safe_divide(p + s, p + s + 2 * (q + r), 1.0)


def sokal_michener_similarity(A, B):
    p, q, r, s = pair_counts(A, B)
    return _safe_divide(p + s, p + q + r + s, 1.0)


SIMILARITIES = {
    "jaccard": jaccard_similarity,
    "rogers_tanimoto": rogers_tanimoto_similarity,
    "sokal_michener": sokal_michener_similarity,
}
DISTANCES = list(SIMILARITIES)


def pairwise_distance(A, B, name):
    """Distance matrix D(A, B) = 1 - similarity(A, B)."""
    if name not in SIMILARITIES:
        raise ValueError(f"Unknown distance '{name}'. Choose from {DISTANCES}")
    return 1.0 - SIMILARITIES[name](A, B)
