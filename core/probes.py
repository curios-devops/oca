"""One ridge probe, standardised, used by every experiment.

Probes decide almost every conclusion in this project, so they are worth getting right
once rather than five times. Two properties matter:

**Features are standardised before the fit.** Ridge penalises the raw coefficient vector,
so with a shared penalty scaled by `trace(X'X)/d` the large-variance dimensions absorb the
whole penalty and the small ones are left effectively unregularised. That is not
hypothetical here: the assembly workspace spans six orders of magnitude in per-dimension
standard deviation (condition number ~1e10), and an unstandardised probe returned position
errors *above chance* -- a fit so wild it was worse than predicting the mean. Z-scoring
with training statistics makes the penalty mean the same thing in every direction.

**Held out in time, never at random.** These are trajectories; a random split leaks the
answer through neighbouring frames.
"""

from __future__ import annotations

import numpy as np


def whiten(X: np.ndarray, cut: int, n_components: int | None = None):
    """Centre, scale, and optionally project to a fixed number of components.

    `n_components` exists to make comparisons between representations *fair*. Decoding
    from a 3456-dimensional mesh state and from a 96-dimensional pooled vector on the same
    1600 training frames is not a test of which holds more information -- the wider one is
    underdetermined and overfits. Measured without it: the mesh state returned a position
    error of 993,925 cells. Projecting every representation onto the same number of
    principal components gives them equal capacity, so a difference means what it claims.
    """
    mu = X[:cut].mean(axis=0)

    def centre(Z):
        return Z - mu

    if n_components is None or n_components >= X.shape[1]:
        # no projection: scale per dimension, with a floor relative to the typical scale.
        # An absolute floor like 1e-8 leaves a dimension of sd 1e-6 amplified a
        # millionfold, which is its own kind of overfitting.
        sd = X[:cut].std(axis=0)
        sd = np.maximum(sd, max(np.median(sd), 1e-12) * 1e-3)
        return lambda Z: centre(Z) / sd

    # PCA on *centred* data, never on per-dimension-whitened data. Whitening first
    # inflates every near-constant direction to unit variance, so the SVD then ranks
    # numerical noise as signal -- measured: a mesh-state decode returning a position
    # error of 217 cells against a chance of 7.8. Centring first keeps the components
    # ordered by genuine variance; the scaling happens afterwards, on the components.
    _, s, vt = np.linalg.svd(centre(X[:cut]), full_matrices=False)
    comp = vt[:n_components].T
    scale = s[:n_components] / np.sqrt(max(cut - 1, 1))
    scale = np.maximum(scale, max(np.median(scale), 1e-12) * 1e-3)
    return lambda Z: (centre(Z) @ comp) / scale


def fit_ridge(X: np.ndarray, Y: np.ndarray, split: float = 0.6, ridge: float = 1e-2,
              n_components: int | None = None):
    """Standardise, fit on the first `split` of time, return (predict_fn, test_slice)."""
    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    n = len(X)
    cut = int(n * split)
    reduce = whiten(X, cut, n_components)

    def prep(Z):
        return np.c_[reduce(Z), np.ones(len(Z))]

    Xtr = prep(X[:cut])
    A = Xtr.T @ Xtr
    A[np.diag_indices_from(A)] += ridge * np.trace(A) / A.shape[0]
    W = np.linalg.solve(A, Xtr.T @ Y[:cut])
    return (lambda Z: prep(Z) @ W), slice(cut, n)


def decode_error(X, Y, split: float = 0.6, ridge: float = 1e-2,
                 n_components: int | None = None) -> np.ndarray:
    """Per-frame Euclidean error of a held-out decode of `Y` from `X`."""
    predict, te = fit_ridge(X, Y, split, ridge, n_components)
    return np.linalg.norm(predict(np.asarray(X)[te]) - np.asarray(Y)[te], axis=1)


def balanced_accuracy(pred: np.ndarray, true: np.ndarray) -> float:
    """Mean per-class accuracy, so a constant guess scores 0.5 on two classes."""
    accs = [float((pred[true == c] == c).mean()) for c in np.unique(true)
            if (true == c).any()]
    return float(np.mean(accs)) if accs else float("nan")


def decode_binary(X, y, split: float = 0.6, ridge: float = 1e-2,
                  n_components: int | None = None):
    """Ridge classifier on {-1,+1}, scored by balanced accuracy on a time split."""
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y)
    if len(X) < 40 or len(np.unique(y)) < 2:
        return None
    target = np.where(y == np.unique(y)[-1], 1.0, -1.0)[:, None]
    predict, te = fit_ridge(X, target, split, ridge, n_components)
    pred = np.where(predict(X[te])[:, 0] > 0, np.unique(y)[-1], np.unique(y)[0])
    return {"balanced_acc": balanced_accuracy(pred, y[te]), "n_test": int(te.stop - te.start)}
