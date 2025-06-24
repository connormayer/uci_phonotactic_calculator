"""
Kneser–Ney / absolute-discount smoothing for bigram count tables.

The function is registered as the “kneser_ney” smoothing_scheme, so it can be
selected from the CLI with, e.g.:

    --smoothing-scheme kneser_ney

Implementation notes
--------------------
* **Order support** – Only bigrams are smoothed; for unigrams or n ≥ 3 the
  original counts are returned unchanged (they will still be normalised later
  by the probability-transform stage).
* **Discount constant** – A single absolute discount D = 0.75 (Chen & Goodman
  1999 recommendation) is applied to every non-zero count.
* **Dense ↔ sparse parity** – Logic is shared; sparse tables are converted to a
  temporary dense matrix, smoothed, then reconverted.  This keeps the code
  compact and guarantees identical numeric results.
* **Return type** – Always returns the *same* container type that was supplied
  (ndarray in → ndarray out, dict in → dict out) so no caller needs to change.

The algorithm preserves the row totals after discounting and back-off mass
redistribution, which means subsequent probability transforms (joint /
conditional) work exactly as before.
"""

from __future__ import annotations

from typing import MutableMapping, Union

import numpy as np
from numpy.typing import NDArray

from ...core.registries import register
from .smoothing import IndexTuple, SparseTable

# Absolute discount value recommended for KN
_D: float = 0.75


# ──────────────────────────────────────────────────────────────────────
# Dense helper
# ──────────────────────────────────────────────────────────────────────
def _kn_dense(table: NDArray[np.float64], *, D: float = _D) -> NDArray[np.float64]:
    """
    Modified Kneser–Ney smoothing for a **bigram** count matrix.

    Parameters
    ----------
    table : np.ndarray, shape (V, V)
        Raw bigram counts.  Axis-0 = history; axis-1 = next symbol.
    D : float, default 0.75
        Absolute discount.  Must be 0 < D < 1.

    Returns
    -------
    np.ndarray
        Smoothed pseudo-counts, same shape and dtype as *table*.
    """
    if table.ndim != 2:  # Only bigrams are handled
        return table

    counts = table
    V = counts.shape[0]

    # Continuation statistics – how many unique histories precede each word?
    continuation = (counts > 0).sum(axis=0)  # shape (V,)
    total_types = continuation.sum() or 1  # avoid div-by-zero
    p_cont = continuation / total_types  # sums to 1.0

    row_tot = counts.sum(axis=1)  # total tokens seen after each history
    n_types = (counts > 0).sum(axis=1)  # unique successors per history

    out = np.zeros_like(counts, dtype=float)

    for h in range(V):
        tot = row_tot[h]
        if tot == 0:
            continue  # unseen history – leave all zero

        lam = (D * n_types[h]) / tot  # back-off weight λ(h)
        discounted = np.maximum(counts[h] - D, 0.0)  # max(c-D,0)

        # Redistribute the held-out mass proportionally to continuation probs
        out[h] = discounted + (lam * tot) * p_cont

    return out


# ──────────────────────────────────────────────────────────────────────
# Sparse helper – wraps dense version for identical maths
# ──────────────────────────────────────────────────────────────────────
def _kn_sparse(
    table: SparseTable, *, D: float = _D
) -> MutableMapping[IndexTuple, float]:
    if not table:
        return table  # nothing to smooth

    # Only apply Kneser–Ney to bigram (order-2) count tables.  If the keys
    # indicate a different n-gram order, return the table unchanged so that
    # downstream components can handle it appropriately.
    sample_key = next(iter(table))
    if len(sample_key) != 2:
        return table  # leave unigram or higher-order counts untouched

    # Infer vocabulary size from the highest index observed
    V = max(max(idx) for idx in table) + 1
    dense = np.zeros((V, V), dtype=float)
    for (i, j), c in table.items():
        dense[i, j] = c

    smoothed = _kn_dense(dense, D=D)

    # Re-serialise to the same (sparse) structure
    new_table: dict[IndexTuple, float] = {}
    nz = np.nonzero(smoothed)
    for i, j in zip(*nz):
        new_table[(int(i), int(j))] = float(smoothed[i, j])

    return new_table


# ──────────────────────────────────────────────────────────────────────
# Public registry entry
# ──────────────────────────────────────────────────────────────────────
@register("smoothing_scheme", "kneser_ney")
def kneser_ney(
    table: Union[NDArray[np.float64], MutableMapping[IndexTuple, float]],
    *,
    vocab: object | None = None,  # kept for API parity; unused here
) -> Union[NDArray[np.float64], MutableMapping[IndexTuple, float]]:
    """
    Entry-point required by the calculator’s smoothing registry.

    The *vocab* parameter is accepted for signature compatibility but is not
    needed in this implementation.
    """
    if isinstance(table, np.ndarray):
        return _kn_dense(table)
    elif isinstance(table, MutableMapping):
        return _kn_sparse(table)
    else:
        raise TypeError(f"Unsupported table type for Kneser–Ney: {type(table)!r}")


__all__ = ["kneser_ney"]
