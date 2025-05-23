"""
Utility helpers for Laplace (+1) smoothing.

Public API
----------
dense_laplace(arr)          – in-place +1 for NumPy tensors
sparse_laplace(table,vocab) – +1 for dict-based sparse tables
apply(cfg, table, vocab)    – single entry-point used by every plugin
"""

from typing import Mapping, MutableMapping, Tuple, TypeAlias

import numpy as np

# ---------------------------------------------------------------------
# Public type aliases — importable by other modules
# ---------------------------------------------------------------------
IndexTuple: TypeAlias = Tuple[int, ...]  # n-gram encoded as index IDs
SparseTable: TypeAlias = MutableMapping[IndexTuple, float]


# ---------------------------------------------------------------------
# Laplace (+1) smoothing for dict-based tables
# ---------------------------------------------------------------------
def sparse_laplace(
    table: SparseTable,
    vocab: Mapping[IndexTuple, float] | set[IndexTuple],
) -> SparseTable:
    """
    Add-one Laplace smoothing for a sparse n-gram table in place.

    Parameters
    ----------
    table : dict[(int, ...), float]
        The counts to be smoothed. Modified in place.
    vocab : Iterable[(int, ...)]
        Every n-gram key that must appear in *table* after smoothing.

    Returns
    -------
    SparseTable
        The same dict instance (convenient for chaining).

    Notes
    -----
    • This helper is intentionally allocation-free for performance.
    • It does *not* normalise the counts; callers handle that.
    """
    for key in vocab:
        table[key] = table.get(key, 0.0) + 1.0
    return table


def dense_laplace(arr: np.ndarray) -> np.ndarray:
    """
    Vectorised +1 Laplace smoothing for dense count tensors.
    The call is in-place-safe: it returns the view `arr` itself with 1.0 added,
    so callers do **not** pay an extra allocation.
    """
    arr += 1.0
    return arr


def apply(table, *, vocab=None):
    """
    Unified Laplace smoothing entry-point used by all plugins.

    Parameters
    ----------
    table : ndarray | MutableMapping
        The counts to be smoothed. Modified in place.
    vocab : For sparse tables, the full n-gram key-set; for dense arrays, ignored
        (default: None).

    Returns
    -------
    The same object instance (dense or sparse) – possibly smoothed.

    Raises
    ------
    TypeError if *table* is neither ndarray nor MutableMapping.
    """
    # Guard: reject negative counts
    if isinstance(table, np.ndarray):
        # Legacy compatibility: replace all negative and -inf counts with zero
        # before Laplace smoothing.
        # This allows legacy_log and 2018-style weighting to work with Laplace
        # smoothing.
        table = np.where(np.isneginf(table) | (table < 0), 0.0, table)
        return dense_laplace(table)
    elif isinstance(table, MutableMapping):
        # Legacy compatibility: replace all negative and -inf counts with zero
        # before Laplace smoothing.
        # This allows legacy_log and 2018-style weighting to work with Laplace
        # smoothing.
        for k, v in list(table.items()):
            if v is None or v < 0 or (isinstance(v, float) and np.isneginf(v)):
                table[k] = 0.0
        if vocab is None:
            raise ValueError("Vocab must be provided for sparse Laplace smoothing.")
        return sparse_laplace(table, vocab)
    else:
        raise TypeError(f"Unknown table type for Laplace smoothing: {type(table)}")


__all__ = [
    "IndexTuple",
    "SparseTable",
    "sparse_laplace",
    "dense_laplace",
    "apply",
]
