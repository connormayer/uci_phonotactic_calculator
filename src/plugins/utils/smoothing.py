"""
Utility helpers for Laplace (+1) smoothing.

Public API
----------
dense_laplace(arr)          – in-place +1 for NumPy tensors
sparse_laplace(table,vocab) – +1 for dict-based sparse tables
apply(cfg, table, vocab)    – single entry-point used by every plugin
"""

import numpy as np
from typing import MutableMapping, Mapping, Tuple, TypeAlias, Any

# ---------------------------------------------------------------------
# Public type aliases — importable by other modules
# ---------------------------------------------------------------------
IndexTuple: TypeAlias = Tuple[int, ...]              # n-gram encoded as index IDs
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

from ...config import Config

def apply(cfg: 'Config', table: Any, vocab=None):
    """
    Unified Laplace helper used by *all* plugins.

    Parameters
    ----------
    cfg   : Config      – honours cfg.smoothing
    table : ndarray | MutableMapping
    vocab : full n-gram key-set for sparse tables (ignored for dense)

    Returns
    -------
    The same object instance (dense or sparse) – possibly smoothed.

    Raises
    ------
    TypeError if *table* is neither ndarray nor MutableMapping.
    """
    if not cfg.smoothing:
        return table
    if hasattr(table, "shape"):                 # NumPy ndarray branch
        return dense_laplace(table)
    if isinstance(table, MutableMapping):       # sparse dict / defaultdict branch
        if vocab is None:
            raise ValueError("apply() requires *vocab* for sparse tables")
        return sparse_laplace(table, vocab)
    raise TypeError(f"smoothing.apply(): unsupported table type {type(table).__name__}")

__all__ = [
    "IndexTuple", "SparseTable",
    "sparse_laplace", "dense_laplace", "apply",
]
