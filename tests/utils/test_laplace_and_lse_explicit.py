"""Explicit checks for Laplace-smoothed bigram constants and LSE equivalence.

These tests implement the quick sanity checks suggested in the recent
spreadsheet-inspection guide:

1.  *Bigram Laplace constant* – when every bigram count is zero, Laplace
    smoothing (+1) should assign the **same** pseudo-count to every cell.
2.  *Log-sum-exp equals log(sum pᵢ)* – aggregator consistency.
"""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray

from uci_phonotactic_calculator.models import aggregate as ag
from uci_phonotactic_calculator.plugins.utils.smoothing import dense_laplace


def test_bigram_laplace_constant() -> None:
    """All-zero bigram matrix → every cell becomes the same +1 count."""
    # Construct a 4×4 zero bigram count matrix (any V ≥ 2 is fine).
    counts: NDArray[np.float64] = np.zeros((4, 4), dtype=np.float64)

    # Apply +1 Laplace smoothing **in place**.
    smoothed = dense_laplace(counts)

    # Pick two *different* bigram cells that were both zero originally.
    p1 = smoothed[0, 1]
    p2 = smoothed[3, 2]

    # They must match exactly (both == 1.0 after smoothing).
    assert math.isclose(p1, p2, rel_tol=0.0, abs_tol=1e-12), (
        f"Laplace did not assign identical pseudo-counts: {p1} vs {p2}"
    )


def test_lse_equals_linear_sum() -> None:
    """`log_sum_exp` must equal `log(sum exp(x))` within float tolerance."""
    # Define some sample log-probabilities.
    log_probs = [math.log(0.05), math.log(0.15), math.log(0.3)]

    lse = ag.logsumexp(log_probs)
    brute = math.log(sum(math.exp(x) for x in log_probs))

    assert math.isclose(lse, brute, rel_tol=1e-9), (
        f"LSE mismatch: {lse} vs brute {brute}"
    )
