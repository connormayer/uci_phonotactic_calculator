"""src/aggregate.py — Strategies for aggregating per‑gram log‑probability components.

All aggregator helpers return log-space values.

Note: Conditional probability always interprets the last axis as the predicted symbol.

min_val and max_val return -inf for empty input, for parity with other helpers.
"""

from typing import Sequence

import numpy as np


def log_product(components: Sequence[float]) -> float:
    """
    Aggregate by summing log‑probabilities (i.e., computing the log of a product).

    Parameters:
      components — a sequence of log‑probability floats.

    Returns:
      The sum of the input components as a float.
    """
    return float(np.sum(components))


def logsumexp(components: Sequence[float]) -> float:
    """
    Aggregate by a numerically stable log‑sum‑exp over probabilities in log‑space:
    log(∑ exp(component_i)).

    Parameters:
      components — a sequence of log‑probability floats.

    Returns:
      A float equal to log(∑ exp(components)), or -inf if all components are -inf
      (i.e., no valid probabilities).
    """
    if not components:
        return float("-inf")
    v = np.array(components, dtype=float)
    m = np.max(v)
    # If the maximum is -inf, all entries are -inf → no valid mass
    if m == float("-inf"):
        return float("-inf")
    # Shift by m for numerical stability: log ∑ exp(x) = m + log ∑ exp(x−m)
    return float(m + np.log(np.sum(np.exp(v - m))))


def linear_sum(components: Sequence[float]) -> float:
    """
    Aggregate by summing *linear-space* probabilities and then return the log.
    All aggregator helpers return log-space values.
    """
    if not components:
        return float("-inf")
    total = np.sum(np.exp(components))
    return float("-inf") if total == 0 else float(np.log(total))


def linear_sum_plus1(components: Sequence[float]) -> float:
    """
    Legacy positional-bigram score: linear Σ P + 1 (still returned in *linear*
    space, **not** logged).  When every component is -inf the sum is 0, so the
    result is exactly 1 — identical to the old script.
    """
    if not components:
        return 1.0  # mirror legacy "+1"
    total = np.sum(np.exp(components))
    return 1.0 + float(total)


def min_val(components: Sequence[float]) -> float:
    """Return the minimum log-value (–∞ for empty)."""
    return float(min(components)) if components else float("-inf")


def max_val(components: Sequence[float]) -> float:
    """Return the maximum log-value (–∞ for empty)."""
    return float(max(components)) if components else float("-inf")


# End of src/aggregate.py
