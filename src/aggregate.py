"""src/aggregate.py — Strategies for aggregating per‑gram log‑probability components."""

import numpy as np
from .config import AggregateMode


def log_product(components: list[float]) -> float:
    """
    Aggregate by summing log‑probabilities (i.e., computing the log of a product).

    Parameters:
      components — a list of log‑probability floats.

    Returns:
      The sum of the input components as a float.
    """
    return float(np.sum(components))


def logsumexp(components: list[float]) -> float:
    """
    Aggregate by a numerically stable log‑sum‑exp over probabilities in log‑space:
    log(∑ exp(component_i)).

    Parameters:
      components — a list of log‑probability floats.

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


def linear_sum(components: list[float]) -> float:
    """
    Aggregate by summing raw probabilities:
    convert each log‑probability to linear space via exp(),
    then sum those probabilities.

    Parameters:
      components — a list of log‑probability floats.

    Returns:
      The sum of the corresponding probabilities as a float,
      or -inf if components is empty.
    """
    if not components:
        return float("-inf")
    # Convert from log space to probabilities and sum them
    return float(np.sum(np.exp(components)))


AGGREGATORS: dict[AggregateMode, callable] = {
    AggregateMode.LOG_PRODUCT: log_product,
    AggregateMode.LOGSUMEXP:   logsumexp,
    AggregateMode.SUM:         linear_sum,
}

# End of src/aggregate.py