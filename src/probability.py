"""probability.py — Log‑space probability transforms for count matrices."""

from collections.abc import Callable
import numpy as np

from .config import ProbMode

# A transform takes a raw count array and returns log‑probabilities
Transform = Callable[[np.ndarray], np.ndarray]

# Registry mapping each ProbMode to its corresponding transform
_PROB_TRANSFORMS: dict[ProbMode, Transform] = {}

def register_prob(mode: ProbMode):
    """
    Decorator to register a new probability transform for a given mode.
    Raises KeyError if the mode is already registered.
    """
    def _decorator(fn: Transform) -> Transform:
        if mode in _PROB_TRANSFORMS:
            raise KeyError(f"Transform already registered for {mode}")
        _PROB_TRANSFORMS[mode] = fn
        return fn
    return _decorator

@register_prob(ProbMode.JOINT)
def _joint(counts: np.ndarray) -> np.ndarray:
    """
    Joint probability: P(gram) = count / total_counts.
    Returns log‑space probabilities. If the input has no counts at all, the
    function returns −∞ everywhere so later aggregation logic can treat those
    grams as unseen.
    """
    if counts.sum() == 0:
        # Corpus may be empty in synthetic/unit‑test scenarios; avoid log(0) warnings.
        return np.full_like(counts, fill_value=-np.inf, dtype=float)

    with np.errstate(divide="ignore"):
        return np.log(counts / counts.sum())

@register_prob(ProbMode.CONDITIONAL)
def _conditional(counts: np.ndarray) -> np.ndarray:
    """
    Conditional probability: P(next | prev) = count / sum_over_rows.
    Returns log‑space probabilities, guarding zero‑column sums.
    """
    col_sums = counts.sum(axis=0, keepdims=True)
    # Prevent division by zero for columns with no counts
    col_sums[col_sums == 0] = 1
    with np.errstate(divide="ignore"):
        return np.log(counts / col_sums)

def counts_to_logprobs(counts: np.ndarray, mode: ProbMode) -> np.ndarray:
    """
    Convert raw counts to log‑probabilities according to the selected mode.

    Parameters:
      counts — raw count matrix or array of shape (...).
      mode   — which ProbMode transform to apply.

    Returns:
      A numpy array of the same shape containing log‑probabilities.
    """
    try:
        transformer = _PROB_TRANSFORMS[mode]
    except KeyError:
        raise ValueError(f"No probability transform registered for mode {mode}")
    return transformer(counts)

# Backward‑compatibility alias
to_probs = counts_to_logprobs

# End of src/probability.py
