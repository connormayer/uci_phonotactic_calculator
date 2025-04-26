import numpy as np
from ...plugins.core import register_prob, BaseTransform
import warnings

@register_prob("joint")
class JointTransform(BaseTransform):
    """Joint probability: P(gram) = count / total."""
    def transform(self, counts: np.ndarray) -> np.ndarray:
        if counts.sum() == 0:
            # Corpus may be empty in synthetic/unit‑test scenarios; avoid log(0) warnings.
            return np.full_like(counts, fill_value=-np.inf, dtype=float)

        with np.errstate(divide="ignore"):
            return np.log(counts / counts.sum())

def joint(counts: np.ndarray) -> np.ndarray:
    """
    Joint probability: P(gram) = count / total_counts.
    Returns log‑space probabilities. If the input has no counts at all, the
    function returns −∞ everywhere so later aggregation logic can treat those
    grams as unseen.
    """
    warnings.warn(
        "joint() is deprecated; use JointTransform().transform() instead.",
        DeprecationWarning, stacklevel=2
    )
    return JointTransform().transform(counts)

    with np.errstate(divide="ignore"):
        return np.log(counts / counts.sum())
