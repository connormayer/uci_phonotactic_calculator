import numpy as np

from ...plugins.core import BaseTransform, register_prob


@register_prob("joint")
class JointTransform(BaseTransform):
    """Joint probability: P(gram) = count / total."""

    def transform(self, counts: np.ndarray) -> np.ndarray:
        if counts.sum() == 0:
            # Corpus may be empty in synthetic/unit‑test scenarios;
            # avoid log(0) warnings.
            return np.full_like(counts, fill_value=-np.inf, dtype=float)

        with np.errstate(divide="ignore"):
            return np.log(counts / counts.sum())
