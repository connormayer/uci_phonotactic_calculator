from typing import cast

import numpy as np
from numpy.typing import NDArray

from ...plugins.core import BaseTransform, register_prob


@register_prob("joint")
class JointTransform(BaseTransform):
    """Joint probability: P(gram) = count / total."""

    def transform(self, counts: NDArray[np.float64]) -> NDArray[np.float64]:
        if counts.sum() == 0:
            # Corpus may be empty in synthetic/unit‑test scenarios;
            # avoid log(0) warnings.
            return cast(
                NDArray[np.float64],
                np.full_like(counts, fill_value=-np.inf, dtype=float),
            )

        with np.errstate(divide="ignore"):
            # Explicitly cast the result to avoid Any return
            return cast(NDArray[np.float64], np.log(counts / counts.sum()))
