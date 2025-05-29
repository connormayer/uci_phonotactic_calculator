from typing import cast

import numpy as np
from numpy.typing import NDArray

from ...plugins.core import BaseTransform, register_prob


@register_prob("conditional")
class ConditionalTransform(BaseTransform):
    """
    Conditional probability: P(next | prev).
    Normalises over the last axis (the predicted symbol).
    """

    def transform(self, counts: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Conditional probability: P(next | prev) = count / sum_over_prediction_axis.
        For n-gram arrays, normalizes over the last axis (axis -1, the predicted symbol)
        so that probabilities sum to 1 for each context (preceding n-1 symbols).
        Returns log‑space probabilities, guarding zero-context sums.
        """
        pred_sum = counts.sum(axis=-1, keepdims=True)
        pred_sum[pred_sum == 0] = 1
        with np.errstate(divide="ignore"):
            # Explicitly cast the result to avoid Any return
            return cast(NDArray[np.float64], np.log(counts / pred_sum))
