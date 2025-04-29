# src/weighting.py
from .registries import register
import math

@register('weight_mode', 'none')
def w_none(freq: float) -> float:
    """Returns 1.0 (unweighted)"""
    return 1.0

@register('weight_mode', 'raw')
def w_raw(freq: float) -> float:
    """Returns frequency (0 becomes 1.0)"""
    return freq or 1.0  # zero-freq → 1.0

@register('weight_mode', 'log')
def w_log(freq: float) -> float:
    """Returns log(freq+1)"""
    return math.log(freq+1)

import math

@register('weight_mode', 'legacy_log')
def w_legacy_log(freq: float) -> float:
    """
    2018 behaviour:
        • log(freq)          when freq > 0   (freq may be < 1 → negative)
        • -inf               when freq == 0  (adds –∞ to the counter)
    """
    return float("-inf") if freq == 0 else math.log(freq)
