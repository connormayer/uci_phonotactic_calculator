"""probability.py — Log‑space probability transforms for count matrices."""

from collections.abc import Callable
import numpy as np
from .plugins.core import get_prob_transform, ProbTransformRegistry, discover_models
from functools import lru_cache




@lru_cache(maxsize=None)
def available_transforms() -> tuple[str, ...]:
    """Return a sorted immutable list of registered transform names."""
    if not ProbTransformRegistry:
        discover_models()
    return tuple(sorted(ProbTransformRegistry))

# End of src/probability.py
