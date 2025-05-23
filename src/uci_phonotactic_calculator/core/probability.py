"""probability.py — Log‑space probability transforms for count matrices."""

from functools import lru_cache

from ..plugins.core import ProbTransformRegistry, discover_models


@lru_cache(maxsize=None)
def available_transforms() -> tuple[str, ...]:
    """Return a sorted immutable list of registered transform names."""
    if not ProbTransformRegistry:
        discover_models()
    return tuple(sorted(ProbTransformRegistry))


# End of src/probability.py
