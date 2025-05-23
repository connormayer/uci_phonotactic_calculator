"""
Position strategies for n-gram models.
"""

from functools import lru_cache
from typing import Protocol, runtime_checkable

from ...core.registries import register, registry


@runtime_checkable
class PositionStrategy(Protocol):
    def bucket(self, gram_idx: int, token_len: int) -> int | None:
        """Return the bucket id for this gram; None → ignore."""
        ...

    def max_buckets(self, token_len: int) -> int:
        """Return the maximum number of buckets for the given token length."""
        ...


class Absolute(PositionStrategy):
    """
    Absolute position: each n-gram is assigned to its absolute index in the token.
    max_buckets() never returns 0; it returns at least 1 to ensure safe array allocation
    even for empty training data.
    """

    def bucket(self, gram_idx: int, token_len: int) -> int | None:
        return gram_idx

    def max_buckets(self, token_len: int) -> int:
        """
        Return the axis length for dense arrays.
        Always ≥ 1 so zero-length training data won't
        create a 0-sized leading dimension.
        """
        return max(token_len, 1)


class Relative(PositionStrategy):
    """
    Relative position: right-aligns buckets so 0 → final gram, 1 → penultimate, etc.
    bucket() must return an int in [0, max_buckets-1]; None suppresses the gram.
    Requires n-gram order at construction.
    """

    def __init__(self, n: int) -> None:
        if n < 1:
            raise ValueError("n-gram order must be ≥ 1")
        self._n = n

    def bucket(self, gram_idx: int, token_len: int) -> int | None:
        # right-align: final gram → 0; skip if not enough symbols
        if token_len < self._n or gram_idx > token_len - self._n:
            return None
        # Edge case: if token_len == _n, only gram_idx == 0 is valid and returns
        # bucket 0;
        # all other gram_idx values are excluded by the guard above. This is correct.
        # relative position from the rightmost possible n-gram
        return token_len - gram_idx - self._n

    def max_buckets(self, token_len: int) -> int:
        """
        Return the axis length for dense arrays.
        Always ≥ 1 so zero-length training data won't
        create a 0-sized leading dimension.
        """
        return max(token_len - self._n + 1, 1)


class NonePos(PositionStrategy):
    def bucket(self, gram_idx: int, token_len: int) -> int | None:
        return 0

    def max_buckets(self, token_len: int) -> int:
        return 1


@lru_cache(maxsize=None)
def _singleton(cls: type[PositionStrategy]) -> PositionStrategy:
    """Return one shared instance per concrete PositionStrategy subclass."""
    return cls()


# ---- public aliases for CLI use ----

# register concrete strategies
register("position_strategy", "absolute")(Absolute)
register("position_strategy", "relative")(Relative)

absolute = Absolute
relative = Relative


# ──────────────────────────────────────────────────────────────
# Registry wiring  +  public accessor
# ──────────────────────────────────────────────────────────────
def get_position_strategy(name: str | None, n: int = 1):
    """
    Return an instance of the requested PositionStrategy or None.

    Parameters
    ----------
    name : {"absolute", "relative", "none", None}
    n    : n-gram order (required only for "relative")

    Examples
    --------
    >>> get_position_strategy("absolute")
    <src.plugins.strategies.position.Absolute object at 0x...>
    >>> get_position_strategy("relative", n=2)
    <src.plugins.strategies.position.Relative object at 0x...>
    >>> get_position_strategy(None) is None
    True
    """
    if name in (None, "", "none"):
        return None
    cls = registry("position_strategy")[name]
    return cls() if name == "absolute" else cls(n)


__all__ = [
    "PositionStrategy",
    "Absolute",
    "Relative",
    "NonePos",
    "absolute",
    "relative",
    "get_position_strategy",
]
