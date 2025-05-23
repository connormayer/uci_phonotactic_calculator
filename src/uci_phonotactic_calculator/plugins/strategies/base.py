"""
src/plugins/strategies/base.py
-------------------------------
Abstract base class for n‑gram counting strategies.
"""

from abc import ABC, abstractmethod
from typing import Any, Optional, Sequence, Tuple


class BaseCounter(ABC):
    """
    Abstract base class for n-gram counters.
    Subclasses must implement accumulate() and finalise().
    Optionally, subclasses may implement accumulate_idx(idx, weight) for fast-path
    pre-indexed n-gram accumulation. The default raises NotImplementedError and
    allows feature-testing via hasattr(counter, "accumulate_idx").
    """

    @abstractmethod
    def accumulate(
        self, token: Sequence[str], weight: Optional[float], **kwargs
    ) -> None:
        """
        Incorporate a single token with its weight into the counts.
        Accepts arbitrary keyword arguments for compatibility (e.g., boundary).
        """
        ...

    @abstractmethod
    def finalise(self) -> Any:
        """
        Return the completed counts data structure (e.g., dict or array).
        """
        ...

    def accumulate_idx(
        self, idx: Tuple[int, ...], weight: float, boundary: str = "#"
    ) -> None:  # pragma: no cover
        """
        Optional fast-path: increment a *pre-indexed* n-gram, with optional boundary
        symbol for consistency.
        Implementers MAY override this for speed; the default
        raises NotImplementedError so callers can feature-test via
        hasattr(counter, "accumulate_idx").
        """
        raise NotImplementedError


# End of src/plugins/strategies/base.py
