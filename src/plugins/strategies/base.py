"""
src/plugins/strategies/base.py
-------------------------------
Abstract base class for n‑gram counting strategies.
"""

from abc import ABC, abstractmethod
from typing import Any

class BaseCounter(ABC):
    """
    Interface for a token‑counting strategy.
    """

    @abstractmethod
    def accumulate(self, token: list[str], weight: float | None) -> None:
        """
        Incorporate a single token with its weight into the counts.
        """
        ...

    @abstractmethod
    def finalise(self) -> Any:
        """
        Return the completed counts data structure (e.g., dict or array).
        """
        ...

# End of src/plugins/strategies/base.py
