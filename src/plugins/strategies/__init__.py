"""
src/plugins/strategies/__init__.py
----------------------------------
Registry of available counting strategies.
"""

from ..core import StrategyRegistry, register_strategy  # re-export
from .base import BaseCounter
from .ngram import NGramCounter
from .position import (
    absolute, relative, get_position_strategy,
    _POSITION_REGISTRY as PositionRegistry,
)

__all__ = [
    "StrategyRegistry", "register_strategy",
    "BaseCounter", "NGramCounter",
    "Absolute", "Relative", "NonePos",
    "absolute", "relative",
    "get_position_strategy", "PositionRegistry",
]

# End of src/plugins/strategies/__init__.py
