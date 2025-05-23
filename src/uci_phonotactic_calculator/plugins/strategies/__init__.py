"""
src/plugins/strategies/__init__.py
----------------------------------
Registry of available counting strategies.
"""

from .base import BaseCounter
from .ngram import NGramCounter
from .position import (
    absolute,
    get_position_strategy,  # new
    relative,
)

__all__ = [
    "BaseCounter",
    "NGramCounter",
    "Absolute",
    "Relative",
    "NonePos",
    "absolute",
    "relative",
    "get_position_strategy",
]

# End of src/plugins/strategies/__init__.py
