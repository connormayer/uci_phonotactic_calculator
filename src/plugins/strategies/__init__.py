"""
src/plugins/strategies/__init__.py
----------------------------------
Registry of available counting strategies.
"""

from .base import BaseCounter
from .ngram import NGramCounter
from .position import absolute, relative
from .position import get_position_strategy  # new

__all__ = [
    "BaseCounter", "NGramCounter",
    "Absolute", "Relative", "NonePos",
    "absolute", "relative",
]
__all__.append('get_position_strategy')

# End of src/plugins/strategies/__init__.py
