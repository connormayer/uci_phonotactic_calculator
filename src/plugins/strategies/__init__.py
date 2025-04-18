"""
src/plugins/strategies/__init__.py
----------------------------------
Registry of available counting strategies.
"""

from typing import Dict, Type
from .base import BaseCounter
from .ngram import NGramCounter

StrategyRegistry: Dict[str, Type[BaseCounter]] = {
    "ngram": NGramCounter,
}

# End of src/plugins/strategies/__init__.py
