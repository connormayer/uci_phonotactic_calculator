"""src/plugins/neighbourhood.py — Neighbourhood density model plugin: fast edit‑distance‑1 neighbor counting."""

from typing import List, Set
from ..types import Symbol
from ..plugins.core import register, BaseModel
from .fallback import FallbackMixin
from functools import lru_cache

@register("neighbourhood")
class NeighbourhoodModel(FallbackMixin, BaseModel):
    """
    Neighbourhood model: for each test token, returns the number of
    training tokens whose Levenshtein distance is exactly 1.
    Uses neighbor‑generation for O(n * |alphabet|) performance.
    """

    def fit(self, corpus):
        """
        Store the training tokens (flattened) for neighbor comparisons.
        """
        # Flatten each token list into a string
        self.training_tokens: List[str] = ["".join(tok) for tok in corpus.tokens]
        # Set for O(1) lookups
        self.training_set: Set[str] = set(self.training_tokens)
        # Alphabet of characters appearing in training strings
        self.alphabet: Set[Symbol] = set("".join(self.training_tokens)) if self.training_tokens else set()
        # Preserve sound index for unseen-symbol checks
        self.sound_index = corpus.sound_index

    @lru_cache(maxsize=32_000)
    def _count_neighbors(self, target: str) -> float:
        """Compute count of edit-distance-1 neighbours for *target*."""
        neighbors = set()
        # deletions
        for i in range(len(target)):
            neighbors.add(target[:i] + target[i+1:])
        # substitutions
        for i, ch in enumerate(target):
            neighbors.update(
                target[:i] + c + target[i+1:]
                for c in self.alphabet if c != ch
            )
        # insertions
        for i in range(len(target) + 1):
            neighbors.update(
                target[:i] + c + target[i:]
                for c in self.alphabet
            )
        return float(sum(1 for n in neighbors if n in self.training_set))

    def score(self, token: list[str]) -> float:
        """
        Return the count of training tokens at edit-distance ==1 from test token.
        If the test token contains any unseen symbol, returns fallback.
        """
        target = "".join(token)
        # Check for unseen symbols
        for s in token:
            if s not in self.sound_index:
                return self._fallback
        return self._count_neighbors(target)

    @classmethod
    def supports(cls, cfg):
        # Neighbourhood model is not n-gram specific, accept any config
        return True

# End of src/plugins/neighbourhood.py
