"""
src/plugins/neighbourhood.py — Neighbourhood density model plugin: fast phoneme-level edit-distance-1 neighbor counting.

Counts words whose phoneme sequence differs by one insertion, deletion or substitution of a phoneme.

Notes:
- All edit operations are at the phoneme level (not character).
- If your corpus encodes a single phoneme as multiple characters, each space-delimited phoneme string is treated as one atomic symbol.
  For correct behavior, ensure each token is a list of phoneme strings.
"""

from typing import List, Set
from ..types import Symbol
from ..config import NeighbourhoodMode
from ..plugins.core import register, BaseModel
from .fallback import FallbackMixin
from functools import lru_cache

@register("neighbourhood")
class NeighbourhoodModel(FallbackMixin, BaseModel):
    """
    Neighbourhood model: for each test token, returns the number of
    training tokens whose Levenshtein distance is exactly 1.
    Uses neighbor-generation for O(n * |alphabet|) performance.
    
    Modes (see NeighbourhoodMode):
      - FULL (default): allows substitution, insertion, and deletion of phonemes.
      - SUBSTITUTION_ONLY: allows only substitutions (no insertions or deletions).
    """

    def fit(self, corpus):
        """
        Store the training tokens (flattened) for neighbor comparisons.
        """
        # Store each token as a tuple of phonemes (immutable & hashable)
        self.training_tokens = [tuple(tok) for tok in corpus.tokens]        # immutable & hashable
        self.training_set    = set(self.training_tokens)                    # O(1) look-ups
        self.alphabet: set[str] = {sym for tok in corpus.tokens for sym in tok}  # distinct phonemes
        # Preserve sound index for unseen-symbol checks
        self.sound_index = corpus.sound_index

    def _generate_neighbors(self, target: tuple[str, ...]) -> set[tuple[str, ...]]:
        mode = self.cfg.neighbourhood_mode
        neighbors: set[tuple[str, ...]] = set()

        if mode is NeighbourhoodMode.FULL or mode is NeighbourhoodMode.SUBSTITUTION_ONLY:
            # substitution (always allowed)
            for i, ph in enumerate(target):
                neighbors.update(
                    target[:i] + (new_ph,) + target[i+1:]
                    for new_ph in self.alphabet if new_ph != ph
                )

        if mode is NeighbourhoodMode.FULL:
            # deletion
            for i in range(len(target)):
                neighbors.add(target[:i] + target[i+1:])
            # insertion
            for i in range(len(target) + 1):
                neighbors.update(
                    target[:i] + (new_ph,) + target[i:]
                    for new_ph in self.alphabet
                )
        return neighbors

    @lru_cache(maxsize=32_000)
    def _count_neighbors(self, target: tuple[str, ...]) -> float:
        """Compute count of edit-distance-1 neighbours for *target* (as tuple)."""
        return float(sum(1 for n in self._generate_neighbors(target) if n in self.training_set))

    def score(self, token: list[str]) -> float:
        """
        Return the count of training tokens at edit-distance ==1 from test token.
        If the test token contains any unseen symbol, returns fallback.
        """
        target = tuple(token)      # phoneme-level key
        # Check for unseen symbols
        for s in token:
            if s not in self.sound_index:
                return self._fallback
        return self._count_neighbors(target)

    @classmethod
    def header(cls, cfg):
        mode = cfg.neighbourhood_mode.value.lower()
        return f"neighbourhood_{mode}"

    @classmethod
    def supports(cls, cfg):
        # Neighbourhood model is not n-gram specific, accept any config
        return True

# End of src/plugins/neighbourhood.py
