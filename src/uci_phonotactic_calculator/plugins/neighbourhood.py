"""
src/plugins/neighbourhood.py — Neighbourhood density model plugin: fast phoneme-level
edit-distance-1 neighbor counting.

Counts words whose phoneme sequence differs by one insertion, deletion or
substitution of a phoneme.

Notes:
- All edit operations are at the phoneme level (not character).
- If your corpus encodes a single phoneme as multiple characters, each
  space-delimited phoneme string is treated as one atomic symbol.
  For correct behavior, ensure each token is a list of phoneme strings.
"""

from functools import lru_cache

from ..cli.utils import slug
from ..core.registries import registry
from .core import BaseModel, register
from .fallback import FallbackMixin


# Separate cached function to avoid memory leaks when used with instance methods
@lru_cache(maxsize=32_000)
def _cached_count_neighbors(
    target: tuple[str, ...],
    mode_name: str,
    alphabet: frozenset[str],  # Updated to frozenset for hashability with lru_cache
    training_set: frozenset[tuple[str, ...]],
) -> float:
    mode_fn = registry("neighbourhood_mode")[mode_name]
    neigh = mode_fn(target, alphabet)
    return float(sum(1 for n in neigh if n in training_set))


@register("neighbourhood")
class NeighbourhoodModel(FallbackMixin, BaseModel):
    """
    Neighbourhood model: for each test token, returns the number of
    training tokens whose Levenshtein distance is exactly 1.
    Uses neighbor-generation for O(n * |alphabet|) performance.

    Modes: 'full', 'substitution_only'.
      - FULL (default): allows substitution, insertion, and deletion of phonemes.
      - SUBSTITUTION_ONLY: allows only substitutions (no insertions or deletions).
    """

    def fit(self, corpus):
        """
        Store the training tokens (flattened) for neighbor comparisons.
        """
        # Store each token as a tuple of phonemes (immutable & hashable)
        self.training_tokens = [
            tuple(tok) for tok in corpus.tokens
        ]  # immutable & hashable
        self.training_set = set(self.training_tokens)  # O(1) look-ups
        self.alphabet: set[str] = {
            sym for tok in corpus.tokens for sym in tok
        }  # distinct phonemes
        # Preserve sound index for unseen-symbol checks
        self.sound_index = corpus.sound_index

    def _count_neighbors(self, target: tuple[str, ...]) -> float:
        # Use a separate cached function to avoid memory leaks with instance methods
        return _cached_count_neighbors(
            target,
            self.cfg.neighbourhood_mode,
            frozenset(
                self.alphabet
            ),  # Convert to frozenset to make it hashable for lru_cache
            frozenset(self.training_set),
        )

    def score(self, token: list[str]) -> float:
        """
        Return the count of training tokens at edit-distance 1.
        If the token contains unseen symbols, return the fallback.
        """
        target = tuple(token)
        if any(s not in self.sound_index for s in token):
            return self._fallback
        return self._count_neighbors(target)

    # ------------------------------------------------------------------
    # CSV header helpers
    # ------------------------------------------------------------------
    @classmethod
    def header(cls, cfg):
        return slug("neighbourhood", cfg.neighbourhood_mode or "full")

    @classmethod
    def supports(cls, cfg):
        # Neighbourhood model accepts any Config
        return True


# End of src/plugins/neighbourhood.py
