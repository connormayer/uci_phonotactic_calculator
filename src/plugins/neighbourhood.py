"""
src/plugins/neighbourhood.py — Neighbourhood density model plugin: fast phoneme-level edit-distance-1 neighbor counting.

Counts words whose phoneme sequence differs by one insertion, deletion or substitution of a phoneme.

Notes:
- All edit operations are at the phoneme level (not character).
- If your corpus encodes a single phoneme as multiple characters, each space-delimited phoneme string is treated as one atomic symbol.
  For correct behavior, ensure each token is a list of phoneme strings.
"""

from ..cli_utils import slug

from ..plugins.core import register, BaseModel
from ..registries import registry
from .fallback import FallbackMixin
from functools import lru_cache

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
        self.training_tokens = [tuple(tok) for tok in corpus.tokens]        # immutable & hashable
        self.training_set    = set(self.training_tokens)                    # O(1) look-ups
        self.alphabet: set[str] = {sym for tok in corpus.tokens for sym in tok}  # distinct phonemes
        # Preserve sound index for unseen-symbol checks
        self.sound_index = corpus.sound_index


    @lru_cache(maxsize=32_000)
    def _count_neighbors(self, target: tuple[str, ...]) -> float:
        mode_fn = registry('neighbourhood_mode')[self.cfg.neighbourhood_mode]
        neigh = mode_fn(target, self.alphabet)
        return float(sum(1 for n in neigh if n in self.training_set))

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
        from ..header_utils import build_header
        return build_header("neighbourhood", cfg)

    @classmethod
    def supports(cls, cfg):
        # Neighbourhood model is not n-gram specific, accept any config
        return True

# End of src/plugins/neighbourhood.py
