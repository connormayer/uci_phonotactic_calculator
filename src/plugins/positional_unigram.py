"""src/plugins/positional_unigram.py — Positional unigram model plugin: position‑specific probabilities for individual sounds."""

import numpy as np
from collections import defaultdict

from ..plugins import register
from .base import BaseModel
from .mixins import TokenWeightMixin, SmoothingMixin


@register("positional_unigram")
class PositionalUnigramModel(TokenWeightMixin, SmoothingMixin, BaseModel):
    """
    Positional unigram model: P(sound | position) with optional weighting and smoothing.
    """

    def fit(self, corpus):
        """
        Build position‑specific probability tables for each sound.
        """
        cfg = self.cfg
        if cfg.ngram_order != 1:
            raise ValueError("PositionalUnigramModel supports ngram_order == 1 only")

        # 1) Accumulate raw counts per position
        pos_counts: dict[int, dict[str, float]] = defaultdict(lambda: defaultdict(float))
        for token, freq in zip(corpus.tokens, corpus.freqs):
            weight = self._w(freq, cfg)
            # TokenWeightMixin now always returns a float
            for idx, sound in enumerate(token):
                pos_counts[idx][sound] += weight

        # 2) For each position, apply smoothing, normalize, and store linear probabilities
        self._pos_probs: dict[int, dict[str, float]] = {}
        for idx, table in pos_counts.items():
            table = self._smooth(table, corpus.sound_index, cfg)
            total = sum(table.values()) or 1.0
            self._pos_probs[idx] = {
                sound: (count / total)
                for sound, count in table.items()
            }

        # 3) Preserve sound index for reference
        self.sound_index = corpus.sound_index

    def score(self, token: list[str]) -> float:
        """
        Sum the positional probabilities for the token, then add 1.
        """
        comps: list[float] = []
        for idx, sound in enumerate(token):
            prob = self._pos_probs.get(idx, {}).get(sound, self._fallback)
            comps.append(prob)

        return 1.0 + sum(comps)

# End of src/plugins/positional_unigram.py