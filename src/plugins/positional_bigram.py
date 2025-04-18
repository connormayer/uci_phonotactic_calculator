"""src/plugins/positional_bigram.py — Positional bigram model plugin: position‑specific probabilities for phoneme pairs."""

import numpy as np
from collections import defaultdict

from ..plugins import register
from .base import BaseModel
from .mixins import TokenWeightMixin, SmoothingMixin


@register("positional_bigram")
class PositionalBigramModel(TokenWeightMixin, SmoothingMixin, BaseModel):
    """
    Positional bigram model: P(bigram | position) with optional weighting and smoothing.
    """

    def fit(self, corpus):
        """
        Train position‑specific bigram probabilities over the given Corpus.
        """
        cfg = self.cfg
        if cfg.ngram_order != 2:
            raise ValueError("PositionalBigramModel supports ngram_order == 2 only")

        # 1) Accumulate raw counts per position‑pair
        pos_counts: dict[tuple[int,int], dict[tuple[str,str], float]] = defaultdict(lambda: defaultdict(float))
        for token, freq in zip(corpus.tokens, corpus.freqs):
            weight = self._w(freq, cfg)
            for i in range(len(token) - 1):
                bigram = (token[i], token[i + 1])
                pos_counts[(i, i + 1)][bigram] += weight

        # 2) Prepare full bigram vocabulary for smoothing
        vocab = corpus.sound_index
        bigram_vocab = [(s1, s2) for s1 in vocab for s2 in vocab]

        # 3) Smooth, normalize, and store linear probabilities
        self._pos_probs: dict[tuple[int,int], dict[tuple[str,str], float]] = {}
        for pos_key, table in pos_counts.items():
            table = self._smooth(table, bigram_vocab, cfg)
            total = sum(table.values()) or 1.0
            self._pos_probs[pos_key] = {
                bg: (count / total)
                for bg, count in table.items()
            }

        # 4) Preserve sound index for scoring
        self.sound_index = corpus.sound_index

    def score(self, token: list[str]) -> float:
        """
        Sum the positional bigram probabilities for the token, then add 1.
        """
        comps: list[float] = []
        for i in range(len(token) - 1):
            prob = self._pos_probs.get((i, i + 1), {}).get((token[i], token[i + 1]), self._fallback)
            comps.append(prob)

        return 1.0 + sum(comps)

# End of src/plugins/positional_bigram.py