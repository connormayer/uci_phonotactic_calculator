"""
src/plugins/bigram.py
---------------------
Bigram model plugin: computes log-probabilities for phoneme pairs.
"""

from ..plugins import register
from .base import BaseModel
from .mixins import TokenWeightMixin, SmoothingMixin
from .strategies.ngram import NGramCounter
from ..probability import counts_to_logprobs
from ..aggregate import AGGREGATORS
from ..corpus import Corpus

@register("bigram")
class BigramModel(TokenWeightMixin, SmoothingMixin, BaseModel):
    """
    Bigram model: P(bigram) or P(next|prev) with optional weighting and smoothing.
    Supports optional additive (+1) smoothing when `cfg.smoothing` is enabled.
    """

    def fit(self, corpus):
        """
        Train a bigram log-probability matrix over the given Corpus.
        """
        cfg = self.cfg
        if cfg.ngram_order != 2:
            raise ValueError("BigramModel supports ngram_order == 2 only")

        counter = NGramCounter(2, corpus.sound_index, cfg)
        for token, freq in zip(corpus.tokens, corpus.freqs):
            weight = self._w(freq, cfg)
            counter.accumulate(token, weight)

        counts = counter.finalise()
        # Optional additive (+1) smoothing across the full bigram matrix
        if cfg.smoothing:
            counts = counts + 1.0

        # Convert raw counts into log-probabilities
        self._probs = counts_to_logprobs(counts, cfg.prob_mode)
        # Preserve index mapping for scoring
        self.sound_index = corpus.sound_index

    def score(self, token: list[str]) -> float:
        """
        Compute the aggregate log-probability of all bigrams in `token`.
        """
        comps: list[float] = []
        for prev, nxt in Corpus.generate_bigrams(token, self.cfg):
            try:
                i = self.sound_index.index(nxt)
                j = self.sound_index.index(prev)
                comps.append(self._probs[i, j])
            except ValueError:
                comps.append(self._fallback)

        aggregator = AGGREGATORS[self.cfg.aggregate_mode]
        return aggregator(comps)

# End of src/plugins/bigram.py