"""
src/plugins/unigram.py
----------------------
Unigram n‑gram model plugin.
"""

import numpy as np

from ..plugins import register
from .base import BaseModel
from .mixins import TokenWeightMixin, SmoothingMixin
from .strategies.ngram import NGramCounter
from ..aggregate import AGGREGATORS

@register("unigram")
class UnigramModel(TokenWeightMixin, SmoothingMixin, BaseModel):
    """
    Unigram model: P(sound) with optional frequency weighting and additive smoothing.
    """

    def fit(self, corpus):
        """
        Build unigram log‑probabilities over the corpus vocabulary.
        """
        cfg = self.cfg
        if cfg.ngram_order != 1:
            raise ValueError("UnigramModel supports ngram_order == 1 only")

        # Count occurrences (with optional log-frequency weighting)
        counter = NGramCounter(1, corpus.sound_index, cfg)
        for token, freq in zip(corpus.tokens, corpus.freqs):
            weight = self._w(freq, cfg)
            counter.accumulate(token, weight)

        # Retrieve raw counts and apply additive smoothing
        counts = counter.finalise()
        counts = self._smooth(counts, corpus.sound_index, cfg)

        # Normalize to log‑probabilities
        total = sum(counts.values())
        self._probs = {
            sound: np.log(count / total)
            for sound, count in counts.items()
        }

        # Preserve sound index for reference
        self.sound_index = corpus.sound_index

    def score(self, token: list[str]) -> float:
        """
        Sum (or log-sum-exp) the log‑probs of each sound in `token`.
        """
        comps: list[float] = []
        for sound in token:
            comps.append(self._probs.get(sound, self._fallback))

        aggregator = AGGREGATORS[self.cfg.aggregate_mode]
        return aggregator(comps)

# End of src/plugins/unigram.py
