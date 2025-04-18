"""
src/plugins/strategies/ngram.py
--------------------------------
Generic n‑gram counting strategy for unigrams and bigrams.
"""

from __future__ import annotations

import numpy as np
from collections import defaultdict
from typing import Any, Sequence, Optional

from .base import BaseCounter
from ...corpus import Corpus
from ...config import Config


class NGramCounter(BaseCounter):
    """
    Count n‑grams of order 1 or 2 over a corpus, with optional token weighting
    and boundary symbols controlled by a Config.
    """

    def __init__(self, order: int, sound_index: list[str], cfg: Config):
        """
        Initialize the counter.

        :param order:      The n‑gram order to count (1 or 2 supported).
        :param sound_index:List of all symbols (vocabulary) used for indexing.
        :param cfg:        Configuration flags for boundary insertion,
                           smoothing, and frequency weighting.
        :raises ValueError: If order is not 1 or 2.
        """
        self.order = order
        self.cfg = cfg
        self.sound_index = sound_index

        if self.order == 1:
            # Unigram counts: symbol → total weight
            self.counts: defaultdict[str, float] = defaultdict(float)
        elif self.order == 2:
            # Bigram counts: next_symbol × prev_symbol matrix
            N = len(sound_index)
            self.counts: np.ndarray = np.zeros((N, N))
        else:
            raise ValueError(f"NGramCounter supports only order 1 or 2, got {order!r}")

    def accumulate(self, token: Sequence[str], weight: Optional[float]) -> None:
        """
        Add counts for a single token sequence.

        :param token: List of symbols for one token.
        :param weight: Numeric weight (log‑freq) or None to skip.
        """
        if weight is None:
            return

        if self.order == 1:
            for s in token:
                self.counts[s] += weight

        else:  # order == 2
            for prev_sym, next_sym in Corpus.generate_bigrams(token, self.cfg):
                i = self.sound_index.index(next_sym)
                j = self.sound_index.index(prev_sym)
                self.counts[i, j] += weight

    def finalise(self) -> Any:
        """
        Return the raw counts structure.

        :return: A defaultdict (for unigrams) or ndarray (for bigrams).
        """
        return self.counts

# End of src/plugins/strategies/ngram.py