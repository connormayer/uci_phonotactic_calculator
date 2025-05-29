"""
src/plugins/strategies/ngram.py — Generic n‑gram counting strategy for any n-gram order.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, MutableMapping, Optional, Sequence, Union

import numpy as np
from numpy.typing import NDArray

from ...core.config import Config
from ...core.corpus import Corpus
from ...core.registries import register
from ...utils.types import IndexTuple
from .base import BaseCounter

# Type for our use case: mapping from index tuples to float values
# This represents the counts of each n-gram index
IndexCountDict = MutableMapping[IndexTuple, float]


@register("count_strategy", "ngram")
class NGramCounter(BaseCounter):
    """
    Count n‑grams of any order over a corpus, with optional token weighting
    and boundary symbols controlled by a Config.
    """

    def __init__(self, order: int, sound_index: list[str], cfg: Config):
        """
        Initialize the counter for any n-gram order.
        Dense (np.ndarray) for order <= 3, sparse (defaultdict) for order > 3.
        """
        self.order = order
        self.cfg = cfg
        self.sound_index = sound_index
        self._dense = order <= 3
        N = len(sound_index)
        self._sym2idx = {s: i for i, s in enumerate(sound_index)}
        if self._dense:
            self.counts: NDArray[np.float64] = np.zeros((N,) * order, dtype=float)
        else:
            # mutable mapping from index-tuples to float counts
            self.counts_dict: IndexCountDict = defaultdict(float)
            # Ensure the type checker knows which attribute we're using
            self._counts_attr = "counts" if self._dense else "counts_dict"

    def accumulate_idx(
        self, idx: IndexTuple, weight: float, boundary: str = "#"
    ) -> None:
        """
        Fast-path: increment a pre-indexed n-gram (expects an index tuple of
        length order).
        The boundary kwarg is accepted for ABC compatibility but is unused here.

        We must not treat falsy numeric weights (0.0, -inf) as 'skip': legacy_log may
        emit -inf and callers may pass 0.0 or negatives.
        Only None means 'skip'.
        """
        if weight is None:  # Only skip if weight is None (not 0.0, -inf, etc.)
            return
        if self._dense:
            self.counts[idx] += weight
        else:
            self.counts_dict[idx] = self.counts_dict[idx] + weight

    def accumulate(
        self, token: Sequence[str], weight: Optional[float], **kwargs: Any
    ) -> None:
        # Extract boundary from kwargs or use default
        boundary = kwargs.get("boundary", "#")
        """
        Add counts for a single token sequence using Corpus.generate_ngrams.

        We must not treat falsy numeric weights (0.0, -inf) as 'skip': legacy_log may
        emit -inf and callers may pass 0.0 or negatives.
        Only None means 'skip'.
        """
        if weight is None:  # Only skip if weight is None (not 0.0, -inf, etc.)
            return
        grams = Corpus.generate_ngrams(
            token,
            self.order,
            self.cfg.boundary_mode,
            index_map=self._sym2idx,
            boundary=boundary,
        )
        for idx in grams:
            if self._dense and -1 in idx:
                continue
            self.accumulate_idx(idx, weight)

    def finalise(self) -> Union[NDArray[np.float64], IndexCountDict]:
        """
        Return the raw counts structure (np.ndarray or CountDict).
        """
        if self._dense:
            return self.counts
        else:
            return self.counts_dict


# End of src/plugins/strategies/ngram.py
