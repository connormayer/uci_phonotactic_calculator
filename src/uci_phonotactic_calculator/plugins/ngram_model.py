"""src/plugins/ngram_model.py — Generic n-gram model plugin for arbitrary order
(>=1)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from ..core.corpus import Corpus
from ..core.registries import registry
from .core import get_prob_transform
from .strategies.position import get_position_strategy

if TYPE_CHECKING:
    from ..core.config import Config

from ..core.header_utils import build_header
from .core import BaseModel, register
from .fallback import FallbackMixin
from .mixins import TokenWeightMixin
from .strategies.position import PositionStrategy


@register("ngram")
class NGramModel(TokenWeightMixin, FallbackMixin, BaseModel):
    order_min: int | None = 1
    """
    Generic n-gram model for arbitrary order n.

    N-gram model supporting both positional and non-positional logic.
    """

    def __init__(self, cfg: "Config"):
        super().__init__(cfg)

        self.strategy: PositionStrategy | None = (
            get_position_strategy(cfg.position_strategy, cfg.ngram_order)
            if self._use_positional()
            else None
        )
        self._boundary = registry("boundary_scheme")[cfg.boundary_scheme]()
        self._counter_cls = registry("count_strategy")[cfg.count_strategy]
        self._is_dense = self.cfg.ngram_order <= 3

    @property
    def dense(self):
        return self._is_dense

    def _use_positional(self) -> bool:
        ps = self.cfg.position_strategy
        return bool(ps and str(ps).lower() != "none")

    @classmethod
    def header(cls, cfg):
        # build_header() already prefixes the n-gram order token (n1, n2 …),
        # so the extra include_order flag is obsolete.
        return build_header("ngram", cfg)

    @classmethod
    def supports(cls, cfg):
        if not super().supports(cfg):
            return False
        # Enforce n <= 2 only for positional models
        if cfg.position_strategy and cfg.ngram_order > 2:
            return False
        # extra rule: forbid conditional P beyond trigram

        return not (cfg.ngram_order > 3 and str(cfg.prob_mode) == "conditional")

    def fit(self, corpus):
        if self._use_positional():
            return self._fit_positional(corpus)
        else:
            return self._fit_plain(corpus)

    def _fit_plain(self, corpus):
        cfg = self.cfg
        self.sound_index = corpus.sound_index
        self._index_map = {s: i for i, s in enumerate(self.sound_index)}
        counter = self._counter_cls(cfg.ngram_order, self.sound_index, cfg)
        # Note: Counter now accumulates any float, including -inf, per new counter
        # semantics (see legacy_log).
        for token, freq in zip(corpus.tokens, corpus.freqs):
            w = self._w(freq)
            counter.accumulate(token, w, boundary=self._boundary)
        self._counts = counter.finalise()
        if cfg.smoothing_scheme != "none":
            self._raw_counts = self._counts.copy()
            self._counts = registry("smoothing_scheme")[cfg.smoothing_scheme](
                self._counts, vocab=self.sound_index
            )
        else:
            self._raw_counts = self._counts
        if self.dense:
            smoothed = self._counts
            tf = get_prob_transform(cfg.prob_mode)
            self._logprobs = tf.transform(smoothed)
        else:
            self._total = sum(self._counts.values()) or 1.0

    def _fit_positional(self, corpus):
        cfg = self.cfg
        if not corpus.tokens:
            self._logprobs = np.array([self._fallback])  # shape (1,)
            self.sound_index = corpus.sound_index
            self._index_map = {}
            return
        max_len_train = max(map(len, corpus.tokens))
        target_len = max_len_train
        left = cfg.boundary_mode in ("both", "prefix")
        right = cfg.boundary_mode in ("both", "suffix")
        pad_total = (left + right) * (cfg.ngram_order - 1)
        eff_len = target_len + pad_total
        buckets = self.strategy.max_buckets(eff_len)
        self.sound_index = corpus.sound_index
        self._index_map = {s: i for i, s in enumerate(self.sound_index)}

        dense_counters = [
            self._counter_cls(cfg.ngram_order, self.sound_index, cfg)
            for _ in range(buckets)
        ]
        for tok, freq in zip(corpus.tokens, corpus.freqs):
            w = self._w(freq)
            grams = Corpus.generate_ngrams(
                tok,
                cfg.ngram_order,
                cfg.boundary_mode,
                index_map=self._index_map,
                boundary=self._boundary,
            )
            for k, idx in enumerate(grams):
                if -1 in idx:
                    continue
                b = self.strategy.bucket(k, eff_len)
                if b is None or b >= buckets:
                    continue
                if hasattr(dense_counters[b], "accumulate_idx"):
                    dense_counters[b].accumulate_idx(idx, w, boundary=self._boundary)
                else:
                    dense_counters[b].accumulate(
                        tuple(self.sound_index[i] for i in idx),
                        w,
                        boundary=self._boundary,
                    )
        self._counts = np.stack([c.finalise() for c in dense_counters])
        self._raw_counts = self._counts.copy()
        self._logprobs = np.empty_like(self._counts, dtype=float)
        for b in range(buckets):
            cnts = self._counts[b]
            if cfg.smoothing_scheme != "none":
                cnts = registry("smoothing_scheme")[cfg.smoothing_scheme](
                    cnts, vocab=self.sound_index
                )
                self._counts[b] = cnts
            if cnts.sum() == 0:
                self._logprobs[b].fill(self._fallback)
                continue
            tf = get_prob_transform(cfg.prob_mode)
            self._logprobs[b] = tf.transform(cnts)
        # Unify logprob structure for sparse positional
        if not hasattr(self, "_logprobs") or not isinstance(self._logprobs, np.ndarray):
            # sparse positional – build a dict keyed by (bucket, *idx)
            self._logprobs = {
                (b, *k): v
                for b, cnts in enumerate(self._counts)
                for k, v in cnts.items()
            }
        return

    def score(self, token: list[str]) -> float:
        if self._use_positional():
            return self._score_positional(token)
        return self._score_plain(token)

    def _score_plain(self, token: list[str]) -> float:
        cfg = self.cfg
        aggregate_fn = registry("aggregate_mode")[cfg.aggregate_mode]
        grams_idx = Corpus.generate_ngrams(
            token,
            cfg.ngram_order,
            cfg.boundary_mode,
            index_map=self._index_map,
            boundary=self._boundary,
        )
        if not grams_idx:
            return self._fallback
        if self.dense:
            comps = [
                self._logprobs[idx] if -1 not in idx else self._fallback
                for idx in grams_idx
            ]
        else:
            if self._total == 0:
                return self._fallback
            comps: list[float] = []
            for idx in grams_idx:
                if -1 in idx:
                    comps.append(self._fallback)
                    continue
                count = self._counts.get(idx, 0.0)
                if count == 0.0:
                    comps.append(self._fallback)
                    continue
                prob = count / self._total
                log_p = np.log(prob) if prob > 0 else self._fallback
                comps.append(log_p)
        if all(c == self._fallback for c in comps):
            return self._fallback
        # Only aggregate if not all fallback
        return aggregate_fn(comps)

    def _score_positional(self, token: list[str]) -> float:
        cfg = self.cfg
        aggregate_fn = registry("aggregate_mode")[cfg.aggregate_mode]
        grams_idx = Corpus.generate_ngrams(
            token,
            cfg.ngram_order,
            cfg.boundary_mode,
            index_map=self._index_map,
            boundary=self._boundary,
        )
        if not grams_idx:
            return self._fallback
        max_len_test = len(token)
        target_len = max_len_test
        left = cfg.boundary_mode in ("both", "prefix")
        right = cfg.boundary_mode in ("both", "suffix")
        pad_total = (left + right) * (cfg.ngram_order - 1)
        eff_len = target_len + pad_total
        comps: list[float] = []
        for k, idx in enumerate(grams_idx):
            bucket = self.strategy.bucket(k, eff_len)
            # Guard: NumPy treats -1 as last index; must bail for unseen symbols
            if -1 in idx:
                comps.append(self._fallback)
                continue
            if bucket is None:
                continue
            if self.dense:
                if bucket < 0 or bucket >= self._logprobs.shape[0]:
                    lp = self._fallback
                else:
                    lp = self._logprobs[(bucket, *idx)]
            else:
                lp = self._logprobs.get((bucket,) + idx, self._fallback)
            comps.append(lp)
        if not comps or all(c == self._fallback for c in comps):
            return self._fallback
        return aggregate_fn(comps)
