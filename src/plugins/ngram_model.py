"""src/plugins/ngram_model.py — Generic n-gram model plugin for arbitrary order (>=1)."""

from .mixins import TokenWeightMixin
from .fallback import FallbackMixin
from ..plugins.core import register, BaseModel
from .strategies.ngram import NGramCounter
from ..aggregate import AGGREGATORS
import numpy as np
from ..plugins.core import get_prob_transform
from ..corpus import Corpus
from .utils import smoothing as sm
from .strategies.position import get_position_strategy, PositionStrategy

@register("ngram")
class NGramModel(TokenWeightMixin, FallbackMixin, BaseModel):
    order_min: int | None = 1
    """
    Generic n-gram model for arbitrary order n.

    N-gram model supporting both positional and non-positional logic.
    """

    def __init__(self, cfg: "Config"):
        super().__init__(cfg)
        self._dense = cfg.ngram_order <= 3  # Prevent AttributeError in _fit_positional
        self.strategy: PositionStrategy | None = (
            get_position_strategy(cfg.position_strategy, cfg.ngram_order)
            if self._use_positional() else None
        )

    def _use_positional(self) -> bool:
        ps = self.cfg.position_strategy
        return bool(ps and str(ps).lower() != "none")

    @staticmethod
    def _weight_token(mode) -> str:
        return mode.value.lower() if mode.name != "NONE" else "unw"

    @classmethod
    def _build_header(cls, cfg):
        parts = ["ngram", f"n{cfg.ngram_order}"]
        if cfg.position_strategy and str(cfg.position_strategy).lower() != "none":  # positional
            parts += [
                "positional",
                cls._weight_token(cfg.weight_mode),
                str(cfg.prob_mode).lower(),
                cfg.aggregate_mode.value.lower(),
                cfg.position_strategy.lower(),
            ]
        else:  # classic
            parts += [
                "smoothed" if cfg.smoothing else "unsmoothed",
                cls._weight_token(cfg.weight_mode),
                "bound" if cfg.use_boundaries else "nobound",
                str(cfg.prob_mode).lower(),
                cfg.aggregate_mode.value.lower(),
            ]
        return "_".join(parts)

    @classmethod
    def header(cls, cfg):
        return cls._build_header(cfg)

    @classmethod
    def supports(cls, cfg):
        if not super().supports(cfg):
            return False
        # Enforce n <= 2 only for positional models
        if cfg.position_strategy and cfg.ngram_order > 2:
            return False
        # extra rule: forbid conditional P beyond trigram
        from ..config import ProbMode
        return not (cfg.ngram_order > 3 and str(cfg.prob_mode) == "conditional")

    def fit(self, corpus):
        if self._use_positional():
            return self._fit_positional(corpus)
        else:
            return self._fit_plain(corpus)

    def _fit_plain(self, corpus):
        cfg = self.cfg
        # Note: Conditional probability always interprets the last axis as the predicted symbol.
        # All n-gram arrays are constructed so that the last axis corresponds to the predicted symbol.
        self.sound_index = corpus.sound_index
        self._index_map = {s: i for i, s in enumerate(self.sound_index)}
        if self._dense:
            N = len(self.sound_index)
            self._counts = np.zeros((N,) * cfg.ngram_order, dtype=float)
        else:
            counter = NGramCounter(cfg.ngram_order, self.sound_index, cfg)
        for token, freq in zip(corpus.tokens, corpus.freqs):
            w = self._w(freq)
            if self._dense:
                grams = Corpus.generate_ngrams(token, cfg.ngram_order,
                                               cfg.use_boundaries,
                                               index_map=self._index_map)
                for idx in grams:
                    if -1 in idx:
                        continue
                    self._counts[idx] += w
            else:
                counter.accumulate(token, w)
        if not self._dense:
            self._counts = counter.finalise()
            if cfg.smoothing:
                self._raw_counts = dict(self._counts)   # snapshot *unsmoothed* counts
            else:
                self._raw_counts = self._counts         # already unsmoothed
            # Apply Laplace (+1) smoothing to the sparse dict, if requested
            if cfg.smoothing:
                import itertools
                sym_vocab  = list(self._index_map.values())            # index IDs 0…N-1
                gram_vocab = itertools.product(sym_vocab, repeat=cfg.ngram_order)
                vocab      = {g for g in gram_vocab}                   # set of all n-grams
                self._counts = sm.apply(cfg, self._counts, vocab)
        if self._dense:
            if not cfg.smoothing:
                self._raw_counts = self._counts.copy()
            smoothed = sm.apply(cfg, self._counts, self.sound_index)
            self._counts = smoothed
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
        max_len = max(map(len, corpus.tokens))
        eff_len = max_len + (cfg.ngram_order - 1)*2 if cfg.use_boundaries else max_len
        buckets = self.strategy.max_buckets(eff_len)
        self.sound_index = corpus.sound_index
        self._index_map = {s: i for i, s in enumerate(self.sound_index)}

        dense_counters = [NGramCounter(cfg.ngram_order, self.sound_index, cfg)
                          for _ in range(buckets)]
        for tok, freq in zip(corpus.tokens, corpus.freqs):
            w = self._w(freq)
            grams = Corpus.generate_ngrams(
                tok, cfg.ngram_order, cfg.use_boundaries, index_map=self._index_map
            )
            for k, idx in enumerate(grams):
                if -1 in idx:
                    continue
                b = self.strategy.bucket(k, eff_len)
                if b is None or b >= buckets:
                    continue
                if hasattr(dense_counters[b], "accumulate_idx"):
                    dense_counters[b].accumulate_idx(idx, w)
                else:
                    dense_counters[b].accumulate(tuple(self.sound_index[i] for i in idx), w)
        self._counts     = np.stack([c.finalise() for c in dense_counters])
        self._raw_counts = self._counts.copy()
        self._logprobs   = np.empty_like(self._counts, dtype=float)
        for b in range(buckets):
            cnts = self._counts[b]
            if cfg.smoothing:
                cnts = sm.apply(cfg, cnts, self.sound_index)
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
                (b, *k): v for b, cnts in enumerate(self._counts)
                          for k, v in cnts.items()
            }
        return


    def score(self, token: list[str]) -> float:
        if self._use_positional():
            return self._score_positional(token)
        return self._score_plain(token)

    def _score_plain(self, token: list[str]) -> float:
        cfg = self.cfg
        grams_idx = Corpus.generate_ngrams(
            token, cfg.ngram_order, cfg.use_boundaries, index_map=self._index_map
        )
        if not grams_idx:
            return self._fallback
        if self._dense:
            comps = [self._logprobs[idx] if -1 not in idx else self._fallback for idx in grams_idx]
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
        return AGGREGATORS[cfg.aggregate_mode](comps)

    def _score_positional(self, token: list[str]) -> float:
        cfg = self.cfg
        grams_idx = Corpus.generate_ngrams(
            token, cfg.ngram_order, cfg.use_boundaries, index_map=self._index_map
        )
        if not grams_idx:
            return self._fallback
        eff_len = len(token) + (cfg.ngram_order - 1)*2 if cfg.use_boundaries else len(token)
        comps: list[float] = []
        for k, idx in enumerate(grams_idx):
            bucket = self.strategy.bucket(k, eff_len)
            # Guard: NumPy treats -1 as last index; must bail for unseen symbols
            if -1 in idx:
                comps.append(self._fallback)
                continue
            if bucket is None:
                continue
            if self._dense:
                if bucket < 0 or bucket >= self._logprobs.shape[0]:
                    lp = self._fallback
                else:
                    lp = self._logprobs[(bucket, *idx)]
            else:
                lp = self._logprobs.get((bucket,) + idx, self._fallback)
            comps.append(lp)
        if not comps or all(c == self._fallback for c in comps):
            return self._fallback
        return AGGREGATORS[cfg.aggregate_mode](comps)
