"""src/variants.py — Enumerate every scoring configuration variant."""

from __future__ import annotations

from itertools import product
from typing import Iterator, NamedTuple

from .config import Config, ProbMode, AggregateMode
from .corpus import Corpus


class Variant(NamedTuple):
    """
    Describes one combination of model name and configuration.

    Attributes:
      header      — CSV column header for this variant
      model_name  — plugin registry key (e.g. 'bigram')
      cfg         — fully-populated Config instance
    """
    header: str
    model_name: str
    cfg: Config


def all_variants(train: Corpus) -> Iterator[Variant]:
    """
    Yield every model/configuration variant for scoring.

    Parameters:
      train — the training corpus (only needed if variants depend on data)

    Yields:
      Variant instances in a deterministic order.
    """
    MODEL_MATRIX = [
        ("unigram",            False, "uni"),
        ("positional_unigram", True,  "uni_pos"),
        ("bigram",             False, "bi"),
        ("positional_bigram",  True,  "bi_pos"),
        ("neighbourhood",      False, "neigh_density"),
    ]

    BOOL = [False, True]
    seen_headers: set[str] = set()

    for plugin, is_pos, prefix in MODEL_MATRIX:
        for smoothing, weighted, boundaries in product(BOOL, BOOL, BOOL):
            for mode in (ProbMode.JOINT, ProbMode.CONDITIONAL):
                # Use only the canonical LOG_PRODUCT aggregator to keep the
                # output CSV manageable (removes logsumexp/rawsum columns)
                for agg in (AggregateMode.LOG_PRODUCT,):
                    # neighbourhood ignores smoothing/weight_by_freq
                    if plugin == "neighbourhood" and (smoothing or weighted):
                        continue

                    cfg = Config.default(
                        positional     = is_pos,
                        smoothing      = smoothing,
                        weight_by_freq = weighted,
                        use_boundaries = boundaries,
                        prob_mode      = mode,
                        aggregate_mode = agg,
                        ngram_order    = 1 if "unigram" in plugin else 2,
                    )

                    # Map SUM → 'rawsum' to avoid header collisions
                    agg_label = "rawsum" if agg is AggregateMode.SUM else agg.value

                    parts = [
                        prefix,
                        "smoothed"       if smoothing  else "prob",
                        "freq_weighted"  if weighted   else "",
                        "bound"          if boundaries else "nobound",
                        mode.value,
                        agg_label,
                    ]
                    header = "_".join(p for p in parts if p)

                    if header in seen_headers:
                        raise AssertionError(f"Duplicate variant header: {header}")
                    seen_headers.add(header)

                    yield Variant(header, plugin, cfg)

# End of src/variants.py