"""src/variants.py — Enumerate every scoring configuration variant.

Supports three execution modes:
- Single model (via --model)
- Full variant grid (default)
- Filtered grid (via --filter KEY=VAL)
"""

from __future__ import annotations

from typing import Iterator, NamedTuple

from ..plugins.core import PluginRegistry, discover_models
from ..plugins.strategies.position import get_position_strategy
from . import probability as probability
from .config import Config
from .corpus import Corpus
from .registries import registry as _r


def _canonical_agg(name: str) -> str:
    return name


discover_models()  # one-time plugin + transform discovery


class Variant(NamedTuple):
    """
    Describes one combination of model name and configuration.

    Attributes:
      header      — CSV column header for this variant
      model_name  — plugin registry key (e.g. 'ngram')
      cfg         — fully-populated Config instance
      strategy    — Position strategy function if positional mode is used,
                   otherwise None
    """

    header: str
    model_name: str
    cfg: Config
    strategy: str | None


def _make_variant(plugin, order, cfg, strategy=None, _seen_headers=None):
    from .header_utils import build_header

    header = build_header(plugin, cfg)
    import os
    import sys

    if _seen_headers is not None:
        if header in _seen_headers or os.environ.get("DEBUG_VARIANTS"):
            print(f"[DEBUG] Header: {header}", file=sys.stderr)
            print(f"[DEBUG] Plugin: {plugin}", file=sys.stderr)
            print(f"[DEBUG] Order: {order}", file=sys.stderr)
            print(f"[DEBUG] Config: {cfg}", file=sys.stderr)
        if header in _seen_headers:
            raise RuntimeError(
                f"Duplicate header generated: {header!r}. "
                "Please adjust build_header() or variant enumeration."
            )
        _seen_headers.add(header)
    return Variant(header, plugin, cfg, strategy)


# First all_variants function removed to fix redefinition issue


def all_variants(
    train: Corpus, filters: dict[str, str] | None = None
) -> Iterator[Variant]:
    """
    Yield every model/configuration variant for scoring.
    Supports full grid, single model, and filtered grid (via ``--filter KEY=VAL``).

    Parameters
    ----------
    train : Corpus
        The training corpus (only needed if variants depend on data).

    Yields
    ------
    Variant
        Instances in a deterministic order.
    """
    if filters:
        from ..cli.main import _matches_filters
    else:

        def _matches_filters(*_):
            return True

    for count_strategy in _r("count_strategy") or {"ngram": None}:
        for order in (1, 2, 3, 4):
            # ── 1) classic (non-positional) variants ──────────────────
            for boundary_mode in _r("boundary_mode"):
                for smoothing_scheme in _r("smoothing_scheme"):
                    for weight_mode in _r("weight_mode"):
                        for mode in probability.available_transforms():
                            for agg in {
                                _canonical_agg(a) for a in _r("aggregate_mode")
                            }:
                                cfg = Config(
                                    ngram_order=order,
                                    boundary_mode=boundary_mode,
                                    smoothing_scheme=smoothing_scheme,
                                    weight_mode=weight_mode,
                                    prob_mode=mode,
                                    aggregate_mode=agg,
                                    position_strategy=None,
                                    count_strategy=count_strategy,
                                )
                                if not PluginRegistry["ngram"].supports(cfg):
                                    continue
                                if _matches_filters(cfg, filters or {}):
                                    yield _make_variant("ngram", order, cfg)
            # ── 2) positional variants (absolute | relative) ──
            for strategy_name in _r("position_strategy"):
                for boundary_mode in ("none", "both"):
                    for weight_mode in _r("weight_mode"):
                        for mode in probability.available_transforms():
                            for agg in _r("aggregate_mode"):
                                cfg = Config.default(
                                    ngram_order=order,
                                    position_strategy=strategy_name,
                                    boundary_mode=boundary_mode,
                                    weight_mode=weight_mode,
                                    prob_mode=mode,
                                    aggregate_mode=agg,
                                    count_strategy=count_strategy,
                                )
                                strategy = get_position_strategy(strategy_name, n=order)
                                if _matches_filters(cfg, filters or {}):
                                    yield _make_variant("ngram", order, cfg, strategy)


def legacy_variants() -> list[Variant]:
    """
    Return the sixteen Variant objects that exactly reproduce the 2018 output.
    Column order: 4 uni, 4 bi, 4 positional-uni, 4 positional-bi.
    """

    def make(**kw):
        return _make_variant("ngram", kw["ngram_order"], Config.default(**kw), None)

    variants = []

    # 1–4  UNIGRAMS  (no boundaries, joint P)
    for weight, smooth in [
        ("none", False),
        ("legacy_log", False),
        ("none", True),
        ("legacy_log", True),
    ]:
        variants.append(
            make(
                ngram_order=1,
                boundary_mode="none",
                weight_mode=weight,
                smoothing_scheme="laplace" if smooth else "none",
                prob_mode="joint",
                aggregate_mode="prod",
            )
        )

    # 5–8  BIGRAMS  (with boundaries, conditional P)
    for weight, smooth in [
        ("none", False),
        ("legacy_log", False),
        ("none", True),
        ("legacy_log", True),
    ]:
        variants.append(
            make(
                ngram_order=2,
                boundary_mode="both",
                weight_mode=weight,
                smoothing_scheme="laplace" if smooth else "none",
                prob_mode="conditional",
                aggregate_mode="prod",
            )
        )

    # helper for positional strategies
    def pos(n, weight, smooth, prob):
        cfg = Config.default(
            ngram_order=n,
            boundary_mode="none",  # positional sets inherit 'none' (no pads)
            weight_mode=weight,
            smoothing_scheme="laplace" if smooth else "none",
            position_strategy="absolute",
            prob_mode=prob,
            aggregate_mode="sum_plus1",
        )
        return _make_variant("ngram", n, cfg, get_position_strategy("absolute", n=n))

    # 9–12  POSITIONAL UNIGRAMS  (absolute, joint P)
    for weight, smooth in [
        ("none", False),
        ("legacy_log", False),
        ("none", True),
        ("legacy_log", True),
    ]:
        variants.append(pos(1, weight, smooth, "joint"))

    # 13–16  POSITIONAL BIGRAMS  (absolute, joint P)
    for weight, smooth in [
        ("none", False),
        ("legacy_log", False),
        ("none", True),
        ("legacy_log", True),
    ]:
        variants.append(pos(2, weight, smooth, "joint"))

    return variants


__all__ = ["Variant", "all_variants", "legacy_variants"]
# Only use _r for registry lookups in this file.

# End of src/variants.py
