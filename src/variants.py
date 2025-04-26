"""src/variants.py — Enumerate every scoring configuration variant.

Supports three execution modes:
- Single model (via --model)
- Full variant grid (default)
- Filtered grid (via --filter KEY=VAL)
"""

from __future__ import annotations

from typing import Iterator, NamedTuple

from .plugins import PluginRegistry
from .plugins.core import get_model
from .config import Config, AggregateMode, WeightMode
from . import probability as probability
from .corpus import Corpus
from .plugins.core import discover_models
from .plugins.strategies.position import get_position_strategy

discover_models()          # one-time plugin + transform discovery


class Variant(NamedTuple):
    """
    Describes one combination of model name and configuration.

    Attributes:
      header      — CSV column header for this variant
      model_name  — plugin registry key (e.g. 'ngram')
      cfg         — fully-populated Config instance
      strategy    — get_position_strategy(strategy_name, n=cfg_variant.ngram_order) if positional, else None
    """
    header: str
    model_name: str
    cfg: Config
    strategy: str | None


def _skip(plugin, weight_mode, agg=None) -> bool:
    """
    Return True if this (plugin, weight_mode, agg) combination should be
    suppressed.  ‘agg’ is accepted for forward-compat even though the
    current rule ignores it.
    """
    return plugin == "neighbourhood" and weight_mode is not WeightMode.NONE


def _make_variant(plugin, order, cfg, strategy=None):
    header = get_model(plugin).header(cfg)
    return Variant(header, plugin, cfg, strategy)


def all_variants(train: Corpus, filters: dict[str, str] | None = None) -> Iterator[Variant]:
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
        from .main import _matches_filters
    else:
        _matches_filters = lambda *_: True
    ORDERS: dict[str, tuple[int, ...]] = {"ngram": (1, 2, 3, 4)}

    for plugin, valid_orders in ORDERS.items():
        for order in valid_orders:
            # ── 1) classic (non-positional) variants ──────────────────
            for smoothing in (False, True):
                for boundaries in (False, True):
                    for weight_mode in WeightMode:
                        for mode in probability.available_transforms():
                            for agg in AggregateMode:
                                if _skip(plugin, weight_mode, agg):
                                    continue
                                cfg = Config(
                                    ngram_order      = order,
                                    smoothing        = smoothing,
                                    use_boundaries   = boundaries,
                                    prob_mode        = mode,
                                    aggregate_mode   = agg,
                                    weight_mode      = weight_mode,
                                    position_strategy=None,
                                )
                                if not PluginRegistry[plugin].supports(cfg):
                                    continue
                                if _matches_filters(cfg, filters or {}):
                                    yield _make_variant(plugin, order, cfg)
            # ── 2) positional variants (absolute | relative) ──
            for strategy_name in ("absolute", "relative"):
                for weight_mode in WeightMode:
                    for mode in probability.available_transforms():
                        for agg in AggregateMode:
                            if _skip(plugin, weight_mode, agg):
                                continue
                            cfg = Config.default(
                                ngram_order=order,
                                position_strategy=strategy_name,
                                use_boundaries=True,
                                smoothing=False,
                                weight_mode=weight_mode,
                                prob_mode=mode,
                                aggregate_mode=agg,
                            )
                            strategy = get_position_strategy(strategy_name, n=order)
                            if _matches_filters(cfg, filters or {}):
                                yield _make_variant(plugin, order, cfg, strategy)

# End of src/variants.py