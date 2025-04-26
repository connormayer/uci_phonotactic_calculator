"""
Canonical column naming
-----------------------
classic    := model "n" order [ "smoothed" | "prob" ]
                              weight-mode
                              [ "bound" | "nobound" ]
                              prob-transform
                              aggregate-mode
positional := model "n" order "positional"
                              weight-mode
                              prob-transform
                              aggregate-mode
                              position-strategy
Every token is lower-case; no token is ever empty, so no filtering step is required.
"""
from __future__ import annotations
from .config import Config

__all__ = ["build_header"]

def _weight_token(mode) -> str:        # mode is a WeightMode enum
    return mode.value.lower() if mode.name != "NONE" else "none"

def build_header(model: str, cfg: Config) -> str:
    """
    Return the canonical CSV column header for a (model, cfg) pair.
    Mirrors variants._make_variant() so every caller is consistent.
    """
    parts: list[str] = [model, f"n{cfg.ngram_order}"]
    if cfg.position_strategy and str(cfg.position_strategy).lower() != "none":  # positional
        parts += [
            "positional",
            _weight_token(cfg.weight_mode),
            str(cfg.prob_mode).lower(),
            cfg.aggregate_mode.value.lower(),
            cfg.position_strategy.lower(),
        ]
    else:                                           # classic
        parts += [
            "smoothed" if cfg.smoothing else "prob",
            _weight_token(cfg.weight_mode),
            "bound" if cfg.use_boundaries else "nobound",
            str(cfg.prob_mode).lower(),
            cfg.aggregate_mode.value.lower(),
        ]
    return "_".join(parts)
