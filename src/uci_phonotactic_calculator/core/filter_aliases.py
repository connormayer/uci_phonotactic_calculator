"""
Canonical mapping for --filter keys.

External modules should import ALIASES and call canonical(key).

Add new entries here; they become immediately visible to
--list-filters and to the CLI parser without touching any
other file.
"""

from __future__ import annotations

from .config import Config

ALIASES: dict[str, str] = {
    # n-gram size
    "n": "ngram_order",
    "order": "ngram_order",
    # weighting
    "weight": "weight_mode",
    "weights": "weight_mode",
    # probability
    "prob": "prob_mode",
    "probability": "prob_mode",
    # aggregation
    "agg": "aggregate_mode",
    "aggregate": "aggregate_mode",
    # smoothing
    "smooth": "smoothing_scheme",
    "smoothing": "smoothing_scheme",
    # padding
    "bound": "boundary_mode",
    "boundary": "boundary_mode",
    # neighbourhood / edit distance
    "neigh": "neighbourhood_mode",
}


def canonical(key: str) -> str:
    """Return the Config attribute for a user-supplied filter key."""
    return ALIASES.get(key.lower(), key.lower())


def matches_filters(cfg: "Config", filters: dict[str, str]) -> bool:
    """Check if a Config object matches the given filter criteria."""
    for key, want in filters.items():
        got = getattr(cfg, key, None)
        if isinstance(got, bool):
            got = "true" if got else "false"
        else:
            got = str(got).lower()
        if got != want:
            return False
    return True
