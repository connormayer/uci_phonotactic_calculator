"""
Canonical mapping for --filter keys.

External modules should import ALIASES and call canonical(key).

Add new entries here; they become immediately visible to
--list-filters and to the CLI parser without touching any
other file.
"""
from __future__ import annotations

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
