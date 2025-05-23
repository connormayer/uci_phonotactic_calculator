# src/header_schema.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable


@dataclass(frozen=True)
class HeaderField:
    attr: str  # Config attribute name
    normalise: Callable[[Any], str]  # value → short token
    default: Any  # sentinel ⇒ token is omitted
    prefix: str | None = None  # optional namespace, e.g. "cs"
    always: bool = False  # emit even if value==default


SCHEMA: tuple[HeaderField, ...] = (
    HeaderField("ngram_order", lambda v: f"n{v}", default=None, always=True),
    # Default strategy is identical to the plug-in name, so emit nothing unless
    # the user selects an alternative (e.g. 'trie' → cs_trie).
    HeaderField(
        "count_strategy",
        lambda v: "" if v == "ngram" else f"cs_{v}",
        default="ngram",
        always=False,
    ),
    HeaderField(
        "position_strategy", lambda v: f"pos_{v or 'none'}", default=None, always=True
    ),
    HeaderField("boundary_mode", lambda v: f"bound_{v}", default=None, always=True),
    HeaderField("smoothing_scheme", lambda v: f"smooth_{v}", default=None, always=True),
    HeaderField("weight_mode", lambda v: f"weight_{v}", default=None, always=True),
    HeaderField("prob_mode", lambda v: f"prob_{v}", default=None, always=True),
    HeaderField(
        "aggregate_mode", lambda v: f"agg_{v or 'none'}", default=None, always=True
    ),
)
