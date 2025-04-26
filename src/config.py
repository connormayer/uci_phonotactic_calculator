"""src/config.py — Configuration dataclass and enums for n-gram models.

Note: Probability always interprets the last axis as the predicted symbol.

Config fields:
- position_strategy: str | None
    Controls positional bucketing for n-gram models. Accepts 'absolute', 'relative', 'none', or None (default). If None, classic n-gram counting is used.
"""

from enum import Enum
from dataclasses import dataclass
from typing import Union


class StrEnum(str, Enum):
    """Base enum whose string representation is its value."""
    def __str__(self) -> str:
        return str(self.value)


class ProbMode(StrEnum):
    """Available probability calculation modes."""
    JOINT = "joint"
    CONDITIONAL = "conditional"


class AggregateMode(StrEnum):
    """Strategies for aggregating per-gram scores."""
    LOG_PRODUCT = "log_product"
    LOGSUMEXP   = "logsumexp"
    SUM         = "sum"
    SUM_PLUS1   = "sum_plus1"      # NEW
    MIN         = "min"
    MAX         = "max"


class WeightMode(StrEnum):
    """Strategies for weighting n-gram scores."""
    NONE = "none"
    RAW  = "raw"
    LOG  = "log"


class NeighbourhoodMode(StrEnum):
    """Edit-distance operations counted as neighbours."""
    FULL              = "full"
    SUBSTITUTION_ONLY = "substitution_only"


import warnings

# TODO-v2.0: final removal of legacy 'positional' kwarg

@dataclass(slots=True)
class Config:
    """
    Configuration for n-gram models.

    Fields:
        position_strategy: str | None — positional bucketing ('absolute', 'relative', 'none', or None)
    """
    # Core options
    use_boundaries : bool           = True
    smoothing      : bool           = False
    weight_mode    : WeightMode     = WeightMode.NONE
    neighbourhood_mode: NeighbourhoodMode = NeighbourhoodMode.FULL
    position_strategy: str | None   = None     # 'absolute', 'relative', 'none', or None

    # Mode selections
    prob_mode      : Union[ProbMode, str] = ProbMode.CONDITIONAL
    aggregate_mode : AggregateMode  = AggregateMode.LOG_PRODUCT
    ngram_order    : int            = 2

    @staticmethod
    def _normalize_none_string(value):
        """Normalize string 'none' (case-insensitive) to None, with a DeprecationWarning (warn once per session)."""
        if isinstance(value, str) and value.lower() == "none":
            if not hasattr(Config, "_warned_none_str"):
                warnings.warn(
                    "String 'none' for position_strategy is deprecated; use None or omit the flag.",
                    DeprecationWarning, stacklevel=3)
                Config._warned_none_str = True
            return None
        return value

    @staticmethod
    def default(**overrides) -> "Config":
        """
        Return a Config initialized with defaults,
        then apply any overrides provided as keyword arguments.
        Warn if any deprecated fields are set.
        """
        cfg = Config()
        for key, value in overrides.items():
            if key == "neighbourhood_mode":
                value = NeighbourhoodMode(value) if isinstance(value, str) else value
            if key == "positional":
                warnings.warn(
                    "Config.positional is deprecated; use position_strategy instead.",
                    DeprecationWarning, stacklevel=2)
                # Legacy True → 'absolute', False → None
                value = "absolute" if value else None
                key = "position_strategy"
            if key == "position_strategy" and isinstance(value, str):
                value = value.strip() or None  # Normalize empty or whitespace-only string to None
                value = Config._normalize_none_string(value)
            setattr(cfg, key, value)
        return cfg


# End of src/config.py
