"""src/config.py — Configuration dataclass and enums for n-gram models."""

from enum import Enum
from dataclasses import dataclass


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


@dataclass(slots=True)
class Config:
    """Holds flags and modes for n-gram model fitting and scoring."""
    # Core options
    positional     : bool           = False
    conditional    : bool           = False
    use_boundaries : bool           = True
    smoothing      : bool           = False
    weight_by_freq : bool           = False

    # Mode selections
    prob_mode      : ProbMode       = ProbMode.CONDITIONAL
    aggregate_mode : AggregateMode  = AggregateMode.LOG_PRODUCT
    ngram_order    : int            = 2

    @staticmethod
    def default(**overrides) -> "Config":
        """
        Return a Config initialized with defaults,
        then apply any overrides provided as keyword arguments.
        """
        cfg = Config()
        for key, value in overrides.items():
            setattr(cfg, key, value)
        return cfg

# End of src/config.py
