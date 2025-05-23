"""src/config.py — Configuration dataclass and enums for n-gram models.

Note: Probability always interprets the last axis as the predicted symbol.

Config fields:
- position_strategy: str | None
    Controls positional bucketing for n-gram models. Accepts 'absolute', 'relative',
    'none', or None (default). If None, classic n-gram counting is used.

"""

import warnings
from dataclasses import dataclass
from typing import Union

from .registries import (
    registry,  # used in Config.boundary_symbol property
    validate_choice,
)

# Note: prob_mode is a free-form string field; no alias needed.

# TODO-v2.0: final removal of legacy 'positional' kwarg


@dataclass(slots=True)
class Config:
    @property
    def boundary_symbol(self):
        return registry("boundary_scheme")[self.boundary_scheme]()

    @staticmethod
    def _normalize_none_string(value):
        """Normalize string 'none' (case-insensitive) to None, with a warning.

        The warning is shown only once per session.
        """
        if isinstance(value, str) and value.lower() == "none":
            if not hasattr(Config, "_warned_none_str"):
                warnings.warn(
                    "String 'none' for position_strategy is deprecated; use None.",
                    DeprecationWarning,
                    stacklevel=3,
                )
                Config._warned_none_str = True
            return None
        return value

    """
    Configuration for n-gram models.

    Fields:
        position_strategy: str | None — positional bucketing ('absolute', 'relative',
            'none', or None)
        weight_mode: str — token weighting strategy
        neighbourhood_mode: str — edit operation set for neighborhood
        prob_mode: str — probability mode (e.g., 'joint', 'conditional')
        aggregate_mode: str — aggregation mode
        ngram_order: int — n-gram order (>=1)
        smoothing_scheme: str — smoothing scheme name (default: 'laplace')
        boundary_mode: str — padding mode: both, prefix, suffix, none
        boundary_scheme: str — boundary symbol scheme
        count_strategy: str — n-gram counting backend
    """
    # Core options
    weight_mode: str = "none"
    neighbourhood_mode: str = "full"
    position_strategy: str | None = None  # 'absolute', 'relative', 'none', or None

    # Mode selections
    prob_mode: Union[str] = "conditional"
    aggregate_mode: str = "prod"
    ngram_order: int = 2
    smoothing_scheme: str = "laplace"
    boundary_mode: str = "both"  # both│prefix│suffix│none
    # Placeholders for future registry categories
    boundary_scheme: str = "word"
    count_strategy: str = "ngram"

    @staticmethod
    def default(**overrides) -> "Config":
        """
        Return a Config initialized with defaults,
        """
        cfg = Config()
        for key, value in overrides.items():
            if key == "position_strategy" and isinstance(value, str):
                value = (
                    value.strip() or None
                )  # Normalize empty or whitespace-only string to None
                value = Config._normalize_none_string(value)
            setattr(cfg, key, value)
        # Validate all registry-backed fields
        cfg.weight_mode = validate_choice("weight_mode", cfg.weight_mode)
        cfg.aggregate_mode = validate_choice("aggregate_mode", cfg.aggregate_mode)
        cfg.neighbourhood_mode = validate_choice(
            "neighbourhood_mode", cfg.neighbourhood_mode
        )
        cfg.position_strategy = (
            None
            if cfg.position_strategy in (None, "", "none")
            else validate_choice("position_strategy", cfg.position_strategy)
        )
        cfg.smoothing_scheme = validate_choice("smoothing_scheme", cfg.smoothing_scheme)
        cfg.boundary_scheme = validate_choice("boundary_scheme", cfg.boundary_scheme)
        cfg.count_strategy = validate_choice("count_strategy", cfg.count_strategy)
        cfg.boundary_mode = validate_choice("boundary_mode", cfg.boundary_mode)
        return cfg


# End of src/config.py
