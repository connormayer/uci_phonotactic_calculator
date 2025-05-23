"""src/header_utils.py — Centralized header construction for all model plugins."""

from ..cli.utils import slug
from ..plugins.core import get_model
from .config import Config
from .header_schema import SCHEMA

_WEIGHT_TAG = {
    "none": "unw",
    "raw": "raw_freq",
    "log": "freq_weighted",
    "legacy_log": "freq_weighted_legacy",
}


def _weight_token(mode: str) -> str:
    return _WEIGHT_TAG.get(mode, mode)


def _smooth_tag(cfg) -> str:
    return (
        "none"
        if cfg.smoothing_scheme == "none"
        else f"{cfg.smoothing_scheme}_smoothing"
    )


# ------------------------------------------------------------------
# Boundary-padding ➜ short, unique token
# ------------------------------------------------------------------
def _boundary_tag(boundary_mode: str | None) -> str:
    """
    Canonicalise a boundary-padding mode into a short token.

    * none   → nobound
    * both   → bound
    * prefix → bound_prefix
    * suffix → bound_suffix
    """
    mode = (boundary_mode or "both").lower()
    if mode == "none":
        return "nobound"
    if mode in ("both", ""):
        return "bound"
    return f"bound_{mode}"


# Aliases for compact header representation
AGGREGATE_ALIASES = {
    "prod": "prod",
    "lse": "lse",
    "sum": "sum",
    "sum_plus1": "sum_plus1",
    "min": "min",
    "max": "max",
}


def _aggregate_alias(mode: str) -> str:
    return mode


def build_header(plugin: str, cfg: "Config") -> str:
    parts: list[str] = [plugin]

    for field in SCHEMA:
        value = getattr(cfg, field.attr, None)
        token = field.normalise(value)
        if token:  # skip empty / suppressed tokens
            parts.append(token)

    # Allow plug-ins to inject extras (see step 3)
    Model = get_model(plugin)
    if hasattr(Model, "extra_header_tokens"):
        parts.extend(Model.extra_header_tokens(cfg))

    return slug(*parts)
