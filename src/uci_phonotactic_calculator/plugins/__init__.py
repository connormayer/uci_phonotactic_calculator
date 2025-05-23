"""Public plugin API – re-exports from plugins.core for back-compat."""

from .core import (
    BaseModel,
    BaseTransform,
    PluginRegistry,
    get_model,
    get_prob_transform,
    register,
)
from .core import discover_models as _core_discover

# Model modules still need auto-import
_core_discover()


def discover_all() -> None:  # legacy name used in variants.py
    _core_discover()


# Convenience export for external callers
__all__ = [
    "register",
    "PluginRegistry",
    "BaseModel",
    "BaseTransform",
    "get_model",
    "get_prob_transform",
    "discover_all",
]
