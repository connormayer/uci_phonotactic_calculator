"""
core.py – central registry for every runtime-discoverable plug-in.

Public decorators
-----------------
• register(name)          – add a *model* class to PluginRegistry
• register_prob(name)     – add a *probability transform* class to ProbTransformRegistry
• register('count_strategy', name) – register a counting strategy

All three return the original class unmodified, so you may stack them
with `@dataclass` or `@functools.total_ordering` as usual.
"""

from __future__ import annotations

import importlib
import pkgutil
import warnings
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Callable, Dict, Type

import numpy as np

warnings.filterwarnings("once", category=DeprecationWarning)

# Make boundary_mode choices available even if users call get_model() without importing
# the parent package.

if TYPE_CHECKING:
    from ..core.config import Config
    from ..core.corpus import Corpus

__all__ = [
    "BaseModel",
    "PluginRegistry",
    "register",
    "ProbTransformRegistry",
    "register_prob",
    "get_model",
    "discover_models",
]

# ------------------------------------------------------------------ #
# 1) Registry + decorator
# ------------------------------------------------------------------ #
PluginRegistry: Dict[str, Type["BaseModel"]] = {}


def register(name: str) -> Callable[[Type["BaseModel"]], Type["BaseModel"]]:
    """Decorator: `@register("bigram")` → class auto-added to PluginRegistry."""

    def _decorator(cls: Type["BaseModel"]) -> Type["BaseModel"]:
        if name in PluginRegistry:
            raise KeyError(f"Plugin ‘{name}’ already registered")
        PluginRegistry[name] = cls
        return cls

    return _decorator


# ---- strategy registry (for extensible counting strategies) ---------


# ------------------------------------------------------------------ #
# 3) Base class with generic capability check
# ------------------------------------------------------------------ #
class BaseModel(ABC):
    # Declarative capability flags
    order_min: int | None = None
    order_max: int | None = None
    requires_dense: bool | None = None  # True = dense-only, False = sparse-only

    def __init__(self, cfg: "Config"):  # quoted to avoid import cycle
        self.cfg = cfg

    @classmethod
    def supports(cls, cfg: "Config") -> bool:  # noqa: D401
        o = cfg.ngram_order
        if cls.order_min is not None and o < cls.order_min:
            return False
        if cls.order_max is not None and o > cls.order_max:
            return False
        if cls.requires_dense is not None:
            want_dense = o <= 3
            if want_dense != cls.requires_dense:
                return False
        return True

    # ------------------------------------------------------------------
    # Optional: plug-ins may inject variant-specific tokens
    # ------------------------------------------------------------------
    @classmethod
    def extra_header_tokens(cls, cfg: "Config") -> list[str]:  # noqa: D401
        return []

    @classmethod
    def header(cls, cfg: "Config") -> str:
        # Provide a default implementation that uses the model's class name
        # and configuration settings to build a header string
        try:
            from ..core.header_utils import build_header

            # Get the model name from registry or use class name
            model_name = next(
                (
                    name
                    for name, model_cls in PluginRegistry.items()
                    if model_cls == cls
                ),
                cls.__name__.lower(),
            )
            return build_header(model_name, cfg)
        except Exception:
            # Last resort fallback - simple name
            return f"{cls.__name__}_{cfg.ngram_order}"

    @abstractmethod
    def fit(self, corpus: "Corpus") -> None: ...
    @abstractmethod
    def score(self, token: list[str]) -> float: ...


# ------------------------------------------------------------------ #
# 3b) Base class for probability transforms
# ------------------------------------------------------------------ #
class BaseTransform(ABC):
    """
    Common interface for probability transforms.
    Implement •transform(counts) → log-probs
             •supports(cfg)       → bool   (optional, mirror BaseModel)
    """

    @staticmethod
    def supports(cfg: "Config") -> bool:  # noqa: D401
        return True  # override if a transform has limits

    @abstractmethod
    def transform(self, counts: "np.ndarray") -> "np.ndarray": ...


# ------------------------------------------------------------------ #
# 4) Discovery helpers
# ------------------------------------------------------------------ #
_DISCOVERED = False


def _discover_submodules(pkg_name: str) -> None:
    """Import every sub-module (non-private) exactly once."""
    mod = importlib.import_module(pkg_name)
    if not hasattr(mod, "__path__"):
        return  # nothing to iterate
    for _, mod_name, is_pkg in pkgutil.iter_modules(mod.__path__):
        if mod_name.startswith("_") or is_pkg:
            continue
        importlib.import_module(f"{pkg_name}.{mod_name}")


def discover_models():
    """
    Import every n-gram model plugin **and** every probability-transform
    module exactly once.  Subsequent calls are no-ops.
    """
    import importlib

    importlib.import_module(
        "uci_phonotactic_calculator.plugins.strategies"
    )  # registers count_strategy entries
    global _DISCOVERED
    if _DISCOVERED:
        return
    # Import every sub-module (non-private) exactly once.
    _discover_submodules(__package__)
    _DISCOVERED = True
    # model plugins live here
    _discover_submodules(f"{__package__}.prob_transforms")  # probability plugins


# ------------------------------------------------------------------ #
# 5) Public accessor
# ------------------------------------------------------------------ #
def get_model(name: str):
    """
    Retrieve a model class by name.
    """
    discover_models()
    try:
        return PluginRegistry[name]
    except KeyError as err:
        raise ValueError(f"Unknown model: {name}") from err


# ---- probability helpers --------------------------------------------
ProbTransformRegistry: dict[str, Type[BaseTransform]] = {}


def register_prob(name: str) -> Callable[[Type[BaseTransform]], Type[BaseTransform]]:
    """Decorator: `@register_prob("joint")` adds class to ProbTransformRegistry."""

    def _decorator(cls: Type[BaseTransform]) -> Type[BaseTransform]:
        if name in ProbTransformRegistry:
            raise KeyError(f"Prob-transform ‘{name}’ already registered")
        ProbTransformRegistry[name] = cls
        return cls

    return _decorator


def get_prob_transform(name: str) -> BaseTransform:
    discover_models()
    key = str(name)  # name is now always a string
    try:
        cls = ProbTransformRegistry[key]
    except KeyError as e:
        avail = ", ".join(sorted(ProbTransformRegistry))
        raise KeyError(f"No prob-transform called ‘{key}’. Available: {avail}") from e
    return cls()
