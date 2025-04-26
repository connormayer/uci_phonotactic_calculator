"""
core.py – central registry for every runtime-discoverable plug-in.

Public decorators
-----------------
• register(name)          – add a *model* class to PluginRegistry
• register_prob(name)     – add a *probability transform* class to ProbTransformRegistry
• register_strategy(name) – (added in PR 2) register a *counting strategy*

All three return the original class unmodified, so you may stack them
with `@dataclass` or `@functools.total_ordering` as usual.
"""
from __future__ import annotations
import importlib, pkgutil, warnings
warnings.filterwarnings("once", category=DeprecationWarning)
from enum import Enum
import numpy as np
from abc import ABC, abstractmethod
from typing import Callable, Dict, Type, TYPE_CHECKING

if TYPE_CHECKING:
    from ..config import Config
    from ..corpus import Corpus

__all__ = [
    "BaseModel",
    "PluginRegistry", "register",
    "ProbTransformRegistry", "register_prob",
    "StrategyRegistry", "register_strategy",
    "get_model", "discover_models",
    "ALIASES",
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
StrategyRegistry: dict[str, "Type[BaseCounter]"] = {}
def register_strategy(name: str) -> Callable[["Type[BaseCounter]"], "Type[BaseCounter]"]:
    """Decorator: `@register_strategy("ngram")` → class auto-added to StrategyRegistry."""
    def _decorator(cls: "Type[BaseCounter]") -> "Type[BaseCounter]":
        if name in StrategyRegistry:
            raise KeyError(f"Strategy ‘{name}’ already registered")
        StrategyRegistry[name] = cls
        return cls
    return _decorator

# ------------------------------------------------------------------ #
# 2) Alias map for deprecated names
# ------------------------------------------------------------------ #
ALIASES = {
    # Deprecated aliases for legacy model names
    "positional_bigram": "ngram",
    "positional_unigram": "ngram",
    "positional_ngram": "ngram",
    "unigram": "ngram",
    "bigram": "ngram",
}

_WARNED_ALIASES: set[str] = set()

def _warn_alias(alias: str):
    if alias not in _WARNED_ALIASES:
        if alias in ("unigram", "bigram"):
            n_val = "1" if alias == "unigram" else "2"
            msg = f"'{alias}' is deprecated; use 'ngram' with -n {n_val}."
        elif alias == "positional_ngram":
            msg = ("'positional_ngram' is deprecated; use 'ngram' plus --position-strategy=absolute/relative/none.")
        else:
            msg = f"'{alias}' is deprecated; use 'ngram'."
        warnings.warn(msg, DeprecationWarning, stacklevel=2)
        _WARNED_ALIASES.add(alias)

# ------------------------------------------------------------------ #
# 3) Base class with generic capability check
# ------------------------------------------------------------------ #
class BaseModel(ABC):
    # Declarative capability flags
    order_min: int | None = None
    order_max: int | None = None
    requires_dense: bool | None = None     # True = dense-only, False = sparse-only

    def __init__(self, cfg: "Config"):     # quoted to avoid import cycle
        self.cfg = cfg

    @classmethod
    def supports(cls, cfg: "Config") -> bool:      # noqa: D401
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
    def supports(cfg: "Config") -> bool:     # noqa: D401
        return True                          # override if a transform has limits

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
        return                                # nothing to iterate
    for _, mod_name, is_pkg in pkgutil.iter_modules(mod.__path__):
        if mod_name.startswith("_") or is_pkg:
            continue
        importlib.import_module(f"{pkg_name}.{mod_name}")

def discover_models() -> None:
    """Import every n-gram model plugin **and** every probability-transform
    module exactly once.  Subsequent calls are no-ops."""
    global _DISCOVERED
    if _DISCOVERED:
        return
    parent_pkg = __name__.rsplit('.', 1)[0]          # "src.plugins"
    _discover_submodules(parent_pkg)                        # model plugins live here
    _discover_submodules(f"{parent_pkg}.prob_transforms")  # probability plugins
    _DISCOVERED = True

# ------------------------------------------------------------------ #
# 5) Public accessor with alias-resolution
# ------------------------------------------------------------------ #
def get_model(name: str):
    """
    Retrieve a model class by name, resolving deprecated aliases and warning if needed.
    """
    discover_models()
    if name in ALIASES:
        _warn_alias(name)
        name = ALIASES[name]
    try:
        return PluginRegistry[name]
    except KeyError as e:
        avail = ", ".join(sorted(PluginRegistry))
        raise KeyError(f"No plugin called ‘{name}’. Available: {avail}") from e

# ---- probability helpers --------------------------------------------
ProbTransformRegistry: dict[str, Type[BaseTransform]] = {}

from .strategies.base import BaseCounter
def register_prob(name: str) -> Callable[[Type[BaseTransform]], Type[BaseTransform]]:
    """Decorator: `@register_prob("joint")` → class auto-added to ProbTransformRegistry."""
    def _decorator(cls: Type[BaseTransform]) -> Type[BaseTransform]:
        if name in ProbTransformRegistry:
            raise KeyError(f"Prob-transform ‘{name}’ already registered")
        ProbTransformRegistry[name] = cls
        return cls
    return _decorator

def get_prob_transform(name: str | Enum) -> BaseTransform:
    discover_models()
    key = str(name)  # converts ProbMode → "joint", "conditional"
    try:
        cls = ProbTransformRegistry[key]
    except KeyError as e:
        avail = ", ".join(sorted(ProbTransformRegistry))
        raise KeyError(f"No prob-transform called ‘{key}’. Available: {avail}") from e
    return cls()
