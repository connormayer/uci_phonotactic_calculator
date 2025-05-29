from collections import defaultdict
from typing import Any, Callable, DefaultDict, Dict, TypeVar

_R: DefaultDict[str, Dict[str, Any]] = defaultdict(dict)  # {category: {name: obj}}


T = TypeVar("T")


def register(category: str, name: str) -> Callable[[T], T]:
    """Decorator: @register('aggregate_mode', 'prod')(fn)."""

    def deco(obj: T) -> T:
        if name in _R[category]:
            raise KeyError(f"{category} “{name}” already registered")
        _R[category][name] = obj
        return obj

    return deco


def registry(category: str) -> Dict[str, Any]:
    """Return the live dict for ‹category› (never raises)."""
    return _R[category]


# ── public helper: validate that a value is registered ────────────
def validate_choice(category: str, value: str) -> str:
    """
    Ensure <value> is registered under <category>; raise if not.
    Returns the value unchanged so the call can be in-line assigned.
    """
    if value not in _R[category]:
        raise ValueError(f"Unknown {category}: {value}")
    return value
