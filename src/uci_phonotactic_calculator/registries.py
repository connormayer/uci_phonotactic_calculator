from collections import defaultdict

_R = defaultdict(dict)  # {category: {name: obj}}


def register(category: str, name: str):
    """Decorator: @register('aggregate_mode', 'prod')(fn)."""

    def deco(obj):
        if name in _R[category]:
            raise KeyError(f"{category} “{name}” already registered")
        _R[category][name] = obj
        return obj

    return deco


def registry(category: str):
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
