"""
Built-in boundary-padding modes.

Each callable receives (token: list[str], pad_sym: str, n: int)
and must return the padded list[str].

The ‘n’ parameter lets a mode decide how many pads are needed
(n-1 for classic n-gram behaviour).
"""

from typing import List

from ..core.registries import register


def _prefix(tok: List[str], pad: str, n: int) -> List[str]:
    return [pad] * (n - 1) + tok


def _suffix(tok: List[str], pad: str, n: int) -> List[str]:
    return tok + [pad] * (n - 1)


def _both(tok: List[str], pad: str, n: int) -> List[str]:
    pad_seq = [pad] * (n - 1)
    return pad_seq + tok + pad_seq


def _none(tok: List[str], pad: str, n: int) -> List[str]:
    return tok  # unchanged


register("boundary_mode", "prefix")(_prefix)
register("boundary_mode", "suffix")(_suffix)
register("boundary_mode", "both")(_both)
register("boundary_mode", "none")(_none)


# ------------------------------------------------------------------
# Introspection helpers – used by models, keep in this central file
# ------------------------------------------------------------------
def needs_prefix(mode: str) -> bool:
    """True if the boundary scheme pads on the left."""
    return mode in ("both", "prefix")


def needs_suffix(mode: str) -> bool:
    """True if the boundary scheme pads on the right."""
    return mode in ("both", "suffix")


__all__ = ["needs_prefix", "needs_suffix"]
