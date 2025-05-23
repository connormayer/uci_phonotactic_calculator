# src/neighbourhood_builtin.py
"""
Built-in edit-distance-1 neighbourhood generators.
Each function returns a *set* of neighbour tokens (tuple[str, …]).
"""

from __future__ import annotations

from typing import Iterable, Set, Tuple

from ..core.registries import register

SymbolT = Tuple[str, ...]


@register("neighbourhood_mode", "full")
def neighbours_full(target: SymbolT, alphabet: Iterable[str]) -> Set[SymbolT]:
    out: Set[SymbolT] = set()
    tlen = len(target)
    # substitutions
    for i, ph in enumerate(target):
        out.update(target[:i] + (a,) + target[i + 1 :] for a in alphabet if a != ph)
    # deletions
    for i in range(tlen):
        out.add(target[:i] + target[i + 1 :])
    # insertions
    for i in range(tlen + 1):
        out.update(target[:i] + (a,) + target[i:] for a in alphabet)
    return out


@register("neighbourhood_mode", "substitution_only")
def neighbours_subst_only(target: SymbolT, alphabet: Iterable[str]) -> Set[SymbolT]:
    out: Set[SymbolT] = set()
    for i, ph in enumerate(target):
        out.update(target[:i] + (a,) + target[i + 1 :] for a in alphabet if a != ph)
    return out
