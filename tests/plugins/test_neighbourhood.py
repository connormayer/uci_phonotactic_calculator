"""
Neighbourhood-density plug-in: explicit verification against the
five-row reference corpus.

Corpus lines (freqs in parens, but freq is ignored by this model):

    t a           (10)
    a t a         (20)
    t a t a       (30)
    a t t a       (20)
    t a a         (10)

Training set (order preserved for readability):

    1.  ('t', 'a')
    2.  ('a', 't', 'a')
    3.  ('t', 'a', 't', 'a')
    4.  ('a', 't', 't', 'a')
    5.  ('t', 'a', 'a')

We will score the target WORD = ('t', 'a', 't', 'a') because it
exists in the set and has interesting neighbours.
"""

from __future__ import annotations

from typing import Set, Tuple

import pytest

from uci_phonotactic_calculator.core.config import Config
from uci_phonotactic_calculator.core.corpus import Corpus
from uci_phonotactic_calculator.plugins.neighbourhood import NeighbourhoodModel

SymbolT = Tuple[str, ...]
TARGET: SymbolT = ("t", "a", "t", "a")  # word we will score


def _train_model(training_path, mode: str) -> NeighbourhoodModel:
    cfg = Config.default(neighbourhood_mode=mode)
    corpus = Corpus(training_path, cfg)
    model = NeighbourhoodModel(cfg)
    model.fit(corpus)
    return model


# ------------------------------------------------------------------ #
# Helpers – compute neighbours by hand so the expected numbers are
#           visible right in the test.
# ------------------------------------------------------------------ #
def full_neighbours(token: SymbolT, alphabet: Set[str]) -> Set[SymbolT]:
    """
    Return the FULL edit-distance-1 neighbour set (subst, ins, del).
    Matches the registry function 'full' but written inline for clarity.
    """
    out = set()
    L = len(token)
    # substitutions
    for i, sym in enumerate(token):
        out.update(token[:i] + (a,) + token[i + 1 :] for a in alphabet if a != sym)
    # deletions
    for i in range(L):
        out.add(token[:i] + token[i + 1 :])
    # insertions
    for i in range(L + 1):
        out.update(token[:i] + (a,) + token[i:] for a in alphabet)
    return out


def subst_only_neighbours(token: SymbolT, alphabet: Set[str]) -> Set[SymbolT]:
    return {
        token[:i] + (a,) + token[i + 1 :]
        for i, sym in enumerate(token)
        for a in alphabet
        if a != sym
    }


# ------------------------------------------------------------------ #
# Tests
# ------------------------------------------------------------------ #
@pytest.mark.parametrize(
    "mode, neighbour_fn",
    [
        ("full", full_neighbours),
        ("substitution_only", subst_only_neighbours),
    ],
)
def test_neighbour_counts_against_hand_math(training_path, mode, neighbour_fn):
    model = _train_model(training_path, mode)

    # Alphabet derived from training corpus (same as model.alphabet)
    alphabet = model.alphabet

    # ---- expected count computed explicitly -----------------------
    expected_set = neighbour_fn(TARGET, alphabet) & model.training_set
    expected_count = float(len(expected_set))

    # ---- model score ---------------------------------------------
    got = model.score(list(TARGET))

    # ---- assertions ----------------------------------------------
    assert got == expected_count, (
        f"\nMode: {mode}"
        f"\nExpected neighbours ({int(expected_count)}): {sorted(expected_set)}"
        f"\nModel returned: {got}"
    )

    # also ensure the type is exactly float for downstream math
    assert isinstance(got, float)
