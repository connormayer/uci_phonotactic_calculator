"""
Explicit verification of the **kneser_ney** smoothing scheme.

We start with a tiny 2-symbol vocabulary:
        prev ⟶ next      x   y
                       ┌──────────┐
            x │         2    1    │
            y │         1    0    │
                       └──────────┘

Absolute discount **D = 0.75** (fixed in the implementation).

Hand maths
----------
Row x (tot = 3, n_types = 2) → λ = D·n_types / tot = 0.5
    discounted:  [2-D, 1-D]  = [1.25, 0.25]
    back-off mass: λ·tot = 1.5
    continuation P:  col totals > 0  →  [2, 1]  so p_cont = [2/3, 1/3]
    final row:  [1.25 + 1.0, 0.25 + 0.5] = **[2.25, 0.75]**

Row y (tot = 1, n_types = 1) → λ = 0.75
    discounted:  [0.25, 0.00]
    back-off mass: 0.75
    final row:  [0.25 + 0.50, 0.00 + 0.25] = **[0.75, 0.25]**

The smoothed matrix should therefore be:

        [[2.25, 0.75],
         [0.75, 0.25]]
"""

from typing import MutableMapping, cast  # Added for type hinting

import numpy as np
from numpy.typing import NDArray  # ✨ right alias for 1-param ndarray

from uci_phonotactic_calculator.core.registries import registry
from uci_phonotactic_calculator.plugins.utils.smoothing import (
    IndexTuple,  # For type hint
)
from uci_phonotactic_calculator.plugins.utils.smoothing_kn import kneser_ney


def test_kn_dense_explicit():
    counts: NDArray[np.float64] = np.array([[2.0, 1.0], [1.0, 0.0]], dtype=np.float64)
    expected: NDArray[np.float64] = np.array(
        [[2.25, 0.75], [0.75, 0.25]], dtype=np.float64
    )

    # kneser_ney returns a Union, so we help mypy by asserting the type
    out_union = kneser_ney(counts.copy())
    assert isinstance(out_union, np.ndarray), "Expected ndarray output"
    out: NDArray[np.float64] = out_union  # Now mypy knows out is an NDArray

    # 1) exact numbers
    assert np.allclose(out, expected)

    # 2) row totals are preserved
    assert np.allclose(out.sum(axis=1), counts.sum(axis=1))


def test_kn_sparse_explicit():
    table_input: dict[IndexTuple, float] = {
        (0, 0): 2.0,
        (0, 1): 1.0,
        (1, 0): 1.0,
    }
    expected: dict[IndexTuple, float] = {
        (0, 0): 2.25,
        (0, 1): 0.75,
        (1, 0): 0.75,
        (1, 1): 0.25,
    }

    out_union = kneser_ney(table_input.copy())
    assert isinstance(out_union, MutableMapping), "Expected MutableMapping output"
    # We can cast here as an alternative to direct assignment after isinstance
    out = cast(MutableMapping[IndexTuple, float], out_union)

    assert out.keys() == expected.keys()
    for k in expected:
        assert abs(out[k] - expected[k]) < 1e-6


def test_kn_registered():
    """The smoother must be discoverable via the global registry."""
    assert "kneser_ney" in registry("smoothing_scheme")
    fn = registry("smoothing_scheme")["kneser_ney"]
    toy_dense = np.ones((2, 2))
    # Test with dense array
    result_dense = fn(toy_dense)
    assert result_dense is not None
    assert isinstance(result_dense, np.ndarray)

    # Test with sparse dict
    toy_sparse: MutableMapping[IndexTuple, float] = {
        (0, 0): 1.0,
        (0, 1): 1.0,
        (1, 0): 1.0,
        (1, 1): 1.0,
    }
    result_sparse = fn(toy_sparse)
    assert result_sparse is not None
    assert isinstance(result_sparse, MutableMapping)
