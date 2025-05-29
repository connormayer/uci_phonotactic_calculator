import numpy as np

from uci_phonotactic_calculator.plugins.utils.smoothing import (
    dense_laplace,
    sparse_laplace,
)


def test_dense_laplace_increments_in_place():
    arr = np.zeros((2, 3))
    view_id = id(arr)
    out = dense_laplace(arr)
    assert id(out) == view_id  # same object, no copy
    assert (out == 1.0).all()  # every cell +1


def test_sparse_laplace_adds_missing_keys_and_increments():
    table = {(0,): 2.0, (1,): 0.0}
    vocab = {(0,), (1,), (2,)}
    sparse_laplace(table, vocab)
    expected = {(0,): 3.0, (1,): 1.0, (2,): 1.0}
    assert table == expected
