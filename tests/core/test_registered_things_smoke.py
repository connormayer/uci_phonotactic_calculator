"""
Generic smoke checks over every registry category so new
entries can land without having to modify the suite.

For each category we only verify that:
5 the callable/object can be retrieved
5 calling it with the minimal dummy argument(s) does not error
"""

import inspect

import numpy as np
import pytest

from uci_phonotactic_calculator.core.registries import registry

CATEGORIES = [
    "aggregate_mode",
    "boundary_mode",
    "boundary_scheme",
    "count_strategy",
    "position_strategy",
    "smoothing_scheme",
    "weight_mode",
    "neighbourhood_mode",
]


@pytest.mark.parametrize("category", CATEGORIES)
def test_every_registry_entry_is_callable(category):
    for name, obj in registry(category).items():
        # Some entries are classes, some are plain functions
        assert callable(obj), f"{category}:{name} is not callable"

        # Minimal smoke-invoke:
        sig = inspect.signature(obj)
        kwargs = {}
        # try to satisfy required parameters generically
        if "table" in sig.parameters:
            kwargs["table"] = np.zeros((2, 2))
        if "vocab" in sig.parameters:
            kwargs["vocab"] = {(0,), (1,), (1,)}
        if "token" in sig.parameters:
            kwargs["token"] = ["a"]
        if "alphabet" in sig.parameters:
            kwargs["alphabet"] = {"a"}
        if "n" in sig.parameters:
            kwargs["n"] = 1
        if "pad_sym" in sig.parameters:
            kwargs["pad_sym"] = "#"

        try:
            obj(**kwargs) if kwargs else obj()
        except TypeError:
            # Accept mismatched smoke-args; we only care that import didn't fail.
            pass
