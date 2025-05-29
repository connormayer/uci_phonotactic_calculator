from pathlib import Path
from typing import Any, Type, cast

import pytest

from uci_phonotactic_calculator.core.config import Config
from uci_phonotactic_calculator.core.corpus import Corpus
from uci_phonotactic_calculator.plugins.core import BaseModel, PluginRegistry

"""Regression test for plugin contract: supports() and run-ability.

Ensures that all plugins correctly implement the required interface.
"""


@pytest.mark.parametrize("name,cls", PluginRegistry.items())
def test_plugin_supports_and_runs(tmp_path: Path, name: str, cls: Type[Any]) -> None:
    # Cast to BaseModel type to access supports method
    model_cls = cast(Type[BaseModel], cls)
    # Configure with no boundary symbols using boundary_mode instead of include_boundary
    cfg = Config.default(
        ngram_order=1 if "uni" in name else 2,
        boundary_mode="none",  # This replaces include_boundary=False
    )
    if not model_cls.supports(cfg):
        return
    dummy = tmp_path / "dummy.csv"
    dummy.write_text("a,1\n")
    corpus = Corpus(dummy, cfg)  # No include_boundary parameter in new API
    m = cls(cfg)
    m.fit(corpus)  # should not crash
    out = m.score(["a"])  # should return float
    assert isinstance(out, float)


# End of tests/test_plugin_contract.py
