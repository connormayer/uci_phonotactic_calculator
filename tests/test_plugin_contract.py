"""Regression test for plugin contract: supports() and run-ability.

Ensures that all plugins correctly implement the required interface.
"""

import pytest

from uci_phonotactic_calculator.config import Config
from uci_phonotactic_calculator.corpus import Corpus
from uci_phonotactic_calculator.plugins import PluginRegistry


@pytest.mark.parametrize("name,cls", PluginRegistry.items())
def test_plugin_supports_and_runs(tmp_path, name, cls):
    # Configure with no boundary symbols using boundary_mode instead of include_boundary
    cfg = Config.default(
        ngram_order=1 if "uni" in name else 2,
        boundary_mode="none",  # This replaces include_boundary=False
    )
    if not cls.supports(cfg):
        return
    dummy = tmp_path / "dummy.csv"
    dummy.write_text("a,1\n")
    corpus = Corpus(dummy, cfg)  # No include_boundary parameter in new API
    m = cls(cfg)
    m.fit(corpus)  # should not crash
    out = m.score(["a"])  # should return float
    assert isinstance(out, float)


# End of tests/test_plugin_contract.py
