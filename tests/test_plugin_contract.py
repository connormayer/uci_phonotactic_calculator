"""tests/test_plugin_contract.py — Regression test for plugin contract: supports() and run-ability."""
import pytest
from ..src.plugins import PluginRegistry
from ..src.config import Config
from ..src.corpus import Corpus

@pytest.mark.parametrize("name,cls", PluginRegistry.items())
def test_plugin_supports_and_runs(tmp_path, name, cls):
    cfg = Config.default(ngram_order=1 if "uni" in name else 2)
    if not cls.supports(cfg):
        return
    dummy = tmp_path / "dummy.csv"
    dummy.write_text("a,1\n")
    corpus = Corpus(dummy, cfg, include_boundary=False)
    m = cls(cfg)
    m.fit(corpus)          # should not crash
    out = m.score(["a"]) # should return float
    assert isinstance(out, float)

# End of tests/test_plugin_contract.py