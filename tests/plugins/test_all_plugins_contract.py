import math

import pytest

from uci_phonotactic_calculator.core.config import Config
from uci_phonotactic_calculator.core.corpus import Corpus
from uci_phonotactic_calculator.plugins.core import PluginRegistry


@pytest.mark.parametrize("name,cls", PluginRegistry.items())
def test_plugin_fits_and_scores(tmp_path, training_path, name, cls):
    """
    • Fit every registered plugin on the shared five-token corpus.
    • Ensure score() returns a *finite* float for an in-vocabulary token
      (i.e. the plugin is really producing numbers, not just -inf).
    """
    # Minimal viable config for the plugin
    order = max(cls.order_min or 1, 1)
    cfg = Config.default(
        ngram_order=order,
        boundary_mode="none",  # simplest pad behaviour
        neighbourhood_mode="full",  # harmless default for non-neighbourhood models
    )

    # Write a throw-away CSV that contains at least one ‘a’ phoneme
    csv = tmp_path / "tiny.csv"
    csv.write_text("a,1\n")

    model = cls(cfg)
    model.fit(Corpus(csv, cfg))

    score = model.score(["a"])
    assert isinstance(score, float)
    # Reject NaN and the explicit fallback –inf used throughout the codebase
    assert not math.isnan(score)
    assert score != float("-inf")
