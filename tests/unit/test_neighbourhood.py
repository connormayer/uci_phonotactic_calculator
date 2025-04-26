"""
Unit tests for the Neighbourhood plugin, focusing on NeighbourhoodMode and phoneme tuple handling.
"""
import pytest
import sys
from pathlib import Path
import tempfile

print(f"[DEBUG] sys.path: {sys.path}")
print(f"[DEBUG] __file__: {__file__}")

# Ensure src is on sys.path for direct pytest invocation
if str(Path(__file__).resolve().parents[2] / "src") not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from plugins.neighbourhood import Neighbourhood, NeighbourhoodMode
from config import Config
from corpus import Corpus

# Helper to create a dummy corpus CSV file and load it
@pytest.fixture
def dummy_corpus(tmp_path):
    data = "word,frequency\npa.ta,1\npata,1\np.ta,1\npat,1\n"
    file = tmp_path / "dummy.csv"
    file.write_text(data)
    cfg = Config.default()
    corpus = Corpus(file, cfg, include_boundary=False)
    return corpus

def test_full_mode_counts_all_edits(dummy_corpus):
    cfg = Config.default(neighbourhood_mode=NeighbourhoodMode.FULL)
    model = Neighbourhood(cfg)
    model.fit(dummy_corpus)
    # 'pata' should have as neighbors: 'pa.ta', 'p.ta', 'pat' (all one edit away)
    # Test tuple handling: all words are split into tuples of phonemes
    assert model._count_neighbors(('p', 'a', 't', 'a')) == 3

def test_substitution_only_mode(dummy_corpus):
    cfg = Config.default(neighbourhood_mode=NeighbourhoodMode.SUBSTITUTION_ONLY)
    model = Neighbourhood(cfg)
    model.fit(dummy_corpus)
    # Only substitutions allowed, so only 'pa.ta' and 'p.ta' are one substitution away from 'pata'
    assert model._count_neighbors(('p', 'a', 't', 'a')) == 2

def test_empty_input(dummy_corpus):
    cfg = Config.default(neighbourhood_mode=NeighbourhoodMode.FULL)
    model = Neighbourhood(cfg)
    model.fit(dummy_corpus)
    # No neighbors for empty string
    assert model._count_neighbors(()) == 0

def test_multi_char_phonemes():
    # Test with multi-character phonemes
    data = "word,frequency\nch.a.t,1\ncha.t,1\nch.at,1\nchat,1\n"
    with tempfile.TemporaryDirectory() as tmpdir:
        file = Path(tmpdir) / "dummy.csv"
        file.write_text(data)
        cfg = Config.default()
        corpus = Corpus(file, cfg, include_boundary=False)
        model = Neighbourhood(cfg)
        model.fit(corpus)
        # 'chat' as ('ch','a','t')
        assert model._count_neighbors(('ch', 'a', 't')) == 3
