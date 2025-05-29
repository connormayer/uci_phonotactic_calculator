"""
Ensure Corpus.generate_ngrams behaves for prefix-only and suffix-only
padding modes (both directions already covered elsewhere).
"""

from uci_phonotactic_calculator.core.corpus import Corpus

TOKEN = ["a", "b", "c"]  # 3-symbol word
PAD = "#"  # default boundary symbol


def _grams(mode):
    return Corpus.generate_ngrams(TOKEN, 2, boundary_mode=mode, boundary=PAD)


def test_prefix_padding():
    grams = _grams("prefix")
    # with n=2 we expect one pad on the left
    assert grams[0] == (PAD, "a")
    assert grams[-1] == ("b", "c")
    assert len(grams) == len(TOKEN)  # 3 bigrams


def test_suffix_padding():
    grams = _grams("suffix")
    # pad appears at the right only
    assert grams[0] == ("a", "b")
    assert grams[-1] == ("c", PAD)
    assert len(grams) == len(TOKEN)  # 3 bigrams
