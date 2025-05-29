from uci_phonotactic_calculator.plugins.strategies.position import (
    Absolute,
    Relative,
    get_position_strategy,
)


def test_absolute_strategy():
    strat = Absolute()
    # token_len 4, bigrams → gram_idx 0..3
    buckets = [strat.bucket(i, 4) for i in range(4)]
    assert buckets == [0, 1, 2, 3]
    assert strat.max_buckets(4) == 4


def test_relative_strategy_bigram():
    strat = Relative(n=2)
    # token_len 4, valid bigram positions 0..2
    buckets = [strat.bucket(i, 4) for i in range(3)]
    # right-aligned: last bigram → bucket 0, etc.
    assert buckets == [2, 1, 0]
    assert strat.bucket(3, 4) is None  # idx beyond last valid bigram
    assert strat.max_buckets(4) == 3  # 4-2+1


def test_get_position_strategy_aliases():
    assert get_position_strategy(None) is None
    assert isinstance(get_position_strategy("absolute"), Absolute)
    assert isinstance(get_position_strategy("relative", n=2), Relative)
