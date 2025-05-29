import math

from uci_phonotactic_calculator.models import aggregate as ag


def test_log_product():
    comps = [math.log(0.1), math.log(0.2)]
    expected = math.log(0.02)
    assert math.isclose(ag.log_product(comps), expected)


def test_logsumexp():
    comps = [math.log(0.1), math.log(0.2)]
    # hand LSE: log(0.1 + 0.2)
    expected = math.log(0.3)
    assert math.isclose(ag.logsumexp(comps), expected)


def test_linear_sum():
    comps = [math.log(0.1), math.log(0.2)]
    expected = math.log(0.3)
    assert math.isclose(ag.linear_sum(comps), expected)


def test_linear_sum_plus1():
    comps = [math.log(0.1), math.log(0.2)]
    expected = 1.0 + 0.3
    assert math.isclose(ag.linear_sum_plus1(comps), expected)


def test_min_max_and_empty():
    comps = [math.log(0.1), math.log(0.2)]
    assert ag.min_val(comps) == min(comps)
    assert ag.max_val(comps) == max(comps)
    assert ag.min_val([]) == float("-inf")
    assert ag.max_val([]) == float("-inf")
