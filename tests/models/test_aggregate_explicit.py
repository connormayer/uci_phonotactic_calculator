"""
Explicit algebra checks for the *built-in* aggregation helpers:
    • prod        Σ log p
    • lse, sum    log Σ p
    • min / max   boundary cases
    • sum_plus1   legacy un-logged score
"""

import math

from uci_phonotactic_calculator.models import aggregate as ag

# Two simple probabilities we’ll reuse everywhere
P1, P2 = 0.1, 0.2
LOG_P1, LOG_P2 = math.log(P1), math.log(P2)


def test_log_product():
    """
    Geometrically: log(P₁·P₂)  ==  log P₁ + log P₂
    Numerical:     log(0.1) + log(0.2)  →  log(0.02)
    """
    expected = math.log(P1 * P2)  # log(0.02)
    got = ag.log_product([LOG_P1, LOG_P2])
    assert math.isclose(got, expected)


def test_logsumexp():
    """
    LSE = log( P₁ + P₂ )
    """
    expected = math.log(P1 + P2)  # log(0.3)
    got = ag.logsumexp([LOG_P1, LOG_P2])
    assert math.isclose(got, expected)


def test_linear_sum():
    """
    linear_sum(): convert log→linear, add, take log.
                  Should equal the LSE above.
    """
    expected = math.log(P1 + P2)
    got = ag.linear_sum([LOG_P1, LOG_P2])
    assert math.isclose(got, expected)


def test_linear_sum_plus1():
    """
    Legacy formula: 1 + Σ Pᵢ   (kept in *linear* space)
    """
    expected = 1.0 + (P1 + P2)  # 1.3
    got = ag.linear_sum_plus1([LOG_P1, LOG_P2])
    assert math.isclose(got, expected)


def test_min_max_and_empty():
    comps = [LOG_P1, LOG_P2]

    assert ag.min_val(comps) == min(comps)  # lower log-p
    assert ag.max_val(comps) == max(comps)  # higher log-p
    assert ag.min_val([]) == float("-inf")  # empty → −∞
    assert ag.max_val([]) == float("-inf")
