"""
Unit tests for the Neighbourhood plugin.

Focuses on validating the integration with neighborhood mode functions.
"""

from uci_phonotactic_calculator.core.config import Config
from uci_phonotactic_calculator.plugins.neighbourhood import NeighbourhoodModel

# Constants for mode names
FULL_MODE = "full"
SUBSTITUTION_ONLY_MODE = "substitution_only"


def test_neighbourhood_model_integration():
    """
    Test that NeighbourhoodModel correctly integrates with registry functions.
    This tests the core functionality without relying on complex corpus handling.
    """
    # Create a simple config
    cfg = Config.default(neighbourhood_mode=FULL_MODE)
    model = NeighbourhoodModel(cfg)

    # Manually set up the model's internal state
    model.training_set = {
        ("p", "a", "t", "a"),
        ("p", "a", ".", "t", "a"),
        ("p", ".", "t", "a"),
        ("p", "a", "t"),
    }
    model.alphabet = {"p", "a", "t", "."}
    model.sound_index = [".", "a", "p", "t"]

    # Test that we can find neighbors for a target
    target = ("p", "a", "t", "a")
    neighbors_count = model._count_neighbors(target)

    # We should find the 3 neighbors from our training set
    # ('p', 'a', '.', 't', 'a'), ('p', '.', 't', 'a'), ('p', 'a', 't')
    assert neighbors_count == 3

    # Test substitution only mode
    cfg = Config.default(neighbourhood_mode=SUBSTITUTION_ONLY_MODE)
    model = NeighbourhoodModel(cfg)
    model.training_set = {
        ("p", "a", "t", "a"),
        ("p", "a", ".", "t", "a"),
        ("p", ".", "t", "a"),
        ("p", "a", "t"),
    }
    model.alphabet = {"p", "a", "t", "."}
    model.sound_index = [".", "a", "p", "t"]

    # The neighbors calculation depends on two things:
    # 1. The substitution_only neighborhood function generating all possible
    #    substitutions
    # 2. How many of those generated neighbors are actually in our training_set
    #
    # In our test data, only one neighbor from our training_set matches a substitution
    # - ('p', '.', 't', 'a') is a valid substitution of ('p', 'a', 't', 'a')
    assert model._count_neighbors(target) == 1

    # Empty input should find 0 neighbors
    assert model._count_neighbors(()) == 0


def test_multi_char_phonemes():
    """Test with multi-character phonemes"""
    cfg = Config.default(neighbourhood_mode=FULL_MODE)
    model = NeighbourhoodModel(cfg)

    # Manually set up the model with multi-char phonemes
    model.training_set = {
        ("ch", "a", "t"),
        ("ch", ".", "a", "t"),
        ("ch", "a", ".", "t"),
        ("ch", "a", "t", "."),
    }
    model.alphabet = {"ch", "a", "t", "."}
    model.sound_index = [".", "a", "ch", "t"]

    # The target 'chat' should have 3 neighbors
    assert model._count_neighbors(("ch", "a", "t")) == 3
