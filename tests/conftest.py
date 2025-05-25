"""
tests/conftest.py - Common fixtures for tests.
Provides fixtures for unit and integration tests, including token frequencies,
sound_index (unique sounds), and dummy CSV files for training and testing.

"""

import pytest
from src.io_utils import read_tokens, WORD_BOUNDARY

# Fixtures for unit tests

TRAINING_FILE = "data/unit_test_data/unit_test_training_data.txt"


@pytest.fixture
def token_freqs():
    return read_tokens(TRAINING_FILE)


@pytest.fixture
def sound_index(token_freqs):
    """
    Returns a sorted list of unique sounds from the token frequencies.
    """
    sounds = sorted({sound for token, _ in token_freqs for sound in token})
    return sounds


@pytest.fixture
def sounds_with_boundary(sound_index):
    """
    Returns the sound_index with the word boundary appended.
    """
    return sound_index + [WORD_BOUNDARY]


# Fixtures for integration tests


@pytest.fixture
def dummy_training_data():
    # Simple CSV content: tokens are space-delimited; optional frequency is provided.
    return "a b,1\nc d,2\n"


@pytest.fixture
def dummy_test_data():
    # Simple CSV content for test data.
    return "a b,1\nx y,1\n"


@pytest.fixture
def dummy_train_file(tmp_path, dummy_training_data):
    file = tmp_path / "dummy_train.csv"
    file.write_text(dummy_training_data, encoding="utf-8")
    return str(file)


@pytest.fixture
def dummy_test_file(tmp_path, dummy_test_data):
    file = tmp_path / "dummy_test.csv"
    file.write_text(dummy_test_data, encoding="utf-8")
    return str(file)


# End of conftest.py
