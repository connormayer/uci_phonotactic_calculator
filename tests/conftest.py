# tests/conftest.py
"""
Make the in-tree `src` package importable when the project has not yet
been installed with `pip install -e .`.
"""

import csv
import sys
import textwrap
from pathlib import Path

import pytest

# Set up the import path before importing from the package
PROJECT_ROOT = Path(__file__).resolve().parents[1]  # …/uci_phonotactic_calculator
SRC_PATH = PROJECT_ROOT / "src"  # …/uci_phonotactic_calculator/src

# Keep the empty-string entry at index 0 (CWD) but ensure local sources
# shadow anything that might be installed globally under the same name.
if SRC_PATH.is_dir() and str(SRC_PATH) not in sys.path:
    sys.path.insert(1, str(SRC_PATH))

# Now we can import from the package
from uci_phonotactic_calculator.core.config import Config  # noqa: E402


@pytest.fixture(scope="session")
def training_path() -> Path:
    """Absolute path to the shared unit-test corpus."""
    return Path(__file__).with_suffix("").parent / "fixtures" / "unit_training_data.txt"


@pytest.fixture
def training_tokens(training_path):
    """List[Tuple[List[str], float]] parsed tokens + freq for easy reuse."""
    with open(training_path, encoding="utf-8") as fh:
        reader = csv.reader(fh)
        return [(row[0].split(" "), float(row[1])) for row in reader]


@pytest.fixture
def unique_sounds(training_tokens):
    """Alphabet + boundary symbol used by legacy explicit-value tests."""
    return sorted({ph for tok, _ in training_tokens for ph in tok}) + ["#"]


@pytest.fixture
def config_factory():
    """
    Helper -> Config.default(**overrides)

    Example:
        cfg = config_factory(ngram_order=2, boundary_mode="none")
    """
    return lambda **kw: Config.default(**kw)


@pytest.fixture
def tiny_csv(tmp_path):
    p = tmp_path / "tiny.csv"
    p.write_text(
        textwrap.dedent(
            """
        a,1
    """
        )
    )
    return p


# End of tests/conftest.py
