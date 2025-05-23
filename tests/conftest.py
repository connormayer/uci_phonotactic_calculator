# tests/conftest.py
"""
Make the in-tree `src` package importable when the project has not yet
been installed with `pip install -e .`.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]  # …/uci_phonotactic_calculator
SRC_PATH = PROJECT_ROOT / "src"  # …/uci_phonotactic_calculator/src

# Keep the empty-string entry at index 0 (CWD) but ensure local sources
# shadow anything that might be installed globally under the same name.
if SRC_PATH.is_dir() and str(SRC_PATH) not in sys.path:
    sys.path.insert(1, str(SRC_PATH))

# End of tests/conftest.py
