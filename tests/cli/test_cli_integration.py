import csv
import sys

# ------------------------------------------------------------------ #
# helpers
# ------------------------------------------------------------------ #
from pathlib import Path
from typing import Any, Sequence


def _run_cli(monkeypatch: Any, tmp_path: Path, argv_tail: Sequence[str]) -> Path:
    """
    Invoke cli.main with a patched sys.argv.
    Always returns the Path of the output CSV produced.
    """
    out = tmp_path / "scores.csv"
    argv = ["ngram_calculator", "--use-demo-data", *argv_tail, str(out)]
    monkeypatch.setattr(sys, "argv", argv, raising=False)

    # import inside function so pytest collection never runs cli.main
    from uci_phonotactic_calculator.cli.main import main as cli_main

    try:
        cli_main()
    except SystemExit as exc:
        # Exit-code 0 == success; any non-zero should surface as a test failure.
        if exc.code not in (0, None):
            raise
    return out


# ------------------------------------------------------------------ #
# 1) legacy 16-column run (+neighbourhood) ---------------------------
# ------------------------------------------------------------------ #
def test_cli_legacy(monkeypatch: Any, tmp_path: Path) -> None:
    out_csv = _run_cli(monkeypatch, tmp_path, [])
    assert out_csv.exists()

    header = next(csv.reader(out_csv.open()))
    # word, word_len + 16 legacy + neighbourhood = 19 columns
    assert header[:2] == ["word", "word_len"]
    assert len(header) == 19


# ------------------------------------------------------------------ #
# 2) explicit model, trigram order ----------------------------------
# ------------------------------------------------------------------ #
def test_cli_single_model_trigram(monkeypatch: Any, tmp_path: Path) -> None:
    out_csv = _run_cli(
        monkeypatch,
        tmp_path,
        ["--model", "ngram", "--ngram-order", "3"],
    )
    header = next(csv.reader(out_csv.open()))
    assert "ngram_n3" in header[2]  # first data column reflects n=3
    # only one data column expected
    assert len(header) == 3
