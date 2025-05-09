# ngram_calculator.py — Legacy CLI shim that delegates to main.py.
import sys
from typing import Sequence, Mapping
from .main import main

def run(
    train_file: str,
    test_file: str,
    output_file: str,
    *,
    model: str | None = None,
    run_all: bool = False,
    filters: Mapping[str, str] | None = None,
    show_progress: bool = False,
    extra_args: Sequence[str] | None = None,
) -> None:
    """
    Programmatic wrapper around the CLI so other Python code
    (e.g. Django) can call the scorer without spawning a subprocess.

    Parameters
    ----------
    train_file, test_file, output_file : paths to the corpus CSVs
    model       : same as ``--model``
    run_all     : same as ``--all``
    filters     : dict like {'weight_mode':'raw'}  → ``--filter weight_mode=raw``
    show_progress : keep rich progress bars (default False for web)
    extra_args  : any extra CLI flags to append verbatim
    """
    argv = ["ngram_calculator", train_file, test_file, output_file]
    if not show_progress:
        argv.append("--no-progress")
    if model:
        argv += ["--model", model]
    if run_all:
        argv.append("--all")
    if filters:
        for k, v in filters.items():
            argv += ["--filter", f"{k}={v}"]
    if extra_args:
        argv.extend(extra_args)

    # Temporarily replace sys.argv and invoke the real CLI entry-point
    old_argv = sys.argv
    try:
        sys.argv = argv
        from .main import main as _main
        _main()
    except SystemExit as exc:          # main() calls sys.exit() freely
        if exc.code not in (0, None):  # re-raise only on failure exit codes
            raise
    finally:
        sys.argv = old_argv

if __name__ == "__main__":
    main()

# End of src/ngram_calculator.py
