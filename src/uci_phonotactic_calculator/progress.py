"""
src/progress.py — Rich Progress helper

Call::     with progress(enabled=True) as p:
               task = p.add_task("Scoring", total=len(items))
               for _ in items: ...

Other modules must only import progress() from here—never rich directly—so swapping libraries later is trivial.

Note for maintainers:
    Each heavy section should be wrapped in its own `with progress(...)` block.
    Rich's Progress context auto-closes and cleans up on exceptions, so no extra exception handling is required for progress bars.
"""
from contextlib import nullcontext
from os import environ
from typing import ContextManager

from rich.progress import (
    Progress, SpinnerColumn, BarColumn, MofNCompleteColumn,
    TimeElapsedColumn, TimeRemainingColumn,
)

def _make() -> Progress:
    return Progress(
        SpinnerColumn(),
        "[progress.description]{task.description}",
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        transient=True,          # erase on completion
    )

def _env_no_progress() -> bool:
    val = environ.get("NO_PROGRESS", "").lower()
    return val not in ("", "0", "false")

def progress(enabled: bool = True) -> ContextManager[Progress]:
    """
    Context manager for a Rich progress bar, or a no-op if disabled.
    Progress is suppressed if either:
      - enabled is False, or
      - the environment variable NO_PROGRESS is set to '1', 'true', or any non-empty non-false value (case-insensitive).
    This allows CI, Docker, or scripts to suppress bars without editing command lines.
    """
    if _env_no_progress() or not enabled:
        # Rich still initialises quickly, but a nullcontext avoids any stderr writes.
        return nullcontext()        # type: ignore[return-value]
    return _make()
