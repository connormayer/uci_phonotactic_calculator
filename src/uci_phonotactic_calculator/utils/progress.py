"""
src/progress.py — Progress helper (Rich default implementation)

DEPRECATED: For custom progress frontends, import from progress_base directly.

Call::     with progress(enabled=True) as p:
               task = p.add_task("Scoring", total=len(items))
               for _ in items: ...

Other modules should only import progress() from here—never rich directly—
so swapping libraries later is trivial.

Note for maintainers:
    Each heavy section should be wrapped in its own `with progress(...)` block.
    Progress contexts auto-close and clean up on exceptions,
    so no extra exception handling is required for progress bars.
"""

from typing import ContextManager

from .progress_base import RichProgress
from .progress_base import progress as base_progress


def progress(enabled: bool = True) -> ContextManager:
    """
    Context manager for a Rich progress bar, or a no-op if disabled.
    Progress is suppressed if either:
      - enabled is False, or
      - the environment variable NO_PROGRESS is set to '1', 'true',
        or any non-empty non-false value (case-insensitive).
    This allows CI, Docker, or scripts to suppress bars without editing command lines.
    """
    return base_progress(enabled)


__all__ = ["progress", "RichProgress"]
