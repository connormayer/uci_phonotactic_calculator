"""
src/utils/progress_base.py — Base Progress abstract class and implementations

This module provides an abstract base class for progress tracking that can be
implemented by different frontend interfaces like Rich (CLI) and Gradio (Web UI).

The core interface is designed to be minimal but complete:
1. Context manager pattern (__enter__, __exit__)
2. Task management (add_task, update)

Implementations are provided for:
- RichProgress: The default implementation using the Rich library for CLI
- GradioProgress: Web UI progress using Gradio's progress component
"""

from abc import ABC, abstractmethod
from contextlib import nullcontext
from os import environ
from typing import ContextManager

from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)


class BaseProgress(ABC):
    @abstractmethod
    def __enter__(self) -> "BaseProgress":
        """Enter the progress context."""
        pass

    @abstractmethod
    def __exit__(self, exc_type, exc, tb):
        """Exit the progress context."""
        pass

    @abstractmethod
    def add_task(self, description: str, total: int | None = None) -> int:
        """Add a new task to the progress tracker."""
        pass

    @abstractmethod
    def update(self, task_id: int, advance: int = 1):
        """Update progress for a specific task."""
        pass


class RichProgress(BaseProgress):
    """
    Rich implementation of the BaseProgress interface.
    Provides a nice progress bar in the terminal.
    """

    def __init__(self, enabled: bool = True):
        self.enabled = enabled and not self._env_no_progress()
        self._progress = self._make() if self.enabled else None

    def __enter__(self):
        if self.enabled:
            self._progress.__enter__()
        return self

    def __exit__(self, exc_type, exc, tb):
        if self.enabled:
            self._progress.__exit__(exc_type, exc, tb)

    def add_task(self, description: str, total: int | None = None) -> int:
        if not self.enabled:
            return 0
        return self._progress.add_task(description, total=total)

    def update(self, task_id: int, advance: int = 1):
        if not self.enabled:
            return
        self._progress.update(task_id, advance=advance)

    def _make(self) -> Progress:
        """Create a Rich Progress instance with standard columns."""
        return Progress(
            SpinnerColumn(),
            "[progress.description]{task.description}",
            BarColumn(),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            transient=True,  # erase on completion
        )

    def _env_no_progress(self) -> bool:
        """Check if progress is disabled via environment variable."""
        val = environ.get("NO_PROGRESS", "").lower()
        return val not in ("", "0", "false")


class GradioProgress(BaseProgress):
    """
    Gradio implementation of the BaseProgress interface.
    Streams status into the Gradio UI.
    """

    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self._g_prog = None  # callable returned by gr.Progress
        self._cm = None  # context-manager object (old API) or None
        self._tasks: dict[int, list[int]] = {}  # task_id -> [current, total]

    def __enter__(self):
        if self.enabled:
            # Lazy import so CLI users don't need Gradio installed
            import gradio as gr

            prog = gr.Progress()  # may or may not be a context-manager

            if hasattr(prog, "__enter__"):  # old API (≤ 4.2)
                self._cm = prog  # remember to close it later
                self._g_prog = prog.__enter__()  # callable update fn
            else:  # new API (≥ 4.3)
                self._cm = None  # nothing to close
                self._g_prog = prog  # prog itself is callable
        return self

    def __exit__(self, exc_type, exc, tb):
        cm = getattr(self, "_cm", None)
        if cm is not None:
            cm.__exit__(exc_type, exc, tb)

    def add_task(self, description: str, total: int | None = None) -> int:
        task_id = len(self._tasks) + 1
        self._tasks[task_id] = [0, total or 0]
        if self._g_prog is not None:
            # The callable has set_description() only on Gradio ≥4.3
            if hasattr(self._g_prog, "set_description"):
                self._g_prog.set_description(description)
            # Some Gradio versions expect an iterable to be registered
            # before numeric updates; gracefully ignore that edge-case.
            try:
                self._g_prog(0, total or 0)
            except IndexError:  # gradio.helpers.Progress.iterables[-1]
                pass
        return task_id

    def update(self, task_id: int, advance: int = 1):
        cur, tot = self._tasks[task_id]
        cur += advance
        self._tasks[task_id][0] = cur
        if self._g_prog is not None:
            self._g_prog(cur, tot)


def progress(enabled: bool = True) -> ContextManager:
    """
    Factory function to create a progress bar (or a no-op if disabled).
    Returns a RichProgress by default.

    Progress is suppressed if either:
    - enabled is False, or
    - the environment variable NO_PROGRESS is set to '1', 'true',
      or any non-empty non-false value (case-insensitive).
    """
    if enabled:
        return RichProgress(enabled)
    return nullcontext()  # type: ignore[return-value]


__all__ = ["BaseProgress", "RichProgress", "GradioProgress", "progress"]
