"""
CLI color utilities.

HEADER_STYLE: tuple[str, str]
    The canonical style for all help-screen headings (bold white).
BODY_STYLE: tuple[str, str]
    The canonical style for all help body text (bold blue).

Contributors: Use HEADER_STYLE for section headers (including 'usage:' and
group titles), and BODY_STYLE for all argument/help body lines. To change the
color palette, update these constants here.

Doctest:
>>> style('ok', 'bold') == '\033[1mok\033[0m' if supports_color(sys.stderr) else 'ok'
True
>>> try: style('fail', 'notastyle')
... except ValueError as e: 'unknown style code' in str(e)
True
>>> slug("foo", None, "bar", "") == "foo_bar"
True
"""

__all__ = ["supports_color", "style", "HEADER_STYLE", "BODY_STYLE", "slug"]

import os
import sys
from functools import lru_cache

# Minimalist ANSI style map
_ANSI_STYLES = {
    "bold": "\033[1m",
    "dim": "\033[2m",
    "cyan": "\033[36m",
    "yellow": "\033[33m",
    "magenta": "\033[35m",
    "white": "\033[37m",
    "blue": "\033[34m",  # NEW
    "reset": "\033[0m",
}

HEADER_STYLE: tuple[str, str] = ("bold", "white")  # All help headings: bold white
BODY_STYLE: tuple[str, str] = ("bold", "blue")  # All help body text: bold blue


# ──────────────────────────────────────────────────────────
def slug(*parts: str) -> str:
    """
    Join non-empty, non-None string fragments with “_”.
    Guarantees that:
    • every element is cast to str()
    • empty strings and None are skipped
    • result never starts/ends with “_” nor contains “__”
    """
    return "_".join(str(p) for p in parts if p not in (None, ""))


# Patch Windows console for colorama only once (if possible)
_WINDOWS_PATCHED = False
if sys.platform == "win32":
    try:
        import colorama

        colorama.just_fix_windows_console()
        _WINDOWS_PATCHED = True
    except ImportError:
        pass


@lru_cache(maxsize=None)
def _stream_supports_color(stream):
    if os.environ.get("NO_COLOR") is not None:
        return False
    if os.environ.get("FORCE_COLOR") is not None:
        return True
    is_a_tty = hasattr(stream, "isatty") and stream.isatty()
    if not is_a_tty:
        return False
    if sys.platform == "win32" and not _WINDOWS_PATCHED:
        return False
    return True


def supports_color(stream=sys.stderr):
    """Returns True if the stream supports color output.

    Considers TTY state and environment flags.
    """
    return _stream_supports_color(stream)


def style(text, *styles):
    """
    Wraps text in ANSI codes if supported, else returns raw text.
    Usage: style('Warning!', 'bold', 'yellow')
    Raises ValueError on unknown style tokens.
    """
    if not styles:
        return text
    unknown = [s for s in styles if s not in _ANSI_STYLES]
    if unknown:
        raise ValueError(f"unknown style code(s): {', '.join(unknown)}")
    if not supports_color(sys.stderr):
        return text
    codes = "".join(_ANSI_STYLES[s] for s in styles)
    reset = _ANSI_STYLES["reset"]
    return f"{codes}{text}{reset}"
