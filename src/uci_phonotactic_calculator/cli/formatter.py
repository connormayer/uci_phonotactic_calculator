import argparse
from argparse import (
    Action,
    RawDescriptionHelpFormatter,
    _MutuallyExclusiveGroup,
)
from typing import Iterable, Optional

from .utils import BODY_STYLE, HEADER_STYLE, style


class ColourHelp(RawDescriptionHelpFormatter):
    def start_section(self, heading: Optional[str]) -> None:
        # Match parent class signature which accepts None
        # Colour section headings
        styled_heading = style(heading or "", *HEADER_STYLE) if heading else None
        super().start_section(styled_heading)

    def _format_usage(
        self,
        usage: str | None,
        actions: Iterable[Action],
        groups: Iterable[_MutuallyExclusiveGroup],
        prefix: str | None,
    ) -> str:
        # This overrides the private method in the parent class
        """
        Return a multi-line usage string where every optional flag appears
        on its own line, e.g.
            usage: main.py
                   [-h]
                   [--no-color]
                   ...
                   train_file test_file output_file
        """
        # Call the parent method to format the basic usage text
        formatted_usage = super()._format_usage(usage, actions, groups, prefix)
        # Strip the 'usage:' prefix and leading/trailing whitespace
        usage = formatted_usage.lstrip("usage:").strip()
        tokens = []
        chunk = []
        depth = 0
        for tok in usage.split():
            depth += tok.count("[") - tok.count("]")
            chunk.append(tok)
            if depth == 0:
                tokens.append(" ".join(chunk))
                chunk.clear()
        if chunk:
            tokens.append(" ".join(chunk))
        prog, *opts = tokens
        indent = " " * 7
        lines = [f"usage: {prog}"]
        lines += [f"{indent}{t}" for t in opts]
        lines[0] = style("usage:", *HEADER_STYLE) + lines[0][6:]
        return "\n".join(lines) + "\n"

    def _format_action(self, action: argparse.Action) -> str:
        parts = super()._format_action(action)
        # Every line inside a section gets the unified body style
        return style(parts, *BODY_STYLE)

    def _get_help_string(self, action: argparse.Action) -> str:
        help_str = action.help
        if help_str is None:
            return ""
        if "%(default)" in help_str:
            default_str = str(action.default)
            help_str = help_str.replace("%(default)s", style(default_str, *BODY_STYLE))
        return help_str
