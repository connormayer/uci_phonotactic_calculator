import argparse

from .utils import BODY_STYLE, HEADER_STYLE, style


class ColourHelp(argparse.RawDescriptionHelpFormatter):
    def start_section(self, heading):
        # Colour section headings
        heading = style(heading, *HEADER_STYLE)
        super().start_section(heading)

    def format_usage(self):
        """
        Return a multi-line usage string where every optional flag appears
        on its own line, e.g.
            usage: main.py
                   [-h]
                   [--no-color]
                   ...
                   train_file test_file output_file
        """
        usage = super().format_usage().lstrip("usage:").strip()
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

    def _format_action(self, action):
        parts = super()._format_action(action)
        # Every line inside a section gets the unified body style
        return style(parts, *BODY_STYLE)

    def _get_help_string(self, action):
        help_str = action.help
        if "%(default)" in help_str:
            default_str = str(action.default)
            help_str = help_str.replace("%(default)s", style(default_str, *BODY_STYLE))
        return help_str
