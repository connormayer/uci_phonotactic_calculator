import argparse

from ..core.registries import registry
from .formatter import ColourHelp
from .utils import BODY_STYLE, HEADER_STYLE, style


def _registry_option(
    group: argparse._ArgumentGroup,
    *,
    flag: str,
    category: str,
    default: str,
    help: str,
    short: str | None = None,
    allow_none: bool = False,
    metavar: str | None = None,  # ← new
):
    choices = sorted(registry(category).keys())
    # always inject the sentinel when requested
    if allow_none and "none" not in choices:
        choices.append("none")
    # guarantee the default is selectable
    display_default = default
    if default is None:
        display_default = "none"
    if display_default not in choices:
        choices.append(display_default)
    # argparse expects all choices as str, and None will break join()
    choices_str = [str(c) if c is not None else "none" for c in choices]
    option_strings = [f"--{flag}"] + ([short] if short else [])
    meta = metavar or "NAME"  # honour caller’s choice
    group.add_argument(
        *option_strings,
        choices=choices_str,
        default=display_default,
        metavar=meta,
        help=f"{help} (choices: {', '.join(choices_str)})",
    )


def _add_extension_flags(parser: argparse.ArgumentParser) -> None:
    from ..core.registries import registry

    ext_group = parser.add_argument_group(style("Extensions", *HEADER_STYLE))
    for _name, ext in registry("cli_ext").items():
        acts = ext.flags()
        for a in acts:
            ext_group._add_action(a)


def build_parser() -> argparse.ArgumentParser:
    # Banner: header uses HEADER_STYLE, bullets must keep BODY_STYLE
    # for visual consistency with help body text.
    header = style("Extensible n-gram scorer Supports:", *HEADER_STYLE)
    bullets = [
        "  Original legacy output (16 + neighbourhood) is now the default",
        "  --all to run the full model/variant grid (was previously default)",
        "  --model to force a single plugin",
        "  --ngram-order (≥1, default 2) to select n-gram order",
        "  --filter KEY=VAL (repeatable) to restrict the variant grid.",
        "  --list-filters to display usable filter keys",
        "  --no-progress to suppress live status",
        "  Zero required flags: python -m src.ngram_calculator "
        "train.csv test.csv out.csv",
    ]
    banner = "\n".join([header] + [style(line, *BODY_STYLE) for line in bullets])
    parser = argparse.ArgumentParser(
        description=banner,
        formatter_class=ColourHelp,
    )
    parser.add_argument(
        "--no-color",
        action="store_true",
        help="Disable coloured output in help and errors.",
    )

    # Core files group
    core = parser.add_argument_group(style("Core files", *HEADER_STYLE))
    core.add_argument(
        "--no-progress",
        dest="progress",
        action="store_false",
        default=True,
        help="Suppress live progress bar/status. Default: show progress bar.",
    )
    core.add_argument("train_file", help="Path to the training corpus CSV")
    core.add_argument("test_file", help="Path to the test corpus CSV")
    core.add_argument("output_file", help="Path to write the scored output CSV")

    # Model selection group
    model = parser.add_argument_group(style("Model selection", *HEADER_STYLE))
    model.add_argument(
        "--model",
        metavar="NAME",
        default=None,
        help="Name of any registered model plugin (run --list-models to view).",
    )
    model.add_argument(
        "--list-models",
        action="store_true",
        help="Print all registered model plugin names and exit.",
    )
    model.add_argument(
        "--list-filters",
        action="store_true",
        help="Print all accepted --filter keys (including aliases) and exit.",
    )
    _registry_option(
        model,
        flag="position-strategy",
        category="position_strategy",
        default=None,
        allow_none=True,
        help="absolute│relative│none",
    )
    _registry_option(
        model,
        flag="neighbourhood-mode",
        category="neighbourhood_mode",
        default="full",
        help="Neighbourhood edit-operation set.",
        metavar="NEIGHBOURHOOD",
    )
    _registry_option(
        model,
        flag="weight-mode",
        category="weight_mode",
        default="none",
        help="Token weighting strategy.",
        metavar="WEIGHT",
    )

    # Scoring flags (now merged)
    _registry_option(
        model,
        flag="aggregate",
        category="aggregate_mode",
        default="prod",
        help="Aggregation strategy (registry ‘aggregate_mode’).",
        metavar="AGGREGATE",
    )
    _registry_option(
        model,
        flag="boundary-scheme",
        category="boundary_scheme",
        default="word",
        help="Symbol to pad tokens with.",
        metavar="BOUNDARY",
    )
    _registry_option(
        model,
        flag="count-strategy",
        category="count_strategy",
        default="ngram",
        help="Counting backend (registry ‘count_strategy’).",
        short="-c",
        metavar="COUNT",
    )
    _registry_option(
        model,
        flag="smoothing-scheme",
        category="smoothing_scheme",
        default="laplace",
        help="Smoothing scheme applied to counts.",
        metavar="SMOOTHING",
    )
    _registry_option(
        model,
        flag="boundary-mode",
        category="boundary_mode",
        default="both",
        allow_none=True,
        help="Padding behaviour: both│prefix│suffix│none",
    )
    model.add_argument(
        "--filter",
        action="append",
        metavar="KEY=VAL",
        help=style(
            "Restrict the grid search to configs whose Config.<KEY> equals <VAL>. "
            "Repeat the flag to combine filters (logical AND). "
            "Examples:  --filter smoothing=true  --filter aggregate=logsumexp "
            "(run --list-filters for details)",
            *BODY_STYLE,
        ),
    )

    # Run-mode group
    runmode = parser.add_argument_group(style("Run-mode", *HEADER_STYLE))

    def legacy_flag(value):
        raise argparse.ArgumentTypeError(
            "--legacy is deprecated: the 16-column output is now the default. "
            "Please remove this flag."
        )

    runmode.add_argument(
        "--legacy",
        action="store_true",
        help=argparse.SUPPRESS,  # Hide from help, but keep for one release
    )
    runmode.add_argument(
        "--all",
        action="store_true",
        help="Run the entire model/variant grid. If omitted, the calculator writes "
        "the original 2018 16-column CSV.",
    )

    # Positional n-gram options group
    pos = parser.add_argument_group(style("Positional n-gram options", *HEADER_STYLE))
    pos.add_argument(
        "-n",
        "--ngram-order",
        type=int,
        default=2,
        metavar="N",
        help="Order of the n-gram model (≥ 1, default 2)",
    )
    group = pos.add_mutually_exclusive_group()
    from ..core.probability import available_transforms

    group.add_argument(
        "--prob-mode", choices=list(available_transforms()), help="Built-in transforms."
    )
    group.add_argument(
        "--prob-transform",
        metavar="NAME",
        help="Name of a custom probability transform registered at runtime",
    )

    # Add Extensions section to help output
    ext_keys = list(registry("cli_ext").keys())
    if ext_keys:
        parser.epilog = (
            (parser.epilog or "")
            + "\n\nExtensions available: "
            + ", ".join(sorted(ext_keys))
        )

    _add_extension_flags(parser)
    # ─────────────────────────────────────────────────────────
    # Give argparse.error() the same colour treatment
    # ─────────────────────────────────────────────────────────
    import sys

    def _styled_error(self, message):
        # usage lines are already colourised by ColourHelp
        self.print_usage(sys.stderr)
        prefix = style("error:", *HEADER_STYLE)
        body = style(message, *BODY_STYLE)
        self.exit(2, f"{prefix} {body}\n")

    # monkey-patch the instance
    parser.error = _styled_error.__get__(parser)
    return parser


if __name__ == "__main__":
    import sys

    parser = build_parser()
    parser.parse_args(sys.argv[1:])  # just trigger the help / parse cycle
