import argparse
from .config import NeighbourhoodMode, ProbMode, AggregateMode, WeightMode

from .cli_utils import style, HEADER_STYLE, BODY_STYLE
from .cli_formatter import ColourHelp

def build_parser() -> argparse.ArgumentParser:
    # Banner: header uses HEADER_STYLE, bullets must keep BODY_STYLE for visual consistency with help body text.
    header = style("Extensible n-gram scorer Supports:", *HEADER_STYLE)
    bullets = [
        "  All model/configuration variants in a grid",
        "  --model to force a single plugin",
        "  --ngram-order (≥1, default 2) to select n-gram order",
        "  Zero required flags: python -m src.ngram_calculator train.csv test.csv out.csv",
    ]
    banner = "\n".join(
        [header] +
        [style(line, *BODY_STYLE) for line in bullets]
    )
    parser = argparse.ArgumentParser(
        description=banner,
        formatter_class=ColourHelp,
    )
    parser.add_argument(
        "--no-color",
        action="store_true",
        help="Disable coloured output in help and errors."
    )

    # Core files group
    core = parser.add_argument_group(style("Core files", *HEADER_STYLE))
    core.add_argument("train_file", help="Path to the training corpus CSV")
    core.add_argument("test_file", help="Path to the test corpus CSV")
    core.add_argument("output_file", help="Path to write the scored output CSV")

    # Model selection group
    model = parser.add_argument_group(style("Model selection", *HEADER_STYLE))
    model.add_argument(
        "--model",
        metavar="NAME",
        default=None,
        help="Name of any registered model plugin (run --list-models to view)."
    )
    model.add_argument(
        "--list-models",
        action="store_true",
        help="Print all registered model plugin names and exit."
    )
    model.add_argument(
        "--position-strategy",
        choices=["absolute", "relative", "none"],
        default=None,
        help="absolute│relative│none — omit the flag for classic n-gram."
    )
    model.add_argument(
        "--neighbourhood-mode",
        choices=[m.value for m in NeighbourhoodMode],
        default=NeighbourhoodMode.FULL.value,
        help="Which edit-distance operations define a neighbour (default: full)."
    )
    model.add_argument(
        "--weight-mode",
        choices=[m.value for m in WeightMode],
        default=WeightMode.NONE.value,
        help="Token weighting: unw (1.0), raw frequency, or log frequency"
    )

    # Scoring flags group
    scoring = parser.add_argument_group(style("Scoring flags", *HEADER_STYLE))
    def build_aggregate_help():
        return ("Aggregation strategy:\n"
                "  " + "\n  ".join(f"{m.value} – {m.name.replace('_', ' ').lower()}" for m in AggregateMode))
    scoring.add_argument(
        "--aggregate",
        type=AggregateMode,
        choices=list(AggregateMode),
        default=AggregateMode.LOG_PRODUCT,
        help=build_aggregate_help()
    )
    scoring.add_argument(
        "--smoothing",
        action="store_true",
        help="Apply add-one smoothing",
    )
    scoring.add_argument(
        "--no-boundaries",
        action="store_true",
        help="Omit boundary symbols in n-gram generation",
    )

    # Positional n-gram options group
    pos = parser.add_argument_group(style("Positional n-gram options", *HEADER_STYLE))
    pos.add_argument(
        "-n", "--ngram-order",
        type=int, default=2, metavar="N",
        help="Order of the n-gram model (≥ 1, default 2)"
    )
    group = pos.add_mutually_exclusive_group()
    group.add_argument(
        "--prob-mode",
        type=ProbMode,
        choices=list(ProbMode),
        help="Built-in transforms: joint | conditional",
    )
    group.add_argument(
        "--prob-transform",
        metavar="NAME",
        help="Name of a custom probability transform registered at runtime",
    )
    return parser
