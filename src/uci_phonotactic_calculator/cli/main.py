# src/main.py — CLI entry point for the extensible n-gram scorer.
import argparse
import csv
import os
import sys
from collections import OrderedDict
from functools import lru_cache

from ..core.config import Config
from ..core.corpus import (
    Corpus,  # Ensure Corpus is available in all scopes
)
from ..core.filter_aliases import ALIASES, canonical
from ..core.variants import all_variants
from ..plugins.core import (
    PluginRegistry,
    get_model,  # Ensure get_model is available in all scopes
)
from ..plugins.core import discover_models as _discover
from ..utils.progress import (
    progress,  # Centralized progress bar logic
)
from .parser import build_parser

# Imported once here – never re-import inside main(), or it will mask the global.


def _matches_filters(cfg: Config, filters: dict[str, str]) -> bool:
    for key, want in filters.items():
        got = getattr(cfg, key, None)
        if isinstance(got, bool):
            got = "true" if got else "false"
        else:
            got = str(got).lower()
        if got != want:
            return False
    return True


# Ensure all plugins are loaded so --model choices/default stay in sync
_discover()


def parse_cfg(args) -> Config:
    """
    Build a Config from parsed arguments, applying overrides to defaults.
    """
    chosen = args.prob_transform or args.prob_mode or "conditional"
    count_strategy = args.count_strategy
    return Config.default(
        prob_mode=chosen,
        aggregate_mode=args.aggregate,
        smoothing_scheme=args.smoothing_scheme,
        weight_mode=args.weight_mode,
        boundary_mode=args.boundary_mode,
        ngram_order=args.ngram_order,
        position_strategy=args.position_strategy,
        neighbourhood_mode=args.neighbourhood_mode,
        count_strategy=count_strategy,
    )


def parse_filters(args) -> dict[str, str]:
    """
    Parse and validate the --filter arguments from CLI args.
    - Accepts None or an empty list and returns an empty dict.
    - Validates each token contains '=', strips whitespace, lower-cases keys/values.
    - Aliases (e.g. aggregate→aggregate_mode) are normalized automatically.
    - Unknown keys (not present on Config) raise argparse.ArgumentTypeError.
    - Boolean strings (true/1, false/0) are normalized to 'true'/'false'.
    Returns a dict[str, str].
    """
    filters = {}
    filter_args = getattr(args, "filter", None)
    if not filter_args:
        return filters
    for token in filter_args:
        if "=" not in token:
            raise argparse.ArgumentTypeError(
                f"Invalid filter: '{token}'. Must be KEY=VAL."
            )
        key, val = token.split("=", 1)
        key_raw = key.strip()  # keep original case for error text
        key = canonical(key_raw)
        val = val.strip().lower()
        # Validate key against Config fields
        if not hasattr(Config(), key):
            valid = sorted(set(list(vars(Config()).keys()) + list(ALIASES.keys())))
            raise argparse.ArgumentTypeError(
                f"Unknown filter key: '{key_raw}'. Valid keys: {', '.join(valid)}"
            )
        if val in ("true", "1"):
            val = "true"
        elif val in ("false", "0"):
            val = "false"
        filters[key] = val
    return filters


def _run_legacy(args, parser):
    # Only the 16 canonical 2018 variants – no grid search!
    from ..core.variants import legacy_variants

    variants = legacy_variants()

    # ── NEW: Neighbourhood-density (full) column ─────────────────────
    from ..core.variants import Variant

    nh_cfg = Config.default(
        neighbourhood_mode="full",
        boundary_mode="none",  # pads are irrelevant for Levenshtein
    )
    nh_header = get_model("neighbourhood").header(nh_cfg)
    variants.append(Variant(nh_header, "neighbourhood", nh_cfg, None))

    headers = [v.header for v in variants]
    header_to_variant = {v.header: v for v in variants}

    @lru_cache(maxsize=None)
    def get_cached_model(header: str):
        var = header_to_variant[header]
        corpus = Corpus(args.train_file, var.cfg)
        model = get_model(var.model_name)(var.cfg)
        model.fit(corpus)
        return model

    test_corpus = Corpus(args.test_file, variants[0].cfg)
    if not test_corpus.tokens:
        parser.error(f"Test file ‘{args.test_file}’ contains no token rows.")

    rows = []
    for tok in test_corpus.tokens:
        row = OrderedDict()
        row["word"] = " ".join(tok)
        row["word_len"] = len(tok)
        for header in headers:
            m = get_cached_model(header)
            score = m.score(tok)
            row[header] = score if score != float("-inf") else float("nan")
        rows.append(row)

    fieldnames = ["word", "word_len"] + headers
    try:
        with open(args.output_file, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames, lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
    except PermissionError as e:
        print(
            f"Permission denied: cannot write to '{args.output_file}': {e}",
            file=sys.stderr,
        )
        sys.exit(1)
    except OSError as e:
        print(f"Error opening output file '{args.output_file}': {e}", file=sys.stderr)
        sys.exit(1)
    sys.exit(0)


def main():
    """
    Parse command-line arguments and execute one of three modes:
    (1) single model, (2) full variant grid, or (3) filtered grid via --filter.
    Trains every specified n-gram variant, scores the test corpus, and writes a unified
    CSV.
    """
    import argparse
    import sys

    # If --help or -h is present, show the full help from the main parser and exit
    if "--help" in sys.argv or "-h" in sys.argv:
        parser = build_parser()
        parser.add_argument(
            "--list-data",
            action="store_true",
            help="List all available packaged data files and exit",
        )
        parser.add_argument(
            "--use-demo-data",
            action="store_true",
            help=(
                "Use demo data files (english.csv, english_test_data.csv) as "
                "train/test "
                "if not provided"
            ),
        )
        parser.parse_args()  # This will print help and exit
        return

    # First check for --use-demo-data and --list-data flags
    # This is a completely different approach - create a custom parser just to detect
    # these flags
    preparser = argparse.ArgumentParser(add_help=False)
    preparser.add_argument("--list-data", action="store_true")
    preparser.add_argument("--use-demo-data", action="store_true")
    # Allow any other args
    preparser.add_argument("args", nargs=argparse.REMAINDER)
    prelim, _ = preparser.parse_known_args()

    # Handle special flags early
    if prelim.list_data:
        from . import demo_data as cli_demo_data

        cli_demo_data.list_data_files()
        sys.exit(0)

    # Now create the real parser for all arguments
    parser = build_parser()
    parser.add_argument(
        "--list-data",
        action="store_true",
        help="List all available packaged data files and exit",
    )
    parser.add_argument(
        "--use-demo-data",
        action="store_true",
        help=(
            "Use demo data files (english.csv, english_test_data.csv) as train/test "
            "if not provided"
        ),
    )

    # Important: if --use-demo-data is specified, make train_file and test_file optional
    if prelim.use_demo_data:
        # Get the action objects for positional args and make them optional
        actions = parser._actions
        for action in actions:
            if action.dest in ["train_file", "test_file"]:
                action.nargs = "?"
                action.required = False

    # Parse the real arguments
    args = parser.parse_args()

    # If --use-demo-data is specified, fill in missing positionals
    if args.use_demo_data:
        from .demo_data import get_demo_paths

        train, test = get_demo_paths()
        if not args.train_file:
            args.train_file = train
        if not args.test_file:
            args.test_file = test

    if getattr(args, "list_filters", False):
        from ..core.filter_aliases import ALIASES

        core_keys = [k for k in Config.__dataclass_fields__ if not k.startswith("_")]
        print("Core keys:")
        for k in sorted(core_keys):
            print(f"  {k}")
        print("\nAliases:")
        for k, v in sorted(ALIASES.items()):
            print(f"  {k} → {v}")
        sys.exit(0)
    from . import utils as cli_utils

    if args.no_color or not cli_utils.supports_color(sys.stderr):
        cli_utils.style = lambda t, *_, **__: t  # type: ignore[assignment]
    # Patch: convert string 'none' to None for position_strategy to avoid
    # DeprecationWarning
    if (
        hasattr(args, "position_strategy")
        and isinstance(args.position_strategy, str)
        and args.position_strategy.lower() == "none"
    ):
        args.position_strategy = None
    filters = parse_filters(args)

    # Mode resolution
    run_all = bool(getattr(args, "all", False))
    run_single = bool(getattr(args, "model", None))
    legacy_mode = not run_all and not run_single

    # --legacy deprecation check
    if getattr(args, "legacy", False):
        parser.error(
            (
                "--legacy is deprecated: the 16-column output is now the default; "
                "simply "
                "drop the flag."
            )
        )

    if legacy_mode:
        _run_legacy(args, parser)  # never returns

    # Invalid combinations
    if run_all and run_single:
        parser.error("Cannot use --all together with --model.")
    if legacy_mode and filters:
        parser.error(
            "--filter may not be used with the default 16-column legacy output mode."
        )

    if args.list_models:
        # Only print primary model keys
        print(", ".join(sorted(set(PluginRegistry))))
        sys.exit(0)
    if args.model:
        try:
            get_model(args.model)
        except KeyError as exc:
            parser.error(str(exc))
    cfg = parse_cfg(args)

    @lru_cache(maxsize=None)
    def get_model_instance(
        model_name: str,
        smoothing_scheme: str,
        weight_mode: str,
        boundary_mode: str,
        prob_mode: str,
        aggregate_mode: str,
        ngram_order: int,
        *,
        position_strategy: str | None = None,
        count_strategy: str = "ngram",
    ):
        """
        Build, train, and cache one variant of the model (positional flag removed).
        """
        cfg_variant = Config.default(
            smoothing_scheme=smoothing_scheme,
            weight_mode=weight_mode,
            boundary_mode=boundary_mode,
            prob_mode=prob_mode,
            aggregate_mode=aggregate_mode,
            ngram_order=ngram_order,
            position_strategy=position_strategy,
            count_strategy=count_strategy,
        )
        train_corpus = Corpus(args.train_file, cfg_variant)
        if not train_corpus.tokens:
            parser.error(f"Training file ‘{args.train_file}’ contains no token rows.")
        model = get_model(model_name)(cfg_variant)
        model.fit(train_corpus)
        return model

    # Load test corpus once (we only need its tokens)
    test = Corpus(
        args.test_file,
        cfg,
    )
    if not test.tokens:
        parser.error(f"Test file ‘{args.test_file}’ contains no token rows.")

    # Prepare output directory
    out_dir = os.path.dirname(args.output_file) or "."
    try:
        os.makedirs(out_dir, exist_ok=True)
    except OSError as e:
        print(f"Error: unable to create directory '{out_dir}': {e}", file=sys.stderr)
        sys.exit(1)

    base_fields = ("word", "word_len")
    if run_single:
        # Derive the single header and variant descriptor up front
        variant_cfg = parse_cfg(args)
        single_header = get_model(args.model).header(variant_cfg)
        variant_headers = [single_header]
        if len(set(variant_headers)) != len(variant_headers):
            dup = [h for h in variant_headers if variant_headers.count(h) > 1][0]
            parser.error(f"Internal error: duplicate CSV header ‘{dup}’.")
        SingleVariant = type(
            "SingleVariant",
            (),
            {
                "model_name": args.model,
                "cfg": variant_cfg,
                "header": single_header,
            },
        )
        single_variant = SingleVariant()
        fieldnames = [*base_fields, *variant_headers]
        rows: list[OrderedDict[str, float | str | int]] = []
        # Progress bar for training loop
        variants_to_run = [single_variant]
    elif run_all:
        # Multi-model: build all headers up front
        variant_headers = [v.header for v in all_variants(test, filters)]
        if len(set(variant_headers)) != len(variant_headers):
            dup = [h for h in variant_headers if variant_headers.count(h) > 1][0]
            parser.error(f"Internal error: duplicate CSV header ‘{dup}’.")
        fieldnames = [*base_fields, *variant_headers]
        rows: list[OrderedDict[str, float | str | int]] = []
        # Progress bar for training loop
        variants_to_run = list(all_variants(test, filters))
    else:
        # Neither single nor all: do nothing (legacy already handled above)
        return

    with progress(args.progress) as bar:
        train_task = None
        if bar is not None:
            train_task = bar.add_task("Training", total=len(variants_to_run))
        for variant in variants_to_run:
            # Build and fit model for each variant
            get_model_instance(
                variant.model_name,
                variant.cfg.smoothing_scheme,
                variant.cfg.weight_mode,
                variant.cfg.boundary_mode,
                variant.cfg.prob_mode,
                variant.cfg.aggregate_mode,
                variant.cfg.ngram_order,
                position_strategy=variant.cfg.position_strategy,
                count_strategy=variant.cfg.count_strategy,
            )
            if bar is not None:
                bar.update(train_task, advance=1)

    # Progress bar for scoring loop
    with progress(args.progress) as bar:
        score_task = None
        if bar is not None:
            score_task = bar.add_task("Scoring", total=len(test.tokens))
        for tok in test.tokens:
            entry = OrderedDict((f, None) for f in fieldnames)
            entry["word"] = " ".join(tok)
            entry["word_len"] = len(tok)
            if run_single:
                model_cls = get_model(single_variant.model_name)
                if not model_cls.supports(single_variant.cfg):
                    if bar is not None:
                        bar.update(score_task, advance=1)
                    continue
                if not _matches_filters(single_variant.cfg, filters):
                    if bar is not None:
                        bar.update(score_task, advance=1)
                    continue
                m = get_model_instance(
                    single_variant.model_name,
                    single_variant.cfg.smoothing_scheme,
                    single_variant.cfg.weight_mode,
                    single_variant.cfg.boundary_mode,
                    single_variant.cfg.prob_mode,
                    single_variant.cfg.aggregate_mode,
                    single_variant.cfg.ngram_order,
                    position_strategy=single_variant.cfg.position_strategy,
                    count_strategy=single_variant.cfg.count_strategy,
                )
                score = m.score(tok)
                entry[single_variant.header] = (
                    score if score != float("-inf") else float("nan")
                )
            else:
                for variant in all_variants(test, filters):
                    model_cls = get_model(variant.model_name)
                    if not model_cls.supports(variant.cfg):
                        continue  # belt-and-suspenders
                    # strategy_name is not needed, position_strategy is passed directly
                    # to get_model_instance below
                    m = get_model_instance(
                        variant.model_name,
                        variant.cfg.smoothing_scheme,
                        variant.cfg.weight_mode,
                        variant.cfg.boundary_mode,
                        variant.cfg.prob_mode,
                        variant.cfg.aggregate_mode,
                        variant.cfg.ngram_order,
                        position_strategy=variant.cfg.position_strategy,
                        count_strategy=variant.cfg.count_strategy,
                    )
                    score = m.score(tok)
                    entry[variant.header] = (
                        score if score != float("-inf") else float("nan")
                    )
            rows.append(entry)
            if bar is not None:
                bar.update(score_task, advance=1)

    # Write out a single CSV: first two cols + one per variant header
    try:
        with open(args.output_file, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames, lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
    except PermissionError as e:
        print(
            f"Permission denied: cannot write to '{args.output_file}': {e}",
            file=sys.stderr,
        )
        sys.exit(1)
    except OSError as e:
        print(f"Error opening output file '{args.output_file}': {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        from .utils import style

        print(style("\n[Interrupted by user]", "yellow", "bold"), file=sys.stderr)

__all__ = ["main"]


# End of src/main.py
