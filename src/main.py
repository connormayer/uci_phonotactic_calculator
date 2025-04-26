"""src/main.py — CLI entry point for the extensible n-gram scorer.

Supports:
- All model/configuration variants in a grid
- --model to force a single plugin
- -n/--ngram-order (≥1, default 2) to select n-gram order
- Zero required flags: python -m src.ngram_calculator train.csv test.csv out.csv
"""

import argparse
import csv
from collections import OrderedDict
from functools import lru_cache
import os
import sys
from typing import Union

from .config import Config, NeighbourhoodMode, ProbMode, AggregateMode, WeightMode
from .corpus import Corpus
from .plugins.core import get_model, PluginRegistry, discover_models as _discover, ALIASES
from .variants import all_variants
# Imported once here – never re-import inside main(), or it will mask the global.

from .plugins.strategies.position import get_position_strategy

# Ensure all plugins are loaded so --model choices/default stay in sync
_discover()


def parse_cfg(args) -> Config:
    """
    Build a Config from parsed arguments, applying overrides to defaults.
    """
    chosen = args.prob_transform or args.prob_mode or ProbMode.CONDITIONAL
    weight_mode = WeightMode(args.weight_mode)
    # Canonicalize position_strategy using Config helper (handles 'none' deprecation)
    position_strategy = Config._normalize_none_string(args.position_strategy)
    neighbourhood_mode = NeighbourhoodMode(args.neighbourhood_mode)
    return Config.default(
        prob_mode      = chosen,
        aggregate_mode = args.aggregate,
        smoothing      = args.smoothing,
        weight_mode    = weight_mode,
        use_boundaries = not args.no_boundaries,
        ngram_order    = args.ngram_order,
        position_strategy = position_strategy,   # NEW
        neighbourhood_mode = neighbourhood_mode,
    )


def main():
    """
    Parse command-line arguments, train every specified n-gram variant,
    score the test corpus, and write a unified CSV.
    """
    parser = argparse.ArgumentParser(description="Extensible n-gram scorer")
    parser.add_argument("train_file",  help="Path to the training corpus CSV")
    parser.add_argument("test_file",   help="Path to the test corpus CSV")
    parser.add_argument("output_file", help="Path to write the scored output CSV")

    parser.add_argument(
        "-n", "--ngram-order",
        type=int, default=2, metavar="N",
        help="Order of the n-gram model (≥ 1, default 2)"
    )

    parser.add_argument(
        "--weight-mode",
        choices=[m.value for m in WeightMode],
        default=WeightMode.NONE.value,
        help="Token weighting: unw (1.0), raw frequency, or log frequency"
    )

    parser.add_argument(
        "--neighbourhood-mode",
        choices=[m.value for m in NeighbourhoodMode],
        default=NeighbourhoodMode.FULL.value,
        help="Which edit-distance operations define a neighbour (default: full)."
    )

    parser.add_argument(
        "--model",
        metavar="NAME",
        default=None,
        help="Name of any registered model plugin (run --list-models to view)."
    )
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="Print all registered model plugin names and exit."
    )
    parser.add_argument(
        "--position-strategy",
        choices=["absolute", "relative", "none"],  # 'none' is deprecated, handled in Config
        default=None,
        help="absolute│relative│none — omit the flag for classic n-gram."
    )
    # NOTE: argparse delivers None (not the string 'None') when this flag is omitted.
    group = parser.add_mutually_exclusive_group()
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
    parser.add_argument(
        "--aggregate",
        type=AggregateMode,
        choices=list(AggregateMode),
        default=AggregateMode.LOG_PRODUCT,
        help=("Aggregation strategy:\n"
              "  " + "\n  ".join(f"{m.value} – {m.name.replace('_', ' ').lower()}" for m in AggregateMode))
    )
    parser.add_argument(
        "--smoothing",
        action="store_true",
        help="Apply add-one smoothing",
    )
    parser.add_argument(
        "--no-boundaries",
        action="store_true",
        help="Omit boundary symbols in n-gram generation",
    )

    args = parser.parse_args()
    if args.list_models:
        # Only print primary model keys (exclude aliases)
        print(", ".join(sorted(set(PluginRegistry) - set(ALIASES))))
        sys.exit(0)
    if args.model:
        try:
            get_model(args.model)
        except KeyError as exc:
            parser.error(str(exc))
    cfg  = parse_cfg(args)

    @lru_cache(maxsize=None)
    def get_model_instance(
        model_name: str,
        smoothing: bool,
        weight_mode: WeightMode,
        use_boundaries: bool,
        prob_mode: Union[ProbMode, str],
        aggregate_mode: AggregateMode,
        ngram_order: int,
        strategy_name: str | None = None,
    ):
        """
        Build, train, and cache one variant of the model (positional flag removed).
        """
        cfg_variant = Config.default(
            smoothing      = smoothing,
            weight_mode    = weight_mode,
            use_boundaries = use_boundaries,
            prob_mode      = prob_mode,
            aggregate_mode = aggregate_mode,
            ngram_order    = ngram_order,
            position_strategy = strategy_name,
        )
        train_corpus = Corpus(args.train_file, cfg_variant, include_boundary=use_boundaries)
        if not train_corpus.tokens:
            parser.error(f"Training file ‘{args.train_file}’ contains no token rows.")
        model = get_model(model_name)(cfg_variant)
        model.fit(train_corpus)
        return model

    # Load test corpus once (we only need its tokens)
    test = Corpus(
        args.test_file,
        cfg,
        include_boundary=cfg.use_boundaries,
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

    run_single = args.model is not None
    base_fields = ("word", "word_len")
    if run_single:
        # Derive the single header and variant descriptor up front
        variant_cfg   = parse_cfg(args)
        single_header = get_model(args.model).header(variant_cfg)
        variant_headers = [single_header]
        if len(set(variant_headers)) != len(variant_headers):
            dup = [h for h in variant_headers if variant_headers.count(h) > 1][0]
            parser.error(f"Internal error: duplicate CSV header ‘{dup}’.")
        SingleVariant = type(
            "SingleVariant", (), {
                "model_name": args.model,
                "cfg":        variant_cfg,
                "header":     single_header,
            }
        )
        single_variant = SingleVariant()
    else:
        # Multi-model: build all headers up front
        variant_headers = [hdr for hdr, *_ in all_variants(test)]
        if len(set(variant_headers)) != len(variant_headers):
            dup = [h for h in variant_headers if variant_headers.count(h) > 1][0]
            parser.error(f"Internal error: duplicate CSV header ‘{dup}’.")
    fieldnames = [*base_fields, *variant_headers]
    rows: list[OrderedDict[str, float | str | int]] = []
    for tok in test.tokens:
        entry = OrderedDict((f, None) for f in fieldnames)
        entry["word"] = " ".join(tok)
        entry["word_len"] = len(tok)
        if run_single:
            model_cls = get_model(single_variant.model_name)
            if not model_cls.supports(single_variant.cfg):
                continue
            m = get_model_instance(
                single_variant.model_name,
                single_variant.cfg.smoothing,
                single_variant.cfg.weight_mode,
                single_variant.cfg.use_boundaries,
                single_variant.cfg.prob_mode,
                single_variant.cfg.aggregate_mode,
                single_variant.cfg.ngram_order,
                strategy_name=single_variant.cfg.position_strategy,
            )
            score = m.score(tok)
            entry[single_variant.header] = score if score != float("-inf") else float("nan")
        else:
            for variant in all_variants(test):
                model_cls = get_model(variant.model_name)
                if not model_cls.supports(variant.cfg):
                    continue  # belt-and-suspenders
                strategy_name = variant.cfg.position_strategy
                m = get_model_instance(
                    variant.model_name,
                    variant.cfg.smoothing,
                    variant.cfg.weight_mode,
                    variant.cfg.use_boundaries,
                    variant.cfg.prob_mode,
                    variant.cfg.aggregate_mode,
                    variant.cfg.ngram_order,
                    strategy_name=strategy_name,
                )
                score = m.score(tok)
                if score == float("-inf"):
                    score = float("nan")
                entry[variant.header] = score
        rows.append(entry)

    # Write out a single CSV: first two cols + one per variant header
    try:
        with open(args.output_file, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames, lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
    except PermissionError as e:
        print(f"Permission denied: cannot write to '{args.output_file}': {e}", file=sys.stderr)
        sys.exit(1)
    except OSError as e:
        print(f"Error opening output file '{args.output_file}': {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

__all__ = ["main"]


# End of src/main.py