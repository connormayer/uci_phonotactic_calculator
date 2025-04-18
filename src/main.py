"""src/main.py — CLI entry point for the extensible n-gram scorer, outputting all model/configuration variants."""

import argparse
import csv
from collections import OrderedDict
from functools import lru_cache
import os
import sys

from .config import Config, ProbMode, AggregateMode
from .corpus import Corpus
from .plugins import get_model, PluginRegistry, _discover
from .variants import all_variants

# Ensure all plugins are loaded so --model choices/default stay in sync
_discover()


def parse_cfg(args) -> Config:
    """
    Build a Config from parsed arguments, applying overrides to defaults.
    """
    return Config.default(
        prob_mode      = args.prob_mode,
        aggregate_mode = args.aggregate,
        smoothing      = args.smoothing,
        weight_by_freq = args.weighted,
        use_boundaries = not args.no_boundaries,
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

    # Legacy flags (now informational only)
    default_model = "bigram" if "bigram" in PluginRegistry else sorted(PluginRegistry)[0]
    parser.add_argument(
        "--model",
        default=default_model,
        choices=sorted(PluginRegistry),
        help="Which model plugin to use",
    )
    parser.add_argument(
        "--prob-mode",
        type=ProbMode,
        choices=list(ProbMode),
        default=ProbMode.CONDITIONAL,
        help="Probability computation mode: joint or conditional",
    )
    parser.add_argument(
        "--aggregate",
        type=AggregateMode,
        choices=list(AggregateMode),
        default=AggregateMode.LOG_PRODUCT,
        help=(
            "Aggregation strategy for per-gram scores:\n"
            "  log_product – sum of log-probabilities\n"
            "  logsumexp   – log of sum of probabilities\n"
            "  sum         – linear sum of probabilities"
        ),
    )
    parser.add_argument(
        "--smoothing",
        action="store_true",
        help="Apply add-one smoothing",
    )
    parser.add_argument(
        "--weighted",
        action="store_true",
        help="Weight counts by log token frequency",
    )
    parser.add_argument(
        "--no-boundaries",
        action="store_true",
        help="Omit boundary symbols in n-gram generation",
    )

    args = parser.parse_args()
    cfg  = parse_cfg(args)

    @lru_cache(maxsize=None)
    def get_model_instance(
        model_name: str,
        positional: bool,
        smoothing: bool,
        weight_by_freq: bool,
        use_boundaries: bool,
        prob_mode: ProbMode,
        aggregate_mode: AggregateMode,
        ngram_order: int,
    ):
        """
        Build, train, and cache one variant of the model.
        """
        cfg_variant = Config.default(
            positional     = positional,
            smoothing      = smoothing,
            weight_by_freq = weight_by_freq,
            use_boundaries = use_boundaries,
            prob_mode      = prob_mode,
            aggregate_mode = aggregate_mode,
            ngram_order    = ngram_order,
        )
        # Include boundary symbol only if that particular variant requests it
        train_corpus = Corpus(
            args.train_file,
            cfg_variant,
            include_boundary=use_boundaries,
        )

        model_cls    = get_model(model_name)
        model        = model_cls(cfg_variant)
        model.fit(train_corpus)
        return model

    # Load test corpus once (we only need its tokens)
    test = Corpus(
        args.test_file,
        cfg,
        include_boundary=cfg.use_boundaries,
    )

    # Prepare output directory
    out_dir = os.path.dirname(args.output_file) or "."
    try:
        os.makedirs(out_dir, exist_ok=True)
    except OSError as e:
        print(f"Error: unable to create directory '{out_dir}': {e}", file=sys.stderr)
        sys.exit(1)

    rows: list[OrderedDict[str, float | str | int]] = []
    for tok in test.tokens:
        entry = OrderedDict([
            ("word",     " ".join(tok)),
            ("word_len", len(tok)),
        ])
        # Score under every variant
        for variant in all_variants(test):
            m     = get_model_instance(
                        variant.model_name,
                        variant.cfg.positional,
                        variant.cfg.smoothing,
                        variant.cfg.weight_by_freq,
                        variant.cfg.use_boundaries,
                        variant.cfg.prob_mode,
                        variant.cfg.aggregate_mode,
                        variant.cfg.ngram_order,
                    )
            score = m.score(tok)
            # turn -inf into nan for clarity
            if score == float("-inf"):
                score = float("nan")
            entry[variant.header] = score

        rows.append(entry)

    # Write out a single CSV: first two cols + one per variant header
    try:
        with open(args.output_file, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=rows[0].keys())
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

# End of src/main.py