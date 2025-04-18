"""src/corpus.py — Corpus loader and bigram generator with optional boundary inclusion."""

from __future__ import annotations
import csv
from pathlib import Path
from typing import Sequence
from dataclasses import dataclass

from .config import Config
from .constants import WORD_BOUNDARY


@dataclass(slots=True)
class Corpus:
    """
    Encapsulates token sequences, their frequencies, and the sound index for n-gram modeling.
    Supports optional inclusion of boundary symbols in the vocabulary.
    """
    tokens: list[list[str]]
    freqs: list[float]
    sound_index: list[str]

    def __init__(
        self,
        file_path: str | Path,
        cfg: Config,
        include_boundary: bool | None = None
    ):
        """
        Load tokens and optional frequencies from a CSV, build the vocabulary,
        and sort it into a sound index.

        Parameters:
          file_path         — Path to the CSV file of tokens
          cfg               — Configuration flags (including cfg.use_boundaries)
          include_boundary  — If True, force boundary symbol into vocabulary;
                              if False, omit it; if None, defer to cfg.use_boundaries.
        """
        self.tokens = []
        self.freqs = []

        # Read CSV of form: "PH1 PH2 PH3",freq?
        with open(file_path, encoding="utf-8") as fh:
            reader = csv.reader(fh)
            for row in reader:
                phonemes = row[0].split(" ")
                self.tokens.append(phonemes)
                freq = float(row[1]) if len(row) == 2 else 0.0
                self.freqs.append(freq)

        # Build vocabulary from all token symbols
        vocab = {sound for token in self.tokens for sound in token}

        # Determine whether to include the boundary symbol
        boundary = cfg.use_boundaries if include_boundary is None else include_boundary
        if boundary:
            vocab.add(WORD_BOUNDARY)

        # Sort into a reproducible index
        self.sound_index = sorted(vocab)

    @staticmethod
    def generate_bigrams(token: Sequence[str], cfg: Config) -> list[tuple[str, str]]:
        """
        Produce bigrams from a token, optionally wrapped in boundary symbols.

        Parameters:
          token — sequence of phoneme strings
          cfg   — configuration indicating whether to add boundaries

        Returns:
          A list of (previous, next) tuples over the (possibly wrapped) token.
        """
        seq = list(token)
        if cfg.use_boundaries:
            seq = [WORD_BOUNDARY] + seq + [WORD_BOUNDARY]
        return list(zip(seq, seq[1:]))

# End of src/corpus.py