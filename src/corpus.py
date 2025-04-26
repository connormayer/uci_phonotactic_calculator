"""src/corpus.py — Corpus loader and bigram generator with optional boundary inclusion."""

from __future__ import annotations
import csv
from pathlib import Path
from typing import Sequence, List, Tuple
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
    def generate_ngrams(
        token: Sequence[str],
        n: int,
        use_boundaries: bool,
        *,
        index_map: dict[str, int] | None = None,
    ) -> List[Tuple]:
        """
        Generate n-grams from a token. Pads with n-1 boundaries on both sides if use_boundaries is True and n > 1.
        If index_map is supplied, returns tuples of ints; otherwise, returns tuples of symbols.
        Returns empty list if sequence too short.

        Boundary tokens are inserted only when n > 1.
        """
        pad = [WORD_BOUNDARY] * (n - 1) if n > 1 else []
        if use_boundaries:
            seq = pad + list(token) + pad
        else:
            seq = list(token)
        if len(seq) < n:
            return []
        grams = [tuple(seq[i:i+n]) for i in range(len(seq) - n + 1)]
        if index_map:
            return [tuple(index_map.get(s, -1) for s in gram) for gram in grams]
        return grams

# End of src/corpus.py