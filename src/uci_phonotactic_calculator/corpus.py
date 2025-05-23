"""src/corpus.py — Corpus loader and bigram generator with optional boundary
inclusion."""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Sequence, Tuple

from .config import Config
from .registries import registry


@dataclass(slots=True)
class Corpus:
    """
    Encapsulates token sequences, their frequencies, and the sound index for
    n-gram modeling.
    Supports optional inclusion of boundary symbols in the vocabulary.
    """

    cfg: Config = field(init=False, repr=False)  # ← new
    tokens: list[list[str]]
    freqs: list[float]
    sound_index: list[str]
    _boundary: str = field(init=False, repr=False)
    boundary_symbol: str = field(init=False)

    def __init__(
        self,
        file_path: str | Path,
        cfg: Config,
    ):
        """
        Load tokens and optional frequencies from a CSV, build the vocabulary,
        and sort it into a sound index.

        Parameters:
          file_path         — Path to the CSV file of tokens
          cfg               — Configuration flags (including cfg.boundary_mode)
        """
        self.cfg = cfg
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

        # Use registry to get the boundary symbol
        pad_sym = registry("boundary_scheme")[cfg.boundary_scheme]()
        # Only add the boundary symbol if boundary_mode != 'none'
        if cfg.boundary_mode != "none":
            vocab.add(pad_sym)
        self._boundary = pad_sym
        self.boundary_symbol = pad_sym  # boundary symbol from config

        # Sort into a reproducible index
        self.sound_index = sorted(vocab)

    def ngrams(
        self, token: Sequence[str], n: int, *, index_map: dict[str, int] | None = None
    ) -> list[tuple]:
        """
        Instance wrapper for generate_ngrams that always uses self._boundary
        and self.cfg.boundary_mode.
        """
        return Corpus.generate_ngrams(
            token,
            n,
            self.cfg.boundary_mode,
            index_map=index_map,
            boundary=self._boundary,
        )

    @staticmethod
    def generate_ngrams(
        token: Sequence[str],
        n: int,
        boundary_mode: str,
        *,
        index_map: dict[str, int] | None = None,
        boundary: str = "#",
    ) -> List[Tuple]:
        """
        Generate n-grams from a token using a registry-driven boundary mode.
        If index_map is supplied, returns tuples of ints; otherwise, returns tuples
        of symbols.
        Returns empty list if sequence too short.
        """
        from .registries import registry

        pad_fn = registry("boundary_mode")[boundary_mode]
        seq = pad_fn(list(token), boundary, n)
        if len(seq) < n:
            return []
        grams = [tuple(seq[i : i + n]) for i in range(len(seq) - n + 1)]
        if index_map:
            return [tuple(index_map.get(s, -1) for s in gram) for gram in grams]
        return grams


# End of src/corpus.py
