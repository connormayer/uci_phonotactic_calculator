"""src/plugins/mixins.py — Mixin classes providing token-frequency weighting and smoothing fallback."""

import numpy as np
from ..constants import MIN_LOG_PROB


class TokenWeightMixin:
    """
    Mixin to compute per-token weights based on token frequency and configuration.
    """

    def _w(self, freq: float, cfg):
        """
        Return the weight for a token of frequency `freq`.
        - If cfg.weight_by_freq: return log(freq) if freq > 0, else 1.0
        - Otherwise: return 1.0
        """
        if cfg.weight_by_freq:
            return np.log(freq) if freq > 0 else 1.0
        return 1.0


class SmoothingMixin:
    """
    Mixin to apply additive (pseudo‑count) smoothing to discrete counts,
    and provide a fallback log‑probability for unseen events.
    """

    def _smooth(self, table, vocabulary, cfg):
        """
        If cfg.smoothing is True, apply additive smoothing:
        - For NumPy arrays: return a new array with +1 in every entry.
        - For dict‑like tables: increment only existing symbols by 1.

        Parameters:
          table       — ndarray or dict of raw counts
          vocabulary  — sequence of symbols to consider for smoothing
          cfg         — configuration flags (cfg.smoothing)
        """
        if not cfg.smoothing:
            return table

        # ndarray branch – add 1 everywhere
        if isinstance(table, np.ndarray):
            return table + 1.0

        # dict‑like branch – only bump existing keys
        for v in vocabulary:
            if v in table:
                table[v] += 1
        return table

    @property
    def _fallback(self):
        """
        Fallback log‑probability to use when an n-gram lookup fails.
        """
        return MIN_LOG_PROB

# End of src/plugins/mixins.py
