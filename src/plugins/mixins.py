"""src/plugins/mixins.py — Token-frequency weighting helper only.

WeightMode behaviors:
  - NONE : 1.0
  - RAW  : frequency (0 becomes 1.0)
  - LOG  : log(max(freq, 1.0))
"""

import math
from ..config import WeightMode

class TokenWeightMixin:
    """
    Mixin to compute per-token weights based on token frequency and configuration.

    WeightMode behaviors:
      - NONE : 1.0
      - RAW  : frequency (0 becomes 1.0)
      - LOG  : log(max(freq, 1.0))

    The method _w(freq) checks self.cfg.weight_mode.
    No explicit cfg argument is required; the mixin expects self.cfg to be present in the instance.
    """

    def _w(self, freq: float) -> float:
        """
        Return the weight for a token based on its frequency, according to self.cfg.weight_mode:
        - NONE: always 1.0
        - RAW: frequency (0 becomes 1.0)
        - LOG: log(max(freq, 1.0))
        """
        if freq < 0:
            raise ValueError("Frequency must be non-negative")
        mode = self.cfg.weight_mode
        if mode is WeightMode.NONE:
            return 1.0
        elif mode is WeightMode.RAW:
            return float(freq or 1.0)
        elif mode is WeightMode.LOG:
            return math.log(max(freq, 1.0))
        else:
            raise ValueError(f"Unknown weight mode: {mode}")


# End of src/plugins/mixins.py
