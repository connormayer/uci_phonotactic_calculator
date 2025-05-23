"""src/plugins/mixins.py — Token-frequency weighting helper only."""

from ..core.registries import registry


class TokenWeightMixin:
    """
    Mixin for models that support token frequency weighting.

    Note:
        legacy_log now returns –∞ for zero counts; callers must be able to
        accumulate that value unchanged.

    The method _w(freq) checks self.cfg.weight_mode.
    No explicit cfg argument is required; the mixin expects self.cfg to be present
    in the instance.

      - none       : 1.0 (unweighted)
      - raw        : frequency (0 becomes 1.0)
      - log        : log(freq+1)
      - legacy_log : log(freq) | -inf when freq == 0
    """

    def _w(self, freq: float) -> float:
        func = registry("weight_mode")[self.cfg.weight_mode]
        return func(freq)


# End of src/plugins/mixins.py
