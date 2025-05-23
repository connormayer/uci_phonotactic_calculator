from ..utils.constants import MIN_LOG_PROB


class FallbackMixin:
    @property
    def _fallback(self) -> float:
        return MIN_LOG_PROB
