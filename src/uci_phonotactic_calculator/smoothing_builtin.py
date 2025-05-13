from .registries import register
from .plugins.utils.smoothing import apply

# ──────────────────────────────────────────────────────
# Sentinel: no smoothing
# ──────────────────────────────────────────────────────
register('smoothing_scheme', 'none')(
    lambda table, *, vocab=None: table   # identity passthrough
)
from uci_phonotactic_calculator.registries import registry

@register('smoothing_scheme', 'laplace')
def laplace(table, *, vocab=None):
    return apply(table, vocab=vocab)
