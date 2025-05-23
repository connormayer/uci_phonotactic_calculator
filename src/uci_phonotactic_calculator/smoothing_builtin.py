from .plugins.utils.smoothing import apply
from .registries import register

# ──────────────────────────────────────────────────────
# Sentinel: no smoothing
# ──────────────────────────────────────────────────────
register("smoothing_scheme", "none")(
    lambda table, *, vocab=None: table  # identity passthrough
)


@register("smoothing_scheme", "laplace")
def laplace(table, *, vocab=None):
    return apply(table, vocab=vocab)
