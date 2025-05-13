"""
Placeholder Kneser–Ney smoother (no-op) – serves as an example for extension
authors.  Real maths deliberately omitted.
"""
from ...registries import register

@register('smoothing_scheme', 'kneser_ney')
def kneser_ney(cfg, table, vocab=None):
    # Simply return counts unchanged
    return table
