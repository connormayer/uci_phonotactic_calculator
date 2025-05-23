"""
Placeholder Kneser–Ney smoother (no-op) – serves as an example for extension
authors.  Real maths deliberately omitted.
"""

from ...core.registries import register


@register("smoothing_scheme", "kneser_ney")
def kneser_ney(table, *, vocab=None):
    # Placeholder: real Kneser–Ney maths not implemented yet
    return table
