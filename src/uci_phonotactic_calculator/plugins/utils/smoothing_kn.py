"""
Placeholder Kneser–Ney smoother (no-op) – serves as an example for extension
authors.  Real maths deliberately omitted.
"""

from typing import Any, Optional

from ...core.registries import register


@register("smoothing_scheme", "kneser_ney")
def kneser_ney(table: Any, *, vocab: Optional[Any] = None) -> Any:
    # Placeholder: real Kneser–Ney maths not implemented yet
    return table
