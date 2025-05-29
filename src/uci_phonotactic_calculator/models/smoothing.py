from typing import Mapping, MutableMapping, Optional, Set, Union

import numpy as np
from numpy.typing import NDArray

from ..core.registries import register
from ..plugins.utils.smoothing import IndexTuple, apply

# ──────────────────────────────────────────────────────
# Sentinel: no smoothing
# ──────────────────────────────────────────────────────
register("smoothing_scheme", "none")(
    # Simple identity passthrough with proper type annotations
    lambda table, *, vocab=None: table  # identity passthrough
)


@register("smoothing_scheme", "laplace")
def laplace(
    table: Union[NDArray[np.float64], MutableMapping[IndexTuple, float]],
    *,
    vocab: Optional[Union[Mapping[IndexTuple, float], Set[IndexTuple]]] = None,
) -> Union[NDArray[np.float64], MutableMapping[IndexTuple, float]]:
    return apply(table, vocab=vocab)
