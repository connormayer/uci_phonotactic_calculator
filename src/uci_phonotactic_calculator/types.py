"""Project-wide aliases for n-gram typing."""

from typing import Mapping, MutableMapping, Tuple, TypeAlias

Symbol: TypeAlias = str  # one phoneme or boundary
Gram: TypeAlias = Tuple[Symbol, ...]  # n-gram in symbol space
Index: TypeAlias = int  # position in sound_index
IndexTuple: TypeAlias = Tuple[Index, ...]  # n-gram in index space

CountDict: TypeAlias = MutableMapping[Gram, float]
SparseLogDict: TypeAlias = Mapping[tuple[int, ...], float]
