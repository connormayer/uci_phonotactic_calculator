"""src/plugins/neighbourhood.py — Neighbourhood density model plugin: fast edit‑distance‑1 neighbor counting."""

from ..plugins import register
from .base import BaseModel

@register("neighbourhood")
class NeighbourhoodModel(BaseModel):
    """
    Neighbourhood model: for each test token, returns the number of
    training tokens whose Levenshtein distance is exactly 1.
    Uses neighbor‑generation for O(n * |alphabet|) performance.
    """

    def fit(self, corpus):
        """
        Store the training tokens (flattened) for neighbor comparisons.
        """
        # Flatten each token list into a string
        self.training_tokens = ["".join(tok) for tok in corpus.tokens]
        # Set for O(1) lookups
        self.training_set = set(self.training_tokens)
        # Alphabet of characters appearing in training strings
        self.alphabet = set("".join(self.training_tokens)) if self.training_tokens else set()
        # Preserve sound index for unseen-symbol checks
        self.sound_index = corpus.sound_index
        # Cache for computed targets
        self._cache: dict[str, float] = {}

    def score(self, token: list[str]) -> float:
        """
        Return the count of training tokens at edit-distance ==1 from test token.
        If the test token contains any unseen symbol, returns fallback.
        """
        target = "".join(token)
        # Check for unseen symbols
        for s in token:
            if s not in self.sound_index:
                return self._fallback

        # Use cache if available
        if target in self._cache:
            return self._cache[target]

        neighbors = set()

        # Deletions
        for i in range(len(target)):
            neighbors.add(target[:i] + target[i+1:])

        # Substitutions
        for i in range(len(target)):
            for c in self.alphabet:
                if c != target[i]:
                    neighbors.add(target[:i] + c + target[i+1:])

        # Insertions
        for i in range(len(target) + 1):
            for c in self.alphabet:
                neighbors.add(target[:i] + c + target[i:])

        # Count how many neighbors appear in the training set
        count = sum(1 for nbr in neighbors if nbr in self.training_set)

        # Cache and return
        self._cache[target] = float(count)
        return float(count)

# End of src/plugins/neighbourhood.py
