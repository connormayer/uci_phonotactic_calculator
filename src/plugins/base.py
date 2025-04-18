"""src/plugins/base.py — Abstract base class for all n‑gram model plugins."""

from abc import ABC, abstractmethod
from ..config import Config

class BaseModel(ABC):
    """
    All plugin model classes must inherit from BaseModel and implement
    fit() and score().
    """

    def __init__(self, cfg: Config):
        """
        Store the user-specified configuration.
        """
        self.cfg = cfg

    def name(self) -> str:
        """
        Return the class name of this model for identification or logging.
        """
        return self.__class__.__name__

    @abstractmethod
    def fit(self, corpus):
        """
        Train internal structures on the given Corpus.
        """
        ...

    @abstractmethod
    def score(self, token: list[str]) -> float:
        """
        Return a numeric score/log-probability for the provided token.
        """
        ...

# End of src/plugins/base.py
