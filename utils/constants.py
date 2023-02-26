from enum import Enum
from dataclasses import dataclass

@dataclass(frozen=True)
class ModelNames:
    NLTK = "nltk"
    NORM_NLTK = "normalized_nltk"


@dataclass(frozen=True)
class PreprocessNames:
    COREF = "coref"
    WITHOUT = "without"
