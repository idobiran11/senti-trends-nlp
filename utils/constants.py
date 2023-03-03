from enum import Enum
from dataclasses import dataclass


@dataclass(frozen=True)
class ModelNames:
    NLTK = "nltk"
    NORM_NLTK = "normalized_nltk"
    NEWS_SENTIMENT = "news_sentiment"


@dataclass(frozen=True)
class PreprocessNames:
    COREF = "coref"
    WITHOUT = "without"


@dataclass(frozen=True)
class SourceNames:
    CNN = "cnn"
    FOX = "fox"
    NYT = "nyt"
    WSJ = "wsj"
