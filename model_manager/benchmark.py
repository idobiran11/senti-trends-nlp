# data on nltk vadar quality https://www.kaggle.com/code/islammohamedd1/assessing-nltk-s-vader-sentiment-analysis-model

from nltk.sentiment import SentimentIntensityAnalyzer
import nltk
from nltk.tokenize import sent_tokenize

_nltk_analyzer = None


def init_nltk():
    global _nltk_analyzer
    nltk.download('vader_lexicon')
    nltk.download('punkt')
    _nltk_analyzer = SentimentIntensityAnalyzer()


def nltk_analyze(text):
    global _nltk_analyzer
    if not _nltk_analyzer:
        init_nltk()
    return _nltk_analyzer.polarity_scores(text)


def sentiment_analysis(name, text):
    init_nltk()
    sentences = sent_tokenize(text)


text = None
with open('model_manager\example_text.txt', 'r', encoding="utf8") as f:
    text = f.read().lower()
sentiment_analysis('trump', text)
