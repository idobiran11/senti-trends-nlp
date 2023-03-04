from main.main_model_run import e2e_handler
from utils.constants import SourceNames, ModelNames, PreprocessNames
from models.news_sentiment import infer

e2e_handler(object_name='trump', left_news_vendor=SourceNames.CNN,
            right_news_vendor=SourceNames.FOX, model=ModelNames.NEWS_SENTIMENT,
            preprocess=PreprocessNames.COREF)
