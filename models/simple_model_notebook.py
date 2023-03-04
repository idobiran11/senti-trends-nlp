# import
from models.plot_stats import plot_stats
from utils.config_neptune import neptune_run
from nltk.tokenize import sent_tokenize
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import math
import nltk
import pandas as pd
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import warnings
from models.utils import run_pipeline, plot_graphs, print_max_min_articles

warnings.filterwarnings("ignore")

OBJECT_NAME = "Netanyahu"
NEWS_VENDOR = "CNN"
FILENAME = "cnn-articles-bibi.csv"

_nltk_analyzer = None

nltk.download('vader_lexicon')
nltk.download('punkt')
_nltk_analyzer = SentimentIntensityAnalyzer()


def plot_graphs(scores, object_name, news_vendor):
    scores_graph = pd.concat([scores.drop(['sentences_score'], axis=1), scores['sentences_score'].apply(pd.Series)],
                             axis=1)
    scores_graph = pd.concat([scores_graph.drop(['text_score'], axis=1), scores_graph['text_score'].apply(pd.Series)],
                             axis=1)
    plot_1 = scores_graph.plot(x="date", y=['neg_s', 'neu_s', 'pos_s', 'compound_s'],
                               kind="line", figsize=(15, 6),
                               title=f'Sentences Model Score for {object_name} on {news_vendor}')
    plot_1.xaxis.set_major_formatter(mdates.DateFormatter('%d-%m'))
    filepath = "plot_1.png"
    plt.savefig(filepath)
    neptune_run[f"build/{news_vendor}-sentences"].upload(filepath)
    plot_2 = scores_graph.plot(x="date", y=['neg', 'neu', 'pos', 'compound'],
                               kind="line", figsize=(15, 6),
                               title=f'Entire Text score for {object_name} on {news_vendor}')
    plot_2.xaxis.set_major_formatter(mdates.DateFormatter('%d-%m'))
    filepath = "plot_2.png"
    plt.savefig(filepath)
    neptune_run[f"build/{news_vendor}-full-text"].upload(filepath)
    plot_3 = scores_graph.plot(x="date", y=['compound', 'compound_s'],
                               kind="line", figsize=(15, 6),
                               title=f'Simple Metric Comparison for {object_name} on {news_vendor}')
    plot_3.xaxis.set_major_formatter(mdates.DateFormatter('%d-%m'))
    filepath = f'data/output_plots/simple_model_{object_name}_{news_vendor}.png'
    plt.savefig(filepath)
    neptune_run[f"build/{news_vendor}-Model-Comparison"].upload(filepath)

    plot_stats(scores_graph, news_vendor)

    return scores_graph


def print_max_min_articles(scores_graph, corpus):
    best_article_s = scores_graph["compound_s"].argmax()
    worse_article_s = scores_graph["compound_s"].argmin()
    best_article = scores_graph["compound"].argmax()
    worse_article = scores_graph["compound"].argmin()
    worse_s = corpus.iloc[worse_article_s]
    best_s = corpus.iloc[best_article_s]
    worse = corpus.iloc[worse_article]
    best = corpus.iloc[best_article]
    print(f"Best Article Title: {best['title']}")
    print(f"Worst Article Title: {worse['title']}")
    print(f"Best_s Article Title: {best_s['title']}")
    print(f"Worst_s Article Title: {worse_s['title']}")


def text_sentence_nltk_handler(object_name, news_vendor, corpus, output_directory="data/output_data", second_object=None):
    def eda(df): return df.rename(columns={'timestamp': 'date'})
    file_path = f"{output_directory}/{news_vendor}_{object_name}_nltk_sentences_sentiment.csv"

    return run_pipeline(eda_func=eda,
                        score_func=calc_scores_on_corpus, object_name=object_name,
                        news_vendor=news_vendor, corpus=corpus, file_path=file_path, second_object=second_object)


def norm_text_sentence_nltk_handler(object_name, news_vendor, corpus, output_directory="data/output_data", second_object=None):
    def eda(df): return df.rename(columns={'timestamp': 'date'})
    def score(df, object_name): return calc_scores_on_corpus(
        df, object_name, normalized=True)
    file_path = f"{output_directory}/{news_vendor}_{object_name}_normalized_nltk_sentences_sentiment.csv"
    return run_pipeline(eda_func=eda,
                        score_func=calc_scores_on_corpus, object_name=object_name,
                        news_vendor=news_vendor, corpus=corpus, file_path=file_path)


def nltk_analyze(text):
    return _nltk_analyzer.polarity_scores(text)


def sentences_split(text):
    sentences = sent_tokenize(text)
    return sentences


def calc_parts_sentiment(texts):
    """
    calc score on text. can be operated on corpus of texts or text splitted to sentences.
    returns each part sentiment
    """
    scores = []
    for text in texts:
        score = nltk_analyze(text)
        scores.append(score)
    return scores


# not using
def calc_total_score(score_list):
    """
    get sentiment_dict. calculate total score.
    """
    total_score = {'neg': 0.0, 'neu': 0.0, 'pos': 0.0, 'compound': 0.0}
    for score in score_list:
        total_score["neg"] += score["neg"]
        total_score["neu"] += score["neu"]
        total_score["pos"] += score["pos"]
        total_score["compound"] += score["compound"]
    return total_score


# not using
def find_sentences_with_word(text_sent, word):
    """
    get a word and text split to sentences.
    return list of sentences containing the word.
    """
    relevant_corpus = []
    scores = []
    for sent in text_sent:
        if word in sent:
            relevant_corpus.append(sent)
            curr_score = nltk_analyze(sent)
            scores.append(curr_score)

    return relevant_corpus, scores


def get_text_score(text_sent, word, normalized=False):
    """
    get a word and text splitted to sentences.
    return list of sentences containing the word, each sentece score and text total score
    """
    relevant_corpus = []
    scores = []
    total_score = {'neg_s': 0.0, 'neu_s': 0.0, 'pos_s': 0.0, 'compound_s': 0.0}
    num_of_sentences = len(text_sent)
    compound_count = 0
    for sent in text_sent:
        if word.lower() in sent.lower():
            relevant_corpus.append(sent)
            curr_score = nltk_analyze(sent)
            scores.append(curr_score)
         #   num_of_sentences += 1

            total_score["neg_s"] += curr_score["neg"]
            total_score["neu_s"] += curr_score["neu"]
            total_score["pos_s"] += curr_score["pos"]
            if not normalized:
                total_score["compound_s"] += curr_score["compound"]
            else:
                normalizing_threshold = 0.0005
                neptune_run["normalizing_threshold"] = normalizing_threshold
                if math.fabs(curr_score["compound"]) > normalizing_threshold:
                    total_score["compound_s"] += curr_score["compound"]
                    compound_count += 1
    total_score["neg_s"] = total_score["neg_s"] / max(num_of_sentences, 1)
    total_score["neu_s"] = total_score["neu_s"] / max(num_of_sentences, 1)
    total_score["pos_s"] = total_score["pos_s"] / max(num_of_sentences, 1)
    if not normalized:
        total_score["compound_s"] = total_score["compound_s"] / \
            num_of_sentences
    else:
        if compound_count > 0:
            total_score["compound_s"] = total_score["compound_s"] / \
                compound_count
        else:
            total_score["compound_s"] = 0
    return relevant_corpus, scores, total_score, num_of_sentences


def calc_scores_on_corpus(corpus, name, normalized=False):
    text_score_df = pd.DataFrame(columns=[
                                 'title', 'date', 'text', 'text_score', 'sentences_score', 'num_of_sentences'])
    for index, row in corpus.iterrows():
        # for row in corpus:
        text = row["text"]
        text = text.lower()
        # calc whole text score
        whole_text_score = nltk_analyze(text)
        # seperate to sentences
        text_sent = sentences_split(text)
        # get score
        relevant_text, relevant_scores, total_score, num_of_sentences = get_text_score(
            text_sent, name, normalized)
        # save row to df
        df = pd.DataFrame(
            {"index": [index], "title": [row["title"]], "text": [row["text"]], "num_of_sentences": [num_of_sentences], "date": [row['date']], "text_score": [whole_text_score],
             "sentences_score": [total_score]})
        text_score_df = text_score_df.append(df)

    return text_score_df


if __name__ == "__main__":
    text_sentence_nltk_handler(
        object_name=OBJECT_NAME, news_vendor=NEWS_VENDOR, filename=FILENAME)
