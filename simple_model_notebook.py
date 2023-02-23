# import
import matplotlib.dates as mdates
import pandas as pd
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

from nltk.tokenize import sent_tokenize

_nltk_analyzer = None

nltk.download('vader_lexicon')
nltk.download('punkt')
_nltk_analyzer = SentimentIntensityAnalyzer()


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


def get_text_score(text_sent, word):
    """
    get a word and text splitted to sentences.
    return list of sentences containing the word, each sentece score and text total score
    """
    relevant_corpus = []
    scores = []
    total_score = {'neg_s': 0.0, 'neu_s': 0.0, 'pos_s': 0.0, 'compound_s': 0.0}
    num_of_senteces = len(text_sent)

    for sent in text_sent:
        if word in sent:
            relevant_corpus.append(sent)
            curr_score = nltk_analyze(sent)
            scores.append(curr_score)

            total_score["neg_s"] += curr_score["neg"]
            total_score["neu_s"] += curr_score["neu"]
            total_score["pos_s"] += curr_score["pos"]
            total_score["compound_s"] += curr_score["compound"]
    total_score["neg_s"] = total_score["neg_s"] / num_of_senteces
    total_score["neu_s"] = total_score["neu_s"] / num_of_senteces
    total_score["pos_s"] = total_score["pos_s"] / num_of_senteces
    total_score["compound_s"] = total_score["compound_s"] / num_of_senteces

    return relevant_corpus, scores, total_score


def clac_score_on_corpus(corpus, name):
    text_score_df = pd.DataFrame(columns=['title', 'date', 'text_score', 'sentences_score'])
    for index, row in corpus.iterrows():
        # for row in corpus:
        text = row["text"]
        text = text.lower()
        # calc whole text score
        whole_text_score = nltk_analyze(text)
        # sperate to senteces
        text_sent = sentences_split(text)
        # get score
        relevant_text, relevant_scores, total_score = get_text_score(text_sent, name)
        # save row to df
        df = pd.DataFrame(
            {"index": [index], "title": [row["title"]], "date": [row['date']], "text_score": [whole_text_score],
             "sentences_score": [total_score]})
        text_score_df = text_score_df.append(df)

    return text_score_df


corpus = pd.read_csv('data/fox-articles-bibi.csv')
corpus.rename(columns={'timestamp': 'date'}, inplace=True)


corpus.head()


corpus.iloc[0, 1]

scores = clac_score_on_corpus(corpus, "netanyahu")

scores.head()
scores.iloc[0, 0]

scores_graph = pd.concat([scores.drop(['sentences_score'], axis=1), scores['sentences_score'].apply(pd.Series)], axis=1)
scores_graph = pd.concat([scores_graph.drop(['text_score'], axis=1), scores_graph['text_score'].apply(pd.Series)],
                         axis=1)

scores_graph.columns


plot_1 = scores_graph.plot(x="date", y=['neg_s', 'neu_s', 'pos_s', 'compound_s'],
                           kind="line", figsize=(15, 6), title='sentences score January 2022')
plot_1.xaxis.set_major_formatter(mdates.DateFormatter('%d-%m'))


plot_2 = scores_graph.plot(x="date", y=['neg', 'neu', 'pos', 'compound'],
                           kind="line", figsize=(15, 6), title='text score January 2022')
plot_2.xaxis.set_major_formatter(mdates.DateFormatter('%d-%m'))

plot_3 = scores_graph.plot(x="date", y=['compound', 'compound_s'],
                           kind="line", figsize=(15, 6), title='metric comparisson January 2022')
plot_3.xaxis.set_major_formatter(mdates.DateFormatter('%d-%m'))



scores_graph.set_index('index')
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



scores_graph.to_csv("cnn_israel_sentiment.csv", index=False)
