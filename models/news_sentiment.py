import pandas as pd
from NewsSentiment import TargetSentimentClassifier
from nltk.tokenize import sent_tokenize
from numpy import tanh
from models.utils import run_pipeline
import pandas as pd
tsc = TargetSentimentClassifier()


# {'neg_s': 0.01972727272727273, 'neu_s': 0.15081818181818185, 'pos_s': 0.011272727272727273, 'compound_s': -0.02520454545454546}


def classify(corpus, name):
    text_classification_df = pd.DataFrame(
        columns=['title', 'date', 'pos', 'neg', 'neu'])
    name = name.lower()
    for index, row in corpus.iloc[:100].iterrows():
        text = row["text"].lower()
        var, label = classify_sent(text, name)
        df = pd.DataFrame({
            "index": [index],
            "title": [row["title"]],
            "date": [row["timestamp"]],
            label: 1,
        })
        for col in ("pos", "neg", "neu"):
            if col != label:
                df[col] = [0]
        text_classification_df = text_classification_df.append(df)

    return text_classification_df


def calc_compound_score(pos, neu, neg):
    return tanh((2 * (pos - neg)) / neu)


def analyze_corpus(corpus, name):
    name = name.lower()
    text_score_df = pd.DataFrame(
        columns=['title', 'date', 'text_score', 'sentences_score'])

    for index, row in corpus.iterrows():
        # for row in corpus:
        text = row["text"]
        text = text.lower()
        # calc whole text score

        # seperate to sentences

        # get score
        label, total_score, num_of_sentences = get_sent_score(
            text, name)
        # save row to df
        df = pd.DataFrame(
            {"index": [index], "title": [row["title"]], "date": [row['date']],
             "text": [text],
             "num_of_sentences": num_of_sentences,
             "text_score": [{"pos": 0, "neg": 0, "neu": 0, "compound": 0, }],
             "sentences_score": [total_score], label+'_lbl': [1]})
        for col in ("pos", "neg", "neu"):
            if col != label:
                df[col+'_lbl'] = [0]
        text_score_df = text_score_df.append(df)

    return text_score_df


def infer(text, name, name_pos=None):
    if not name_pos:
        name_pos = text.find(name)
    start = text[: name_pos]
    end = text[name_pos+len(name):]
    if len(text) > 600:
        if len(start) > 200:
            start = start[-200:]
        if len(end) > 200:
            end = end[: 200]

    return to_nltk_style(tsc.infer_from_text(start, name, end))


def to_nltk_style(scores):
    res = {}
    nltk_keys = {"negative": ("neg_s", "neg"), "neutral": (
        "neu_s", "neu"), "positive": ("pos_s", "pos")}
    for score in scores:
        key1, key2 = nltk_keys[score['class_label']]
        res[key1] = score['class_prob']
        res[key2] = score['class_prob']
    return res


def classify_sent(text, name):
    relevant_corpus = []
    votes = {
        'neg': 0,
        'pos': 0,
        'neu': 0,
    }

    total_score = {'neg_s': 0.0, 'neu_s': 0.0, 'pos_s': 0.0, 'compound_s': 0.0}
    text_sent = sent_tokenize(text)
    num_of_relevant = 0

    for text in text_sent:
        name_pos = text.find(name)
        if name_pos > -1:
            num_of_relevant += 1
            relevant_corpus.append(text)
            curr_score = infer(text, name, name_pos)
            curr_score = {key: value for key,
                          value in curr_score.items() if key[-1] != 's'}
            votes[max(curr_score.items(), key=lambda item: item[1])[0]] += 1

    return relevant_corpus, max(votes.items(), key=lambda item: item[1])[0]


def get_sent_score(text, name):
    """
    get a word and text splitted to sentences.
    return list of sentences containing the word, each sentece score and text total score
    """
    def normalziation(num_r, num_t): return min(1, 0.5 + 8 * (num_r / num_t))

    def calc_ccore(score, num_r, num_t): return ((score / num_r) *
                                                 normalziation(num_r, num_t))
    relevant_corpus = []
    scores = []
    total_score = {'neg_s': 0.0, 'neu_s': 0.0, 'pos_s': 0.0, 'compound_s': 0.0}
    text_sent = sent_tokenize(text)
    num_of_sentences = len(text_sent)
    num_of_relevant = 0
    votes = {
        'neg': 0,
        'pos': 0,
        'neu': 0,
    }
    for text in text_sent:

        name_pos = text.find(name)
        if name_pos > -1:
            num_of_relevant += 1
            relevant_corpus.append(text)
            curr_score = infer(text, name, name_pos)
            scores.append(curr_score)

            total_score["neg_s"] += curr_score["neg"]
            total_score["neu_s"] += curr_score["neu"]
            total_score["pos_s"] += curr_score["pos"]

            total_score["compound_s"] += calc_compound_score(
                curr_score["pos"], curr_score["neu"], curr_score["neg"])

            curr_score = {key: value for key,
                          value in curr_score.items() if key[-1] != 's'}
            votes[max(curr_score.items(), key=lambda item: item[1])[0]] += 1

    if num_of_relevant > 0:
        total_score["neg_s"] = calc_ccore(
            total_score["neg_s"], num_of_relevant, num_of_sentences)
        total_score["neu_s"] = calc_ccore(
            total_score["neu_s"], num_of_relevant, num_of_sentences)
        total_score["pos_s"] = calc_ccore(
            total_score["pos_s"], num_of_relevant, num_of_sentences)
        total_score["compound_s"] = calc_ccore(
            total_score["compound_s"], num_of_relevant, num_of_sentences)

    return max(votes.items(), key=lambda item: item[1])[0], total_score, num_of_relevant

# df = pd.read_csv("data/fox-articles-netanyahu.csv")


def rm_chars(st):
    new_st = ''
    for i, char in enumerate(st):
        if char in ('\n', '\xa0'):
            continue
        elif (i < len(st) - 1) and (char == ' ' and st[i+1] == ' '):
            continue
        else:
            new_st += char
    return new_st


def news_sentiment_handler(object_name, news_vendor, corpus, output_directory="data/output_data"):

    def eda(df: pd.DataFrame):

        df.text = df.text.apply(rm_chars)
        df = df.rename(columns={'timestamp': 'date'})

        return df

    file_path = f"{output_directory}/{object_name}_{news_vendor}_news_sentiment.csv"

    return run_pipeline(eda_func=eda,
                        score_func=analyze_corpus, object_name=object_name,
                        news_vendor=news_vendor, corpus=corpus, file_path=file_path)
