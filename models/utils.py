
from utils.config_neptune import neptune_run
import pandas as pd
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import warnings
from models.plot_stats import plot_stats


def run_pipeline(eda_func, score_func, object_name, news_vendor, corpus, file_path):
    if eda_func:
        corpus = eda_func(corpus)

    scores = score_func(corpus, object_name)
    scores_graph = plot_graphs(scores, object_name, news_vendor)
    scores_graph.set_index('index')
    print_max_min_articles(scores_graph, corpus)
    scores_graph.to_csv(file_path, index=False)
    neptune_run[f'eval/{news_vendor}_sentiment'].upload(file_path)
    return scores_graph


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
