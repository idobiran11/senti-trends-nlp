from utils.config_neptune import neptune_run
import pandas as pd
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import warnings
from models.plot_stats import plot_stats


def run_pipeline(eda_func, score_func, object_name, news_vendor, corpus, file_path, second_object=None):
    if eda_func:
        corpus = eda_func(corpus)

    scores_1 = score_func(corpus, object_name)
    scores_graph = plot_graphs(scores_1, object_name, news_vendor)
    scores_graph.set_index('index')
    print_max_min_articles(scores_graph, corpus)
    if second_object:
        scores_2 = score_func(corpus, second_object)
        scores_graph_2 = plot_graphs(scores_2, object_name, news_vendor)
        scores_graph_2.set_index('index')
        print_max_min_articles(scores_graph_2, corpus)
        compare_plots(object_name, second_object, scores_graph, scores_graph_2, news_vendor)
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


def compare_plots(object_name, second_object, scores_graph, scores_graph_2, news_vendor):
    scores_graph = scores_graph.rename(columns={'compound_s': f'{object_name}_compound_s'})
    scores_graph_2 = scores_graph_2.rename(columns={'compound_s': f'{second_object}_compound_s'})
    merged_df = pd.merge(scores_graph, scores_graph_2, on='title', how='left')
    merged_df = merged_df.drop(merged_df[merged_df[f'{second_object}_compound_s'] == 0].index)
    # add a column to indicate which value is larger
    merged_df['larger'] = merged_df[f'{object_name}_compound_s'] > merged_df[f'{second_object}_compound_s']

    # plot the 'a' and 'b' columns as lines
    ax = merged_df[[f'{object_name}_compound_s', f'{second_object}_compound_s']].plot(kind='line')

    # add labels to the plot
    ax.set_xlabel('Row')
    ax.set_ylabel('Value')
    ax.set_title('Comparison of Columns A and B')

    # add markers to indicate which value is larger
    for i, row in merged_df.iterrows():
        if row['larger']:
            ax.scatter(i, row[f'{object_name}_compound_s'], marker='^', color='green')
        else:
            ax.scatter(i, row[f'{second_object}_compound_s'], marker='v', color='red')

    filepath = f'{news_vendor}-compare-{object_name}-{second_object}'
    plt.savefig(filepath)
    neptune_run[f"{filepath}"].upload(filepath)
    # show the plot
    plt.show()

    # calculate the percentage where 'a' is larger
    pct_a_larger = (merged_df[f'{object_name}_compound_s'] > merged_df[f'{second_object}_compound_s']).mean() * 100

    # print the percentage
    print(f'{pct_a_larger:.2f}% of rows where {object_name} is larger than {second_object}')
