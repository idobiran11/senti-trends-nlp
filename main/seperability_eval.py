"""
evaluation-
seperability metric
asumptions:
1. we can identify subjects that the public opinion about can be seperated between the different sides of the political spectrum.
2. crertain media sources can be identified with one side of the political spectrum
"""
import logging
import os
import jenkspy
import matplotlib.dates as mdates
import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from Simple_model.simple_model_notebook import text_sentence_nltk_handler


def create_logger():
    # Set up the logger
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    # Create a file handler
    handler = logging.FileHandler('data/logger/tests.log')
    handler.setLevel(logging.INFO)

    # Create a formatter and add it to the handler
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)

    # Add the handler to the logger
    logger.addHandler(handler)
    return logger


logger = create_logger()


def e2e_handler(object_name: str, left_news_vendor: str, right_news_vendor: str, model: str):
    if model == "nltk":
        left_df = text_sentence_nltk_handler(object_name=object_name, news_vendor=left_news_vendor,
                                             filename=f'{left_news_vendor}-articles-{object_name}.csv')
        right_df = text_sentence_nltk_handler(object_name=object_name, news_vendor=right_news_vendor,
                                              filename=f'{right_news_vendor}-articles-{object_name}.csv')
    left = relevant_data(left_df, ["date", "compound_s"])
    right = relevant_data(right_df, ["date", "compound_s"])
    print(f"Shape Left df: {left.shape}")
    print(f"Shape Right df: {right.shape}")
    left, right = align_time_period(left, right)
    print(f"Updated Shape Left df: {left.shape}")
    print(f"Updated Shape Right df: {right.shape}")
    left['source'] = left_news_vendor
    right['source'] = right_news_vendor
    avg_left = left['score'].mean()
    avg_right = right['score'].mean()
    print(f'Average left Score: {avg_left}')
    print(f'Average right Score: {avg_right}')
    if avg_left > avg_right:
        top = 'left'
    else:
        top = 'right'
    print(f"Updated again Shape Left df: {left.shape}")
    print(f"Updated again Shape Right df: {right.shape}")
    result = pd.concat([left, right])
    result['Date'] = pd.to_datetime(result['Date'])
    sns.boxplot(data=result, x=mdates.date2num(result.Date), y='source')
    plt.title(f'{object_name}_{left_news_vendor}_{right_news_vendor}_{model} Distribution over time after equalizing')
    plt.xlabel('Time (to num)')
    plt.show()
    sns.scatterplot(data=result, x="Date", y="score", hue="source")
    plt.title(f'{object_name}_{left_news_vendor}_{right_news_vendor}_{model} Distribution over score and time')
    plt.show()
    sns.set(rc={'figure.figsize': (11.7, 8.27)})
    plt.show()
    sns.lineplot(data=result, x="Date", y="score", hue="source")
    plt.title(f'{object_name}_{left_news_vendor}_{right_news_vendor}_{model} main Distribution over score and time')
    plt.show()
    sns.set(rc={'figure.figsize': (8, 6)})
    sns.boxplot(data=result, x="score", y="source")
    plt.title(f'{object_name}_{left_news_vendor}_{right_news_vendor}_{model} Score Distribution by source')
    plt.show()
    sns.stripplot(data=result, x="score", y="source", hue='source')
    plt.title(f'{object_name}_{left_news_vendor}_{right_news_vendor}_{model} Score Distribution by source')
    plt.show()

    threshold = jenkspy.jenks_breaks(result['score'], n_classes=2)  # supposed to give me the best threshold

    print(f'Threshold: {threshold[1]}')

    plt.figure(figsize=(5, 6))
    sns.stripplot(x='score', data=result, hue='source', jitter=True, alpha=0.65, edgecolor='none')
    plt.title(f'{object_name}_{left_news_vendor}_{right_news_vendor} Clustered score plot')
    sns.despine()
    locs, labels = plt.xticks()
    plt.xticks(locs, map(lambda x: "%.1f" % x, locs))
    plt.xlabel('score')
    plt.yticks([])
    plt.vlines(threshold[1], ymax=1, ymin=-1)
    plt.show()

    if top == 'left':
        result["cluster"] = np.where(result['score'] > threshold[1], 'left', 'right')
    else:
        result["cluster"] = np.where(result['score'] > threshold[1], 'right', 'left')

    sns.scatterplot(data=result, y='score', x='Date', hue='source', style='cluster', edgecolor='none', alpha=0.65)
    plt.title(f'{object_name}_{left_news_vendor}_{right_news_vendor} By Labelled Cluster ')
    plt.show()

    purity = purity_score(result["source"], result["cluster"], object_name, left_news_vendor, right_news_vendor)

    print(f"Purity Score: {purity}")

    logger.info(f'Model: {model}, Object: {object_name}, News Vendors: {left_news_vendor} | {right_news_vendor}, Purity: {purity}')


def relevant_data(data, columns=[]):
    relevant = data[columns]
    relevant['date'] = pd.to_datetime(relevant['date'])
    relevant['Date'] = relevant['date'].dt.date
    relevant = relevant.drop('date', axis=1)
    relevant = relevant.rename(columns={'compound_s': 'score'})
    return relevant


def align_time_period(left, right):
    left['Date'] = pd.to_datetime(left['Date'])
    right['Date'] = pd.to_datetime(right['Date'])
    l_min_date = left['Date'].min()
    r_min_date = right['Date'].min()
    l_max_date = left['Date'].max()
    r_max_date = right['Date'].max()
    upper_border = min(l_max_date, r_max_date)
    lower_border = max(l_min_date, r_min_date)
    right = right[right.Date < upper_border]
    right = right[right.Date > lower_border]
    left = left[left.Date < upper_border]
    left = left[left.Date > lower_border]
    return left, right


def purity_score(y_true, y_pred, object_name, left_news_vendor, right_news_vendor):
    # compute contingency matrix (also called confusion matrix)
    contingency_matrix = metrics.cluster.contingency_matrix(y_true, y_pred)
    disp = metrics.ConfusionMatrixDisplay(confusion_matrix=contingency_matrix)
    disp.plot()
    plt.title(f'{object_name}_{left_news_vendor}_{right_news_vendor} Confusion Matrix')
    plt.show()
    # return purity
    return np.sum(np.amax(contingency_matrix, axis=0)) / np.sum(contingency_matrix)


if __name__ == "__main__":
    e2e_handler(object_name=os.environ.get('OBJECT_NAME'), left_news_vendor=os.environ.get('LEFT_NEWS'),
                right_news_vendor=os.environ.get('RIGHT_NEWS'), model=os.environ.get('MODEL'))
