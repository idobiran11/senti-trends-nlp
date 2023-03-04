"""
evaluation-
seperability metric
asumptions:
1. we can identify subjects that the public opinion about can be seperated between the different sides of the political spectrum.
2. crertain media sources can be identified with one side of the political spectrum
"""
from os import environ
import logging
import os
import jenkspy
import matplotlib.dates as mdates
import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from models.simple_model_notebook import text_sentence_nltk_handler, norm_text_sentence_nltk_handler, sent_norm_text_sentence_nltk_handler
from preprocess.coreference_resolution import coref_preprocess, no_preprocess
from utils.constants import ModelNames, PreprocessNames, SourceNames
from utils.config_neptune import neptune_run, neptune
from models.news_sentiment import news_sentiment_handler


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


def e2e_handler(object_name: str, left_news_vendor: str, right_news_vendor: str, model: str, preprocess: str, second_object_name: str = None):
    algo_run = AlgoRun(object_name, left_news_vendor,
                       right_news_vendor, model, preprocess, second_object_name)
    left_input_file = f'{left_news_vendor}-articles-{object_name}.csv'
    right_input_file = f'{right_news_vendor}-articles-{object_name}.csv'
    left_preprocessed_df = algo_run.preprocess_function(news_vendor=left_news_vendor, object_name=object_name,
                                                        input_csv_name=left_input_file)
    right_preprocessed_df = algo_run.preprocess_function(news_vendor=right_news_vendor, object_name=object_name,
                                                         input_csv_name=right_input_file)
    left_df = algo_run.model_function(object_name=object_name, news_vendor=left_news_vendor,
                                      corpus=left_preprocessed_df, second_object=second_object_name)
    right_df = algo_run.model_function(object_name=object_name, news_vendor=right_news_vendor,
                                       corpus=right_preprocessed_df, second_object=second_object_name)

    print_create_eval_plots(object_name, left_news_vendor,
                            right_news_vendor, model, left_df, right_df)

    neptune_run.stop()


class AlgoRun:

    def __init__(self, object_name: str, left_news_vendor: str, right_news_vendor: str, model: str, preprocess: str, second_object: str):
        self.initialize_neptune_run(object_name, left_news_vendor, right_news_vendor, model,
                                    preprocess, second_object, neptune_run)
        self.object_name = object_name
        if left_news_vendor not in SourceNames.__dict__.values():
            raise Exception("Non existent news vendor: " + left_news_vendor)
        if right_news_vendor not in SourceNames.__dict__.values():
            raise Exception("Non existent news vendor: " + right_news_vendor)

        self.left_news_vendor = left_news_vendor
        self.right_news_vendor = right_news_vendor
        self.model_name = model
        self.preprocess_name = preprocess
        if self.preprocess_name == PreprocessNames.COREF:
            self.preprocess_function = coref_preprocess
        elif self.preprocess_name == PreprocessNames.WITHOUT:
            self.preprocess_function = no_preprocess
        else:
            raise Exception("Non existent preprocess")
        if self.model_name == ModelNames.NLTK:
            self.model_function = text_sentence_nltk_handler
        elif self.model_name == ModelNames.NORM_NLTK:
            self.model_function = norm_text_sentence_nltk_handler
        elif self.model_name == ModelNames.SENT_NORM_NLTK:
            self.model_function = sent_norm_text_sentence_nltk_handler
        elif self.model_name == ModelNames.NEWS_SENTIMENT:
            self.model_function = news_sentiment_handler
        else:
            raise Exception("No known model name set")

    @ staticmethod
    def initialize_neptune_run(object_name: str, left_news_vendor: str, right_news_vendor: str, model: str,
                               preprocess: str, second_object: str, run):

        run['object_name'] = object_name
        run['second_object_name'] = second_object
        run['left_news_vendor'] = left_news_vendor
        run['right_news_vendor'] = right_news_vendor
        run['model'] = model
        run['preprocess'] = preprocess


def print_create_eval_plots(object_name: str, left_news_vendor: str, right_news_vendor: str, model: str, left_df,
                            right_df):

    left = relevant_data(left_df, ["date", "compound_s"])
    right = relevant_data(right_df, ["date", "compound_s"])

    if left.empty or right.empty:
        print("Empty data frame")
        return

    print(f"Shape Left df: {left.shape}")
    print(f"Shape Right df: {right.shape}")
    # left, right = align_time_period(left, right)
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
    fig = sns.boxplot(data=result, x=mdates.date2num(result.Date), y='source')
    plt.title(
        f'{object_name}_{left_news_vendor}_{right_news_vendor}_{model} Distribution over time after equalizing')
    plt.xlabel('Time (to num)')
    plt.show()
    figure = fig.figure
    neptune_run['eval/evaluation-plots-distirbution'].upload(
        neptune.types.File.as_image(figure))
    sns.scatterplot(data=result, x="Date", y="score", hue="source")
    plt.title(
        f'{object_name}_{left_news_vendor}_{right_news_vendor}_{model} Distribution over score and time')
    plt.show()
    figure = fig.figure
    neptune_run['eval/evaluation-plots-score-time'].upload(
        neptune.types.File.as_image(figure))
    sns.set(rc={'figure.figsize': (11.7, 8.27)})
    fig = sns.lineplot(data=result, x="Date", y="score", hue="source")
    plt.title(
        f'{object_name}_{left_news_vendor}_{right_news_vendor}_{model} main Distribution over score and time')
    figure = fig.figure
    neptune_run['eval/evaluation-plots-score-time-main'].upload(
        neptune.types.File.as_image(figure))
    sns.set(rc={'figure.figsize': (8, 6)})
    plt.show()
    fig = sns.boxplot(data=result, x="score", y="source")
    plt.title(
        f'{object_name}_{left_news_vendor}_{right_news_vendor}_{model} Score Distribution by source')
    figure = fig.figure
    neptune_run['eval/evaluation-plots-score-time-source'].upload(
        neptune.types.File.as_image(figure))
    plt.show()
    fig = sns.stripplot(data=result, x="score", y="source", hue='source')
    plt.title(
        f'{object_name}_{left_news_vendor}_{right_news_vendor}_{model} Score Distribution by source')
    figure = fig.figure
    neptune_run['eval/evaluation-distribution-by-source'].upload(
        neptune.types.File.as_image(figure))
    plt.show()
    # supposed to give me the best threshold
    threshold = jenkspy.jenks_breaks(result['score'], n_classes=2)

    print(f'Threshold: {threshold[1]}')

    plt.figure(figsize=(5, 6))
    fig = sns.stripplot(x='score', data=result, hue='source',
                        jitter=True, alpha=0.65, edgecolor='none')
    plt.title(
        f'{object_name}_{left_news_vendor}_{right_news_vendor} Clustered score plot')
    sns.despine()
    locs, labels = plt.xticks()
    plt.xticks(locs, map(lambda x: "%.1f" % x, locs))
    plt.xlabel('score')
    plt.yticks([])
    plt.vlines(threshold[1], ymax=1, ymin=-1)
    figure = fig.figure
    neptune_run['eval/evaluation-clustered-score-plot'].upload(
        neptune.types.File.as_image(figure))
    plt.show()

    if top == 'left':
        result["cluster"] = np.where(
            result['score'] > threshold[1], 'left', 'right')
    else:
        result["cluster"] = np.where(
            result['score'] > threshold[1], 'right', 'left')

    fig = sns.scatterplot(data=result, y='score', x='Date',
                          hue='source', style='cluster', edgecolor='none', alpha=0.65)
    plt.title(
        f'{object_name}_{left_news_vendor}_{right_news_vendor} By Labelled Cluster ')
    figure = fig.figure
    neptune_run['eval/evaluation-labelled-cluster'].upload(
        neptune.types.File.as_image(figure))
    plt.show()

    purity = purity_score(result["source"], result["cluster"],
                          object_name, left_news_vendor, right_news_vendor)
    neptune_run["purity"] = purity

    print(f"Purity Score: {purity}")

    print(
        f'Model: {model}, Object: {object_name}, News Vendors: {left_news_vendor} | {right_news_vendor}, Purity: {purity}')


def relevant_data(data, columns=[]):
    relevant = data[columns]
    relevant['date'] = pd.to_datetime(relevant['date'])
    relevant['Date'] = relevant['date'].dt.date
    relevant = relevant.drop('date', axis=1)
    relevant = relevant.rename(columns={'compound_s': 'score'})
    return relevant


def align_time_period(left, right):
    original_left = left
    original_right = right
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
    if left.empty or right.empty:
        return original_left, original_right

    return left, right


def purity_score(y_true, y_pred, object_name, left_news_vendor, right_news_vendor):
    # compute contingency matrix (also called confusion matrix)
    contingency_matrix = metrics.cluster.contingency_matrix(y_true, y_pred)
    disp = metrics.ConfusionMatrixDisplay(confusion_matrix=contingency_matrix)
    disp.plot()
    plt.title(
        f'{object_name}_{left_news_vendor}_{right_news_vendor} Confusion Matrix')
    plt.savefig('confmatrix.jpg')
    neptune_run['eval/evaluation-confusion-matrix'].upload('confmatrix.jpg')
    # return purity
    return np.sum(np.amax(contingency_matrix, axis=0)) / np.sum(contingency_matrix)


if __name__ == "__main__":
    e2e_handler(object_name=os.environ.get('OBJECT_NAME'), left_news_vendor=os.environ.get('LEFT_NEWS'),
                right_news_vendor=os.environ.get('RIGHT_NEWS'), model=os.environ.get('MODEL'),
                preprocess=os.environ.get("PREPROCESS"), second_object_name=os.environ.get("SECOND_OBJECT"))
