import requests
import datetime
from newsapi import NewsApiClient
from bs4 import BeautifulSoup
import pandas as pd

# up to 100 requests per day
NEWS_API_KEY = '612ad5a9a6cd45a699e3ffb64d606d17'

SEARCH_TERM = 'Trump'

NOW = datetime.datetime.now()
YEAR_AGO = (NOW - datetime.timedelta(days=365)).strftime("%Y%m%d")


def api_request():
    newsapiobj = NewsApiClient(api_key=NEWS_API_KEY)

    all_articles = newsapiobj.get_everything(q=SEARCH_TERM,
                                             sources='cnn',
                                             from_param='2023-01-02',
                                             to='2023-02-01',
                                             language='en',
                                             sort_by='relevancy',
                                             page=6)
    return all_articles


def create_dataframe():
    columns = ['date', 'source', 'title', 'author', 'url']
    df = pd.DataFrame(columns=columns)
    return df


def add_data_to_df(df, request):
    for article in request.get('articles'):
        source = article.get('source')
        date = article.get('publishedAt')
        title = article.get('title')
        author = article.get('author')
        url = article.get('url')
        df = df.append({'date': date, 'source': source, 'title': title, 'author': author, 'url': url}, ignore_index=True)
    return df


def save_df(df):
    df.to_csv(f'{datetime.datetime.now()}data.csv')


def main():
    request = api_request()
    df = create_dataframe()
    df = add_data_to_df(df, request)
    save_df(df)


if __name__ == "__main__":
    main()
