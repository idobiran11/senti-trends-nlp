import requests
import datetime
from newsapi import NewsApiClient
from bs4 import BeautifulSoup
import pandas as pd
get = requests.get
# up to 100 requests per day
NEWS_API_KEY = '612ad5a9a6cd45a699e3ffb64d606d17'

SEARCH_TERM = 'israel'

NOW = datetime.datetime.now()
YEAR_AGO = (NOW - datetime.timedelta(days=365)).strftime("%Y%m%d")


def api_request():
    newsapiobj = NewsApiClient(api_key=NEWS_API_KEY)

    all_articles = newsapiobj.get_everything(q=SEARCH_TERM,
                                             sources='fox-news',  # cnn
                                             from_param='2023-01-20',
                                             #  to='2023-02-01',
                                             language='en',
                                             #  sort_by='relevancy',
                                             #  page=6
                                             )
    return all_articles['articles'], all_articles['totalResults']


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
        df = df.append({'date': date, 'source': source, 'title': title,
                       'author': author, 'url': url}, ignore_index=True)
    return df


def save_df(df):
    df.to_csv(f'{datetime.datetime.now()}data.csv')


def main():
    articles, total = api_request()
    data = pd.DataFrame(data={}, columns=[
        'text', 'title', 'source', 'timestamp', 'url'])
    for article in articles:
        print(article['title'])
        text = Article.get_full_article(
            article['url'], article['source']['id'])
        if text is None:
            continue
        article_data = {'title': [article['title']],
                        'url': [article['url']],
                        'text': [text],
                        'source': [article['source']['name']],
                        'timestamp': [article['publishedAt']]}

        data = pd.concat([data, pd.DataFrame(data=article_data)],
                         ignore_index=True)
    csv_name = 'data/fox-articles-israel.csv'
    data.to_csv(csv_name, index=False)
    print(f'saved {data.shape[0]} articles out of {total} to {csv_name}')


def get_webpage(url):
    # url = "https://www.nytimes.com/2023/01/20/us/politics/abortion-republicans-roe-v-wade.html"

    payload = {}
    headers = {
        'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/109.0.0.0 Safari/537.36',
        'Cookie': 'nyt-a=mLCOJ2cKWCmTY32dxmO6kP; nyt-b3-traceid=cb262ae7a0634b68a4ba648e4e53eaa2; nyt-gdpr=0; nyt-geo=IL; nyt-purr=cfhhcfhhhukfhu; nyt-us=0; datadome=4MEKMQAVxvpWQg3wmcYGJ1oHb1lJp5YXd9gthTMI_3C3jHg1u7LzMhNtMkWOF5cNEFkULtjf3odWDEJTeUUCZ2NeEQ1mj~MUpHyKR52lTfGbWa74mchKzNjoOSQOnVGa'
    }
    res = get(url, headers=headers, data=payload)
    if res.status_code != 200:
        raise Exception('Error loading: {}'.format(url))
    return res.text


class Article:
    def __init__(self, title, text, url, source, ):
        self.text = text
        self.title = title
        self.url = url
        self.source = source

    @ staticmethod
    def get_full_article(url, source):
        page = get_webpage(url)
        soup = BeautifulSoup(page, 'html.parser')

        if source == 'fox-news':
            relevant_section = soup.find('div', {'class': 'article-body'})
            if relevant_section is None:
                return None
            else:
                return relevant_section.get_text()
        elif source == 'cnn':
            relevant_section = soup.find('section', {'id': 'body-text'})
            if relevant_section is None:
                return None
            else:
                return relevant_section.get_text()
        else:
            raise Exception('Source not supported')


if __name__ == "__main__":
    main()
