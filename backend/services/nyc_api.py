import datetime
from requests import get, request
from bs4 import BeautifulSoup
from pynytimes import NYTAPI
import re
import pandas as pd

# Being Used:
SEARCH_FOR = "Trump"
DATE_NOW = datetime.datetime.now()
BEGIN_DATE = datetime.datetime(year=2022, month=1, day=1)
END_DATE = datetime.datetime(year=2023, month=1, day=31)
SOURCES_LIST = ["New York Times",
                "AP",
                "Reuters",
                "International Herald Tribune"]

# Not currently used, but can be added as lists in options like here:
# https://pynytimes.michadenheijer.com/search/article-search#options
TYPE_OF_MATERIAL = ["News Analysis", "News", "An Analysis"]
NEWS_DESK = ["Politics", "Foreign", "Business Day",
             "National", "Business", "Culture", "U.S.", "World"]

NUM_ARTICLES_PER_SOURCE = 30
API_KEY = 'P5BOlzt1xIFW4USj5SRoN4jl5hYQAAyQ'
SECRET_KEY = 'zkJPCUJiSwFWJ62x'


def main_function(search_term, begin_date, end_date, sources_list, num_articles_per_source):
    newsapi = NYTAPI(API_KEY, parse_dates=True)
    article_dict = {}
    for source in sources_list:
        article_dict[source] = []
        articles = newsapi.article_search(
            query=search_term,  # Search for articles about search term
            results=num_articles_per_source,  # Return this number of articles per source
            # Search for articles in between begin and end dates
            dates={
                "begin": begin_date,
                "end": end_date
            },
            options={
                "sort": "oldest",  # Sort by oldest options
                # Return articles from the following source
                "sources": [source]})

        data = pd.DataFrame(data={}, columns=[
                            'text', 'title', 'source', 'news_desk', 'type_of_material', 'date', 'url'])
        crnt = {}
        for article_data in articles:
            title = article_data['headline']['main']
            crnt['title'] = [re.sub(r"[^a-zA-Z0-9\s]", "", title)]

            crnt['url'] = [article_data['web_url']]
            # first_paragraph = article_data['lead_paragraph']
            crnt['date'] = [article_data['pub_date']]
            crnt['news_desk'] = [article_data['news_desk']]
            crnt['type_of_material'] = [article_data['type_of_material']]
            crnt['text'] = [Article.get_full_article(crnt['url'][0])]
            data = pd.concat([data, pd.DataFrame(data=crnt)],
                             ignore_index=True)
        data.to_csv('data/articles.csv', index=False)

    # algorithm(article_dict)


def algorithm(article_dict):
    pass


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
    def __init__(self, title, text, url, source, news_desk, type_of_material, date):
        self.text = text
        self.title = title
        self.url = url
        self.source = source
        self.date = date
        self.news_desk = news_desk
        self.type_of_material = type_of_material

    @ staticmethod
    def get_full_article(url):
        page = get_webpage(url)
        soup = BeautifulSoup(page, 'html.parser')
        text = soup.find('section', {'name': 'articleBody'}).get_text()
        return text


if __name__ == '__main__':
    main_function(SEARCH_FOR, BEGIN_DATE, END_DATE,
                  SOURCES_LIST, NUM_ARTICLES_PER_SOURCE)
