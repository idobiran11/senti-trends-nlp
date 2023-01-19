import datetime
import urllib.request
from bs4 import BeautifulSoup
from pynytimes import NYTAPI

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
NEWS_DESK = ["Politics", "Foreign", "Business Day", "National", "Business", "Culture", "U.S.", "World"]

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
        for article in articles:
            title = article['headline']['print_headline']
            url = article['web_url']
            first_paragraph = article['lead_paragraph']
            date = article['pub_date']
            news_desk = article['news_desk']
            type_of_material = article['type_of_material']
            article_dict[source].append(Article(title, url, source, news_desk, type_of_material, first_paragraph, date))
    algorithm(article_dict)

def algorithm(article_dict):
    pass


class Article:
    def __init__(self, title, url, source, news_desk, type_of_material, first_paragraph, date):
        self.title = title
        self.url = url
        self.source = source
        self.date = date
        self.first_paragraph = first_paragraph
        self.news_desk = news_desk
        self.type_of_material = type_of_material

    @staticmethod
    def get_full_article(url):
        html = urllib.request.urlopen(url).read()
        soup = BeautifulSoup(html, 'html.parser')
        text = soup.find('section', {'name': 'articleBody'}).get_text()
        return text


if __name__ == '__main__':
    main_function(SEARCH_FOR, BEGIN_DATE, END_DATE, SOURCES_LIST, NUM_ARTICLES_PER_SOURCE)
