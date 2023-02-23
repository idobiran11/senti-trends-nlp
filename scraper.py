from bs4 import BeautifulSoup
from requests import get
from datetime import datetime
import pandas as pd
from sys import argv


def get_the_f_time(time_string):
    time_string = "March 8, 2021 2:33am EST"

    if time_string[-3:] in ('EST', 'EDT'):
        time_string = time_string[:-4]

    commaIndex = time_string.find(',')
    colonIndex = time_string.find(':')

    if time_string[commaIndex - 2] == ' ':
        time_string = (time_string[:commaIndex - 1] +
                       '0' + time_string[commaIndex - 1:])

    if time_string[colonIndex - 1] == ' ':
        time_string = (time_string[:colonIndex] +
                       '0' + time_string[colonIndex:])
    return datetime.strptime(time_string, "%B %d, %Y %I:%M%p")


object_name = 'Netanyahu'

start_date = '20220301'  # YYYYMMDD
end_date = '20230103'  # YYYYMMDD


def get_json(url, params={}):
    res = get(url, params=params)
    if (res.status_code != 200):
        print(f"WARN: Request to {url} failed, res:\n {res.text}")
    return res.json()


class fox_scraper:
    base_url = "https://api.foxnews.com/search/web"
    source_name = 'fox'

    def __init__(self, object_name, start_date, end_date):
        self.object_name = object_name
        self.start_date = start_date.replace('/', '')
        self.end_date = end_date.replace('/', '')

    def get_articles(self):
        page_num = 1
        self.articles = []

        next_page_index = 1

        params = {
            "q": self.object_name,
            "sort": f"date:r:{self.start_date}:{self.end_date}"}

        while (next_page_index is not None):

            res = get_json(self.base_url, params=params)
            total_count = res['queries']['request'][0].get(
                'totalResults', None) or 0

            if total_count == 0:
                print(
                    f"no articles for object {self.object_name} in dates {self.start_date}:{self.end_date}")
                break

            print(
                f"next page: found {total_count} results for {self.object_name} in {self.source_name}")

            for i, item in enumerate(res['items']):
                print(
                    f"scraping article {int(next_page_index)+ i}: {item['title']}")
                article = self._get_article(item)
                if article:
                    self.articles.append(self._get_article(item))

            next_page = res['queries'].get('nextPage', [None])[0]
            next_page_index = next_page['startIndex'] if next_page else None
            params['start'] = next_page_index
        return self

    def _get_article(self, item):
        article = {}
        article['url'] = item['link']
        article['source'] = self.source_name
        article['title'] = item['title']
        if not self._is_valid(article):
            return None
        article['text'], article['timestamp'] = self._get_article_data(
            article['url'])

        if not article['text']:
            return None

        return article

    def _get_article_data(self, article_url):
        """Returns  (text, date)"""
        res = get(article_url)
        soup = BeautifulSoup(res.text, 'html.parser')
        temp = soup.find('div', class_='article-body')
        if not temp:
            print(" -- skipping article, no article-body")
            return None, None
        text = temp.text

        temp = soup.find('div', 'article-date')
        temp = temp.find('time') if temp else None
        if not temp:
            print(" -- skipping article, no article-date")
            return None, None
        date = get_the_f_time(temp.text)
        return text, date

    def _is_valid(self, article):
        if 'video' in article['url'].split('/'):
            print(f" -- skipping video: {article['url']}")
            return False
        return True

    def to_csv(self):
        if not self.articles:
            print("no articles to save")
            return
        df = pd.DataFrame(self.articles)
        df.to_csv(
            f'data/{self.source_name}-articles-{self.object_name}.csv', index=False)


sources = {
    'fox': fox_scraper
    # 'cnn': cnn_scraper
}

try:
    fox_scraper("netanyahu", "2022/01/01", "2023/23/02").get_articles()
except Exception as e:
    print(e)
finally:
    print("saving to csv")
    fox_scraper.to_csv()

# if __name__ == "__main__":
#     if len(argv) != 5:
#         print("Usage: python scraper.py <source> <object_name> <start_date> <end_date>")
#         print("Example: python scraper.py netanyahu 2023/01/01 2023/01/10")
#         exit(1)
#     source, object_name, start_date, end_date = argv[1:]

#     if source in sources:
#         scraper = sources[source](object_name, start_date, end_date)
#     else:
#         print(f"source {source} not supported. use one of {sources.keys()}")
#         exit(1)
#     scraper = fox_scraper(object_name, start_date, end_date)
#     scraper.get_articles()
#     scraper.to_csv()
