from time import sleep
from bs4 import BeautifulSoup
from requests import get
from parsel import Selector
from datetime import datetime, timedelta
import pandas as pd
from sys import argv

d = pd.read_csv('data/fox-articles-trump.csv')
pass


def get_the_f_time(time_string):
    original = time_string
    time_string = time_string.strip()
    time_string = time_string.strip('\n')
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

    time_string = time_string.replace('Published ', '')
    time_string = time_string.replace('Updated on ', '')
    time_string = time_string.strip()

    return datetime.strptime(time_string, "%B %d, %Y %I:%M%p")


def get_json(url, params={}):
    res = get(url, params=params)
    if (res.status_code != 200):
        print(f"WARN: Request to {url} failed, res:\n {res.text}")
        return None
    return res.json()


class scraper:
    articles = []
    object_name = None
    start_date = None
    end_date = None

    def __init__(self, object_name, start_date, end_date):
        self.object_name = object_name
        self.start_date = start_date
        self.end_date = end_date

    def to_csv(self):
        if not self.articles:
            print("no articles to save")
            return
        df = pd.DataFrame(self.articles)
        df = df.drop_duplicates(subset=['url'])
        df.to_csv(
            f'data/{self.source_name}-articles-{self.object_name}.csv', index=False)


class fox_scraper(scraper):
    base_url = "https://api.foxnews.com/search/web"
    source_name = 'fox'
    unique = set()

    def __init__(self, object_name, start_date, end_date=None):
        start_date = start_date.replace('/', '')
        if end_date:
            end_date = end_date.replace('/', '')
        else:
            end_date = datetime.now().strftime("%Y%m%d")
        super().__init__(object_name, start_date, end_date)

    def get_articles(self):
        step = 30
        const_start_date = datetime.strptime(self.start_date, "%Y%m%d")
        const_end_date = datetime.strptime(self.end_date, "%Y%m%d")
        number_of_days = (datetime.today() - const_start_date).days
        for days_from_start in range(0, number_of_days,  step):
            self.start_date = (
                const_start_date + timedelta(days=days_from_start)).strftime("%Y%m%d")
            self.end_date = (min(
                [const_start_date + timedelta(days=days_from_start + step), const_end_date])).strftime("%Y%m%d")

            self._get_articles()
            if self.end_date == datetime.today().strftime("%Y%m%d"):
                break
        return self

    def _get_articles(self):
        page_num = 1

        next_page_index = 1

        params = {
            "q": self.object_name,
            "sort": f"date:r:{self.start_date}:{self.end_date}"}

        while (next_page_index is not None):

            res = get_json(self.base_url, params=params)
            if not res:
                print(" -- unexpedted end of pages")
                break
            total_count = int(res['queries']['request'][0].get(
                'totalResults', None) or 0)

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
                    self.articles.append(article)

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
            print(f" -- skipping article, no article-body: {article_url}")
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
        url_parts = article['url'].split('/')
        if 'video' in url_parts or 'radio' in url_parts:
            print(f" -- skipping video/podcast: {article['url']}")
            return False
        return True


class cnn_scraper(scraper):
    base_url = "https://search.api.cnn.io/content"
    source_name = 'cnn'

    def __init__(self, object_name, start_date):
        start_date = datetime.strptime(start_date, "%Y/%m/%d")
        super().__init__(object_name, start_date, None)

    def get_articles(self):
        count = 1
        total_count = None
        outdated_count = 0
        params = {
            "q": self.object_name,
            "size": 50,
            "sort": "newest",
            "from": 1,
            "type": "article",
            "page": 1,
        }

        while ((not total_count) or count <= total_count) and outdated_count < 100:
            res = get_json(self.base_url, params=params)
            if not res:
                return
            if not total_count:
                total_count = res['meta']['of']
                print(
                    f"found {total_count} results for {self.object_name} in {self.source_name}")
            print(f"page {params['page']}")
            for item in res['result']:
                print(f"scraping article {count}: {item['headline']}")
                article = self._get_article(item)
                if article:
                    if article['timestamp'] >= self.start_date:
                        self.articles.append(article)
                    else:
                        print(" -- skipping article, not in date range")
                        outdated_count += 1
                count += 1
            params['page'] += 1
            params['from'] += 50
        return self

    def _get_article(self, item):
        article = {}
        article['url'] = item['url']
        article['source'] = self.source_name
        article['title'] = item['headline']
        article['text'] = item['body']
        article['timestamp'] = datetime.fromisoformat(
            item['firstPublishDate'].split('T')[0])

        return article


class wsj_scraper(scraper):
    base_url = 'https://www.wsj.com/search'

    def __init__(self, object_name, start_date):
        # start_date = datetime.strptime(start_date, "%Y/%m/%d")
        super().__init__(object_name, start_date, None)

    def get_articles(self):
        params = {
            "query": self.object_name,
            "sort": "date-desc",
            "startDate": self.start_date,
            "source": "wsjie, blog, autowire, wsjpro"
        }

        res = get_json(self.base_url, params=params)
        urls = self._get_article_urls(res)
        for url in urls:
            print(f"scraping article: {url}")
            article = self._get_article(url)
            if article:
                self.articles.append(article)
        return self

    def _get_article(self, url):
        pass

    def _get_article_urls(self, res):
        selector = Selector(text=res.text)
        all_urls = selector.xpath(
            "//article/div[contains(@class,'search-result')]//a/@href").extract()
        return all_urls


sources = {
    'fox': fox_scraper,
    'cnn': cnn_scraper,
    'wsj': wsj_scraper,
}

cnn = None

if __name__ == "__main__":
    if len(argv) != 4:
        print("Usage: python scraper.py <source> <object_name> <start_date>")
        print("Example: python scraper.py netanyahu 2023/01/01")
    else:
        source, object_name, start_date, end_date = argv[1:]

        if source in sources:
            news_scraper = sources[source](object_name, start_date)
        else:
            print(
                f"source {source} not supported. use one of {sources.keys()}")
            exit(1)
        news_scraper.get_articles().to_csv()

    source = 'cnn'
    news_scraper = sources[source]("trump", "2021/01/01",).get_articles()

    print("saving to csv")
    news_scraper.to_csv()
    print('waiting a bit to make sure', end='', )
    for i in range(4):
        print('.', end='')
        sleep(0.5)
        print('')
    print("trying to read csv")
    try:
        d = pd.read_csv(f'data/{source}-articles-netanyahu.csv')
        if (not d.empty):
            print("ido ze oved!!")
        else:
            print("ido ata zodek!!")
    except:
        print("ido ata zodek!!")
