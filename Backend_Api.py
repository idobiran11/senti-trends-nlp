import requests
import datetime


class NewYorkTimesAPI:
    API_KEY = 'P5BOlzt1xIFW4USj5SRoN4jl5hYQAAyQ'
    SECRET_KEY = 'zkJPCUJiSwFWJ62x'
    DATE_NOW = datetime.datetime.now()

    def __init__(self, search_term, search_time_backwards_days):
        self.search_term = search_term
        self.search_time_days = (self.DATE_NOW - datetime.timedelta(days=search_time_backwards_days)).strftime("%Y%m%d")

    def make_request(self):
        url = f"https://api.nytimes.com/svc/search/v2/articlesearch.json?q={self.search_term}&begin_date={self.search_time_days}&api-key={self.API_KEY}"
        response = requests.get(url)
        data = response.json()
        return data


if __name__ == '__main__':
    search_term = input("Enter a search term: ")
    search_time_backwards_days = int(input("Enter the number of days to search back: "))
    ny_times_api = NewYorkTimesAPI(search_term, search_time_backwards_days)
    data = ny_times_api.make_request()
    print(data)