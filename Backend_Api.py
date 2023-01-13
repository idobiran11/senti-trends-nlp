import requests
import datetime

API_KEY = 'P5BOlzt1xIFW4USj5SRoN4jl5hYQAAyQ'
SECRET_KEY = 'zkJPCUJiSwFWJ62x'

# Set the search term
search_term = "Donald Trump"

# Set the date range (past year)
now = datetime.datetime.now()
past_year = (now - datetime.timedelta(days=365)).strftime("%Y%m%d")

# Make the request
url = f"https://api.nytimes.com/svc/search/v2/articlesearch.json?q={search_term}&begin_date={past_year}&api-key={api_key}"
response = requests.get(url)

# Parse the JSON response
data = response.json()