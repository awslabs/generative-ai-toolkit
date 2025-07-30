"""
File for the available tools the agent can call.
"""

from datetime import datetime, timedelta

import requests
from pydantic import BaseModel

TICKETMASTER_API_KEY = "XXXX"  # ToDo: replace with your own Ticketmaster API key
OPEN_WEATHER_API_KEY = "XXX"  # ToDo: replace with your own openweathermap.org API key
NEWS_API_KEY = "XXX"  # ToDo: replace with your own newsapi.org API key


def get_weather(city_name: str):
    """
    Checks the weather at a given location.

    Parameters
    ----------
    city_name : string
        The name of the city to check the weather for
    """
    base_url = "http://api.openweathermap.org/data/2.5/weather?"
    complete_url = f"{base_url}appid={OPEN_WEATHER_API_KEY}&q={city_name}&units=metric"
    response = requests.get(complete_url)
    weather_data = response.json()
    weather = weather_data["weather"][0]["main"].lower()
    return f"The weather in {city_name} is {weather}"


class Article(BaseModel):
    """
    helper class to store data of an article
    """

    # source: Source
    # author: str | None
    title: str
    description: str
    # url: HttpUrl
    # urlToImage: HttpUrl | None
    publishedAt: str  # noqa: N815
    # content: str


def get_news(city_name: str, k: int = 5):
    """
    Gets the current news headlines for a city.

    Parameters
    ------
    city_name: string
    The name of the city
    k: int
    The number of articles to retrieve (default: 5)
    """

    base_url = "https://newsapi.org/v2/everything?"
    yesterday = datetime.now() - timedelta(days=1)
    yesterday_formatted = yesterday.strftime("%Y-%m-%d")
    complete_url = f"{base_url}q={city_name}&from={yesterday_formatted}&sortBy=publishedAt&apiKey={NEWS_API_KEY}"
    response = requests.get(complete_url)

    return [Article(**x).model_dump() for x in response.json()["articles"][:k]]


def search_events(city_name: str, k: int = 10):
    """
    Gets upcoming events in a city (e.g. concerts, comedy shows, etc.)
    for which tickets can be bought.
    This uses the Ticketmaster API.

    Parameters
    ------
    city_name: string
      The name of the city
    k: int
      The number of events to retrieve (default: 10)
    """

    url = "https://app.ticketmaster.com/discovery/v2/events.json"
    params = {
        "classificationName": "music",
        "apikey": TICKETMASTER_API_KEY,
        "city": city_name,
        "size": k,
    }

    response = requests.get(url, params=params)
    if response.status_code != 200:
        raise Exception(
            f"API request failed with status code: {response.status_code}: {response.text}"
        )

    data = response.json()
    events = []
    for event in data.get("_embedded", {}).get("events", []):
        name = event.get("name", "No Title")
        date = (
            event.get("dates", {}).get("start", {}).get("localDate", "No Date Provided")
        )
        genre = (
            event.get("classifications", [{}])[0]
            .get("genre", {})
            .get("name", "No Genre")
        )
        event_id = event.get("id", "No ID")
        events.append(
            {
                "id": event_id,
                "name": name,
                "genre": genre,
                "date": date,
            }
        )
    return events
