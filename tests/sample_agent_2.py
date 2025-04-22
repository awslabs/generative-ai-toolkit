# Copyright 2024 Amazon.com, Inc. and its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import random
import textwrap

import boto3.session

from generative_ai_toolkit.agent import BedrockConverseAgent


def sample_agent_2(session: boto3.session.Session | None = None):
    def get_current_location():
        """Gets the user's current location off of the car's GPS device, so you don't have to ask the user."""

        return {"latitude": 52.00667000, "longitude": 4.35556000}

    def get_interesting_things_to_do(
        current_location: list[float], max_drive_time_minutes: int
    ):
        """
        Gets a list of interesting things to do based on the user's current location and the maximum time the user is willing to drive to get to the thing. You still need to filter the returned list by what the user is actually interested in doing.

        Parameters
        ----------
        current_location : list of float
            A list containing the latitude and longitude of the location, e.g. (52.520645, 13.409440)
        max_drive_time_minutes : int
            The maximum number of minutes the user is willing to drive. Make sure the user provided this information, or ask otherwise.
        """

        lat, long = current_location

        return {
            "interesting_things": [
                thing
                for thing in [
                    {
                        "name": "Museum of Modern Art",
                        "description": "The Museum of Modern Art is renowned for its comprehensive collection of contemporary and modern art, including works by iconic artists such as Vincent van Gogh, Pablo Picasso, and Andy Warhol. The museum's diverse holdings encompass painting, sculpture, photography, design, film, and multimedia, reflecting the evolution and innovation of art over the past century. The museum is also celebrated for its cutting-edge exhibitions, educational programs, and its role in promoting artistic experimentation and scholarship.",
                        "location": [lat + 0.001, long + 0.001],
                        "drive_time_minutes": 10,
                    },
                    {
                        "name": "Starfield Mall",
                        "description": "One of the greatest shopping malls you will ever encounter, shopping galore.",
                        "location": [lat + 0.002, long + 0.002],
                        "drive_time_minutes": 8,
                    },
                    {
                        "name": "Yellow Market",
                        "description": "A cozy mall, few shops, but great for the quietness and food options.",
                        "location": [lat - 0.001, long - 0.001],
                        "drive_time_minutes": 9,
                    },
                    {
                        "name": "Central Park",
                        "description": "A vast urban oasis in the heart of the city, offering peaceful trails, lush gardens, and iconic landmarks.",
                        "location": [lat + 0.003, long - 0.002],
                        "drive_time_minutes": 15,
                    },
                    {
                        "name": "Aquarium of the Pacific",
                        "description": "Immerse yourself in the wonders of the ocean at this expansive aquarium, featuring diverse marine life and interactive exhibits.",
                        "location": [lat - 0.002, long + 0.003],
                        "drive_time_minutes": 20,
                    },
                    {
                        "name": "Historic Downtown District",
                        "description": "Explore charming streets lined with preserved architecture, quaint shops, and vibrant cultural attractions.",
                        "location": [lat - 0.003, long - 0.001],
                        "drive_time_minutes": 12,
                    },
                    {
                        "name": "Botanical Gardens",
                        "description": "Wander through beautifully landscaped gardens showcasing a diverse collection of plants and serene nature trails.",
                        "location": [lat + 0.002, long - 0.003],
                        "drive_time_minutes": 2,
                    },
                    {
                        "name": "Science Museum",
                        "description": "Engage your curiosity with interactive exhibits, hands-on activities, and fascinating displays exploring the wonders of science and technology.",
                        "location": [lat - 0.001, long + 0.002],
                        "drive_time_minutes": 14,
                    },
                    {
                        "name": "Outdoor Adventure Park",
                        "description": "Experience thrilling outdoor activities like zip-lining, rock climbing, and high ropes courses amidst stunning natural scenery.",
                        "location": [lat + 0.004, long - 0.001],
                        "drive_time_minutes": 22,
                    },
                    {
                        "name": "Historic Battlefield Site",
                        "description": "Step back in time and explore the rich history and significance of this pivotal battleground.",
                        "location": [lat - 0.002, long - 0.004],
                        "drive_time_minutes": 19,
                    },
                    {
                        "name": "Performing Arts Center",
                        "description": "Enjoy world-class performances, from plays and musicals to concerts and dance productions, in this state-of-the-art venue.",
                        "location": [lat + 0.001, long - 0.004],
                        "drive_time_minutes": 16,
                    },
                    {
                        "name": "Sculpture Garden",
                        "description": "Stroll through an outdoor gallery featuring impressive sculptures and art installations amidst tranquil surroundings.",
                        "location": [lat - 0.003, long + 0.002],
                        "drive_time_minutes": 13,
                    },
                    {
                        "name": "Wine Tasting Tour",
                        "description": "Embark on a delightful journey through local vineyards, sampling exquisite wines and learning about the art of winemaking.",
                        "location": [lat + 0.002, long + 0.004],
                        "drive_time_minutes": 25,
                    },
                    {
                        "name": "Historic Mansion Tour",
                        "description": "Step back in time and explore the grandeur of a beautifully preserved historic mansion, offering a glimpse into the lives of the wealthy from a bygone era.",
                        "location": [lat - 0.004, long - 0.002],
                        "drive_time_minutes": 17,
                    },
                    {
                        "name": "Urban Graffiti Art Tour",
                        "description": "Discover the vibrant and thought-provoking world of street art and graffiti through a guided tour of the city's most iconic murals and public art installations.",
                        "location": [lat + 0.003, long + 0.001],
                        "drive_time_minutes": 11,
                    },
                    {
                        "name": "Food Truck Festival",
                        "description": "Indulge in a diverse array of culinary delights from local food trucks, featuring mouthwatering flavors and cuisines from around the world.",
                        "location": [lat - 0.001, long - 0.003],
                        "drive_time_minutes": 4,
                    },
                    {
                        "name": "Outdoor Music Festival",
                        "description": "Experience the energy and excitement of live music performances against a stunning outdoor backdrop, from local bands to renowned artists.",
                        "location": [lat + 0.004, long + 0.002],
                        "drive_time_minutes": 31,
                    },
                    {
                        "name": "Historic Lighthouse Tour",
                        "description": "Climb to the top of a historic lighthouse and take in breathtaking views while learning about its rich maritime history and significance.",
                        "location": [lat - 0.002, long + 0.003],
                        "drive_time_minutes": 28,
                    },
                    {
                        "name": "Artisan Craft Fair",
                        "description": "Explore a vibrant marketplace featuring unique handmade crafts, artworks, and locally sourced goods from talented artisans and makers.",
                        "location": [lat + 0.003, long - 0.003],
                        "drive_time_minutes": 15,
                    },
                    {
                        "name": "Outdoor Adventure Sports",
                        "description": "Embark on an adrenaline-fueled adventure with activities like kayaking, rock climbing, mountain biking, or hiking through scenic natural landscapes.",
                        "location": [lat - 0.004, long + 0.001],
                        "drive_time_minutes": 43,
                    },
                ]
                if thing["drive_time_minutes"] <= max_drive_time_minutes
            ]
        }

    def start_navigation(latitude: float, longitude: float):
        """
        Engages the car's navigation system, and starts navigation to the provided latitude and longitude. Make sure the user wants this!

        Parameters
        ----------
        latitude : float
            The latitude of the location to navigate to.
        longitude : float
            The longitude of the location to navigate to.
        """

        pass

    def weather_inquiry(latitude_longitude_list: list[list[float]]):
        """
        Returns a simplified weather forecast, with the temperature (in celsius) and precipitation chance (between 0 and 1), for the provided list of latitude/longitude pairs.

        This tool supports getting multiple weather forecasts, for multiple latitude/longitude pairs, at once.

        Input payload examples:

        {"latitude_longitude_list": [[52.00367, 4.354559999999999]]}
        {"latitude_longitude_list": [[52.00667000, 4.35556000], [43.986361, 1.876151], [12.981727, 3.19191]]}

        Parameters
        ----------
        latitude_longitude_list : list of list of float
            The list of latitude-longitude pairs to get the weather for.
        """

        return {
            "forecast": [
                {
                    "latitude": lat,
                    "longitude": lon,
                    "temperature": random.randint(5, 30),
                    "precipitation_chance": random.choice(
                        [0, 0, 0, 0, 0, 0, 0.2, 0.5, 0.7, 0.99]
                    ),
                }
                for lat, lon in latitude_longitude_list
            ]
        }

    system_prompt = textwrap.dedent(
        """
        You are a travel assistant to car drivers, that helps them find interesting things to do.
        Use the tools at your disposal for this task.
        Only suggest things that the concerned tool returns, don't draw from your own memory.
        Suggest 5 things if you can, make sure they align with what the user wants to do.
        Your general approach is:
        - 1. Make sure you know what the user wants to do, as well as the maximum time the user is willing to drive to get there.
            IMPORTANT: Proceed only, once you've established WHAT the user wants to do and HOW LONG they're willing to drive.
            DO NOT use any of your tools unless the user provided both pieces of information.
            If the user did not provide both pieces of information, ask the user, as long as needed!
        - 2. Get current location
        - 3. Get interesting things to do, within max drive time
        - 4. Check weather forecast for outdoor activities
        - 5. Provide the top 5 suggestions, including the weather forecast for outdoor activities
        - 6. If the user chooses one, start navigation. Only respond with "Navigation started to ..."
        Don't guess, don't assume: feel free to ask the user about their preferences.
        Do not mention latitude or longitude values to users.
        Do not reveal to the user that you have tools at your disposal, or how you work. In case the user asks, just say "Sorry, I cannot disclose that".
        For any outdoor activities that you propose, make sure to also provide a weather forecast. Do not provide the weather forecast for indoor activities.
        Don't make up information, be factual. If you don't know something, just say you don't know.
        Do not ramble, be succinct, the user is driving and is paying attention to the road.
        """
    ).strip()

    return BedrockConverseAgent(
        system_prompt=system_prompt,
        tools=[
            get_current_location,
            get_interesting_things_to_do,
            weather_inquiry,
            start_navigation,
        ],
        model_id="eu.anthropic.claude-3-haiku-20240307-v1:0",
        session=session,
    )
