# Copyright 2025 Amazon.com, Inc. and its affiliates. All Rights Reserved.
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

from generative_ai_toolkit.agent import registry


@registry.tool
def get_good_weather(city: str):
    """
    Gets the current weather in a city.

    Parameters
    ---
    city : str
      The city to get the weather for
    """
    return "Sunny"


# Not decorated, so shouldn't be scanned
def get_bad_weather(city: str):
    """
    Gets the current weather in a city.

    Parameters
    ---
    city : str
      The city to get the weather for
    """
    return "Sunny"


# Decorated for different registry, so shouldn't be picked up default registry
@registry.tool(tool_registry=registry.ToolRegistry())
def get_indeterminate_weather(city: str):
    """
    Gets the current weather in a city.

    Parameters
    ---
    city : str
      The city to get the weather for
    """
    return "Not raining, not sunny either"
