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

from colorama import Fore, Style


def pretty_print_json(data, *, indent=0, output=print):
    """
    Pretty prints a JSON object with color coding for keys and values.
    """
    spaces = " " * indent
    if isinstance(data, dict):
        output(f"{spaces}{Fore.YELLOW}{{")
        for key, value in data.items():
            output(f"{spaces}  {Fore.YELLOW}{key}{Style.RESET_ALL}: ", end="")
            pretty_print_json(value, indent=indent + 4)
        output(f"{spaces}{Fore.YELLOW}}}")
    elif isinstance(data, list):
        output(f"{spaces}{Fore.GREEN}[")
        for item in data:
            pretty_print_json(item, indent=indent + 4)
        output(f"{spaces}{Fore.GREEN}]{Style.RESET_ALL}")
    else:
        output(f"{Fore.CYAN}{data}{Style.RESET_ALL}")
