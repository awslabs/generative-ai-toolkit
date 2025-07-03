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


import json
from datetime import date, datetime, time
from types import SimpleNamespace


class DefaultJsonEncoder(json.JSONEncoder):
    """
    A JSON encoder that is a bit more smart, and lenient, than Python's native JSON encoder
    """

    def default(self, o):
        if isinstance(o, date | datetime | time):
            return o.isoformat()
        if isinstance(o, SimpleNamespace):
            return o.__dict__
        elif hasattr(o, "__json__") and callable(o.__json__):
            return o.__json__()
        return str(o)
