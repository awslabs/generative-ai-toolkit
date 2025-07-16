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
from decimal import Decimal
from typing import Any

from generative_ai_toolkit.utils.json import DefaultJsonEncoder


class DynamoDbMapper(DefaultJsonEncoder):
    """
    Helper class to convert Decimal to float, which boto3 dynamodb doesn't do out of the box :(
    """

    def default(self, o):
        if isinstance(o, Decimal):
            if o.as_integer_ratio()[1] == 1:
                return int(o)
            return float(o)
        return super().default(o)

    @classmethod
    def to_dynamo(cls, data: Any) -> Any:
        return json.loads(json.dumps(data, cls=cls), parse_float=Decimal)

    @classmethod
    def from_dynamo(cls, data: Any) -> Any:
        return json.loads(
            json.dumps(data, cls=cls),
        )
