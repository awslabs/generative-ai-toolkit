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

import datetime
import re
from collections.abc import Mapping, Sequence
from decimal import Decimal
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mypy_boto3_dynamodb.type_defs import TableAttributeValueTypeDef

from boto3.dynamodb.types import Binary


class DynamoDbMapper:
    @classmethod
    def serialize(  # noqa: PLR0911
        cls,
        value: "Mapping[str, TableAttributeValueTypeDef] | TableAttributeValueTypeDef | Sequence[Mapping[str, TableAttributeValueTypeDef]]",
    ):
        if isinstance(value, dict):
            return {k: cls.serialize(v) for k, v in value.items()}
        elif isinstance(value, list):
            return [cls.serialize(v) for v in value]
        elif isinstance(value, set):
            return {cls.serialize(v) for v in value}
        if isinstance(value, datetime.datetime):
            if value.tzinfo is None:
                value = value.replace(tzinfo=datetime.UTC)  # type: ignore
            return value.isoformat().replace("+00:00", "Z")
        elif isinstance(value, float):
            return Decimal(str(value))
        elif isinstance(value, BaseException):
            return repr(value)
        else:
            return value

    @classmethod
    def deserialize(  # noqa: PLR0911
        cls,
        value: "Mapping[str, TableAttributeValueTypeDef] | TableAttributeValueTypeDef | Sequence[Mapping[str, TableAttributeValueTypeDef]]",
    ):
        if isinstance(value, dict):
            return {k: cls.deserialize(v) for k, v in value.items()}
        elif isinstance(value, list):
            return [cls.deserialize(v) for v in value]
        elif isinstance(value, set):
            return {cls.deserialize(v) for v in value}
        elif isinstance(value, Binary):
            return bytes(value)  # type: ignore
        elif isinstance(value, Decimal):
            if value % 1 == 0:
                return int(value)
            return float(value)
        elif isinstance(value, str) and re.fullmatch(
            r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?Z", value
        ):
            return datetime.datetime.fromisoformat(value.replace("Z", "+00:00"))
        else:
            return value
