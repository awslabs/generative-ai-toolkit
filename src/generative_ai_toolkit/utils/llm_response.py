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

import json
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mypy_boto3_bedrock_runtime.type_defs import ConverseResponseTypeDef


def get_text(response: "ConverseResponseTypeDef"):
    message = response["output"].get("message")
    if message:
        for msg in message["content"]:
            if "text" in msg:
                return msg["text"]
    raise ValueError(f"No text found in response: `{response}`")


def json_parse(response: "ConverseResponseTypeDef"):
    text = get_text(response).strip()
    try:
        return json.loads(text.replace("\n", " "))
    except json.decoder.JSONDecodeError as e:
        raise Exception(f"Could not JSON parse response: `{text}`") from e
