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
from datetime import UTC, date, datetime, time

from generative_ai_toolkit.agent import BedrockConverseAgent
from generative_ai_toolkit.utils.json import DefaultJsonEncoder


class Foo:
    def __json__(self):
        return {
            "foo": "bar",
            "bar": datetime(2021, 1, 1, 1, 1, 1, tzinfo=UTC),
            "baz": date(2022, 2, 2),
            "qux": time(1, 2, 3, 4),
        }


def test_tool_result_json_encoder():
    foo = Foo()

    assert (
        json.dumps(foo, cls=DefaultJsonEncoder)
        == '{"foo": "bar", "bar": "2021-01-01T01:01:01+00:00", "baz": "2022-02-02", "qux": "01:02:03.000004"}'
    )


def test_agent_tool_json_response(mock_bedrock_converse):
    agent = BedrockConverseAgent(
        model_id="anthropic.claude-3-sonnet-20240229-v1:0",
        session=mock_bedrock_converse.session(),
    )

    def foo():
        """
        Return a Foo
        """
        return Foo()

    agent.register_tool(foo)
    mock_bedrock_converse.add_output(
        tool_use_output=[{"name": "foo", "input": {}, "toolUseId": "abc123"}]
    )
    mock_bedrock_converse.add_output(text_output=["done"])
    agent.converse("test")
    tool_result_msg = agent.messages[-2]
    assert tool_result_msg["content"][0] == {
        "toolResult": {
            "content": [
                {
                    "json": {
                        "toolResponse": {
                            "bar": "2021-01-01T01:01:01+00:00",
                            "baz": "2022-02-02",
                            "foo": "bar",
                            "qux": "01:02:03.000004",
                        },
                    },
                },
            ],
            "status": "success",
            "toolUseId": "abc123",
        },
    }


def test_agent_tool_json_response_custom_encoder(mock_bedrock_converse):
    class MyJsonEncoder(json.JSONEncoder):
        def default(self, o):
            if isinstance(o, Foo):
                return "That's a foo!"
            return super().default(o)

    agent = BedrockConverseAgent(
        model_id="anthropic.claude-3-sonnet-20240229-v1:0",
        session=mock_bedrock_converse.session(),
        tool_result_json_encoder=MyJsonEncoder,
    )

    def foo():
        """
        Return a Foo
        """
        return Foo()

    agent.register_tool(foo)
    mock_bedrock_converse.add_output(
        tool_use_output=[{"name": "foo", "input": {}, "toolUseId": "abc123"}]
    )
    mock_bedrock_converse.add_output(text_output=["done"])
    agent.converse("test")
    tool_result_msg = agent.messages[-2]
    assert tool_result_msg["content"][0] == {
        "toolResult": {
            "content": [
                {
                    "json": {
                        "toolResponse": "That's a foo!",
                    },
                },
            ],
            "status": "success",
            "toolUseId": "abc123",
        },
    }
