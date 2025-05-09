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

import textwrap

from generative_ai_toolkit.agent.agent import BedrockConverseAgent


def test_converse_stream_with_manual_mock(
    mock_bedrock_converse_stream_spaghetti_recipe_session,
):
    agent = BedrockConverseAgent(
        model_id="anthropic.claude-3-7-sonnet-20250219-v1:0",
        session=mock_bedrock_converse_stream_spaghetti_recipe_session,
        additional_model_request_fields={
            "reasoning_config": {"type": "enabled", "budget_tokens": 1024}
        },
    )

    def get_recipe(dish: str):
        """
        Get a recipe for the user

        Parameters
        ------
        dish: str
          The name of the dish to get a recipe for
        """
        return "Eggs, panceta, spaghetti, olive oil, good luck"

    agent.register_tool(get_recipe)

    res = agent.converse_stream("How should I make Spaghetti Carbonara?")

    expected = textwrap.dedent(
        """
        <thinking>
        The user is asking for a recipe for Spaghetti Carbonara. I have a tool available called `get_recipe` that can provide recipes.

        The required parameter for this function is:
        - dish: The name of the dish to get a recipe for

        In this case, the dish is "Spaghetti Carbonara". This is clearly stated in the user's request, so I can call the function with this parameter.
        </thinking>

        I can help you with a recipe for Spaghetti Carbonara! Let me get that for you.
        Here is the recipe bla bla
        """
    ).lstrip()

    assert "".join(res) == expected

    assert agent.messages == [
        {
            "role": "user",
            "content": [{"text": "How should I make Spaghetti Carbonara?"}],
        },
        {
            "role": "assistant",
            "content": [
                {
                    "reasoningContent": {
                        "reasoningText": {
                            "text": 'The user is asking for a recipe for Spaghetti Carbonara. I have a tool available called `get_recipe` that can provide recipes.\n\nThe required parameter for this function is:\n- dish: The name of the dish to get a recipe for\n\nIn this case, the dish is "Spaghetti Carbonara". This is clearly stated in the user\'s request, so I can call the function with this parameter.',
                            "signature": "ErcBCkgIAxABGAIiQGe2vlYmGavCTzCi6P+Ur23sKQwAgQ1SXGV+Qo++w+vr2Rwn1mCyu706GfqjmKp7lx6CnFfHBJj3fFp7kTlJB6MSDGLA/3wQ6NXSnYVtPxoM1TWYaoh7l8fbOTRHIjCQkiTU4Lz3b2j/VCc0jUJoY6kAr0XPv4HYlrof3Xtx9Bc46guGHv/H35kJVfQwhowqHRuvQD9PFtZHu35tnx3VNZagLfRzPeK+MFxzER+y",
                        }
                    }
                },
                {
                    "text": "I can help you with a recipe for Spaghetti Carbonara! Let me get that for you."
                },
                {
                    "toolUse": {
                        "toolUseId": "tooluse_W0JfmJb6Si61QMXXy0ynYw",
                        "name": "get_recipe",
                        "input": {"dish": "Spaghetti Carbonara"},
                    }
                },
            ],
        },
        {
            "role": "user",
            "content": [
                {
                    "toolResult": {
                        "toolUseId": "tooluse_W0JfmJb6Si61QMXXy0ynYw",
                        "status": "success",
                        "content": [
                            {
                                "json": {
                                    "toolResponse": "Eggs, panceta, spaghetti, olive oil, good luck"
                                }
                            }
                        ],
                    }
                }
            ],
        },
        {
            "content": [
                {
                    "text": "Here is the recipe bla bla",
                },
            ],
            "role": "assistant",
        },
    ]


def test_converse_stream_with_mock(mock_bedrock_converse):
    agent = BedrockConverseAgent(
        model_id="anthropic.claude-3-7-sonnet-20250219-v1:0",
        session=mock_bedrock_converse.session(),
        additional_model_request_fields={
            "reasoning_config": {"type": "enabled", "budget_tokens": 1024}
        },
    )

    def get_recipe(dish: str):
        """
        Get a recipe for the user

        Parameters
        ------
        dish: str
          The name of the dish to get a recipe for
        """
        return "Eggs, panceta, spaghetti, olive oil, good luck"

    agent.register_tool(get_recipe)

    mock_bedrock_converse.add_output(
        reasoning_output=[
            'The user is asking for a recipe for Spaghetti Carbonara. I have a tool available called `get_recipe` that can provide recipes.\n\nThe required parameter for this function is:\n- dish: The name of the dish to get a recipe for\n\nIn this case, the dish is "Spaghetti Carbonara". This is clearly stated in the user\'s request, so I can call the function with this parameter.',
        ],
        text_output=[
            "I can help you with a recipe for Spaghetti Carbonara! Let me get that for you."
        ],
        tool_use_output=[
            {
                "name": "get_recipe",
                "input": {"dish": "Spaghetti Carbonara"},
                "toolUseId": "tooluse_W0JfmJb6Si61QMXXy0ynYw",
            }
        ],
    )
    mock_bedrock_converse.add_output(text_output=["Here is the recipe bla bla"])

    res = agent.converse_stream("How should I make Spaghetti Carbonara?")

    expected = textwrap.dedent(
        """
        <thinking>
        The user is asking for a recipe for Spaghetti Carbonara. I have a tool available called `get_recipe` that can provide recipes.

        The required parameter for this function is:
        - dish: The name of the dish to get a recipe for

        In this case, the dish is "Spaghetti Carbonara". This is clearly stated in the user's request, so I can call the function with this parameter.
        </thinking>

        I can help you with a recipe for Spaghetti Carbonara! Let me get that for you.
        Here is the recipe bla bla
        """
    ).lstrip()

    assert "".join(res) == expected

    assert agent.messages == [
        {
            "role": "user",
            "content": [{"text": "How should I make Spaghetti Carbonara?"}],
        },
        {
            "role": "assistant",
            "content": [
                {
                    "reasoningContent": {
                        "reasoningText": {
                            "text": 'The user is asking for a recipe for Spaghetti Carbonara. I have a tool available called `get_recipe` that can provide recipes.\n\nThe required parameter for this function is:\n- dish: The name of the dish to get a recipe for\n\nIn this case, the dish is "Spaghetti Carbonara". This is clearly stated in the user\'s request, so I can call the function with this parameter.',
                            "signature": "4f5bb65b0c12f9c26d1c953063eb8595f3d45fbbe9ba6f123cc234e74c71ba94",
                        }
                    }
                },
                {
                    "text": "I can help you with a recipe for Spaghetti Carbonara! Let me get that for you."
                },
                {
                    "toolUse": {
                        "toolUseId": "tooluse_W0JfmJb6Si61QMXXy0ynYw",
                        "name": "get_recipe",
                        "input": {"dish": "Spaghetti Carbonara"},
                    }
                },
            ],
        },
        {
            "role": "user",
            "content": [
                {
                    "toolResult": {
                        "toolUseId": "tooluse_W0JfmJb6Si61QMXXy0ynYw",
                        "status": "success",
                        "content": [
                            {
                                "json": {
                                    "toolResponse": "Eggs, panceta, spaghetti, olive oil, good luck"
                                }
                            }
                        ],
                    }
                }
            ],
        },
        {
            "content": [
                {
                    "text": "Here is the recipe bla bla",
                },
            ],
            "role": "assistant",
        },
    ]


def test_converse_no_thinking(mock_bedrock_converse):
    agent = BedrockConverseAgent(
        model_id="anthropic.claude-3-7-sonnet-20250219-v1:0",
        session=mock_bedrock_converse.session(),
        additional_model_request_fields={
            "reasoning_config": {"type": "enabled", "budget_tokens": 1024}
        },
        include_reasoning_text_within_thinking_tags=False,
    )

    def get_recipe(dish: str):
        """
        Get a recipe for the user

        Parameters
        ------
        dish: str
          The name of the dish to get a recipe for
        """
        return "Eggs, panceta, spaghetti, olive oil, good luck"

    agent.register_tool(get_recipe)

    mock_bedrock_converse.add_output(
        reasoning_output=[
            'The user is asking for a recipe for Spaghetti Carbonara. I have a tool available called `get_recipe` that can provide recipes.\n\nThe required parameter for this function is:\n- dish: The name of the dish to get a recipe for\n\nIn this case, the dish is "Spaghetti Carbonara". This is clearly stated in the user\'s request, so I can call the function with this parameter.',
        ],
        text_output=[
            "I can help you with a recipe for Spaghetti Carbonara! Let me get that for you."
        ],
        tool_use_output=[
            {
                "name": "get_recipe",
                "input": {"dish": "Spaghetti Carbonara"},
                "toolUseId": "tooluse_W0JfmJb6Si61QMXXy0ynYw",
            }
        ],
    )
    mock_bedrock_converse.add_output(text_output=["Here is the recipe bla bla"])

    res = agent.converse_stream("How should I make Spaghetti Carbonara?")

    expected = textwrap.dedent(
        """
        I can help you with a recipe for Spaghetti Carbonara! Let me get that for you.
        Here is the recipe bla bla
        """
    ).lstrip()

    assert "".join(res) == expected


def test_converse_stream_no_thinking(mock_bedrock_converse):
    agent = BedrockConverseAgent(
        model_id="anthropic.claude-3-7-sonnet-20250219-v1:0",
        session=mock_bedrock_converse.session(),
        additional_model_request_fields={
            "reasoning_config": {"type": "enabled", "budget_tokens": 1024}
        },
        include_reasoning_text_within_thinking_tags=False,
    )

    def get_recipe(dish: str):
        """
        Get a recipe for the user

        Parameters
        ------
        dish: str
          The name of the dish to get a recipe for
        """
        return "Eggs, panceta, spaghetti, olive oil, good luck"

    agent.register_tool(get_recipe)

    mock_bedrock_converse.add_output(
        reasoning_output=[
            'The user is asking for a recipe for Spaghetti Carbonara. I have a tool available called `get_recipe` that can provide recipes.\n\nThe required parameter for this function is:\n- dish: The name of the dish to get a recipe for\n\nIn this case, the dish is "Spaghetti Carbonara". This is clearly stated in the user\'s request, so I can call the function with this parameter.',
        ],
        text_output=[
            "I can help you with a recipe for Spaghetti Carbonara! Let me get that for you."
        ],
        tool_use_output=[
            {
                "name": "get_recipe",
                "input": {"dish": "Spaghetti Carbonara"},
                "toolUseId": "tooluse_W0JfmJb6Si61QMXXy0ynYw",
            }
        ],
    )
    mock_bedrock_converse.add_output(text_output=["Here is the recipe bla bla"])

    res = agent.converse_stream("How should I make Spaghetti Carbonara?")

    expected = textwrap.dedent(
        """
        I can help you with a recipe for Spaghetti Carbonara! Let me get that for you.
        Here is the recipe bla bla
        """
    ).lstrip()

    assert "".join(res) == expected
