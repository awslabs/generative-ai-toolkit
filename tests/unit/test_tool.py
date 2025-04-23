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

import pytest

from generative_ai_toolkit.agent import BedrockConverseTool, Tool


def test_tool():

    def sample_function():
        """This is a sample function"""
        return "Hello World"

    tool: Tool = BedrockConverseTool(sample_function)
    assert tool.tool_spec == {
        "name": "sample_function",
        "description": "This is a sample function",
        "inputSchema": {"json": {"type": "object", "properties": {}}},
    }
    assert tool.invoke() == "Hello World"


def test_tool_with_multiline_description():

    def sample_function():
        """
        This is a sample function.

        And it's great!
        """
        return "Hello World"

    tool: Tool = BedrockConverseTool(sample_function)
    assert tool.tool_spec == {
        "name": "sample_function",
        "description": "This is a sample function.\n\nAnd it's great!",
        "inputSchema": {"json": {"type": "object", "properties": {}}},
    }
    assert tool.invoke() == "Hello World"


def test_tool_with_parameters():

    def sample_function_with_parameters(parameter1: str, parameter2: int):
        """
        This is a sample function with parameters.

        And it's great!

        Parameters
        ---
          parameter1 : str
            The first parameter
          parameter2 : int
            The second parameter
        """
        return f"Hello World {parameter1} {parameter2}"

    tool: Tool = BedrockConverseTool(sample_function_with_parameters)
    assert tool.tool_spec == {
        "name": "sample_function_with_parameters",
        "description": "This is a sample function with parameters.\n\nAnd it's great!",
        "inputSchema": {
            "json": {
                "type": "object",
                "properties": {
                    "parameter1": {
                        "type": "string",
                        "description": "The first parameter",
                    },
                    "parameter2": {
                        "type": "integer",
                        "description": "The second parameter",
                    },
                },
                "required": ["parameter1", "parameter2"],
            }
        },
    }
    assert tool.invoke(parameter1="foo", parameter2=1) == "Hello World foo 1"


def test_tool_with_optional_parameters():

    def sample_function_with_optional_parameters_1(
        parameter1: str, parameter2: int = 1
    ):
        """
        This is a sample function with optional parameters

        Parameters
        ---
          parameter1 : str
            The first parameter
          parameter2 : int
            The second parameter
        """
        return f"Hello World {parameter1} {parameter2}"

    tool1: Tool = BedrockConverseTool(sample_function_with_optional_parameters_1)

    def sample_function_with_optional_parameters_2(
        parameter1: str, parameter2: int | None = None
    ):
        """
        This is a sample function with optional parameters

        Parameters
        ---
          parameter1 : str
            The first parameter
          parameter2 : int
            The second parameter
        """
        if parameter2 is None:
            parameter2 = 1
        return f"Hello World {parameter1} {parameter2}"

    tool2: Tool = BedrockConverseTool(sample_function_with_optional_parameters_2)

    for index, tool in enumerate([tool1, tool2]):
        assert tool.tool_spec == {
            "name": f"sample_function_with_optional_parameters_{index + 1}",
            "description": "This is a sample function with optional parameters",
            "inputSchema": {
                "json": {
                    "type": "object",
                    "properties": {
                        "parameter1": {
                            "type": "string",
                            "description": "The first parameter",
                        },
                        "parameter2": {
                            "type": "integer",
                            "description": "The second parameter",
                        },
                    },
                    "required": [
                        "parameter1",
                    ],
                }
            },
        }
        assert tool.invoke(parameter1="foo") == "Hello World foo 1"


def test_tool_without_docstring():

    def sample_function_without_docstring():
        return "Hello World"

    with pytest.raises(
        ValueError, match="Function must have a docstring in order to be used as tool"
    ):
        BedrockConverseTool(sample_function_without_docstring)
