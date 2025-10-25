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
from types import NoneType
from typing import Literal

import pytest

from generative_ai_toolkit.agent import BedrockConverseAgent, BedrockConverseTool, Tool
from generative_ai_toolkit.test import Expect


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


def test_explicit_tool_spec_no_parameters(mock_bedrock_converse):
    agent = BedrockConverseAgent(
        model_id="dummy", session=mock_bedrock_converse.session()
    )
    agent.register_tool(
        lambda: "Sunny",
        tool_spec={
            "name": "get_weather",
            "description": "Gets the current weather",
            "inputSchema": {},
        },
    )

    assert "get_weather" in agent.tools
    mock_bedrock_converse.add_output(tool_use_output=[{"name": "get_weather"}])
    mock_bedrock_converse.add_output(text_output=["Sunny"])
    agent.converse("What is the weather?")
    Expect(agent.traces).tool_invocations.to_include("get_weather").with_input({})


def test_explicit_tool_spec_with_parameters(mock_bedrock_converse):
    agent = BedrockConverseAgent(
        model_id="dummy", session=mock_bedrock_converse.session()
    )
    agent.register_tool(
        lambda preferred_weather: f"Not {preferred_weather}",
        tool_spec={
            "name": "get_weather",
            "description": "Gets the current weather",
            "inputSchema": {
                "json": {
                    "type": "object",
                    "properties": {
                        "preferred_weather": {
                            "type": "string",
                            "description": "The preferred weather",
                        },
                    },
                    "required": [
                        "preferred_weather",
                    ],
                }
            },
        },
    )

    assert "get_weather" in agent.tools
    mock_bedrock_converse.add_output(
        tool_use_output=[
            {"name": "get_weather", "input": {"preferred_weather": "Sunny"}}
        ]
    )
    mock_bedrock_converse.add_output(text_output=["Not sunny"])
    agent.converse("What is the weather? I'd like it to be sunny.")
    Expect(agent.traces).tool_invocations.to_include("get_weather").with_input(
        {"preferred_weather": "Sunny"}
    )


def test_tool_with_multiline_parameter_descriptions(mock_bedrock_converse):
    """
    Test that multi-line parameter descriptions are correctly parsed and included
    in the tool spec when invoking tools through the agent.
    """

    def analyze_data(
        data_source: str, filter_criteria: str, output_format: str = "json"
    ):
        """
        Analyzes data from a specified source with filtering capabilities.

        This is a more detailed description of the function.

        Parameters
        ----------
        data_source : str
            The source location of the data to analyze.
            This could be a file path, URL, or database connection string.
            Multiple formats are supported.
        filter_criteria : str
            The criteria to filter the data before analysis.
            Supports SQL-like syntax for complex filtering operations.
            Can include multiple conditions joined by AND/OR operators.
        output_format : str
            The desired output format for the analysis results.

            Common formats include json, csv, or xml.

        Examples
        --------
        This is an example.

        Other section
        --------
        Some other section
        """
        return f"Analyzed {data_source} with filter '{filter_criteria}' in {output_format} format"

    agent = BedrockConverseAgent(
        model_id="dummy", session=mock_bedrock_converse.session()
    )
    agent.register_tool(analyze_data)

    tool_spec = agent.tools["analyze_data"].tool_spec
    assert (
        tool_spec.get("description")
        == textwrap.dedent(
            """
            Analyzes data from a specified source with filtering capabilities.

            This is a more detailed description of the function.

            Examples
            --------
            This is an example.

            Other section
            --------
            Some other section
            """
        ).strip()
    )

    json_schema = tool_spec["inputSchema"].get("json")
    assert json_schema is not None
    properties = json_schema["properties"]

    assert (
        properties["data_source"]["description"]
        == textwrap.dedent(
            """
            The source location of the data to analyze.
            This could be a file path, URL, or database connection string.
            Multiple formats are supported.
            """
        ).strip()
    )
    assert (
        properties["filter_criteria"]["description"]
        == textwrap.dedent(
            """
            The criteria to filter the data before analysis.
            Supports SQL-like syntax for complex filtering operations.
            Can include multiple conditions joined by AND/OR operators.
            """
        ).strip()
    )
    assert (
        properties["output_format"]["description"]
        == textwrap.dedent(
            """
            The desired output format for the analysis results.

            Common formats include json, csv, or xml.
            """
        ).strip()
    )

    mock_bedrock_converse.add_output(
        tool_use_output=[
            {
                "name": "analyze_data",
                "input": {
                    "data_source": "/data/sales.csv",
                    "filter_criteria": "region='US' AND year=2024",
                },
            }
        ]
    )
    mock_bedrock_converse.add_output(
        text_output=["The analysis shows strong sales growth."]
    )

    agent.converse("Analyze the sales data")

    Expect(agent.traces).tool_invocations.to_include("analyze_data").with_input(
        {
            "data_source": "/data/sales.csv",
            "filter_criteria": "region='US' AND year=2024",
        }
    )

    llm_invocation = next(
        trace
        for trace in agent.traces
        if trace.attributes.get("ai.trace.type") == "llm-invocation"
    )
    tool_spec = llm_invocation.attributes["ai.llm.request.tool.config"]["tools"][0][
        "toolSpec"
    ]
    assert tool_spec == {
        "name": "analyze_data",
        "description": "Analyzes data from a specified source with filtering capabilities.\n\nThis is a more detailed description of the function.\n\nExamples\n--------\nThis is an example.\n\nOther section\n--------\nSome other section",
        "inputSchema": {
            "json": {
                "type": "object",
                "properties": {
                    "data_source": {
                        "type": "string",
                        "description": "The source location of the data to analyze.\nThis could be a file path, URL, or database connection string.\nMultiple formats are supported.",
                    },
                    "filter_criteria": {
                        "type": "string",
                        "description": "The criteria to filter the data before analysis.\nSupports SQL-like syntax for complex filtering operations.\nCan include multiple conditions joined by AND/OR operators.",
                    },
                    "output_format": {
                        "type": "string",
                        "description": "The desired output format for the analysis results.\n\nCommon formats include json, csv, or xml.",
                    },
                },
                "required": ["data_source", "filter_criteria"],
            }
        },
    }


def test_tool_with_none_type():
    """Test that type(None) is correctly mapped to null."""

    def return_none(value: NoneType):
        """
        Returns none value.

        Parameters
        ----------
        value : NoneType
            A none value
        """
        return None

    tool: Tool = BedrockConverseTool(return_none)
    assert tool.tool_spec == {
        "name": "return_none",
        "description": "Returns none value.",
        "inputSchema": {
            "json": {
                "type": "object",
                "properties": {
                    "value": {
                        "type": "null",
                        "description": "A none value",
                    },
                },
                "required": ["value"],
            }
        },
    }


def test_tool_with_literal_string_type():
    """Test that Literal string types include allowed values in description."""

    def set_color(color: Literal["red", "green", "blue"]):
        """
        Sets the color.

        Parameters
        ----------
        color : Literal["red", "green", "blue"]
            The color to set
        """
        return f"Color set to {color}"

    tool: Tool = BedrockConverseTool(set_color)
    assert tool.tool_spec == {
        "name": "set_color",
        "description": "Sets the color.",
        "inputSchema": {
            "json": {
                "type": "object",
                "properties": {
                    "color": {
                        "type": "string",
                        "description": 'The color to set\n\nAllowed values: "red", "green", "blue"',
                    },
                },
                "required": ["color"],
            }
        },
    }
    assert tool.invoke(color="red") == "Color set to red"


def test_tool_with_literal_integer_type():
    """Test that Literal integer types include allowed values in description."""

    def set_level(level: Literal[1, 2, 3, 4, 5]):
        """
        Sets the level.

        Parameters
        ----------
        level : Literal[1, 2, 3, 4, 5]
            The level to set
        """
        return f"Level set to {level}"

    tool: Tool = BedrockConverseTool(set_level)
    assert tool.tool_spec == {
        "name": "set_level",
        "description": "Sets the level.",
        "inputSchema": {
            "json": {
                "type": "object",
                "properties": {
                    "level": {
                        "type": "integer",
                        "description": "The level to set\n\nAllowed values: 1, 2, 3, 4, 5",
                    },
                },
                "required": ["level"],
            }
        },
    }
    assert tool.invoke(level=3) == "Level set to 3"


def test_tool_with_literal_boolean_type():
    """Test that Literal boolean types include allowed values in description."""

    def set_flag(flag: Literal[True]):
        """
        Sets the flag.

        Parameters
        ----------
        flag : Literal[True]
            The flag to set (must be True)
        """
        return f"Flag set to {flag}"

    tool: Tool = BedrockConverseTool(set_flag)
    assert tool.tool_spec == {
        "name": "set_flag",
        "description": "Sets the flag.",
        "inputSchema": {
            "json": {
                "type": "object",
                "properties": {
                    "flag": {
                        "type": "boolean",
                        "description": "The flag to set (must be True)\n\nAllowed values: True",
                    },
                },
                "required": ["flag"],
            }
        },
    }
    assert tool.invoke(flag=True) == "Flag set to True"


def test_tool_with_optional_literal_type():
    """Test that Optional[Literal[...]] types work correctly."""

    def set_mode(mode: Literal["fast", "slow", "medium"] | None = None):
        """
        Sets the processing mode.

        Parameters
        ----------
        mode : Literal["fast", "slow", "medium"] | None
            The processing mode
        """
        return f"Mode set to {mode}"

    tool: Tool = BedrockConverseTool(set_mode)
    assert tool.tool_spec == {
        "name": "set_mode",
        "description": "Sets the processing mode.",
        "inputSchema": {
            "json": {
                "type": "object",
                "properties": {
                    "mode": {
                        "type": "string",
                        "description": 'The processing mode\n\nAllowed values: "fast", "slow", "medium"',
                    },
                },
            }
        },
    }
    assert tool.invoke(mode="fast") == "Mode set to fast"
    assert tool.invoke() == "Mode set to None"


def test_tool_with_optional_none_type():
    """Test that optional None parameters work correctly."""

    def process_value(value: str, optional_none: None | str = None):
        """
        Process a value with an optional None parameter.

        Parameters
        ----------
        value : str
            The value to process
        optional_none : None | str
            An optional parameter that can be None or string
        """
        return f"Processed: {value}, Optional: {optional_none}"

    tool: Tool = BedrockConverseTool(process_value)
    assert tool.tool_spec == {
        "name": "process_value",
        "description": "Process a value with an optional None parameter.",
        "inputSchema": {
            "json": {
                "type": "object",
                "properties": {
                    "value": {
                        "type": "string",
                        "description": "The value to process",
                    },
                    "optional_none": {
                        "type": "string",
                        "description": "An optional parameter that can be None or string",
                    },
                },
                "required": ["value"],
            }
        },
    }
    assert tool.invoke(value="test") == "Processed: test, Optional: None"
    assert (
        tool.invoke(value="test", optional_none="data")
        == "Processed: test, Optional: data"
    )


def test_tool_with_mixed_literal_and_regular_types():
    """Test tool with both Literal and regular types."""

    def configure_system(
        name: str, mode: Literal["auto", "manual"], max_retries: int = 3
    ):
        """
        Configures the system.

        Parameters
        ----------
        name : str
            The system name
        mode : Literal["auto", "manual"]
            The operation mode
        max_retries : int
            Maximum number of retries
        """
        return f"Configured {name} in {mode} mode with {max_retries} retries"

    tool: Tool = BedrockConverseTool(configure_system)
    assert tool.tool_spec == {
        "name": "configure_system",
        "description": "Configures the system.",
        "inputSchema": {
            "json": {
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "The system name",
                    },
                    "mode": {
                        "type": "string",
                        "description": 'The operation mode\n\nAllowed values: "auto", "manual"',
                    },
                    "max_retries": {
                        "type": "integer",
                        "description": "Maximum number of retries",
                    },
                },
                "required": ["name", "mode"],
            }
        },
    }
    assert (
        tool.invoke(name="MySystem", mode="auto")
        == "Configured MySystem in auto mode with 3 retries"
    )
