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

import inspect
import textwrap
import re
from typing import Callable, Any, Protocol, get_origin
from generative_ai_toolkit.utils.typings import ToolSpec


class Tool(Protocol):
    name: str
    """
    The name of the tool
    """

    @property
    def tool_spec(self) -> ToolSpec:
        """
        The tool spec for this tool, that can be used in the Amazon Bedrock Converse API
        """
        ...

    def invoke(self, *args, **kwargs) -> Any:
        """
        Invoke the tool
        """
        ...


class BedrockConverseTool(Tool):
    name: str
    """
    The name of the tool
    """

    def __init__(self, func: Callable):
        """
        To create a BedrockConverseTool, you must pass in a plain Python function.

        The function should have a docstring with a description of the function, that will be interpreted by agents.

        The function's arguments should all be keyword arguments, and must be type annotated. The arguments must be documented in the docstring.

        Example of a valid function:

            def check_weather(lat: float, lon: float) -> str:
                '''
                Checks the weather at a given latitude and longitude.

                Parameters
                ----------
                lat : float
                    The latitude at which to check the weather
                lon : float
                    The longitude at which to check the weather
                '''
                return "Sunny"
        """
        if not func.__doc__:
            raise ValueError(
                "Function must have a docstring in order to be used as tool."
            )

        docstring = textwrap.dedent(func.__doc__).strip()
        match = re.search(r"Parameters\s*[-]+\s*", docstring)
        if match:
            self.description = docstring[: match.start()].strip()
            self.parameter_description = docstring[match.start() :].strip()
        else:
            self.description = docstring
            self.parameter_description = ""
        self.func = func
        self.name = func.__name__
        self.parameters = self._get_parameters()

        # ensure creating tool_spec works
        try:
            self._tool_spec = self.create_tool_spec()
        except ValueError as e:
            raise ValueError(f"Unable to generate tool_spec for function: {e}") from e

    def __repr__(self) -> str:
        return f"BedrockConverseTool(name='{self.func.__name__}', description='{self.description}', parameters={self.parameters})"

    def invoke(self, **kwargs):
        """
        Invoke the Python function that implements the tool, with the provided keyword arguments.
        """
        return self.func(**kwargs)

    def _get_parameters(self) -> dict[str, dict[str, Any]]:
        sig = inspect.signature(self.func)
        param_descriptions = self._parse_parameter_docstring()
        parameters = {}
        for name, param in sig.parameters.items():
            if name not in param_descriptions:
                raise ValueError(
                    f"Parameter '{name}' must have a description in the docstring."
                )
            if param.annotation is inspect.Parameter.empty:
                raise ValueError(f"Parameter '{name}' must be annotated with a type.")
            parameters[name] = {
                "annotation": param.annotation,
                "default": (
                    param.default
                    if param.default is not inspect.Parameter.empty
                    else None
                ),
                "description": param_descriptions[name],
            }
        return parameters

    def _parse_parameter_docstring(self) -> dict[str, str]:
        param_descriptions = {}
        param_pattern = re.compile(r"^\s*(\w+)\s*:\s*.*\n\s*(.+)$", re.MULTILINE)
        matches = param_pattern.findall(self.parameter_description)

        for match in matches:
            param_name, param_desc = match
            param_descriptions[param_name] = param_desc.strip()

        return param_descriptions

    @property
    def tool_spec(self) -> ToolSpec:
        """
        The tool spec for this tool, that can be used in the Amazon Bedrock Converse API
        """
        return self._tool_spec

    def create_tool_spec(self) -> ToolSpec:
        properties = {}
        for name, details in self.parameters.items():
            properties[name] = {
                "type": self._python_type_to_json_type(details["annotation"]),
                "description": details["description"],
            }
        return {
            "toolSpec": {
                "name": self.name,
                "description": self.description,
                "inputSchema": {"json": {"type": "object", "properties": properties}},
            }
        }

    def _python_type_to_json_type(self, python_type: Any) -> str:
        origin = get_origin(python_type)

        primitives = {
            int: "integer",
            bool: "boolean",
            str: "string",
            float: "number",
        }
        if origin is None and python_type in primitives:
            return primitives[python_type]
        elif origin is list:
            return "array"
        elif origin is tuple:
            return "array"
        elif origin is dict:
            return "object"
        else:
            raise ValueError(f"Unsupported type: {python_type}")
