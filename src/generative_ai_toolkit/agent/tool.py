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

import inspect
import re
import textwrap
from collections.abc import Callable
from types import UnionType
from typing import (
    TYPE_CHECKING,
    Any,
    Protocol,
    Union,
    get_args,
    get_origin,
    runtime_checkable,
)

if TYPE_CHECKING:
    from mypy_boto3_bedrock_runtime.type_defs import ToolSpecificationTypeDef


@runtime_checkable
class Tool(Protocol):

    @property
    def tool_spec(self) -> "ToolSpecificationTypeDef":
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

    def __init__(
        self, func: Callable, *, tool_spec: "ToolSpecificationTypeDef | None" = None
    ):
        """
        To create a BedrockConverseTool, you must pass in a plain Python function.

        The function must be documented in a compatible way, or you must pass in a `tool_spec` explicitly.

        To document your function in a compatible way:

        - The function should have a docstring with a description of the function, that will be interpreted by agents.

        - The function's arguments should all be keyword arguments, and must be type annotated. The arguments must be documented in the docstring.

        - Example of a valid function:

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
        self.func = func
        if tool_spec:
            self._tool_spec = tool_spec
            return

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
        self.parameters, self.required_parameters = self._get_parameters()

        # ensure creating tool_spec works
        try:
            self._tool_spec = self.create_tool_spec()
        except ValueError as e:
            raise ValueError(f"Unable to generate tool_spec for function: {e}") from e

    def __repr__(self) -> str:
        return f"BedrockConverseTool(name='{self.func.__name__}', tool_spec={self.tool_spec})"

    def invoke(self, **kwargs):
        """
        Invoke the Python function that implements the tool, with the provided keyword arguments.
        """
        return self.func(**kwargs)

    def _get_parameters(self) -> tuple[dict[str, dict[str, Any]], list[str]]:
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
        return parameters, [
            param.name
            for param in sig.parameters.values()
            if param.default is inspect.Parameter.empty
        ]

    def _parse_parameter_docstring(self) -> dict[str, str]:
        param_descriptions = {}
        param_pattern = re.compile(r"^\s*(\w+)\s*:\s*.*\n\s*(.+)$", re.MULTILINE)
        matches = param_pattern.findall(self.parameter_description)

        for match in matches:
            param_name, param_desc = match
            param_descriptions[param_name] = param_desc.strip()

        return param_descriptions

    @property
    def tool_spec(self) -> "ToolSpecificationTypeDef":
        """
        The tool spec for this tool, that can be used in the Amazon Bedrock Converse API
        """
        return self._tool_spec

    def create_tool_spec(self) -> "ToolSpecificationTypeDef":
        properties = {}
        for name, details in self.parameters.items():
            properties[name] = {
                "type": self._python_type_to_json_type(details["annotation"]),
                "description": details["description"],
            }
        json = {
            "type": "object",
            "properties": properties,
        }
        tool_spec: ToolSpecificationTypeDef = {
            "name": self.func.__name__,
            "description": self.description,
            "inputSchema": {"json": json},
        }
        if self.required_parameters:
            json["required"] = self.required_parameters
        return tool_spec

    def _python_type_to_json_type(self, python_type: Any) -> str:
        origin = get_origin(python_type)

        # Handle optionality unions, e.g. List[str] | None
        if origin in (UnionType, Union):
            not_none = [t for t in get_args(python_type) if t is not type(None)]
            if len(not_none) == 1:
                return self._python_type_to_json_type(not_none[0])
            raise ValueError(f"Unsupported union type: {python_type}")

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
