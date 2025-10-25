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
from types import NoneType, UnionType
from typing import (
    TYPE_CHECKING,
    Any,
    Literal,
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
        # Look for NumPy-style Parameters section: "Parameters" followed by dashes/equals (e.g., "---" or "===")
        parameter_section_match = re.search(r"Parameters\s*[-=]{3,}\s*", docstring)
        if parameter_section_match:
            # Find where the Parameters section ends (when the next section starts)
            # Pattern matches NumPy-style section headers like "Examples\n---" or "Returns\n==="
            # \n           - newline before section name
            # ([A-Z][\w\s]+) - section name starting with capital letter, followed by word chars/spaces
            # \n           - newline after section name
            # [-=]{3,}     - at least 3 dashes or equals for underline
            # \s*          - optional trailing whitespace
            sections_pattern = r"\n([A-Z][\w\s]+)\n[-=]{3,}\s*"

            # Search for sections after the Parameters section
            next_section_match = re.search(
                sections_pattern, docstring[parameter_section_match.start() :]
            )

            if next_section_match:
                # There's a section after Parameters
                params_end = (
                    parameter_section_match.start() + next_section_match.start()
                )

                # Description = before Parameters + after Parameters section
                description_before = docstring[
                    : parameter_section_match.start()
                ].strip()
                description_after = docstring[params_end:].strip()
                self.description = description_before + "\n\n" + description_after

                # Parameter description = just the Parameters section
                self.parameter_description = docstring[
                    parameter_section_match.start() : params_end
                ].strip()
            else:
                # No section after Parameters, everything after is parameters
                self.description = docstring[: parameter_section_match.start()].strip()
                self.parameter_description = docstring[
                    parameter_section_match.start() :
                ].strip()
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

            # Extract literal values if this is a Literal type
            literal_values = self._extract_literal_values(param.annotation)

            parameters[name] = {
                "annotation": param.annotation,
                "default": (
                    param.default
                    if param.default is not inspect.Parameter.empty
                    else None
                ),
                "description": param_descriptions[name],
                "literal_values": literal_values,
            }
        return parameters, [
            param.name
            for param in sig.parameters.values()
            if param.default is inspect.Parameter.empty
        ]

    def _parse_parameter_docstring(self) -> dict[str, str]:
        """
        Parse a NumPy-style Parameters section into {param_name: multi-line description}.
        - Allows blank lines inside descriptions.
        - Stops before the next param with the same indent or section end.
        - Handles final lines without a trailing newline.
        """
        # Complex regex to match NumPy-style parameter entries with multi-line descriptions
        # Example it matches:
        #     param_name : str
        #         This is the description line 1.
        #         Description line 2.
        #
        #         Description line 3 after blank line.
        param_pattern = re.compile(
            r"^"  # Match at start of line (with re.MULTILINE, this is any line start)
            r"(?P<indent>[ \t]*)"  # Capture the indentation (spaces/tabs) of the parameter line
            r"(?P<name>\w+)"  # Capture the parameter name (letters, digits, underscore)
            r"\s*:\s*"  # Match colon with optional whitespace around it
            r".*"  # Match the rest of the line (type annotation, etc.)
            r"\r?\n"  # Match newline (handles both Unix \n and Windows \r\n)
            r"(?P<desc>("  # Start capturing the description block as a group
            r"^(?P=indent)[ \t]+[^\r\n]*(?:\r?\n|\Z)"  # Description line: same indent + extra spaces/tab + content + newline/end
            r"|"  # OR
            r"^[ \t]*(?:\r?\n|\Z)"  # A blank line (only whitespace) + newline/end
            r")+)",  # One or more description/blank lines (the + makes it required)
            re.MULTILINE,  # ^ and $ match line boundaries, not just string boundaries
        )

        src = self.parameter_description or ""
        results: dict[str, str] = {}

        for m in param_pattern.finditer(src):
            name = m.group("name")
            desc_block = m.group("desc")

            # Remove the base indentation from description lines
            # (?m) - multiline mode for ^
            # ^    - start of each line
            # (?:...) - non-capturing group
            # {re.escape(m.group("indent"))} - the exact indent of the param line (e.g., "    ")
            # [ \t]+ - followed by at least one space or tab (the description indent)
            # This removes "    " (param indent) + " " (desc indent) from each line
            desc_block = re.sub(
                rf"(?m)^(?:{re.escape(m.group('indent'))}[ \t]+)",
                "",  # Replace with empty string (remove the indentation)
                desc_block,
            )

            # Trim only outer whitespace; keep internal blank lines for formatting.
            results[name] = desc_block.strip()

        return results

    def _extract_literal_values(self, python_type: Any) -> list[Any] | None:
        """
        Extract literal values from a Literal type annotation.
        Handles Optional[Literal[...]] and Literal[...] | None patterns.
        Returns None if not a Literal type.
        """
        origin = get_origin(python_type)

        # Check if it's a Literal type directly
        if origin is Literal:
            return list(get_args(python_type))

        # Check if it's an Optional[Literal[...]] or Literal[...] | None
        if origin in (UnionType, Union):
            args = get_args(python_type)
            # Filter out None types
            not_none = [t for t in args if t is not NoneType]
            # If there's exactly one non-None type and it's a Literal
            if len(not_none) == 1 and get_origin(not_none[0]) is Literal:
                return list(get_args(not_none[0]))

        return None

    @property
    def tool_spec(self) -> "ToolSpecificationTypeDef":
        """
        The tool spec for this tool, that can be used in the Amazon Bedrock Converse API
        """
        return self._tool_spec

    def create_tool_spec(self) -> "ToolSpecificationTypeDef":
        properties = {}
        for name, details in self.parameters.items():
            description = details["description"]

            # If this parameter has literal values, append them to the description
            if details.get("literal_values"):
                literal_values = details["literal_values"]
                # Format the values nicely (with quotes for strings)
                formatted_values = ", ".join(
                    f'"{v}"' if isinstance(v, str) else str(v) for v in literal_values
                )
                description = (
                    f"{description.rstrip()}\n\nAllowed values: {formatted_values}"
                )

            properties[name] = {
                "type": self._python_type_to_json_type(details["annotation"]),
                "description": description,
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

    def _python_type_to_json_type(self, python_type: Any) -> str:  # noqa: PLR0911
        origin = get_origin(python_type)

        # Handle optional unions like str | None
        if origin in (UnionType, Union):
            not_none = [t for t in get_args(python_type) if t is not NoneType]
            if len(not_none) == 1:
                return self._python_type_to_json_type(not_none[0])
            raise ValueError(f"Unsupported union type: {python_type}")

        # Handle Literal types - infer base type from first literal value
        if origin is Literal:
            literal_values = get_args(python_type)
            if not literal_values:
                raise ValueError(f"Literal type has no values: {python_type}")
            first_value = literal_values[0]
            if isinstance(first_value, bool):
                return "boolean"
            elif isinstance(first_value, int):
                return "integer"
            elif isinstance(first_value, str):
                return "string"
            elif isinstance(first_value, float):
                return "number"
            else:
                raise ValueError(f"Unsupported literal value type: {type(first_value)}")

        primitives = {
            int: "integer",
            bool: "boolean",
            str: "string",
            float: "number",
            NoneType: "null",
        }

        # Handle primitive types
        if python_type in primitives:
            return primitives[python_type]

        # Handle collection generics
        if origin in (list, tuple) or python_type in (list, tuple):
            return "array"
        if origin is dict or python_type is dict:
            return "object"

        raise ValueError(f"Unsupported type: {python_type}")
