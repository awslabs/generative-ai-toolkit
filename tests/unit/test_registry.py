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


from generative_ai_toolkit.agent.registry import (
    DEFAULT_TOOL_REGISTRY,
    ToolRegistry,
    tool,
)


def test_tool_registry_add():
    """Test adding tools to the registry."""
    registry = ToolRegistry()

    def sample_tool1():
        return "Tool 1 result"

    def sample_tool2():
        return "Tool 2 result"

    registry.add(sample_tool1)
    assert len(registry) == 1
    assert registry[0] is sample_tool1

    registry.add(sample_tool2)
    assert len(registry) == 2
    assert registry[0] is sample_tool1
    assert registry[1] is sample_tool2


def test_tool_decorator_simple():
    """Test the @tool decorator without arguments."""

    # Create a clean test environment
    test_registry = ToolRegistry()

    # Test the simple decorator syntax
    @tool(tool_registry=test_registry)
    def sample_decorated_tool():
        """Sample tool."""
        return "Tool result"

    # Verify the function still works
    assert sample_decorated_tool() == "Tool result"

    # Verify the function was registered
    assert len(test_registry) == 1
    assert test_registry[0] is sample_decorated_tool


def test_tool_decorator_with_default_registry():
    """Test the @tool decorator with the default registry."""
    # Reset the default registry to ensure clean test state
    assert len(DEFAULT_TOOL_REGISTRY) == 0
    try:

        @tool
        def sample_tool_with_default_registry():
            """Sample tool with default registry."""
            return "Tool with default registry"

        # Verify the function still works
        assert sample_tool_with_default_registry() == "Tool with default registry"

        # Verify the function was registered in the default registry
        assert len(DEFAULT_TOOL_REGISTRY) == 1
        assert DEFAULT_TOOL_REGISTRY[0] is sample_tool_with_default_registry
    finally:
        # Restore the default registry
        DEFAULT_TOOL_REGISTRY.clear()


def test_tool_recursive_import():
    """Test that functions marked with @tool can actually be found"""
    import tools_registry_test.weather  # noqa: PLC0415

    assert len(DEFAULT_TOOL_REGISTRY) == 0
    ToolRegistry.recursive_import(tools_registry_test.weather)
    assert len(DEFAULT_TOOL_REGISTRY) == 1
    from tools_registry_test.weather.get_weather import (  # noqa: PLC0415
        get_good_weather,
    )

    assert DEFAULT_TOOL_REGISTRY[0] == get_good_weather


def test_tool_recursive_import_non_default_registry():
    """Test that functions marked with @tool(tool_registry=xyz) can actually be found"""
    import tools_registry_test.weather2  # noqa: PLC0415
    from tools_registry_test.other.other_registry import other_registry  # noqa: PLC0415

    assert len(other_registry) == 0
    ToolRegistry.recursive_import(tools_registry_test.weather2)
    assert len(other_registry) == 1
    from tools_registry_test.weather2.get_weather import (  # noqa: PLC0415
        get_indeterminate_weather,
    )

    assert other_registry[0] == get_indeterminate_weather


def test_tool_multi_registry():
    """Test that functions marked with @tool(tool_registry=[1,2]) can actually be found"""
    from tools_registry_test.other.other_registry import (  # noqa: I001, PLC0415
        yet_another_registry,
    )
    assert len(yet_another_registry) == 0
    import tools_registry_test.common.common_tool  # noqa: F401, PLC0415
    assert len(yet_another_registry) == 1
    assert (
        yet_another_registry[0] == tools_registry_test.common.common_tool.my_common_tool
    )
