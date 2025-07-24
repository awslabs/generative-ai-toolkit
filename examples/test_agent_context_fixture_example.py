"""
Copyright 2025 Amazon.com, Inc. and its affiliates. All Rights Reserved.

Licensed under the Amazon Software License (the "License").
You may not use this file except in compliance with the License.
A copy of the License is located at

  https://aws.amazon.com/asl/

or in the "license" file accompanying this file. This file is distributed
on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either
express or implied. See the License for the specific language governing
permissions and limitations under the License.
"""

###
# Example showing how to use AgentContext.set_test_context() for test fixtures.
###

import pytest

from generative_ai_toolkit.context import AgentContext, AuthContext


# Simple pytest fixture using the helper function
@pytest.fixture
def agent_context():
    return AgentContext.set_test_context()


# More customized fixture
@pytest.fixture
def custom_agent_context():
    return AgentContext.set_test_context(
        conversation_id="custom-test-id",
        auth_context=AuthContext(principal_id="user123"),
    )


# Example tool that uses context
def example_tool(message: str) -> str:
    """Example tool that accesses agent context"""
    context = AgentContext.current()
    return f"User {context.auth_context['principal_id']} in conversation {context.conversation_id}: {message}"


# Test using the fixture
def test_tool_with_context(agent_context):
    result = example_tool("Hello")
    assert "test" in result
    assert "Hello" in result


def test_tool_with_custom_context(custom_agent_context):
    result = example_tool("Hello")
    assert "user123" in result
    assert "custom-test-id" in result
