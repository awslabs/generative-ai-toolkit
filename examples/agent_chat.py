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
# The purpose of this example is to demonstrate the chat UI.
#
# Start this script by running: python examples/agent_chat.py
#
# This runs a fully mocked sample agent and shows it in the chat UI,
# so you can chat with it. The agent will always just echo back what you say,
# and invoke its tools (and one invocation will always error).
###

import textwrap

from mypy_boto3_bedrock_runtime.type_defs import (
    ConverseRequestTypeDef,
    ConverseStreamRequestTypeDef,
)

from generative_ai_toolkit.agent import BedrockConverseAgent
from generative_ai_toolkit.test.mock import MockBedrockConverse
from generative_ai_toolkit.ui import chat_ui

mock = MockBedrockConverse(stream_delay_between_tokens=0.01)


def response_generator(
    mock: MockBedrockConverse,
    request: ConverseRequestTypeDef | ConverseStreamRequestTypeDef,
):
    if "toolConfig" in request and "tools" in request["toolConfig"]:
        mock.add_output(
            tool_use_output=[
                {"name": tool["toolSpec"]["name"], "input": {"test_input": "foobar"}}
                for tool in request["toolConfig"]["tools"]
                if "toolSpec" in tool
            ]
        )
    if "messages" in request:
        last_msg = request["messages"][-1]["content"][0].get("text")
        mock.add_output(f"You said: {last_msg}")


mock.response_generator = response_generator


md = """
Some text

```
# Some markdown
```

Some other text
"""


def test_tool(test_input: str):
    """
    A test tool.

    Parameters
    ---
    test_input : str
      The input to the tool
    """
    return md


def test_tool2(test_input: str):
    """
    A test tool.

    Parameters
    ---
    test_input : str
      The input to the tool
    """
    return {"md": md}


def test_tool3(test_input: str):
    """
    A test tool.

    Parameters
    ---
    test_input : str
      The input to the tool
    """
    raise RuntimeWarning("This is a warning")


agent = BedrockConverseAgent(
    system_prompt=textwrap.dedent(
        """
        You are a helpful assistant.
        """
    ),
    model_id="eu.anthropic.claude-3-7-sonnet-20250219-v1:0",
    session=mock.session(),
)
agent.register_tool(test_tool)
agent.register_tool(test_tool2)
agent.register_tool(test_tool3)

agent.converse("Hello!")

demo = chat_ui(agent)
demo.launch()
