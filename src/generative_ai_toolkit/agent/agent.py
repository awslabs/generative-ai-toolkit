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

import json
import time
from typing import Any, Callable, Iterable, Mapping, Protocol, Sequence, cast

import boto3.session
import boto3
import botocore.exceptions

from generative_ai_toolkit.utils.typings import (
    MessageContent,
    NonStreamingResponse,
    Message,
    LlmRequest,
    StreamingResponse,
    ContentBlockDelta,
)
from generative_ai_toolkit.agent.tool import Tool, BedrockConverseTool
from generative_ai_toolkit.conversation_history import (
    ConversationHistory,
    InMemoryConversationHistory,
)
from generative_ai_toolkit.tracer import AgentTracer, SingleConversationTracer, Trace
from generative_ai_toolkit.utils.logging import logger


class Agent(Protocol):
    @property
    def model_id(self) -> str:
        """
        The LLM model_id of the agent
        """
        ...

    @property
    def system_prompt(self) -> str | None:
        """
        The system prompt of the agent
        """
        ...

    @property
    def tools(self) -> dict[str, Tool]:
        """
        The tools that have been registered with the agent.
        The agent can decide to use these tools during conversations.
        """
        ...

    @property
    def messages(self) -> Sequence[Message]:
        """
        Get the messages sent to the agent so far (for the current conversation)
        """
        ...

    @property
    def conversation_id(self) -> str:
        """
        Get the conversation id of the agent.
        """
        ...

    @property
    def traces(self) -> Sequence[Trace]:
        """
        Get the collected traces so far (for the current conversation)
        """
        ...

    def set_conversation_id(self, conversation_id: str) -> None:
        """
        Set the conversation id of the agent.
        """
        ...

    @property
    def auth_context(self) -> str | None:
        """
        The current auth context of the agent.
        """
        ...

    def set_auth_context(self, auth_context: str | None) -> None:
        """
        Set the auth context of the agent.
        """
        ...

    def reset(self) -> None:
        """
        Reset the state of the agent, e.g. in order to start a new conversation.
        (This does not unregister tools)
        """
        ...

    def register_tool(self, tool: Tool) -> Tool:
        """
        Register a tool with the agent.
        The agent can decide to use these tools during conversations.
        """
        ...

    def converse(self, user_input: str, tools: Sequence[Tool] | None = None) -> str:
        """
        Start or continue a conversation with the agent and return the agent's response as string.

        If you provide tools, that list of tools supersedes any tools that have been registered with the agent (but otherwise does not force their use).

        The agent may decide to use tools, and will do so autonomously (limited by the max_successive_tool_invocations that you've set on the agent).
        """
        ...

    def converse_stream(
        self, user_input: str, tools: Sequence[Tool] | None = None
    ) -> Iterable[str]:
        """
        Start or continue a conversation with the agent and return the agent's response as an iterable of strings.

        The caller must consume this iterable and collect all strings and concatenate them to get the full response.

        The iterable ends when the agent requests new user input, and then you should call this function again with the new user input.

        If you provide tools, that list of tools supersedes any tools that have been registered with the agent (but otherwise does not force their use).

        The agent may decide to use tools, and will do so autonomously (limited by the max_successive_tool_invocations that you've set on the agent).
        """
        ...


class BedrockConverseAgent(Agent):
    _model_id: str
    _system_prompt: str | None
    _tools: dict[str, Tool]
    _conversation_history: ConversationHistory
    _tracer: AgentTracer

    def __init__(
        self,
        *,
        model_id: str,
        system_prompt: str | None = None,
        maxTokens: int | None = None,
        temperature: float | None = None,
        topP: float | None = None,
        stop_sequences: list[str] | None = None,
        conversation_history: ConversationHistory
        | Callable[..., ConversationHistory]
        | None = None,
        tracer: AgentTracer | None = None,
        session: boto3.session.Session | None = None,
        tools: Sequence[Callable] | None = None,
        max_successive_tool_invocations: int = 10,
    ) -> None:
        """
        Create an Agent that will use the Bedrock Converse API to operate.

        Parameters
        ----------
        system_prompt : string
            The system prompt to start the conversation with the agent.
        model_id : string
            The model to use for the agent.
        maxTokens : int, optional
            The maximum number of tokens to generate in the response, by default None
            (no limit)
            Note: this is a hard limit, the actual response may be shorter.
        temperature : float, optional
            The temperature to use for the agent, by default None
            (use the model's default)
        topP : float, optional
            The top_p value to use for the agent, by default None
            (use the model's default)
        stop_sequences : list[str], optional
            The stop sequences to use for the agent, by default None
            (use the model's default)
        conversation_history : a ConversationHistory instance, or a Callable that returns a ConversationHistory instance, optional
            The conversation history to use for the agent, by default InMemoryConversationHistory
        tracer : AgentTracer, optional
            The tracer to use for the agent, by default NoopAgentTracer (which does not trace)
        session : boto3.session.Session, optional
            The AWS session to use for the Bedrock Converse API, by default None (use the default session)
        tools : Sequence[Callable], optional
            The tools to register with the agent, by default None
        max_successive_tool_invocations : int, optional
            The maximum number of successive tool invocations allowed, by default 10
        """
        self._system_prompt = system_prompt
        self._model_id = model_id
        self._tools = {}
        self.bedrock_client = (session or boto3).client("bedrock-runtime")
        if callable(conversation_history):
            self._conversation_history = conversation_history()
        elif conversation_history:
            self._conversation_history = conversation_history
        else:
            self._conversation_history = InMemoryConversationHistory()
        self._tracer = tracer or SingleConversationTracer()
        self.default_inference_config = {}
        if maxTokens is not None:
            self.default_inference_config["maxTokens"] = maxTokens
        if temperature is not None:
            self.default_inference_config["temperature"] = temperature
        if topP is not None:
            self.default_inference_config["topP"] = topP
        if stop_sequences is not None:
            self.default_inference_config["stopSequences"] = stop_sequences
        if tools:
            for tool in tools:
                self.register_tool(tool)
        if max_successive_tool_invocations < 0:
            raise ValueError("max_successive_tool_invocations must be positive")
        self.max_converse_iterations = max_successive_tool_invocations + 1

    @property
    def model_id(self):
        return self._model_id

    @property
    def system_prompt(self):
        return self._system_prompt

    @property
    def tools(self):
        """
        The tools that have been registered with the agent.
        The agent can decide to use these tools during conversations.
        """
        return self._tools

    @property
    def messages(self):
        """
        Get the messages sent to the agent so far (for the current conversation)
        """
        return self._conversation_history.messages

    @property
    def conversation_id(self):
        """
        Get the conversation id of the agent.
        """
        return self._conversation_history.conversation_id

    @property
    def traces(self):
        """
        Get the collected traces so far (for the current conversation)
        """
        return self._tracer.get_traces(
            self._conversation_history.conversation_id, auth_context=self.auth_context
        )

    def set_conversation_id(self, conversation_id: str):
        """
        Set the conversation id of the agent.
        """
        self._conversation_history.set_conversation_id(conversation_id)

    @property
    def auth_context(self):
        """
        The current auth context of the agent.
        """
        return self._conversation_history.auth_context

    def set_auth_context(self, auth_context: str | None) -> None:
        """
        Set the auth context of the agent.
        """
        self._conversation_history.set_auth_context(auth_context)

    def reset(self):
        """
        Reset the state of the agent, e.g. in order to start a new conversation.
        (This does not unregister tools)
        """
        self._conversation_history.reset()

    def register_tool(self, tool: Callable | Tool) -> Tool:
        """
        Register a tool with the agent.
        The agent can decide to use these tools during conversations.
        If you provide a Python function (Callable), it will be converted to a BedrockConverseTool for you.
        """
        if callable(tool):
            tool = BedrockConverseTool(tool)
        self._tools[tool.name] = tool
        return tool

    def _invoke_tool(self, msg: MessageContent, tools: Mapping[str, Tool]):
        if "toolUse" not in msg:
            raise ValueError("Invalid tool usage response.")
        tool_use = msg["toolUse"]
        tool_error = None
        start = time.perf_counter()
        try:
            tool_name = tool_use["name"]
            tool = tools[tool_name]
            tool_response = tool.invoke(**tool_use["input"])
            if not isinstance(tool_response, dict):
                tool_response = {"tool_response": tool_response}
        except Exception as err:
            tool_response = {
                "message": f"Error invoking tool {tool_use['name']}: {err}"
            }
            tool_error = err
        finally:
            latency_ms = int(1000 * (time.perf_counter() - start))

        self._tracer.trace(
            self._conversation_history.conversation_id,
            "TOOL",
            {
                "tool_name": tool_use["name"],
                "tool_use_id": tool_use["toolUseId"],
                "tool_input": tool_use["input"],
            },
            {"tool_response": tool_response, "latency_ms": latency_ms},
            auth_context=self.auth_context,
        )
        self._conversation_history.add_message(
            {
                "role": "user",
                "content": [
                    {
                        "toolResult": {
                            "toolUseId": tool_use["toolUseId"],
                            "status": "success" if not tool_error else "error",
                            "content": [{"json": tool_response}],
                        }
                    },
                ],
            },
        )

    def converse(
        self, user_input: str, tools: Sequence[Callable | Tool] | None = None
    ) -> str:
        """
        Start or continue a conversation with the agent and return the agent's response as string.

        If you provide tools, that list of tools supersedes any tools that have been registered with the agent (but otherwise does not force their use).

        The agent may decide to use tools, and will do so autonomously (limited by the max_successive_tool_invocations that you've set on the agent).
        """

        # If the current instance has an override for converse_stream, use that one instead
        if (
            type(self).converse is BedrockConverseAgent.converse
            and type(self).converse_stream is not BedrockConverseAgent.converse_stream
        ):
            return "".join(self.converse_stream(user_input, tools=tools))

        if not user_input:
            raise ValueError("Missing user input")

        self._conversation_history.add_message(
            {
                "role": "user",
                "content": [
                    {
                        "text": user_input,
                    },
                ],
            },
        )

        request: LlmRequest = {
            "modelId": self.model_id,
            "inferenceConfig": self.default_inference_config,
            "messages": list(self._conversation_history.messages),
        }
        if self.system_prompt:
            request["system"] = [
                {
                    "text": self.system_prompt,
                },
            ]
        tools_available = self.tools
        if tools is not None:
            tools = [
                BedrockConverseTool(tool) if callable(tool) else tool for tool in tools
            ]
            tools_available = (
                {tool.name: tool for tool in tools} if tools is not None else {}
            )
        if tools_available:
            request["toolConfig"] = {
                "tools": [tool.tool_spec for tool in tools_available.values()],
            }

        texts: list[str] = []
        for _ in range(self.max_converse_iterations):
            # Call LLM
            try:
                response = cast(
                    NonStreamingResponse,
                    self.bedrock_client.converse(**cast(Any, request)),
                )
            except botocore.exceptions.ClientError as err:
                logger.error(str(err), request=request, response=err.response)
                raise
            self._tracer.trace(
                self._conversation_history.conversation_id,
                "LLM",
                request,
                response,
                auth_context=self.auth_context,
            )

            # Capture text to show user
            for message in response["output"]["message"]["content"]:
                if "text" in message:
                    texts.append(message["text"])

            self._conversation_history.add_message(response["output"]["message"])

            if response["stopReason"] == "tool_use":
                last_msg = response["output"]["message"]["content"][-1]
                self._invoke_tool(last_msg, tools=tools_available)
                request["messages"] = list(self._conversation_history.messages)
                continue

            elif response["stopReason"] == "end_turn":
                concatenated = "\n".join(texts)
                texts = []
                return concatenated

            raise Exception("Unexpected LLM response")
        else:
            raise Exception(
                "Too many successive tool invocations:{self.max_converse_iterations} "
            )

    def converse_stream(
        self, user_input: str, tools: Sequence[Callable | Tool] | None = None
    ) -> Iterable[str]:
        """
        Start or continue a conversation with the agent and return the agent's response as an iterable of strings.

        The caller must consume this iterable and collect all strings and concatenate them to get the full response.

        The iterable ends when the agent requests new user input, and then you should call this function again with the new user input.

        If you provide tools, that list of tools supersedes any tools that have been registered with the agent (but otherwise does not force their use).

        The agent may decide to use tools, and will do so autonomously (limited by the max_successive_tool_invocations that you've set on the agent).
        """

        # If the current instance has an override for converse, use that one instead
        if (
            type(self).converse_stream is BedrockConverseAgent.converse_stream
            and type(self).converse is not BedrockConverseAgent.converse
        ):
            return self.converse(user_input, tools=tools)

        if not user_input:
            raise ValueError("Missing user input")

        self._conversation_history.add_message(
            {
                "role": "user",
                "content": [
                    {
                        "text": user_input,
                    },
                ],
            },
        )

        request: LlmRequest = {
            "modelId": self.model_id,
            "inferenceConfig": self.default_inference_config,
            "messages": list(self._conversation_history.messages),
        }
        if self.system_prompt:
            request["system"] = [
                {
                    "text": self.system_prompt,
                },
            ]
        tools_available = self.tools
        if tools is not None:
            tools = [
                BedrockConverseTool(tool) if callable(tool) else tool for tool in tools
            ]
            tools_available = (
                {tool.name: tool for tool in tools} if tools is not None else {}
            )
        if tools_available:
            request["toolConfig"] = {
                "tools": [tool.tool_spec for tool in tools_available.values()],
            }

        def update_content_block(
            content_blocks: list[MessageContent], content_block_delta: ContentBlockDelta
        ):
            content_block_index = content_block_delta["contentBlockIndex"]
            delta = content_block_delta["delta"]
            if len(content_blocks) < content_block_index + 1:
                content_blocks.append(delta)
            else:
                content_block = content_blocks[content_block_index]
                if "text" in delta:
                    if "text" not in content_block:
                        raise ValueError("Expect existing text content block")
                    content_block["text"] += delta["text"]
                elif "toolUse" in delta:
                    if "toolUse" not in content_block:
                        raise ValueError("Expect existing toolUse content block")
                    content_block["toolUse"]["input"] = (
                        content_block["toolUse"].get("input", "")
                        + delta["toolUse"]["input"]
                    )

        texts: list[str] = [""]
        for _ in range(self.max_converse_iterations):
            # Call LLM
            try:
                response = cast(
                    StreamingResponse,
                    self.bedrock_client.converse_stream(**cast(Any, request)),
                )
            except botocore.exceptions.ClientError as err:
                logger.error(str(err), request=request, response=err.response)
                raise

            metadata = None
            stop_reason = None

            message: Message | None = None
            content_blocks: list[MessageContent] = []

            for stream_event in response["stream"]:
                if "messageStart" in stream_event:
                    message = {
                        "role": stream_event["messageStart"]["role"],
                        "content": content_blocks,
                    }
                elif "contentBlockStart" in stream_event:
                    content_block = stream_event["contentBlockStart"]["start"]
                    content_blocks.append(content_block)

                elif "contentBlockDelta" in stream_event:
                    update_content_block(
                        content_blocks, stream_event["contentBlockDelta"]
                    )
                    text = stream_event["contentBlockDelta"]["delta"].get("text")
                    if text:
                        texts[-1] = text
                        yield text

                elif "messageStop" in stream_event:
                    stop_reason = stream_event["messageStop"]["stopReason"]
                    if texts[-1]:
                        texts.append("")
                    yield "\n"

                elif "metadata" in stream_event:
                    metadata = stream_event["metadata"]

            if not metadata:
                raise ValueError("Incomplete response stream: missing metadata")
            if not stop_reason:
                raise ValueError("Incomplete response stream: missing stop_reason")
            if message is None:
                message = {
                    "role": "assistant",
                    "content": [],
                }

            self._tracer.trace(
                self._conversation_history.conversation_id,
                "LLM",
                request,
                NonStreamingResponse(
                    metrics=metadata["metrics"],
                    usage=metadata["usage"],
                    ResponseMetadata=response["ResponseMetadata"],
                    stopReason=stop_reason,
                    output={"message": message},
                ),
                auth_context=self.auth_context,
            )
            for cb in content_blocks:
                if "toolUse" in cb:
                    tool_input = cb["toolUse"]["input"]
                    cb["toolUse"]["input"] = (
                        json.loads(str(tool_input)) if tool_input else {}
                    )

            self._conversation_history.add_message(message)

            if stop_reason == "tool_use":
                for msg in content_blocks:
                    if "toolUse" in msg:
                        self._invoke_tool(msg, tools=tools_available)
                request["messages"] = list(self._conversation_history.messages)
                continue

            elif stop_reason == "end_turn":
                break

            raise Exception("Unexpected LLM response")
        else:
            raise Exception("Too many successive tool invocations")
