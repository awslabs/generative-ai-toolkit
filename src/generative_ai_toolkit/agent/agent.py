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

from collections.abc import Callable, Iterable, Sequence
from threading import Event
from typing import (
    TYPE_CHECKING,
    Any,
    Literal,
    Protocol,
    Unpack,
    overload,
    runtime_checkable,
)

from generative_ai_toolkit.agent.tool import (
    Tool,
)
from generative_ai_toolkit.context import AuthContext
from generative_ai_toolkit.conversation_history import (
    ConversationHistory,
)
from generative_ai_toolkit.tracer import (
    Trace,
    Tracer,
)
from generative_ai_toolkit.tracer.context import (
    TraceContext,
    TraceContextUpdate,
)

if TYPE_CHECKING:
    from mypy_boto3_bedrock_runtime.type_defs import (
        MessageUnionTypeDef,
        ToolSpecificationTypeDef,
    )


@runtime_checkable
class Agent(Tool, Protocol):
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
    def conversation_history(self) -> ConversationHistory:
        """
        Get the conversation history instance of the agent.
        """
        ...

    @property
    def messages(self) -> Sequence["MessageUnionTypeDef"]:
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
    def subcontext_id(self) -> str | None:
        """
        The current subcontext id (if any) of the agent.
        """
        ...

    @property
    def tracer(self) -> Tracer:
        """
        Get the tracer instance of the agent
        """
        ...

    @property
    def traces(self) -> Sequence[Trace]:
        """
        Get the collected traces so far (for the current conversation)
        """
        ...

    @property
    def trace_context(self) -> TraceContext:
        """
        Get the trace context of the agent
        """
        ...

    def set_trace_context(
        self, **update: Unpack[TraceContextUpdate]
    ) -> Callable[[], None]:
        """
        Set the trace context of the agent
        """
        ...

    def set_conversation_id(
        self, conversation_id: str, *, subcontext_id: str | None = None
    ) -> None:
        """
        Set the current conversation id and subcontext_id (if any) of the agent.
        """
        ...

    @property
    def auth_context(self) -> AuthContext:
        """
        The current auth context of the agent.
        """
        ...

    def set_auth_context(self, **auth_context: Unpack[AuthContext]) -> None:
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

    def register_tool(
        self, tool: Callable | Tool, tool_spec: "ToolSpecificationTypeDef | None" = None
    ) -> Tool:
        """
        Register a tool with the agent.
        The agent can decide to use these tools during conversations.
        If you provide a Python function (Callable), it will be converted to a `Tool` for you.
        In order to make that work, it must be documented in a compatible way (as mandated by your Agent implementation).
        Alternatively, pass in a `tool_spec` explicitly, alongside your Python function.
        """
        ...

    def converse(
        self,
        user_input: str | None,
        tools: Sequence[Tool] | None = None,
        stop_event: Event | None = None,
    ) -> str:
        """
        Start or continue a conversation with the agent and return the agent's response as string.

        Parameters
        ----------
        user_input : str | None
            The user input to add to the conversation history. Must be:
            - A non-empty string: will be added to the agent's conversation history and a response will be generated
            - None: use when the user input was already added to the agent's conversation history
            Empty strings are not allowed and will raise a ValueError.
        tools : Sequence[Tool] | None, optional
            Tools to use for this conversation. If provided, this list supersedes any tools that have been
            registered with the agent (but otherwise does not force their use).
        stop_event : Event | None, optional
            An event that can be used to abort the generation of the assistant's response.

        Returns
        -------
        str
            The agent's response as a string.

        Notes
        -----
        The agent may decide to use tools, and will do so autonomously (limited by the
        max_successive_tool_invocations that you've set on the agent).
        """
        ...

    @overload
    def converse_stream(
        self,
        user_input: str | None,
        stream: Literal["text"] = "text",
        tools: Sequence[Tool] | None = None,
        stop_event: Event | None = None,
    ) -> Iterable[str]:
        """
        Start or continue a conversation with the agent.

        Parameters
        ----------
        user_input : str | None
            The user input to add to the conversation history. Must be:
            - A non-empty string: will be added to the agent's conversation history and a response will be generated
            - None: use when the user input was already added to the agent's conversation history
            Empty strings are not allowed and will raise a ValueError.
        stream : Literal["text"], optional
            Stream mode. Default is "text" which streams text chunks.
        tools : Sequence[Tool] | None, optional
            Tools to use for this conversation. If provided, this list supersedes any tools that have been
            registered with the agent (but otherwise does not force their use).
        stop_event : Event | None, optional
            An event that can be used to abort the generation of the assistant's response.

        Yields
        ------
        str
            Response fragments (text chunks) as they are produced.

        Notes
        -----
        - The caller must consume this iterable fully for the agent to progress.
        - The iterable ends when the agent requests new user input, and then you should call this function again with the new user input.
        - The agent may decide to use tools, and will do so autonomously (limited by the max_successive_tool_invocations that you've set on the agent).
        """
        ...

    @overload
    def converse_stream(
        self,
        user_input: str | None,
        stream: Literal["traces"],
        tools: Sequence[Tool] | None = None,
        stop_event: Event | None = None,
    ) -> Iterable[Trace]:
        """
        Start or continue a conversation with the agent.

        Parameters
        ----------
        user_input : str | None
            The user input to add to the conversation history. Must be:
            - A non-empty string: will be added to the agent's conversation history and a response will be generated
            - None: use when the user input was already added to the agent's conversation history
            Empty strings are not allowed and will raise a ValueError.
        stream : Literal["traces"]
            Stream mode. Set to "traces" to stream traces.
        tools : Sequence[Tool] | None, optional
            Tools to use for this conversation. If provided, this list supersedes any tools that have been
            registered with the agent (but otherwise does not force their use).
        stop_event : Event | None, optional
            An event that can be used to abort the generation of the assistant's response.

        Yields
        ------
        Trace
            Traces as they are produced by the agent and its tools.

        Notes
        -----
        - The caller must consume this iterable fully for the agent to progress.
        - The iterable ends when the agent requests new user input, and then you should call this function again with the new user input.
        - The agent may decide to use tools, and will do so autonomously (limited by the max_successive_tool_invocations that you've set on the agent).
        """
        ...

    def invoke(self, *args, **kwargs) -> Any:
        """
        Invoke the agent as tool. This method is used, when the agent is registered as a tool with another agent.
        """
        ...

    @property
    def tool_spec(self) -> "ToolSpecificationTypeDef":
        """
        The tool specification of the agent, that allows it to be registered as a tool with another agent.
        """
        ...
