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

import contextvars
import json
import re
import sys
import traceback
import weakref
from collections import defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from concurrent.futures import Executor, ThreadPoolExecutor
from contextlib import ExitStack
from queue import Queue
from threading import Event, Thread
from typing import (
    TYPE_CHECKING,
    Any,
    Literal,
    Protocol,
    Unpack,
    overload,
    runtime_checkable,
)

import boto3
import boto3.session
import botocore.exceptions
from botocore.config import Config

from generative_ai_toolkit.agent.tool import (
    BedrockConverseTool,
    Tool,
)
from generative_ai_toolkit.context import AgentContext, AuthContext
from generative_ai_toolkit.conversation_history import (
    ConversationHistory,
    InMemoryConversationHistory,
)
from generative_ai_toolkit.tracer import (
    ChainableTracer,
    InMemoryTracer,
    IterableTracer,
    QueueTracer,
    TeeTracer,
    Trace,
    Tracer,
    traced,
)
from generative_ai_toolkit.tracer.context import (
    TraceContext,
    TraceContextUpdate,
)
from generative_ai_toolkit.utils.json import DefaultJsonEncoder

if TYPE_CHECKING:
    from mypy_boto3_bedrock_runtime import BedrockRuntimeClient
    from mypy_boto3_bedrock_runtime.type_defs import (
        ContentBlockOutputTypeDef,
        ConverseRequestTypeDef,
        ConverseStreamOutputTypeDef,
        ConverseStreamRequestTypeDef,
        GuardrailStreamConfigurationTypeDef,
        InferenceConfigurationTypeDef,
        MessageOutputTypeDef,
        MessageUnionTypeDef,
        PerformanceConfigurationTypeDef,
        PromptVariableValuesTypeDef,
        ToolResultBlockUnionTypeDef,
        ToolSpecificationTypeDef,
    )


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

    def set_conversation_id(self, conversation_id: str) -> None:
        """
        Set the conversation id of the agent.
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
        user_input: str,
        tools: Sequence[Tool] | None = None,
        stop_event: Event | None = None,
    ) -> str:
        """
        Start or continue a conversation with the agent and return the agent's response as string.

        If you provide tools, that list of tools supersedes any tools that have been registered with the agent (but otherwise does not force their use).

        The agent may decide to use tools, and will do so autonomously (limited by the max_successive_tool_invocations that you've set on the agent).
        """
        ...

    @overload
    def converse_stream(
        self,
        user_input: str,
        stream: Literal["text"] = "text",
        tools: Sequence[Tool] | None = None,
        stop_event: Event | None = None,
    ) -> Iterable[str]:
        """
        Start or continue a conversation with the agent.

        Response fragments (text chunks) are yielded as they are produced.

        The caller must consume this iterable fully for the agent to progress.

        The iterable ends when the agent requests new user input, and then you should call this function again with the new user input.

        If you provide tools, that list of tools supersedes any tools that have been registered with the agent (but otherwise does not force their use).

        The agent may decide to use tools, and will do so autonomously (limited by the max_successive_tool_invocations that you've set on the agent).
        """
        ...

    @overload
    def converse_stream(
        self,
        user_input: str,
        stream: Literal["traces"],
        tools: Sequence[Tool] | None = None,
        stop_event: Event | None = None,
    ) -> Iterable[Trace]:
        """
        Start or continue a conversation with the agent.

        Traces are yielded as they are produced by the agent and its tools.

        The caller must consume this iterable fully for the agent to progress.

        The iterable ends when the agent requests new user input, and then you should call this function again with the new user input.

        If you provide tools, that list of tools supersedes any tools that have been registered with the agent (but otherwise does not force their use).

        The agent may decide to use tools, and will do so autonomously (limited by the max_successive_tool_invocations that you've set on the agent).
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


class BedrockConverseAgent(Agent):
    _model_id: str
    _system_prompt: str | None
    _tools: dict[str, Tool]
    _conversation_history: ConversationHistory
    _tracer: ChainableTracer

    # class attribute to track the tracer and conversation history instances used,
    # to prevent accidental double usage, see below.
    _instances_used: set[weakref.ReferenceType] = set()

    def __init__(
        self,
        *,
        model_id: str,
        system_prompt: str | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        stop_sequences: list[str] | None = None,
        guardrail_identifier: str | None = None,
        guardrail_version: str | None = None,
        guardrail_trace: Literal["disabled", "enabled", "enabled_full"] | None = None,
        guardrail_stream_processing_mode: Literal["sync", "async"] | None = None,
        conversation_history: (
            ConversationHistory | Callable[..., ConversationHistory] | None
        ) = None,
        additional_model_request_fields: Mapping[str, Any] | None = None,
        prompt_variables: Mapping[str, "PromptVariableValuesTypeDef"] | None = None,
        additional_model_response_field_paths: Sequence[str] | None = None,
        request_metadata: Mapping[str, str] | None = None,
        performance_config: "PerformanceConfigurationTypeDef | None" = None,
        tracer: Tracer | Callable[..., Tracer] | None = None,
        session: boto3.session.Session | None = None,
        bedrock_client: "BedrockRuntimeClient | None" = None,
        tools: Sequence[Callable] | None = None,
        max_successive_tool_invocations: int = 30,
        executor: Executor | None = None,
        tool_result_json_encoder: type[json.JSONEncoder] | None = None,
        include_reasoning_text_within_thinking_tags=True,
        name: str | None = None,
        description: str | None = None,
        input_schema: dict[str, Any] | None = None,
        converse_implementation: Literal["converse", "converse-stream"] | None = None,
    ) -> None:
        """
        Create an Agent that will use the Bedrock Converse API to operate.

        Parameters
        ----------
        model_id : str
            The model identifier to use for the agent
        system_prompt : str | None, optional
            A prompt that provides instructions or context to the model about the task it should perform, or the persona it should adopt during the conversation.
        max_tokens : int | None, optional
            The maximum number of tokens to allow in the generated response. The default value is the maximum allowed value for the model that you are using.
        temperature : float | None, optional
            The likelihood of the model selecting higher-probability options while generating a response. A lower value makes the model more likely to choose higher-probability options, while a higher value makes the model more likely to choose lower-probability options.
        top_p : float | None, optional
            The percentage of most-likely candidates that the model considers for the next token. For example, if you choose a value of 0.8 for topP, the model selects from the top 80% of the probability distribution of tokens that could be next in the sequence.
        stop_sequences : list[str] | None, optional
            A list of stop sequences. A stop sequence is a sequence of characters that causes the model to stop generating the response.
        guardrail_identifier : str | None, optional
            Identifier for the guardrail to apply
        guardrail_version : str | None, optional
            Version of the guardrail to use
        guardrail_trace : Literal["disabled", "enabled", "enabled_full"], optional
            The trace behavior for the guardrail.
        guardrail_stream_processing_mode : Literal["sync", "async"] | None, optional
            Guardrail processing mode
        conversation_history : ConversationHistory | Callable[..., ConversationHistory] | None, optional
            Storage for conversation state, by default InMemoryConversationHistory
        additional_model_request_fields : Mapping[str, Any] | None, optional
            Additional fields for model requests
        prompt_variables : Mapping[str, PromptVariableValuesTypeDef] | None, optional
            Contains a map of variables in a prompt from Prompt management to objects containing the values to fill in for them when running model invocation. This field is ignored if you don't specify a prompt resource in the modelId field.
        additional_model_response_field_paths : Sequence[str] | None, optional
            Additional model parameters field paths to return in the response. Converse and ConverseStream return the requested fields as a JSON Pointer object in the additionalModelResponseFields field. Example: [ "/stop_sequence" ]
        request_metadata : Mapping[str, str] | None, optional
            Key-value pairs that you can use to filter invocation logs.
        performance_config : Literal["standard", "optimized"] | None, optional
            To use a latency-optimized version of the model, set to optimized
        tracer : Tracer | Callable[..., Tracer] | None, optional
            Tracer for monitoring agent behavior, by default InMemoryTracer (wrapped in TeeTracer)
        session : boto3.session.Session | None, optional
            AWS session for Bedrock API calls, by default None (use default session)
        bedrock_client : BedrockRuntimeClient | None, optional
            Boto3 client for "bedrock-runtime", by default None (use provided session to create a client)
        tools : Sequence[Callable] | None, optional
            Tools available to the agent
        max_successive_tool_invocations : int, optional
            Maximum number of consecutive tool calls, by default 30
        executor : Executor | None, optional
            Executor for parallelizing tool invocations. By default, a ThreadPoolExecutor with 8 workers is used.
        tool_result_json_encoder : type[json.JSONEncoder], optional
            Custom JSON Encoder to encode tool results with, prior to sending them to the LLM
        include_reasoning_text_within_thinking_tags : bool
            Should reasoning texts be included (within <thinking> tags) in the output of converse and converse_stream? (Default: True)
        name : str | None, optional
            Name of the agent when used as a tool
        description : str | None, optional
            Description of the agent when used as a tool
        input_schema : dict[str, Any] | None, optional
            JSON schema for the input when the agent is used as a tool. If not provided, the agent is assumed to take its input as as single string.
        converse_implementation : Literal["converse", "converse-stream"] | None, optional
            The Amazon Bedrock API to use. By default this matches the agent method you invoke on the agent: if you invoke `agent.converse()` that will use the Converse API (https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_Converse.html), and if you invoke `agent.converse_stream()` that will use the ConverseStream API (https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_ConverseStream.html).
            By setting `converse_implementation` to either `"converse"` or `"converse-stream"`, you can force usage of either the Converse or ConverseStream API. For example, if you set `converse_implementation` to `"converse"`, then even when you invoke `agent.converse_stream()` that will actually use the Converse API, not ConverseStream.
            An example where this may be useful to you, is where you want to call `agent.converse_stream()` (e.g. for streaming traces) with a foundational model that doesn't support ConverseStream with tools (see https://docs.aws.amazon.com/bedrock/latest/userguide/conversation-inference-supported-models-features.html), so you need to force usage of the Converse API.
        """
        self._system_prompt = system_prompt
        self._model_id = model_id
        self._tools = {}
        self.bedrock_client: BedrockRuntimeClient = bedrock_client or (
            session or boto3
        ).client(
            "bedrock-runtime",
            config=Config(
                read_timeout=120,  # Default is 60, which can be a tad short for LLM responses
                tcp_keepalive=True,
            ),
        )
        if not conversation_history:
            self._conversation_history = InMemoryConversationHistory()
        else:
            if callable(conversation_history):
                conversation_history = conversation_history()
            ref = weakref.ref(conversation_history)
            if ref in self._instances_used:
                raise RuntimeError(
                    f"Cannot use the same ConversationHistory instance {ref()} across multiple agent instances. "
                    "Instead, pass a new ConversationHistory instance, or a ConversationHistory factory."
                )
            else:
                self._instances_used.add(ref)
            self._conversation_history = conversation_history
            weakref.finalize(self, self._prune_instances_used)
        if not tracer:
            self._tracer = TeeTracer().add_tracer(InMemoryTracer(memory_size=50))
        else:
            if callable(tracer):
                tracer = tracer()
            ref = weakref.ref(tracer)
            if ref in self._instances_used:
                raise RuntimeError(
                    f"Cannot use the same Tracer instance {ref()} across multiple agent instances. "
                    "Instead, pass a new Tracer instance, or a Tracer factory."
                )
            else:
                self._instances_used.add(ref)
            if isinstance(tracer, ChainableTracer):
                self._tracer = tracer
            else:
                self._tracer = TeeTracer().add_tracer(tracer)
            weakref.finalize(self, self._prune_instances_used)
        resource_attributes = self.tracer.context.resource_attributes
        if "service.name" not in resource_attributes:
            self.tracer.set_context(
                resource_attributes={
                    "service.name": self.__class__.__name__,
                    **resource_attributes,
                }
            )
        self.default_inference_config: InferenceConfigurationTypeDef = {}
        if max_tokens is not None:
            self.default_inference_config["maxTokens"] = max_tokens
        if temperature is not None:
            self.default_inference_config["temperature"] = temperature
        if top_p is not None:
            self.default_inference_config["topP"] = top_p
        if stop_sequences is not None:
            self.default_inference_config["stopSequences"] = stop_sequences
        self.default_guardrail_config: GuardrailStreamConfigurationTypeDef | None = None
        if guardrail_identifier and guardrail_version:
            self.default_guardrail_config = {
                "guardrailIdentifier": guardrail_identifier,
                "guardrailVersion": guardrail_version,
            }
            if guardrail_trace:
                self.default_guardrail_config["trace"] = guardrail_trace
            if guardrail_stream_processing_mode:
                self.default_guardrail_config["streamProcessingMode"] = (
                    guardrail_stream_processing_mode
                )
        self.default_model_request_fields = additional_model_request_fields
        self.default_prompt_variables = prompt_variables
        self.default_model_response_field_paths = additional_model_response_field_paths
        self.default_request_metadata = request_metadata
        self.default_performance_config = performance_config
        if tools:
            for tool in tools:
                self.register_tool(tool)
        if max_successive_tool_invocations < 0:
            raise ValueError("max_successive_tool_invocations must be positive")
        self.max_converse_iterations = max_successive_tool_invocations + 1
        self.executor = executor or ThreadPoolExecutor(
            max_workers=8, thread_name_prefix="tool-invocation"
        )
        self.tool_result_json_encoder: type[json.JSONEncoder] = (
            tool_result_json_encoder or DefaultJsonEncoder
        )
        self.include_reasoning_text_within_thinking_tags = (
            include_reasoning_text_within_thinking_tags
        )
        self.name = name
        "Name of the agent when used as a tool"

        self.description = description
        "Description of the agent when used as a tool"

        self.input_schema = input_schema
        "JSON schema (override) for the input when the agent is used as a tool"

        self.converse_implementation = converse_implementation

    @classmethod
    def _prune_instances_used(cls):
        cls._instances_used = {r for r in cls._instances_used if r() is not None}

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
    def conversation_history(self):
        """
        Get the conversation history instance of the agent.
        """
        return self._conversation_history

    @property
    def messages(self):
        """
        Get the messages sent to the agent so far (for the current conversation)
        """
        with self._tracer.trace(
            "conversation-history-list", span_kind="CLIENT"
        ) as span:
            span.add_attribute("ai.trace.type", "conversation-history-list")
            span.add_attribute(
                "ai.conversation.history.implementation",
                repr(self._conversation_history),
            )
            span.add_attribute("peer.service", "memory:short-term")
            messages = self._conversation_history.messages
            span.add_attribute("ai.conversation.history.messages", messages)
            return messages

    def _add_message(self, msg: "MessageUnionTypeDef") -> None:
        """
        Add a message to the conversation history
        """
        with self._tracer.trace("conversation-history-add", span_kind="CLIENT") as span:
            span.add_attribute("ai.trace.type", "conversation-history-add")
            span.add_attribute("ai.conversation.history.message", msg)
            span.add_attribute(
                "ai.conversation.history.implementation",
                repr(self._conversation_history),
            )
            span.add_attribute("peer.service", "memory:short-term")
            self._conversation_history.add_message(msg)

    @property
    def conversation_id(self):
        """
        Get the conversation id of the agent.
        """
        return self._conversation_history.conversation_id

    @property
    def tracer(self):
        """
        Get the tracer instance of the agent
        """
        return self._tracer

    @property
    def trace_context(self) -> TraceContext:
        """
        Get the trace context of the agent
        """
        return self.tracer.context

    def set_trace_context(
        self, **update: Unpack[TraceContextUpdate]
    ) -> Callable[[], None]:
        """
        Set the trace context of the agent
        """
        return self.tracer.set_context(**update)

    @property
    def traces(self):
        """
        Get the collected traces so far (for the current conversation)

        This recurses into all subagents that have been involved.
        """

        def get_traces_recursive(
            agent: AgentAsTool, attribute_filter: Mapping[str, Any]
        ):
            traces = list(agent.tracer.get_traces(attribute_filter=attribute_filter))

            subagent_invocations: dict[AgentAsTool, set[str]] = defaultdict(set)
            for trace in traces:
                if trace.attributes.get(
                    "ai.trace.type"
                ) == "tool-invocation" and trace.attributes.get("ai.tool.subagent"):
                    subagent_name = trace.attributes["ai.tool.name"]
                    subagent = agent.tools[subagent_name]
                    if not isinstance(subagent, AgentAsTool):
                        continue
                    subagent_invocations[subagent].add(trace.span_id)

            for subagent, span_ids in subagent_invocations.items():
                for span_id in span_ids:
                    subagent_traces = get_traces_recursive(
                        subagent,
                        {
                            **attribute_filter,
                            "ai.agent.hierarchy.parent.span.id": span_id,
                        },
                    )
                    traces.extend(subagent_traces)
            return traces

        return sorted(
            get_traces_recursive(
                self,
                attribute_filter={
                    "ai.conversation.id": self._conversation_history.conversation_id,
                    "ai.auth.context": self._conversation_history.auth_context,
                },
            ),
            key=lambda trace: trace.started_at,
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

    def set_auth_context(self, **auth_context: Unpack[AuthContext]) -> None:
        """
        Set the auth context of the agent.
        """
        self._conversation_history.set_auth_context(**auth_context)

    def reset(self):
        """
        Reset the state of the agent, e.g. in order to start a new conversation.
        (This does not unregister tools)
        """
        self._conversation_history.reset()

    def register_tool(
        self, tool: Callable | Tool, tool_spec: "ToolSpecificationTypeDef | None" = None
    ) -> Tool:
        """
        Register a tool with the agent.
        The agent can decide to use these tools during conversations.
        If you provide a Python function (Callable), it will be converted to a `BedrockConverseTool` for you.
        In order to make that work, it must be documented in a compatible way (see `BedrockConverseTool`).
        Alternatively, pass in a `tool_spec` explicitly, alongside your Python function.
        """
        if not isinstance(tool, Tool):
            tool = BedrockConverseTool(tool, tool_spec=tool_spec)
        self._tools[tool.tool_spec["name"]] = tool
        return tool

    @staticmethod
    def shorten_bedrock_model_id(model_id: str, prefix="", sep=":") -> str | None:
        match = re.match(r"^([a-z]{2}\.)?([a-z]+)\.([a-z0-9-]+?)(-\d{8})?-v", model_id)
        parts = [prefix] if prefix else []
        if match:
            parts.append(match.group(3))
        return sep.join(parts)

    def _invoke_tools(
        self,
        messages: Sequence["ContentBlockOutputTypeDef"],
        tools: Mapping[str, Tool],
        stop_event: Event | None,
    ) -> list["ToolResultBlockUnionTypeDef"]:
        if len(messages) == 1:
            return [
                AgentContext(
                    auth_context=self.auth_context,
                    tracer=self.tracer,
                    conversation_id=self.conversation_id,
                    stop_event=stop_event,
                )
                .copy_context()
                .run(self._invoke_tool, messages[0], tools, stop_event)
            ]
        return list(
            self.executor.map(
                lambda msg, ctx: ctx.run(self._invoke_tool, msg, tools, stop_event),
                messages,
                (
                    AgentContext(
                        auth_context=self.auth_context,
                        tracer=self.tracer,
                        conversation_id=self.conversation_id,
                        stop_event=stop_event,
                    ).copy_context()
                    for _ in messages
                ),
            )
        )

    def _invoke_tool(
        self,
        msg: "ContentBlockOutputTypeDef",
        tools: Mapping[str, Tool],
        stop_event: Event | None,
    ) -> "ToolResultBlockUnionTypeDef":
        if stop_event and stop_event.is_set():
            raise RuntimeError("Aborting as stop_event is set")
        if "toolUse" not in msg:
            raise ValueError("Invalid tool usage.")
        tool_use = msg["toolUse"]
        tool_name = tool_use["name"]
        with self._tracer.trace(tool_name, span_kind="CLIENT") as trace:
            trace.add_attribute("peer.service", f"tool:{tool_name}")
            trace.add_attribute("ai.trace.type", "tool-invocation")
            trace.add_attribute("ai.tool.name", tool_name)
            trace.add_attribute("ai.tool.use.id", tool_use["toolUseId"])
            trace.add_attribute("ai.tool.input", tool_use["input"])
            trace.emit_snapshot()

            tool_error = None
            tool_response_json: Any = None
            try:
                tool = tools.get(tool_name)
                if not tool:
                    raise ValueError(f"Unknown tool: {tool_name}")
                if isinstance(tool, AgentAsTool):
                    trace.add_attribute("ai.tool.subagent", True)
                    trace.emit_snapshot()
                    tool.set_auth_context(**self.auth_context)
                    tool.set_conversation_id(self.conversation_id)
                    tool.set_trace_context(
                        span=trace,
                        span_attributes={
                            **tool.tracer.context.span_attributes,
                            "ai.agent.hierarchy.parent.span.id": trace.span_id,
                        },
                    )
                    tool_response = tool.invoke(
                        **tool_use["input"], stop_event=stop_event
                    )
                else:
                    tool_response = tool.invoke(**tool_use["input"])
                trace.add_attribute("ai.tool.output", tool_response)
                if not isinstance(tool_response, dict):
                    tool_response = {"toolResponse": tool_response}
                # Ensure tool response can be represented as JSON
                # (otherwise boto3 will throw errors upon calling converse)
                tool_response_json = json.loads(
                    json.dumps(tool_response, cls=self.tool_result_json_encoder)
                )
            except Exception as err:
                tool_error = err
                trace.add_attribute("ai.tool.error", err)
                trace.add_attribute(
                    "ai.tool.error.traceback",
                    "".join(traceback.format_exception(*sys.exc_info())),
                )

            return {
                "toolUseId": tool_use["toolUseId"],
                "status": "success" if not tool_error else "error",
                "content": [
                    {
                        "json": (
                            tool_response_json
                            if tool_response_json
                            else {
                                "message": (
                                    f"Error invoking tool: {tool_error}"
                                    if tool_error
                                    else "Success"
                                )
                            }
                        )
                    }
                ],
            }

    @traced("converse", span_kind="SERVER")
    def converse(
        self,
        user_input: str,
        tools: Sequence[Callable | Tool] | None = None,
        stop_event: Event | None = None,
    ) -> str:
        """
        Start or continue a conversation with the agent and return the agent's response as string.

        If you provide tools, that list of tools supersedes any tools that have been registered with the agent (but otherwise does not force their use).

        The agent may decide to use tools, and will do so autonomously (limited by the max_successive_tool_invocations that you've set on the agent).
        """

        current_trace = self._tracer.current_trace
        current_trace.add_attribute("ai.trace.type", "converse")
        current_trace.add_attribute(
            "ai.conversation.id", self.conversation_id, inheritable=True
        )
        current_trace.add_attribute(
            "ai.auth.context", self.auth_context, inheritable=True
        )
        if self.name:
            current_trace.add_attribute("ai.agent.name", self.name, inheritable=True)
        current_trace.add_attribute("ai.user.input", user_input)
        current_trace.emit_snapshot()

        if self.converse_implementation == "converse-stream":
            return "".join(
                self.converse_stream(user_input, tools=tools, stop_event=stop_event)
            )

        if not user_input:
            raise ValueError("Missing user input")

        self._add_message(
            {
                "role": "user",
                "content": [
                    {
                        "text": user_input,
                    },
                ],
            },
        )

        request: ConverseRequestTypeDef = {
            "modelId": self.model_id,
            "inferenceConfig": self.default_inference_config,
            "messages": list(self.messages),
        }
        if self.default_guardrail_config:
            request["guardrailConfig"] = self.default_guardrail_config
        if self.system_prompt:
            request["system"] = [
                {
                    "text": self.system_prompt,
                },
            ]
        if self.default_model_request_fields:
            request["additionalModelRequestFields"] = self.default_model_request_fields
        if self.default_model_response_field_paths:
            request["additionalModelResponseFieldPaths"] = (
                self.default_model_response_field_paths
            )
        if self.default_prompt_variables:
            request["promptVariables"] = self.default_prompt_variables
        if self.default_request_metadata:
            request["requestMetadata"] = self.default_request_metadata
        if self.default_performance_config:
            request["performanceConfig"] = self.default_performance_config
        tools_available = self.tools
        if tools is not None:
            tools = [
                BedrockConverseTool(tool) if callable(tool) else tool for tool in tools
            ]
            tools_available = (
                {tool.tool_spec["name"]: tool for tool in tools}
                if tools is not None
                else {}
            )
        if tools_available:
            request["toolConfig"] = {
                "tools": [
                    {"toolSpec": tool.tool_spec} for tool in tools_available.values()
                ],
            }

        texts: list[str] = []
        for i in range(self.max_converse_iterations):
            with self._tracer.trace(f"cycle-{i}") as cycle_trace:
                cycle_trace.add_attribute("ai.trace.type", "cycle")
                cycle_trace.add_attribute("ai.agent.cycle.nr", i, inheritable=True)

                if stop_event and stop_event.is_set():
                    current_trace.add_attribute("ai.conversation.aborted", True)
                    concatenated = "\n".join(texts)
                    current_trace.add_attribute("ai.agent.response", concatenated)
                    return concatenated

                with self._tracer.trace("llm-invocation", span_kind="CLIENT") as trace:
                    model_id = request["modelId"]
                    trace.add_attribute(
                        "peer.service",
                        self.shorten_bedrock_model_id(model_id, prefix="llm"),
                    )
                    trace.add_attribute("ai.trace.type", "llm-invocation")
                    trace.add_attribute(
                        "ai.llm.request.inference.config", request["inferenceConfig"]
                    )
                    trace.add_attribute("ai.llm.request.messages", request["messages"])
                    trace.add_attribute("ai.llm.request.model.id", model_id)
                    trace.add_attribute("ai.llm.request.system", request.get("system"))
                    trace.add_attribute(
                        "ai.llm.request.tool.config", request.get("toolConfig")
                    )
                    if "guardrailConfig" in request:
                        trace.add_attribute(
                            "ai.llm.request.guardrail.config",
                            request["guardrailConfig"],
                        )

                    if "additionalModelRequestFields" in request:
                        trace.add_attribute(
                            "ai.llm.request.additional.model.request.fields",
                            request["additionalModelRequestFields"],
                        )

                    if "additionalModelResponseFieldPaths" in request:
                        trace.add_attribute(
                            "ai.llm.request.additional.model.response.field.paths",
                            request["additionalModelResponseFieldPaths"],
                        )

                    if "promptVariables" in request:
                        trace.add_attribute(
                            "ai.llm.request.prompt.variables",
                            request["promptVariables"],
                        )

                    if "requestMetadata" in request:
                        trace.add_attribute(
                            "ai.llm.request.request.metadata",
                            request["requestMetadata"],
                        )

                    if "performanceConfig" in request:
                        trace.add_attribute(
                            "ai.llm.request.performance.config",
                            request["performanceConfig"],
                        )
                    trace.emit_snapshot()

                    try:
                        response = self.bedrock_client.converse(**request)
                        trace.add_attribute(
                            "ai.llm.response.output", response["output"]
                        )
                        trace.add_attribute(
                            "ai.llm.response.stop.reason", response["stopReason"]
                        )
                        trace.add_attribute("ai.llm.response.usage", response["usage"])
                        trace.add_attribute(
                            "ai.llm.response.metrics", response["metrics"]
                        )
                        if "trace" in response:
                            trace.add_attribute(
                                "ai.llm.response.trace", response["trace"]
                            )
                        if "performanceConfig" in response:
                            trace.add_attribute(
                                "ai.llm.response.performance.config",
                                response["performanceConfig"],
                            )
                        trace.emit_snapshot()

                    except botocore.exceptions.ClientError as err:
                        trace.add_attribute("ai.llm.response.error", err.response)
                        raise

            # Capture text to show user
            message = response["output"].get("message")
            if message:
                cycle_text = ""
                for msg_content in message["content"]:
                    if self.include_reasoning_text_within_thinking_tags:
                        reasoning_text = (
                            msg_content.get("reasoningContent", {})
                            .get("reasoningText", {})
                            .get("text")
                        )
                        if reasoning_text:
                            cycle_text += f"<thinking>\n{reasoning_text}\n</thinking>\n"
                    if "text" in msg_content and msg_content["text"]:
                        cycle_text += msg_content["text"]
                if cycle_text:
                    texts.append(cycle_text)
                    cycle_trace.add_attribute("ai.agent.cycle.response", cycle_text)
                    cycle_trace.emit_snapshot()
                    current_trace.add_attribute(
                        "ai.agent.response",
                        "\n".join(texts),
                    )
                    current_trace.emit_snapshot()
                self._add_message(message)

                if response["stopReason"] == "tool_use":
                    tool_results = self._invoke_tools(
                        [
                            msg_content
                            for msg_content in message["content"]
                            if "toolUse" in msg_content
                        ],
                        tools_available,
                        stop_event=stop_event,
                    )
                    self._add_message(
                        {
                            "role": "user",
                            "content": [
                                {
                                    "toolResult": tool_result,
                                }
                                for tool_result in tool_results
                            ],
                        },
                    )
                    request["messages"] = list(self.messages)
                    continue

                elif response["stopReason"] in (
                    "end_turn",
                    "max_tokens",
                    "stop_sequence",
                    "guardrail_intervened",
                    "content_filtered",
                ):
                    concatenated = "\n".join(texts)
                    return concatenated

        raise Exception(
            "Too many successive tool invocations:{self.max_converse_iterations} "
        )

    @overload
    def converse_stream(
        self,
        user_input: str,
        stream: Literal["text"] = "text",
        tools: Sequence[Callable | Tool] | None = None,
        stop_event: Event | None = None,
    ) -> Iterable[str]:
        """
        Start or continue a conversation with the agent.

        Response fragments (text chunks) are yielded as they are produced.

        The caller must consume this iterable fully for the agent to progress.

        The iterable ends when the agent requests new user input, and then you should call this function again with the new user input.

        If you provide tools, that list of tools supersedes any tools that have been registered with the agent (but otherwise does not force their use).

        The agent may decide to use tools, and will do so autonomously (limited by the max_successive_tool_invocations that you've set on the agent).
        """
        ...

    @overload
    def converse_stream(
        self,
        user_input: str,
        stream: Literal["traces"],
        tools: Sequence[Callable | Tool] | None = None,
        stop_event: Event | None = None,
    ) -> Iterable[Trace]:
        """
        Start or continue a conversation with the agent.

        Traces are yielded as they are produced by the agent and its tools.

        The caller must consume this iterable fully for the agent to progress.

        The iterable ends when the agent requests new user input, and then you should call this function again with the new user input.

        If you provide tools, that list of tools supersedes any tools that have been registered with the agent (but otherwise does not force their use).

        The agent may decide to use tools, and will do so autonomously (limited by the max_successive_tool_invocations that you've set on the agent).
        """
        ...

    def converse_stream(
        self,
        user_input: str,
        stream: Literal["traces"] | Literal["text"] = "text",
        tools: Sequence[Callable | Tool] | None = None,
        stop_event: Event | None = None,
    ) -> Iterable[str] | Iterable[Trace]:
        gen = self._converse_stream(
            user_input=user_input, tools=tools, stop_event=stop_event
        )
        if stream == "text":
            yield from gen
        elif stream == "traces":
            with ExitStack() as exit_stack:
                tracer = IterableTracer()
                exit_stack.enter_context(self._tracer.temporary_tracer(tracer))
                self.add_queue_tracer_to_sub_agents(
                    agent=self, parents=(), queue=tracer.queue, exit_stack=exit_stack
                )

                ctx = contextvars.copy_context()
                thread = Thread(
                    target=ctx.run,
                    args=[self._consume_traced_iterable, gen, tracer],
                    daemon=True,
                    name="converse_stream",
                )
                thread.start()
                try:
                    yield from tracer
                finally:
                    thread.join()

        else:
            raise ValueError(f"stream must be 'text' or 'traces', but got {stream}")

    @classmethod
    def add_queue_tracer_to_sub_agents(
        cls,
        *,
        exit_stack: ExitStack,
        agent: "AgentAsTool",
        parents: tuple[str, ...],
        queue: Queue,
    ):
        for tool_name, tool in agent.tools.items():
            if isinstance(tool, AgentAsTool):
                cls.add_queue_tracer_to_sub_agents(
                    agent=tool,
                    parents=(*parents, tool_name),
                    queue=queue,
                    exit_stack=exit_stack,
                )
                if isinstance(tool.tracer, ChainableTracer):
                    sub_agent_tracer = QueueTracer(queue=queue)
                    exit_stack.enter_context(
                        tool.tracer.temporary_tracer(sub_agent_tracer)
                    )

    @traced("converse-stream", span_kind="SERVER")
    def _converse_stream(
        self,
        user_input: str,
        tools: Sequence[Callable | Tool] | None = None,
        stop_event: Event | None = None,
    ) -> Iterable[str]:
        current_trace = self._tracer.current_trace
        current_trace.add_attribute("ai.trace.type", "converse-stream")
        current_trace.add_attribute(
            "ai.conversation.id", self.conversation_id, inheritable=True
        )
        current_trace.add_attribute(
            "ai.auth.context", self.auth_context, inheritable=True
        )
        if self.name:
            current_trace.add_attribute("ai.agent.name", self.name, inheritable=True)
        current_trace.add_attribute("ai.user.input", user_input)
        current_trace.emit_snapshot()

        if self.converse_implementation == "converse":
            yield self.converse(user_input, tools=tools, stop_event=stop_event)
            return

        if not user_input:
            raise ValueError("Missing user input")

        self._add_message(
            {
                "role": "user",
                "content": [
                    {
                        "text": user_input,
                    },
                ],
            },
        )

        request: ConverseStreamRequestTypeDef = {
            "modelId": self.model_id,
            "inferenceConfig": self.default_inference_config,
            "messages": self.messages,
        }
        if self.default_guardrail_config:
            request["guardrailConfig"] = self.default_guardrail_config
        if self.system_prompt:
            request["system"] = [
                {
                    "text": self.system_prompt,
                },
            ]
        if self.default_model_request_fields:
            request["additionalModelRequestFields"] = self.default_model_request_fields
        if self.default_model_response_field_paths:
            request["additionalModelResponseFieldPaths"] = (
                self.default_model_response_field_paths
            )
        if self.default_prompt_variables:
            request["promptVariables"] = self.default_prompt_variables
        if self.default_request_metadata:
            request["requestMetadata"] = self.default_request_metadata
        if self.default_performance_config:
            request["performanceConfig"] = self.default_performance_config
        tools_available = self.tools
        if tools is not None:
            tools = [
                BedrockConverseTool(tool) if callable(tool) else tool for tool in tools
            ]
            tools_available = (
                {tool.tool_spec["name"]: tool for tool in tools}
                if tools is not None
                else {}
            )
        if tools_available:
            request["toolConfig"] = {
                "tools": [
                    {"toolSpec": tool.tool_spec} for tool in tools_available.values()
                ],
            }

        texts: list[str] = []
        for i in range(self.max_converse_iterations):
            with self._tracer.trace(f"cycle-{i}") as cycle_trace:
                cycle_trace.add_attribute("ai.trace.type", "cycle")
                cycle_trace.add_attribute("ai.agent.cycle.nr", i, inheritable=True)

                if stop_event and stop_event.is_set():
                    current_trace.add_attribute("ai.conversation.aborted", True)
                    return

                texts.append("")

                with self._tracer.trace("llm-invocation", span_kind="CLIENT") as trace:
                    model_id = request["modelId"]
                    trace.add_attribute(
                        "peer.service",
                        self.shorten_bedrock_model_id(model_id, prefix="llm"),
                    )
                    trace.add_attribute("ai.trace.type", "llm-invocation")
                    trace.add_attribute(
                        "ai.llm.request.inference.config", request["inferenceConfig"]
                    )
                    trace.add_attribute("ai.llm.request.messages", request["messages"])
                    trace.add_attribute("ai.llm.request.model.id", model_id)
                    trace.add_attribute("ai.llm.request.system", request.get("system"))
                    trace.add_attribute(
                        "ai.llm.request.tool.config", request.get("toolConfig")
                    )
                    if "guardrailConfig" in request:
                        trace.add_attribute(
                            "ai.llm.request.guardrail.config",
                            request["guardrailConfig"],
                        )

                    if "additionalModelRequestFields" in request:
                        trace.add_attribute(
                            "ai.llm.request.additional.model.request.fields",
                            request["additionalModelRequestFields"],
                        )

                    if "additionalModelResponseFieldPaths" in request:
                        trace.add_attribute(
                            "ai.llm.request.additional.model.response.field.paths",
                            request["additionalModelResponseFieldPaths"],
                        )

                    if "promptVariables" in request:
                        trace.add_attribute(
                            "ai.llm.request.prompt.variables",
                            request["promptVariables"],
                        )

                    if "requestMetadata" in request:
                        trace.add_attribute(
                            "ai.llm.request.request.metadata",
                            request["requestMetadata"],
                        )

                    if "performanceConfig" in request:
                        trace.add_attribute(
                            "ai.llm.request.performance.config",
                            request["performanceConfig"],
                        )
                    trace.emit_snapshot()

                    try:
                        response = self.bedrock_client.converse_stream(**request)
                    except botocore.exceptions.ClientError as err:
                        trace.add_attribute("ai.llm.response.error", err.response)
                        raise

                    metadata = None
                    stop_reason = None

                    content_blocks: list[ContentBlockOutputTypeDef] = []
                    message: MessageOutputTypeDef = {
                        "role": "assistant",
                        "content": content_blocks,
                    }
                    stream_events: dict[int, list[ConverseStreamOutputTypeDef]] = (
                        defaultdict(list)
                    )

                    is_reasoning = False
                    for stream_event in response["stream"]:
                        if stop_event and stop_event.is_set():
                            current_trace.add_attribute("ai.conversation.aborted", True)
                            response["stream"].close()
                            return
                        if "contentBlockStart" in stream_event:
                            index = stream_event["contentBlockStart"][
                                "contentBlockIndex"
                            ]
                            stream_events[index].append(stream_event)

                        elif "contentBlockDelta" in stream_event:
                            index = stream_event["contentBlockDelta"][
                                "contentBlockIndex"
                            ]
                            stream_events[index].append(stream_event)

                            text = stream_event["contentBlockDelta"]["delta"].get(
                                "text"
                            )
                            if text:
                                texts[-1] += text
                                cycle_trace.add_attribute(
                                    "ai.agent.cycle.response", texts[-1]
                                )
                                cycle_trace.emit_snapshot()
                                current_trace.add_attribute(
                                    "ai.agent.response",
                                    "\n".join(text for text in texts if text),
                                )
                                current_trace.emit_snapshot()
                                yield text

                            if self.include_reasoning_text_within_thinking_tags:
                                reasoning_content = stream_event["contentBlockDelta"][
                                    "delta"
                                ].get("reasoningContent")
                                if reasoning_content:
                                    reasoning_text = reasoning_content.get("text")
                                    if reasoning_text:
                                        if not is_reasoning:
                                            texts[-1] += "<thinking>\n"
                                            cycle_trace.add_attribute(
                                                "ai.agent.cycle.response", texts[-1]
                                            )
                                            cycle_trace.emit_snapshot()
                                            current_trace.add_attribute(
                                                "ai.agent.response",
                                                "\n".join(
                                                    text for text in texts if text
                                                ),
                                            )
                                            current_trace.emit_snapshot()
                                            is_reasoning = True
                                            yield "<thinking>\n"
                                        texts[-1] += reasoning_text
                                        cycle_trace.add_attribute(
                                            "ai.agent.cycle.response", texts[-1]
                                        )
                                        cycle_trace.emit_snapshot()
                                        current_trace.add_attribute(
                                            "ai.agent.response",
                                            "\n".join(text for text in texts if text),
                                        )
                                        current_trace.emit_snapshot()
                                        yield reasoning_text
                                    reasoning_signature = reasoning_content.get(
                                        "signature"
                                    )
                                    if reasoning_signature:
                                        if is_reasoning:
                                            texts[-1] += "\n</thinking>\n\n"
                                            cycle_trace.add_attribute(
                                                "ai.agent.cycle.response", texts[-1]
                                            )
                                            cycle_trace.emit_snapshot()
                                            current_trace.add_attribute(
                                                "ai.agent.response",
                                                "\n".join(
                                                    text for text in texts if text
                                                ),
                                            )
                                            current_trace.emit_snapshot()
                                            is_reasoning = False
                                            yield "\n</thinking>\n\n"

                        elif "contentBlockStop" in stream_event:
                            index = stream_event["contentBlockStop"][
                                "contentBlockIndex"
                            ]
                            content_blocks.append(
                                self.squash_stream_events(stream_events[index])
                            )

                        elif "messageStart" in stream_event:
                            pass

                        elif "messageStop" in stream_event:
                            stop_reason = stream_event["messageStop"]["stopReason"]
                            yield "\n"

                        elif "metadata" in stream_event:
                            metadata = stream_event["metadata"]
                        else:
                            raise Exception(f"Unsupported stream event {stream_event}")

                    if not metadata:
                        raise ValueError("Incomplete response stream: missing metadata")
                    if not stop_reason:
                        raise ValueError(
                            "Incomplete response stream: missing stop_reason"
                        )

                    trace.add_attribute("ai.llm.response.output", {"message": message})
                    trace.add_attribute("ai.llm.response.stop.reason", stop_reason)
                    trace.add_attribute("ai.llm.response.usage", metadata["usage"])
                    trace.add_attribute("ai.llm.response.metrics", metadata["metrics"])
                    if "trace" in metadata:
                        trace.add_attribute("ai.llm.response.trace", metadata["trace"])
                    if "performanceConfig" in metadata:
                        trace.add_attribute(
                            "ai.llm.response.performance.config",
                            metadata["performanceConfig"],
                        )
                    trace.emit_snapshot()

                if stop_event and stop_event.is_set():
                    current_trace.add_attribute("ai.conversation.aborted", True)
                    return

                if not texts[-1]:
                    texts.pop()
                self._add_message(message)

                if stop_reason == "tool_use":
                    tool_results = self._invoke_tools(
                        [message for message in content_blocks if "toolUse" in message],
                        tools_available,
                        stop_event=stop_event,
                    )
                    self._add_message(
                        {
                            "role": "user",
                            "content": [
                                {
                                    "toolResult": tool_result,
                                }
                                for tool_result in tool_results
                            ],
                        },
                    )
                    request["messages"] = self.messages
                    continue

                elif stop_reason in (
                    "end_turn",
                    "max_tokens",
                    "stop_sequence",
                    "guardrail_intervened",
                    "content_filtered",
                ):
                    break

        else:
            raise Exception("Too many successive tool invocations")

    @staticmethod
    def squash_stream_events(
        accumulated: "list[ConverseStreamOutputTypeDef]",
    ):
        current_block = {
            "text": "",
            "toolUse": {
                "name": "",
                "toolUseId": "",
                "input": "",
            },
            "reasoningContent": {
                "text": "",
                "signature": "",
                "redactedContent": "",
            },
        }
        for cb in accumulated:
            if "contentBlockStart" in cb:
                tool_use = cb["contentBlockStart"]["start"].get("toolUse")
                if tool_use:
                    current_block["toolUse"]["name"] = tool_use["name"]
                    current_block["toolUse"]["toolUseId"] = tool_use["toolUseId"]
            elif "contentBlockDelta" in cb:
                delta = cb["contentBlockDelta"]["delta"]
                if "text" in delta:
                    current_block["text"] += delta["text"]
                elif "toolUse" in delta and "input" in delta["toolUse"]:
                    current_block["toolUse"]["input"] = (
                        current_block["toolUse"].get("input", "")
                        + delta["toolUse"]["input"]
                    )
                elif "reasoningContent" in delta:
                    if "text" in delta["reasoningContent"]:
                        current_block["reasoningContent"]["text"] += delta[
                            "reasoningContent"
                        ]["text"]
                    if "signature" in delta["reasoningContent"]:
                        current_block["reasoningContent"]["signature"] = delta[
                            "reasoningContent"
                        ]["signature"]
                    if "redactedContent" in delta["reasoningContent"]:
                        current_block["reasoningContent"]["redactedContent"] = delta[
                            "reasoningContent"
                        ]["redactedContent"]

        content_block: ContentBlockOutputTypeDef = {}
        if current_block["toolUse"]["toolUseId"]:
            content_block["toolUse"] = {
                "name": current_block["toolUse"]["name"],
                "toolUseId": current_block["toolUse"]["toolUseId"],
                "input": (
                    json.loads(current_block["toolUse"]["input"])
                    if current_block["toolUse"].get("input")
                    else {}
                ),
            }
        if current_block["text"]:
            content_block["text"] = current_block["text"]

        if current_block["reasoningContent"]["text"]:
            content_block["reasoningContent"] = {
                "reasoningText": {
                    "text": current_block["reasoningContent"]["text"],
                    "signature": current_block["reasoningContent"]["signature"],
                },
            }
            redacted_content = current_block["reasoningContent"].get("redactedContent")
            if redacted_content:
                content_block["reasoningContent"]["redactedContent"] = redacted_content

        return content_block

    def invoke(self, *args, **kwargs) -> Any:
        """
        Invoke an agent as tool.

        This method is meant to be called by supervisor agents.

        To invoke an agent directly, use converse_stream() or converse().
        """
        if not self.input_schema:
            user_input = kwargs.pop("user_input")
        else:
            params = {}
            for param_name in self.input_schema["properties"]:
                if param_name in kwargs:
                    param_value = kwargs.pop(param_name)
                    params[param_name] = param_value
            user_input = (
                f"Your input is:\n\n{json.dumps(params, cls=DefaultJsonEncoder)}"
            )
        return "".join(self.converse_stream(*args, **kwargs, user_input=user_input))

    @property
    def tool_spec(self) -> "ToolSpecificationTypeDef":
        if not self.name:
            raise RuntimeError(
                "Missing name. When using an agent as tool, the agent must be instantiated with a name"
            )

        if not self.description:
            raise RuntimeError(
                "Missing description. When using an agent as tool, the agent must be instantiated with a description"
            )
        return {
            "name": self.name,
            "description": self.description,
            "inputSchema": {
                "json": self.input_schema
                or {
                    "type": "object",
                    "properties": {
                        "user_input": {
                            "type": "string",
                            "description": "The input to the agent",
                        },
                    },
                    "required": ["user_input"],
                }
            },
        }

    @staticmethod
    def _consume_traced_iterable(iterable: Iterable[str], tracer: IterableTracer):
        try:
            for _ in iterable:
                pass
        finally:
            tracer.shutdown()


@runtime_checkable
class AgentAsTool(Protocol):
    @property
    def tools(self) -> dict[str, Tool]:
        """
        The tools that have been registered with the agent.
        The agent can decide to use these tools during conversations.
        """
        ...

    @property
    def tracer(self) -> Tracer:
        """
        Get the tracer instance of the agent
        """
        ...

    def set_trace_context(
        self, **update: Unpack[TraceContextUpdate]
    ) -> Callable[[], None]:
        """
        Set the trace context of the agent
        """
        ...

    def set_conversation_id(self, conversation_id: str) -> None:
        """
        Set the conversation id of the agent.
        """
        ...

    def set_auth_context(self, **auth_context: Unpack[AuthContext]) -> None:
        """
        Set the auth context of the agent.
        """
        ...
