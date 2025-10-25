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
import textwrap
import traceback
import warnings
import weakref
from collections import defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from concurrent.futures import Executor, ThreadPoolExecutor
from contextlib import ExitStack
from threading import Event, Lock, Thread
from typing import (
    TYPE_CHECKING,
    Any,
    Literal,
    Protocol,
    TypeGuard,
    TypeVar,
    Unpack,
    overload,
    runtime_checkable,
)

import boto3
import boto3.session
import botocore.exceptions
from botocore.config import Config

from generative_ai_toolkit.agent.agent import Agent
from generative_ai_toolkit.agent.bedrock_converse_stream import (
    BedrockConverseStreamEventContentBlockHandler,
)
from generative_ai_toolkit.agent.tool import (
    BedrockConverseTool,
    Tool,
)
from generative_ai_toolkit.context import AgentContext, AuthContext
from generative_ai_toolkit.conversation_history import (
    ConversationHistory,
    InMemoryConversationHistory,
)
from generative_ai_toolkit.exceptions import StopEventAbortError
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
from generative_ai_toolkit.utils.stop_event import (
    invoke_cancellable,
    with_placeholder_emit,
)
from generative_ai_toolkit.utils.ulid import Ulid

if TYPE_CHECKING:
    from mypy_boto3_bedrock_runtime import BedrockRuntimeClient
    from mypy_boto3_bedrock_runtime.type_defs import (
        ContentBlockOutputTypeDef,
        ConverseRequestTypeDef,
        ConverseStreamRequestTypeDef,
        GuardrailStreamConfigurationTypeDef,
        InferenceConfigurationTypeDef,
        MessageOutputTypeDef,
        MessageUnionTypeDef,
        PerformanceConfigurationTypeDef,
        PromptVariableValuesTypeDef,
        ToolResultBlockUnionTypeDef,
        ToolResultContentBlockUnionTypeDef,
        ToolSpecificationTypeDef,
    )


class BedrockConverseAgent(Agent):
    _model_id: str
    _system_prompt: str | None
    _tools: dict[str, Tool]
    _tools_lock: Lock
    _conversation_history: ConversationHistory
    _tracer: ChainableTracer

    # class attribute to track the conversation history instances used,
    # to prevent accidental double usage, see below.
    _instances_used: set[weakref.ReferenceType] = set()

    class _QueueTracer(QueueTracer):
        """
        Dummy subclass of QueueTracer
        Allows us to use isinstance() to recognize QueueTracers we created for streaming traces
        """

        pass

    class _IterableTracer(IterableTracer):
        """
        Dummy subclass of IterableTracer
        Allows us to use isinstance() to recognize IterableTracers we created for streaming traces
        """

        pass

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
        tools: Sequence[Callable | Tool] | None = None,
        max_successive_tool_invocations: int = 30,
        executor: Executor | None = None,
        tool_result_json_encoder: type[json.JSONEncoder] | None = None,
        include_reasoning_text_within_thinking_tags=True,
        name: str | None = None,
        description: str | None = None,
        input_schema: dict[str, Any] | None = None,
        converse_implementation: Literal["converse", "converse-stream"] | None = None,
        stop_event_abort_message: str = "\n***\nABORTED: The generation of the assistant's response was aborted\n***",
    ) -> None:
        """
        Create an Agent that will use the Bedrock Converse API to operate.

        Parameters
        ----------
        model_id : str
            The model identifier to use for the agent. Pick one from https://docs.aws.amazon.com/bedrock/latest/userguide/models-supported.html
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
        stop_event_abort_message : str, optional
            The message to return when a stop_event is triggered and the generation is aborted.
        """
        self._spawn_args = {
            k: v for k, v in locals().items() if k not in {"self", "tools", "tracer"}
        }
        self._system_prompt = system_prompt
        self._model_id = model_id
        self._tools = {}
        self._tools_lock = Lock()
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
            self._tracer = TeeTracer().add_tracer(InMemoryTracer())
        else:
            if callable(tracer):
                tracer = tracer()
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
        self.stop_event_abort_message = stop_event_abort_message

        self._subagent_lock: dict[str, Lock] = {}

    def spawn(self):
        """
        Spawn a new independent instance of the agent.

        Creates a new agent instance using the original __init__ arguments that were used to
        create this agent. The spawned agent will have fresh internal state (auth_context,
        conversation_id, and subcontext_id will be unset) but will share the same tracer
        instance and have access to the same registered tools as the parent agent.

        This method is useful when you need multiple parallel instances of an agent operating
        independently, such as when registering an agent as a tool with another agent.

        Returns
        -------
        BedrockConverseAgent
            A new independent instance of the agent with fresh internal state but sharing
            the tracer and having copies of all registered tools.

        Notes
        -----
        - The spawned agent shares the same tracer as the parent agent
        - All tools registered with the parent agent are copied to the spawned instance
        - Conversation history is not shared; the spawned agent starts with a fresh history
        - The spawned agent uses the same AWS session/Bedrock client configuration
        """
        spawned = type(self)(
            **self._spawn_args, tools=list(self.tools.values()), tracer=self.tracer
        )

        return spawned

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
        with self._tools_lock:
            return self._tools.copy()

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
    def subcontext_id(self) -> str | None:
        """
        The current subcontext id (if any) of the agent.
        """
        return self._conversation_history.subcontext_id

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
            traces = {
                trace.span_id: trace
                for trace in agent.tracer.get_traces(attribute_filter=attribute_filter)
            }

            subagent_invocations: dict[AgentAsTool, dict[str, Trace]] = defaultdict(
                dict
            )
            for trace in traces.values():
                if (
                    trace.attributes.get("ai.trace.type") == "tool-invocation"
                    and "ai.tool.subagent.subcontext.id" in trace.attributes
                ):
                    subagent_name = trace.attributes["ai.tool.name"]
                    subagent = agent.tools[subagent_name]
                    if not isinstance(subagent, AgentAsTool):
                        continue
                    subagent_invocations[subagent][trace.span_id] = trace

            for subagent, subagent_parent_traces in subagent_invocations.items():
                for span_id, parent_trace in subagent_parent_traces.items():
                    subagent_traces = get_traces_recursive(
                        subagent,
                        {
                            "ai.auth.context": attribute_filter["ai.auth.context"],
                            "ai.conversation.id": attribute_filter[
                                "ai.conversation.id"
                            ],
                            "ai.subcontext.id": parent_trace.attributes[
                                "ai.tool.subagent.subcontext.id"
                            ],
                            "ai.agent.hierarchy.parent.span.id": span_id,
                        },
                    )
                    # Add links to parent traces
                    for subagent_trace in subagent_traces.values():
                        if (
                            not subagent_trace.parent_span
                            and "ai.agent.hierarchy.parent.span.id"
                            in subagent_trace.attributes
                        ):
                            subagent_trace.parent_span = subagent_parent_traces[
                                subagent_trace.attributes[
                                    "ai.agent.hierarchy.parent.span.id"
                                ]
                            ]
                    traces.update(subagent_traces)
            return traces

        return sorted(
            get_traces_recursive(
                self,
                attribute_filter={
                    "ai.conversation.id": self._conversation_history.conversation_id,
                    "ai.subcontext.id": self._conversation_history.subcontext_id,
                    "ai.auth.context": self._conversation_history.auth_context,
                },
            ).values(),
            key=lambda trace: trace.started_at,
        )

    def set_conversation_id(
        self, conversation_id: str, *, subcontext_id: str | None = None
    ) -> None:
        """
        Set the current conversation id and subcontext_id (if any) of the agent.
        """
        self._conversation_history.set_conversation_id(
            conversation_id, subcontext_id=subcontext_id
        )

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
        elif isinstance(tool, AgentAsTool) and not isinstance(tool, Spawnable):
            self._subagent_lock[tool.tool_spec["name"]] = Lock()

        with self._tools_lock:
            self._tools[tool.tool_spec["name"]] = tool

        return tool

    @staticmethod
    def _shorten_bedrock_model_id(model_id: str, prefix="", sep=":") -> str:
        match = re.match(r"^([a-z]+\.)?([a-z]+)\.([a-z0-9-]+?)(-\d{8})?-v", model_id)
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
                    context_key=self.conversation_history.context_key,
                    agent=self,
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
                        context_key=self.conversation_history.context_key,
                        agent=self,
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
        if "toolUse" not in msg:
            raise ValueError("Invalid tool usage.")
        tool_use = msg["toolUse"]
        if stop_event and stop_event.is_set():
            return {
                "toolUseId": tool_use["toolUseId"],
                "status": "error",
                "content": [{"text": "The tool invocation was aborted"}],
            }
        tool_name = tool_use["name"]
        with self._tracer.trace(tool_name, span_kind="CLIENT") as trace:
            trace.add_attribute("peer.service", f"tool:{tool_name}")
            trace.add_attribute("ai.trace.type", "tool-invocation")
            trace.add_attribute("ai.tool.name", tool_name)
            trace.add_attribute("ai.tool.use.id", tool_use["toolUseId"])
            trace.add_attribute("ai.tool.input", tool_use["input"])
            trace.emit_snapshot()

            tool_error = None
            tool_response_content: Sequence[ToolResultContentBlockUnionTypeDef] = []
            try:
                tool = tools.get(tool_name)
                if not tool:
                    raise ValueError(f"Unknown tool: {tool_name}")
                if isinstance(tool, AgentAsTool):
                    subcontext_id = tool_use["input"].get("subcontext_id")
                    if not subcontext_id or not isinstance(subcontext_id, str):
                        subcontext_id = Ulid().ulid
                    trace.add_attribute("ai.tool.subagent.subcontext.id", subcontext_id)
                    trace.emit_snapshot()

                    with ExitStack() as exit_stack:
                        if isinstance(tool, Spawnable):
                            tool = tool.spawn()
                        else:
                            # The agent cannot be invoked in parallel, so acquire lock:
                            exit_stack.enter_context(self._subagent_lock[tool_name])

                        # If we're using a _QueueTracer or _IterableTracer,
                        # i.e. we're processing converse_stream(..., stream="traces"),
                        # then we should also attach a _QueueTracer to the subagent,
                        # sp that its traces are collected into the original _IterableTracer
                        if isinstance(tool.tracer, ChainableTracer):
                            queue = next(
                                (
                                    tracer.queue
                                    for tracer in reversed(self.tracer.tracers)
                                    if isinstance(
                                        tracer,
                                        self._QueueTracer | self._IterableTracer,
                                    )
                                ),
                                None,
                            )
                            if queue:
                                # Check if the subagent already has a tracer for this queue
                                # to prevent duplicate traces (e.g., in self-invocation scenarios)
                                already_has_queue = any(
                                    isinstance(
                                        tracer, self._QueueTracer | self._IterableTracer
                                    )
                                    and tracer.queue is queue
                                    for tracer in tool.tracer.tracers
                                )
                                if not already_has_queue:
                                    exit_stack.enter_context(
                                        tool.tracer.temporary_tracer(
                                            self._QueueTracer(queue=queue)
                                        )
                                    )

                        tool.set_auth_context(**self.auth_context)
                        tool.set_conversation_id(
                            self.conversation_id, subcontext_id=subcontext_id
                        )
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
                    tool_response = invoke_cancellable(
                        stop_event=stop_event,
                        method=tool.invoke,
                        kwargs=tool_use["input"],
                    )
                trace.add_attribute("ai.tool.output", tool_response)
                if self._is_tool_result_content_block_sequence(tool_response):
                    tool_response_content = tool_response
                else:
                    # Return tool response as JSON object to the LLM
                    if not isinstance(tool_response, dict):
                        tool_response = {"toolResponse": tool_response}
                    # Ensure tool response can be represented as JSON
                    # (otherwise boto3 will throw errors upon calling converse)
                    tool_response_json = json.loads(
                        json.dumps(tool_response, cls=self.tool_result_json_encoder)
                    )
                    tool_response_content = [{"json": tool_response_json}]
            except Exception as err:
                tool_error = err
                tool_response_content = [
                    {"text": (f"Error invoking tool: {tool_error}")}
                ]
                trace.add_attribute("ai.tool.error", repr(tool_error))
                trace.add_attribute(
                    "ai.tool.error.traceback",
                    "".join(traceback.format_exception(*sys.exc_info())),
                )

            return {
                "toolUseId": tool_use["toolUseId"],
                "status": "success" if not tool_error else "error",
                "content": tool_response_content,
            }

    @staticmethod
    def _is_tool_result_content_block_sequence(  # noqa: PLR0911
        seq: Sequence[Any],
    ) -> "TypeGuard[Sequence[ToolResultContentBlockUnionTypeDef]]":
        if not isinstance(seq, Sequence) or isinstance(seq, str | bytes):
            return False

        for item in seq:
            if not isinstance(item, Mapping):
                return False

            if not any(
                key in item for key in ("json", "text", "image", "video", "document")
            ):
                return False

            if "json" in item and not isinstance(item["json"], Mapping):
                return False

            if "text" in item and not isinstance(item["text"], str):
                return False

            if "image" in item:
                image = item["image"]
                if not isinstance(image, Mapping) or "format" not in image:
                    return False

            if "video" in item:
                video = item["video"]
                if not isinstance(video, Mapping) or "format" not in video:
                    return False

            if "document" in item:
                document = item["document"]
                if not isinstance(document, Mapping) or "format" not in document:
                    return False

        return True

    @traced("converse", span_kind="SERVER")
    def converse(
        self,
        user_input: str | None,
        tools: Sequence[Callable | Tool] | None = None,
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
        tools : Sequence[Callable | Tool] | None, optional
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

        current_trace = self._tracer.current_trace
        current_trace.add_attribute("ai.trace.type", "converse")
        current_trace.add_attribute(
            "ai.conversation.id", self.conversation_id, inheritable=True
        )
        current_trace.add_attribute(
            "ai.subcontext.id", self.subcontext_id, inheritable=True
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

        if user_input == "":
            raise ValueError("Missing user input")

        if user_input is not None:
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

                with self._tracer.trace("llm-invocation", span_kind="CLIENT") as trace:
                    model_id = request["modelId"]
                    trace.add_attribute(
                        "peer.service",
                        self._shorten_bedrock_model_id(model_id, prefix="llm"),
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
                        response = invoke_cancellable(
                            stop_event=stop_event,
                            method=self.bedrock_client.converse,
                            kwargs=request,
                        )
                    except StopEventAbortError:
                        current_trace.add_attribute("ai.conversation.aborted", True)

                        text = self.stop_event_abort_message
                        texts.append(text)
                        cycle_trace.add_attribute("ai.agent.cycle.response", text)
                        current_trace.add_attribute(
                            "ai.agent.response",
                            "\n".join(texts),
                        )

                        # Create synthetic message and add to history
                        synthetic_message: MessageOutputTypeDef = {
                            "role": "assistant",
                            "content": [{"text": text}],
                        }
                        trace.add_attribute(
                            "ai.llm.response.output",
                            {"message": synthetic_message},
                        )
                        self._add_message(synthetic_message)

                        concatenated = "\n".join(texts)
                        return concatenated
                    except botocore.exceptions.ClientError as err:
                        trace.add_attribute("ai.llm.response.error", err.response)
                        raise

                    trace.add_attribute("ai.llm.response.output", response["output"])
                    trace.add_attribute(
                        "ai.llm.response.stop.reason", response["stopReason"]
                    )
                    trace.add_attribute("ai.llm.response.usage", response["usage"])
                    trace.add_attribute("ai.llm.response.metrics", response["metrics"])
                    if "trace" in response:
                        trace.add_attribute("ai.llm.response.trace", response["trace"])
                    if "performanceConfig" in response:
                        trace.add_attribute(
                            "ai.llm.response.performance.config",
                            response["performanceConfig"],
                        )

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
                                cycle_text += (
                                    f"<thinking>\n{reasoning_text}\n</thinking>\n"
                                )
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

                    tool_invoke_instructions = [
                        msg_content
                        for msg_content in message["content"]
                        if "toolUse" in msg_content
                    ]
                    if tool_invoke_instructions:
                        tool_results = self._invoke_tools(
                            tool_invoke_instructions,
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

                if response["stopReason"] in (
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
        user_input: str | None,
        stream: Literal["text"] = "text",
        tools: Sequence[Callable | Tool] | None = None,
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
        tools : Sequence[Callable | Tool] | None, optional
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
        tools: Sequence[Callable | Tool] | None = None,
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
        tools : Sequence[Callable | Tool] | None, optional
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

    def converse_stream(
        self,
        user_input: str | None,
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
            tracer = self._IterableTracer()
            with self._tracer.temporary_tracer(tracer):
                ctx = contextvars.copy_context()
                thread = Thread(
                    target=ctx.run,
                    args=[self._consume_iterable_and_shutdown_tracer, gen, tracer],
                    daemon=False,
                    name="converse_stream",
                )
                thread.start()
                try:
                    yield from tracer
                finally:
                    thread.join(timeout=5.0)
                    if thread.is_alive():
                        warnings.warn(
                            "The thread that was streaming traces couldn't be joined (stuck thread)",
                            RuntimeWarning,
                            stacklevel=2,
                        )

        else:
            raise ValueError(f"stream must be 'text' or 'traces', but got {stream}")

    @traced("converse-stream", span_kind="SERVER")
    def _converse_stream(
        self,
        user_input: str | None,
        tools: Sequence[Callable | Tool] | None = None,
        stop_event: Event | None = None,
    ) -> Iterable[str]:
        current_trace = self._tracer.current_trace
        current_trace.add_attribute("ai.trace.type", "converse-stream")
        current_trace.add_attribute(
            "ai.conversation.id", self.conversation_id, inheritable=True
        )
        current_trace.add_attribute(
            "ai.subcontext.id", self.subcontext_id, inheritable=True
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

        if user_input == "":
            raise ValueError("Missing user input")

        if user_input is not None:
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

                texts.append("")

                with self._tracer.trace("llm-invocation", span_kind="CLIENT") as trace:
                    model_id = request["modelId"]
                    trace.add_attribute(
                        "peer.service",
                        self._shorten_bedrock_model_id(model_id, prefix="llm"),
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
                        response = invoke_cancellable(
                            stop_event=stop_event,
                            method=self.bedrock_client.converse_stream,
                            kwargs=request,
                        )
                    except StopEventAbortError:
                        response = None
                    except botocore.exceptions.ClientError as err:
                        trace.add_attribute("ai.llm.response.error", err.response)
                        raise

                    metadata = None
                    stop_reason = None
                    content_block_handler = (
                        BedrockConverseStreamEventContentBlockHandler()
                    )

                    if response:
                        is_reasoning = False
                        for index, stream_event in enumerate(
                            with_placeholder_emit(
                                response["stream"], stop_event=stop_event
                            )
                        ):
                            trace.add_attribute(
                                "ai.llm.response.stream.events", index + 1
                            )
                            if stop_event and stop_event.is_set():
                                break
                            if stream_event is None:
                                continue
                            if "contentBlockStart" in stream_event:
                                content_block_handler.process_stream_event(stream_event)
                                trace.add_attribute(
                                    "ai.llm.response.output",
                                    {
                                        "message": content_block_handler.get_message(
                                            provisional=True
                                        )
                                    },
                                )
                                trace.emit_snapshot()

                            elif "contentBlockDelta" in stream_event:
                                content_block_handler.process_stream_event(stream_event)
                                trace.add_attribute(
                                    "ai.llm.response.output",
                                    {
                                        "message": content_block_handler.get_message(
                                            provisional=True
                                        )
                                    },
                                )
                                trace.emit_snapshot()

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
                                    reasoning_content = stream_event[
                                        "contentBlockDelta"
                                    ]["delta"].get("reasoningContent")
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
                                                "\n".join(
                                                    text for text in texts if text
                                                ),
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
                                content_block_handler.process_stream_event(stream_event)
                                trace.add_attribute(
                                    "ai.llm.response.output",
                                    {
                                        "message": content_block_handler.get_message(
                                            provisional=True
                                        )
                                    },
                                )
                                trace.emit_snapshot()
                            elif "messageStart" in stream_event:
                                pass

                            elif "messageStop" in stream_event:
                                stop_reason = stream_event["messageStop"]["stopReason"]
                                yield "\n"

                            elif "metadata" in stream_event:
                                metadata = stream_event["metadata"]
                                trace.add_attribute(
                                    "ai.llm.response.stop.reason", stop_reason
                                )
                                trace.add_attribute(
                                    "ai.llm.response.usage", metadata["usage"]
                                )
                                trace.add_attribute(
                                    "ai.llm.response.metrics", metadata["metrics"]
                                )
                                if "trace" in metadata:
                                    trace.add_attribute(
                                        "ai.llm.response.trace", metadata["trace"]
                                    )
                                if "performanceConfig" in metadata:
                                    trace.add_attribute(
                                        "ai.llm.response.performance.config",
                                        metadata["performanceConfig"],
                                    )
                                trace.emit_snapshot()
                            else:
                                raise Exception(
                                    f"Unsupported stream event {stream_event}"
                                )

                    if stop_event and stop_event.is_set():
                        current_trace.add_attribute("ai.conversation.aborted", True)

                        text = self.stop_event_abort_message
                        texts[-1] += text
                        cycle_trace.add_attribute("ai.agent.cycle.response", texts[-1])
                        cycle_trace.emit_snapshot()
                        current_trace.add_attribute(
                            "ai.agent.response",
                            "\n".join(text for text in texts if text),
                        )
                        message = content_block_handler.get_message(provisional=True)
                        message["content"] = [
                            cb for cb in message["content"] if "toolUse" not in cb
                        ]
                        message["content"].append({"text": text})
                        trace.add_attribute(
                            "ai.llm.response.output",
                            {"message": message},
                        )
                        self._add_message(message)
                        yield text
                        Thread(
                            target=lambda r=response: r and r["stream"].close(),
                            daemon=True,
                        ).start()
                        return

                    if not metadata:
                        raise ValueError("Incomplete response stream: missing metadata")
                    if not stop_reason:
                        raise ValueError(
                            "Incomplete response stream: missing stop_reason"
                        )
                    message = content_block_handler.get_message()
                    if not message["content"]:
                        raise ValueError("LLM response has empty content")

                    trace.add_attribute(
                        "ai.llm.response.output",
                        {"message": message},
                    )

                if not texts[-1]:
                    texts.pop()
                self._add_message(message)

                tool_invoke_instructions = [
                    content_block
                    for content_block in content_block_handler.finalized_blocks
                    if "toolUse" in content_block
                ]
                if tool_invoke_instructions:
                    tool_results = self._invoke_tools(
                        tool_invoke_instructions,
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

        # Remove subcontext_id (if any) as the parameter should have been used before calling invoke(),
        # as argument to set_conversation_id(..., subcontext_id=...)
        # It is not actually an input parameter of converse_stream()
        kwargs.pop("subcontext_id", None)

        response = "".join(self.converse_stream(*args, **kwargs, user_input=user_input))
        return json.dumps({"response": response, "subcontext_id": self.subcontext_id})

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
                        "subcontext_id": {
                            "type": "string",
                            "description": textwrap.dedent(
                                """
                                Identifier of the subcontext with conversational memory that the agent is working in.

                                Only provide this for follow-up messages to the agent, after its initial response.
                                In that case, provide the same subcontext_id that the agent provided in its initial response,
                                so that the agent will use the right conversational memory, and thus may understand what you're referring to!

                                For independent requests to the agent, do not provide subcontext_id. This is more efficient,
                                because the agent can then operate with an empty short term memory, unburdened by prior conversation history.
                                """
                            ).strip(),
                        },
                    },
                    "required": ["user_input"],
                }
            },
        }

    @staticmethod
    def _consume_iterable_and_shutdown_tracer(
        iterable: Iterable[str], tracer: IterableTracer
    ):
        try:
            for _ in iterable:
                pass
        finally:
            tracer.shutdown()


C = TypeVar("C")


@runtime_checkable
class Spawnable(Protocol):
    def spawn(self: C) -> C:
        """
        Spawn a new independent instance of self.
        """
        ...


@runtime_checkable
class AgentAsTool(Protocol):
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

    def set_conversation_id(
        self, conversation_id: str, *, subcontext_id: str | None = None
    ) -> None:
        """
        Set the current conversation id and subcontext_id (if any) of the agent.
        """
        ...

    def set_auth_context(self, **auth_context: Unpack[AuthContext]) -> None:
        """
        Set the auth context of the agent.
        """
        ...
