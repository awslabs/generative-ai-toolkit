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

import textwrap
from collections.abc import Callable, Iterable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime
from typing import (
    TYPE_CHECKING,
    Any,
    Literal,
    Protocol,
    TypedDict,
    cast,
    runtime_checkable,
)

import boto3

from generative_ai_toolkit.agent import Tool
from generative_ai_toolkit.metrics.measurement import Measurement
from generative_ai_toolkit.tracer.tracer import (
    Trace,
    TraceScope,
)
from generative_ai_toolkit.utils.llm_response import get_text

if TYPE_CHECKING:
    from mypy_boto3_bedrock_runtime.type_defs import MessageOutputTypeDef


class CaseTrace(Trace):
    case: "Case"
    case_nr: int
    run_nr: int
    permutation: Mapping[str, Any] | None
    permutation_nr: int

    def __init__(
        self,
        *,
        # CaseTrace specific attributes:
        case: "Case",
        case_nr: int,
        run_nr: int,
        permutation_nr: int,
        permutation: Mapping[str, Any] | None,
        # Trace attributes:
        span_name: str,
        span_kind: Literal["INTERNAL", "SERVER", "CLIENT"] = "INTERNAL",
        trace_id: str | None = None,
        span_id: str | None = None,
        parent_span: "Trace | None" = None,
        started_at: datetime | None = None,
        ended_at: datetime | None = None,
        attributes: dict[str, Any] | None = None,
        span_status: Literal["UNSET", "OK", "ERROR"] = "UNSET",
        resource_attributes: Mapping[str, Any] | None = None,
        scope: "TraceScope | None" = None,
    ) -> None:
        super().__init__(
            span_name=span_name,
            span_kind=span_kind,
            trace_id=trace_id,
            span_id=span_id,
            parent_span=parent_span,
            started_at=started_at,
            ended_at=ended_at,
            attributes=attributes,
            resource_attributes=resource_attributes,
            span_status=span_status,
            scope=scope,
        )
        self.case = case
        self.case_nr = case_nr
        self.run_nr = run_nr
        self.permutation = permutation
        self.permutation_nr = permutation_nr


class _AgentLike(Protocol):
    def converse(self, user_input: str) -> Any: ...

    @property
    def traces(self) -> Sequence[Trace]: ...


@runtime_checkable
class _AgentLikeWithReset(_AgentLike, Protocol):

    def reset(self) -> None: ...


@runtime_checkable
class _AgentLikeWithMessages(_AgentLike, Protocol):

    @property
    def messages(self) -> Sequence["MessageOutputTypeDef"]: ...


AgentLike = _AgentLike | _AgentLikeWithReset | _AgentLikeWithMessages


@dataclass
class CaseTraceInfo:
    case_nr: int = 0
    run_nr: int = 0
    permutation: Mapping[str, Any] | None = None
    permutation_nr: int = 0


ValidatorFunc = Callable[[Sequence[CaseTrace]], str | Sequence[str] | None]


class Case:
    _user_inputs: list[str]
    overall_expectations: str | None
    expected_agent_responses_per_turn: list[Sequence[str]]
    converse_kwargs: Mapping
    validate: ValidatorFunc | Sequence[ValidatorFunc] | None
    user_input_producer: Callable[[Sequence["MessageOutputTypeDef"]], str] | None

    def __init__(
        self,
        user_inputs: str | Sequence[str] | None = None,
        *,
        name: str | None = None,
        user_input_producer: (
            Callable[[Sequence["MessageOutputTypeDef"]], str] | None
        ) = None,
        overall_expectations: str | None = None,
        converse_kwargs: Mapping | None = None,
        validate: ValidatorFunc | Sequence[ValidatorFunc] | None = None,
    ) -> None:
        if not user_inputs:
            self._user_inputs = []
        elif type(user_inputs) is str:
            self._user_inputs = [user_inputs]
        else:
            self._user_inputs = list(user_inputs)
        self.user_input_producer = user_input_producer
        self.overall_expectations = overall_expectations
        self.expected_agent_responses_per_turn = []
        self.validate = validate
        self.converse_kwargs = converse_kwargs if converse_kwargs is not None else {}
        self.name = name

    def __repr__(self) -> str:
        return f"Case(name={repr(self.name)},user_inputs={self._user_inputs})"

    def as_case_trace(
        self, trace: Trace, case_trace_info: CaseTraceInfo | None = None
    ) -> CaseTrace:
        if case_trace_info is None:
            case_trace_info = CaseTraceInfo()
        return CaseTrace(
            # Trace attributes:
            span_name=trace.span_name,
            span_kind=trace.span_kind,
            trace_id=trace.trace_id,
            span_id=trace.span_id,
            parent_span=trace.parent_span,
            started_at=trace.started_at,
            ended_at=trace.ended_at,
            attributes=dict(trace.attributes),
            resource_attributes=trace.resource_attributes,
            span_status=trace.span_status,
            scope=trace.scope,
            # CaseTrace specific attributes:
            case=self,
            case_nr=case_trace_info.case_nr,
            run_nr=case_trace_info.run_nr,
            permutation=case_trace_info.permutation,
            permutation_nr=case_trace_info.permutation_nr,
        )

    def run(
        self,
        agent: AgentLike | Callable[[], AgentLike],
        case_trace_info: CaseTraceInfo | None = None,
    ) -> Sequence[CaseTrace]:
        """
        Run through the case with the supplied agent, by feeding it one user input at a time and awaiting the agent's response.

        Either supply an agent, or a factory function that returns an agent.
        If you supply an agent (and not a factory function), the agent will be reset() first.

        Returns the traces for this conversation, and stores a reference to the case in each trace.
        """
        if callable(agent):
            _agent = agent()
        else:
            _agent = agent
            if isinstance(_agent, _AgentLikeWithReset):
                _agent.reset()
        if self._user_inputs:
            for user_input in self._user_inputs:
                _agent.converse(user_input, **self.converse_kwargs)
        if self.user_input_producer:
            if not isinstance(_agent, _AgentLikeWithMessages):
                raise ValueError(
                    "user_input_producer can only be used with agents that implement the messages property"
                )
            while user_input := self.user_input_producer(_agent.messages):
                _agent.converse(user_input, **self.converse_kwargs)
        return [self.as_case_trace(trace, case_trace_info) for trace in _agent.traces]

    def add_turn(
        self,
        user_input: str,
        expected_agent_responses: Sequence[str],
    ):
        if len(self._user_inputs) != len(self.expected_agent_responses_per_turn):
            raise ValueError(
                "Cannot add turn for case with different nr of user_inputs and nr of expected_agent_responses"
            )
        self._user_inputs.append(user_input)
        self.expected_agent_responses_per_turn.append(expected_agent_responses)

    @classmethod
    def _for_agent_tool(
        cls,
        *,
        tool: Tool,
        agent_system_prompt="You are a helpful AI assistant",
        model_id="anthropic.claude-3-haiku-20240307-v1:0",
        language="en_US",
        case_name: str | None = None,
    ):
        bedrock_client = boto3.client("bedrock-runtime")
        response = bedrock_client.converse(
            modelId=model_id,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "text": textwrap.dedent(
                                """
                                The agent's system prompt, and the tool's specification are described below.
                                Create a user utterance that would make the LLM agent use the tool.
                                The user utterance should be detailed enough so that the agent would be able to directly infer all tool parameter values.
                                Of course the user would not literally instruct the agent to call the tool. The user would simply convey their intent, the thing the tool usage should achieve.

                                <agent_system_prompt>
                                {agent_system_prompt}
                                </agent_system_prompt>

                                <tool_spec>
                                {tool_spec}
                                </tool_spec>

                                The user utterance should be in the following language: {language}

                                Return only the proposed user utterance, and nothing else. Don't wrap the utterance in quotes.
                                """
                            )
                            .format(
                                agent_system_prompt=agent_system_prompt,
                                tool_spec=tool.tool_spec,
                                language=language,
                            )
                            .strip()
                        }
                    ],
                }
            ],
            system=[
                {
                    "text": textwrap.dedent(
                        """
                        You are an expert at creating sample user utterances, that are used for testing LLM based agents.
                        Thus, you are good at pretending to be a human, and speak as they would.
                        """
                    ).strip()
                }
            ],
        )
        return Case(
            name=case_name or f"Tool use: {tool.tool_spec["name"]}",
            user_inputs=[get_text(response)],
        )

    @classmethod
    def for_agent_tools(
        cls,
        *,
        tools: Iterable[Tool],
        languages=("en_US",),
        agent_system_prompt="You are a helpful AI assistant",
        model_id="anthropic.claude-3-haiku-20240307-v1:0",
        case_name: str | None = None,
    ) -> list["Case"]:
        futs = []
        with ThreadPoolExecutor() as executor:
            for tool in tools:
                for language in languages:
                    fut = executor.submit(
                        cls._for_agent_tool,
                        agent_system_prompt=agent_system_prompt,
                        language=language,
                        tool=tool,
                        model_id=model_id,
                        case_name=case_name,
                    )
                    futs.append(fut)
        return [fut.result() for fut in futs]


def case(
    name: str | None = None,
    *,
    user_inputs: Sequence[str] | None = None,
    overall_expectations: str | None = None,
    converse_kwargs: Mapping | None = None,
    user_input_producer: (
        Callable[[Sequence["MessageOutputTypeDef"]], str] | None
    ) = None,
):
    def decorator(func: ValidatorFunc):
        return Case(
            name=name or func.__name__,
            user_inputs=user_inputs,
            overall_expectations=overall_expectations,
            converse_kwargs=converse_kwargs,
            user_input_producer=user_input_producer,
            validate=func,
        )

    return decorator


class ConversationMessage(TypedDict):
    role: Literal["user", "assistant"]
    text: str


def user_conversation_from_trace(trace: Trace):
    if trace.attributes.get("ai.trace.type") != "llm-invocation":
        raise ValueError("Trace did not capture an LLM invocation")
    return user_conversation_from_messages(
        (
            *trace.attributes["ai.llm.request.messages"],
            trace.attributes["ai.llm.response.output"]["message"],
        )
    )


def user_conversation_from_messages(messages: Iterable["MessageOutputTypeDef"]):
    user_conversation: list[ConversationMessage] = []

    for msg in messages:
        texts: list[str] = []
        for part in msg["content"]:
            if "text" not in part:
                continue
            texts.append(part["text"])
        if texts:
            text = "\n".join(texts)
            if user_conversation and user_conversation[-1]["role"] == msg["role"]:
                user_conversation[-1]["text"] += "\n" + text
            else:
                user_conversation.append({"role": msg["role"], "text": text})

    return user_conversation


class UserInputProducer:
    def __init__(
        self,
        *,
        user_intent: str,
        model_id="anthropic.claude-3-haiku-20240307-v1:0",
        language="en_US",
        max_nr_turns: int = 5,
    ):
        self.bedrock_client = boto3.client("bedrock-runtime")
        self.language = language
        self.model_id = model_id
        self.user_intent = user_intent
        self.max_nr_turns = max_nr_turns

    def _format_conversation_history(
        self, messages: Sequence["MessageOutputTypeDef"]
    ) -> str:
        turns = []
        for msg in user_conversation_from_messages(messages):
            turns.append(
                f""" <conversation_turn role="{msg["role"]}">{msg["text"]}</conversation_turn>"""
            )

        return (
            textwrap.dedent(
                """
            <conversation_history>
            {conversation_history}
            </conversation_history>
            """
            )
            .format(conversation_history="\n".join(turns))
            .strip()
        )

    def _should_stop_conversation(
        self, messages: Sequence["MessageOutputTypeDef"] | None = None
    ) -> bool:
        if not messages:
            return False
        conversation_messages = user_conversation_from_messages(messages)
        if int(len(conversation_messages) / 2) >= self.max_nr_turns:
            return True
        text = (
            textwrap.dedent(
                """
                Here's the user's current intent:

                <user_intent>
                {user_intent}
                </user_intent>

                Here is a conversation between a user and an assistant:

                {conversation_history}

                Return "ASK USER" if the assistant needs input from the user, for example:

                  - if the assistant asks the user for confirmation
                  - if the assistant asks the user for additional information to (further) clarify the user's request.

                Return "ABORT", if continuing the conversation doesn't make sense anymore, for example:

                  - if the user's intent was reasonably satisfied by the assistant; the assistant did what the user asked.
                  - if the user wants to end the conversation
                  - if continuing the conversation becomes pointless

                Your response should be "ASK USER" or "ABORT", followed by your reasoning, for example:

                  - ABORT: the user's intent was satisfied, they wanted XYZ and they got XYZ. The assistant doesn't need to confirm.
                  - ABORT: the user explicitly asks to abort/stop/discontinue the conversation
                  - ABORT: the user implicitly signals they want to end the conversation, by saying e.g. just "OK" or "Thank you"
                  - ASK USER: the assistant needs the user to clarify XYZ
                  - ASK USER: the assistant requests the user for confirmation
                  - ASK USER: the assistant wants to know if the user has any other requests

                Give your response now, in the format explained above.
                """
            )
            .format(
                conversation_history=self._format_conversation_history(messages),
                user_intent=self.user_intent,
            )
            .strip()
        )

        response = self.bedrock_client.converse(
            modelId=self.model_id,
            messages=[
                {
                    "role": "user",
                    "content": [{"text": text}],
                }
            ],
            system=[
                {
                    "text": textwrap.dedent(
                        """
                        You are an expert at judging conversations between users and assistants.
                        Your job is to determine if the user should provide additional input to the assistant.
                        Don't bother users unnecessarily; if they got what they asked for, the assistant doesn't need more input from them.
                        """
                    ).strip()
                }
            ],
        )
        response = get_text(response)
        intent_satisfied = "ABORT" in response
        return intent_satisfied

    def __call__(self, messages: Sequence["MessageOutputTypeDef"] | None = None) -> str:
        if messages:
            if self._should_stop_conversation(messages):
                return ""
            conversation_history_text = textwrap.dedent(
                """
                Take into account that the user has already been talking with the agent:

                {conversation_history}
                """
            ).format(conversation_history=self._format_conversation_history(messages))
        else:
            conversation_history_text = ""

        text = (
            textwrap.dedent(
                """
                Here's the user's current intent.

                <user_intent>
                {user_intent}
                </user_intent>

                Generate an utterance, as if you are a user with that intent, to make the agent achieve that intent.
                {conversation_history_text}

                If the agent asks more information from the user, make up a concise and to-the-point answer on behalf of the user.

                Return only the proposed user utterance, and nothing else. Don't wrap the utterance in quotes.
                """
            )
            .format(
                user_intent=self.user_intent,
                language=self.language,
                conversation_history_text=conversation_history_text,
            )
            .strip()
        )

        response = self.bedrock_client.converse(
            modelId=self.model_id,
            messages=[
                {
                    "role": "user",
                    "content": [{"text": text}],
                }
            ],
            system=[
                {
                    "text": textwrap.dedent(
                        """
                        You are an expert at creating user utterances as input for LLM based agents. Your utterances are used to test these agents.
                        """
                    ).strip()
                }
            ],
        )
        return get_text(response)


class _PassFail:
    measurement_name_passed = "ValidationPassed"
    measurement_name_failed = "ValidationFailed"

    def passed(self):
        return Measurement(
            name=self.measurement_name_passed, value=1, validation_passed=True
        )

    def failed(self, validation_messages: str | Sequence[str]):
        return Measurement(
            name=self.measurement_name_failed,
            value=1,
            additional_info={
                "validation_messages": (
                    [validation_messages]
                    if isinstance(validation_messages, str)
                    else validation_messages
                )
            },
            validation_passed=False,
        )

    def validate_conversation(
        self, traces: Sequence[Trace], case_: Case
    ) -> Measurement | Sequence[Measurement] | None:
        if case_.validate is None:
            return
        validate = (
            case_.validate if isinstance(case_.validate, Sequence) else [case_.validate]
        )
        validation_messages: list[str] = []
        for validator in validate:
            try:
                result = validator(cast(Sequence[CaseTrace], traces))
            except Exception as e:
                result = str(e)
            if result:
                if isinstance(result, str):
                    validation_messages.append(result)
                else:
                    validation_messages.extend(result)
        return (
            self.passed()
            if not validation_messages
            else self.failed(validation_messages)
        )


PassFail = _PassFail()


class Expect:
    def __init__(self, traces: Sequence[Trace]) -> None:
        if not traces:
            raise ValueError("traces must not be an empty list")
        self.traces = traces

    @property
    def user_input(self):
        """
        Make assertions about the user input.
        """

        user_inputs = [
            trace.attributes["ai.user.input"]
            for trace in self.traces
            if "ai.user.input" in trace.attributes
        ]
        return _StringAssertor(user_inputs)

    @property
    def agent_text_response(self):
        """
        Make assertions about the agent's response
        """

        agent_text_responses = [
            trace.attributes["ai.agent.response"]
            for trace in self.traces
            if "ai.agent.response" in trace.attributes
        ]
        return _StringAssertor(agent_text_responses)

    @property
    def tool_invocations(self):
        """
        Make assertions about tool invocations (traces with the attribute "ai.tool.name")
        """

        tool_traces = [
            trace for trace in self.traces if "ai.tool.name" in trace.attributes
        ]
        return _ToolAssertor(tool_traces)


class _StringAssertor:

    _at: int
    base_value: str
    base_values: Sequence[str]

    def __init__(self, base_values: str | Sequence[str], at=-1) -> None:
        self.base_values = [base_values] if type(base_values) is str else base_values
        self._at = at
        self.base_value = self.base_values[at] if self.base_values else ""

    def at(self, index: int) -> "_StringAssertor":
        return _StringAssertor(self.base_values, index)

    def to_equal(self, value: str) -> None:
        assert self.base_value == value, f"'{self.base_value}' != '{value}'"

    def to_include(self, value: str) -> None:
        assert (
            value in self.base_value
        ), f"'{self.base_value}' does not include '{value}'"

    def to_not_include(self, value: str) -> None:
        assert value not in self.base_value, f"'{self.base_value}' includes '{value}'"

    def to_have_length(self, length: int | None = None):
        expected_txt = length if length is not None else "of at least 1"
        assert (
            len(self.base_value) == length
            if length is not None
            else len(self.base_value) > 0
        ), f"Expected '{self.base_value}' to have length {expected_txt}, but it has length {len(self.base_value)}"


class _ToolAssertor:
    def __init__(self, tool_traces: Sequence[Trace]) -> None:
        self.tool_traces = tool_traces

    def to_have_length(self, length: int | None = None):
        expected_txt = length if length is not None else "at least one"
        assert (
            len(self.tool_traces) == length
            if length is not None
            else len(self.tool_traces) > 0
        ), f"Expected {expected_txt} tool invocation(s), but encountered {len(self.tool_traces)}"

    def to_include(self, tool_name: str, *, with_error: bool | None | str = False):
        tool_invocations = [
            trace
            for trace in self.tool_traces
            if tool_name == trace.attributes["ai.tool.name"]
        ]
        if not tool_invocations:
            raise AssertionError(f"Tool {tool_name} was not invoked")
        if with_error is None:
            return _ToolInputOutputAssertor(tool_invocations)
        errors = [
            trace.attributes["ai.tool.error"]
            for trace in tool_invocations
            if "ai.tool.error" in trace.attributes
        ]
        if with_error:
            if not any(errors):
                raise AssertionError(f"Tool {tool_name} did not raise an error")
            if type(with_error) is str:
                assert any(error for error in errors if with_error in error)
        elif any(errors):
            raise AssertionError(f"Tool {tool_name} raised an error: {errors[0]}")
        return _ToolInputOutputAssertor(tool_invocations)

    def to_not_include(self, tool_name: str):
        for trace in self.tool_traces:
            if trace.attributes["ai.tool.name"] == tool_name:
                raise AssertionError(f"Tool {tool_name} was invoked")


class _ToolInputOutputAssertor:
    def __init__(self, tool_traces: Sequence[Trace]) -> None:
        self.tool_traces = tool_traces

    def with_input(self, expected_input: Any):
        tool_name = self.tool_traces[0].attributes["ai.tool.name"]
        if len(self.tool_traces) > 1:
            message = f"Tool {tool_name} was not invoked with input '{expected_input}'"
        else:
            actual_input = self.tool_traces[0].attributes["ai.tool.input"]
            message = f"Tool {tool_name} was invoked with input '{actual_input}' but expected '{expected_input}'"
        assert any(
            trace.attributes["ai.tool.input"] == expected_input
            for trace in self.tool_traces
            if "ai.tool.input" in trace.attributes
        ), message
        return self

    def with_output(self, expected_output: Any):
        tool_name = self.tool_traces[0].attributes["ai.tool.name"]
        if len(self.tool_traces) > 1:
            message = f"Tool {tool_name} did not return output '{expected_output}'"
        else:
            actual_output = self.tool_traces[0].attributes["ai.tool.output"]
            message = f"Tool {tool_name} returned output '{actual_output}' but expected '{expected_output}'"
        assert any(
            trace.attributes["ai.tool.output"] == expected_output
            for trace in self.tool_traces
            if "ai.tool.output" in trace.attributes
        ), message
        return self
