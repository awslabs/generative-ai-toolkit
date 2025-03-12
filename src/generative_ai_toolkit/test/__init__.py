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

from dataclasses import asdict, dataclass
import textwrap
from typing import (
    Any,
    Callable,
    Iterable,
    Mapping,
    Protocol,
    Sequence,
    cast,
    runtime_checkable,
)
from concurrent.futures import ThreadPoolExecutor

import boto3

from generative_ai_toolkit.tracer.tracer import (
    Trace,
    LlmTrace,
    ToolTrace,
    user_conversation_from_messages,
)
from generative_ai_toolkit.metrics.measurement import Measurement
from generative_ai_toolkit.agent import Tool
from generative_ai_toolkit.utils.llm_response import get_text, NonStreamingResponse
from generative_ai_toolkit.utils.typings import Message


@dataclass(kw_only=True, frozen=True, repr=False)
class LlmCaseTrace(LlmTrace):
    case: "Case"
    case_nr: int
    run_nr: int
    permutation: Mapping[str, Any] | None

    def __repr__(self) -> str:
        return repr_case_trace(self)


@dataclass(kw_only=True, frozen=True, repr=False)
class ToolCaseTrace(ToolTrace):
    case: "Case"
    case_nr: int
    run_nr: int
    permutation: Mapping[str, Any] | None

    def __repr__(self) -> str:
        return repr_case_trace(self)


CaseTrace = LlmCaseTrace | ToolCaseTrace


def repr_case_trace(trace: CaseTrace) -> str:
    return f"CaseTrace(to={trace.to}, conversation_id={trace.conversation_id}, trace_id={trace.trace_id}), case_name={trace.case.name}, case_nr={trace.case_nr}, run_nr={trace.run_nr})"


class _AgentLike(Protocol):
    def converse(self, user_input: str) -> Any: ...

    @property
    def traces(self) -> Iterable[Trace]: ...


@runtime_checkable
class _AgentLikeWithReset(Protocol):
    def converse(self, user_input: str) -> Any: ...

    @property
    def traces(self) -> Iterable[Trace]: ...

    def reset(self) -> None: ...


@runtime_checkable
class _AgentLikeWithMessages(Protocol):
    def converse(self, user_input: str) -> Any: ...

    @property
    def traces(self) -> Iterable[Trace]: ...

    @property
    def messages(self) -> Sequence[Message]: ...


AgentLike = _AgentLike | _AgentLikeWithReset | _AgentLikeWithMessages


@dataclass
class CaseTraceInfo:
    case_nr: int = 0
    run_nr: int = 0
    permutation: Mapping[str, Any] | None = None


ValidatorFunc = Callable[[Sequence[CaseTrace]], str | Sequence[str] | None]


class Case:
    _user_inputs: list[str]
    overall_expectations: str | None
    expected_agent_responses_per_turn: list[Sequence[str]]
    converse_kwargs: Mapping
    validate: ValidatorFunc | Sequence[ValidatorFunc] | None
    user_input_producer: Callable[[Sequence[Message]], str] | None

    def __init__(
        self,
        name: str,
        *,
        user_inputs: Sequence[str] | None = None,
        user_input_producer: Callable[[Sequence[Message]], str] | None = None,
        overall_expectations: str | None = None,
        converse_kwargs: Mapping | None = None,
        validate: ValidatorFunc | Sequence[ValidatorFunc] | None = None,
    ) -> None:
        self._user_inputs = list(user_inputs) if user_inputs is not None else []
        self.user_input_producer = user_input_producer
        self.overall_expectations = overall_expectations
        self.expected_agent_responses_per_turn = []
        self.validate = validate
        self.converse_kwargs = converse_kwargs if converse_kwargs is not None else {}
        self.name = name

    def __repr__(self) -> str:
        return f'Case(name="{self.name}",user_inputs={self._user_inputs})'

    def as_case_trace(
        self, trace: Trace, case_trace_info: CaseTraceInfo | None = None
    ) -> CaseTrace:
        if case_trace_info is None:
            case_trace_info = CaseTraceInfo()
        if isinstance(trace, LlmTrace):
            return LlmCaseTrace(
                **asdict(trace),
                case=self,
                case_nr=case_trace_info.case_nr,
                run_nr=case_trace_info.run_nr,
                permutation=case_trace_info.permutation,
            )
        elif isinstance(trace, ToolTrace):
            return ToolCaseTrace(
                **asdict(trace),
                case=self,
                case_nr=case_trace_info.case_nr,
                run_nr=case_trace_info.run_nr,
                permutation=case_trace_info.permutation,
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
            messages = (
                _agent.messages if isinstance(_agent, _AgentLikeWithMessages) else []
            )
            while user_input := self.user_input_producer(messages):
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
            name=case_name or f"Tool use: {tool.name}",
            user_inputs=[get_text(cast(NonStreamingResponse, response))],
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
    user_input_producer: Callable[[Sequence[Message]], str] | None = None,
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

    def _format_conversation_history(self, messages: Sequence[Message]) -> str:
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
        self, messages: Sequence[Message] | None = None
    ) -> bool:
        if not messages:
            return False
        conversation_messages = user_conversation_from_messages(messages)
        if len(conversation_messages) > self.max_nr_turns:
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

                Return "USER INPUT NEEDED" if the assistant STILL requires information from the user. Return "USER INPUT OPTIONAL" otherwise.
                Only return "USER INPUT NEEDED" or "USER INPUT OPTIONAL".
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
                        """
                    ).strip()
                }
            ],
        )
        intent_satisfied = (
            "optional" in get_text(cast(NonStreamingResponse, response)).lower()
        )
        return intent_satisfied

    def __call__(self, messages: Sequence[Message] | None = None) -> str:
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
        return get_text(cast(NonStreamingResponse, response))


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

    def _to_equal(self, a, b):
        assert a == b, f"{a} != {b}"

    @property
    def agent_text_response(self):
        last_trace = self.traces[-1]
        assert last_trace.to == "LLM", "Last trace is not an LLM trace"
        assert last_trace.response["output"]["message"]["role"] == "assistant"
        agent_messages = [
            c for c in last_trace.response["output"]["message"]["content"]
        ]
        agent_text_response = [c["text"] for c in agent_messages if "text" in c][0]
        return _StringAssertor(agent_text_response)

    @property
    def tool_invocations(self):
        tool_traces = [trace for trace in self.traces if trace.to == "TOOL"]
        return _ToolAssertor(tool_traces)


class _StringAssertor:

    def __init__(self, base_value: str) -> None:
        self.base_value = base_value

    def to_equal(self, value: str) -> None:
        assert self.base_value == value

    def to_include(self, value: str) -> None:
        assert value in self.base_value

    def to_have_length(self, length=1):
        assert len(self.base_value) >= length


class _ToolAssertor:
    def __init__(self, tool_traces: Sequence[ToolTrace]) -> None:
        self.tool_traces = tool_traces

    def to_include(self, tool_name: str, *, with_error: bool | None | str = False):
        tool_invocations = [
            trace
            for trace in self.tool_traces
            if tool_name == trace.request["tool_name"]
        ]
        if not tool_invocations:
            raise AssertionError(f"Tool {tool_name} was not invoked")
        if with_error is None:
            pass
        errors = [
            trace.response["tool_error"]
            for trace in tool_invocations
            if trace.response.get("tool_error")
        ]
        if with_error:
            if not any(errors):
                raise AssertionError(f"Tool {tool_name} did not raise an error")
            if type(with_error) is str:
                assert any(error for error in errors if with_error in error)
        else:
            if any(errors):
                raise AssertionError(f"Tool {tool_name} raised an error: {errors[0]}")
