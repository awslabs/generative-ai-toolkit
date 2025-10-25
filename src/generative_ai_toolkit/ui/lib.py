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

import asyncio
import html
import json
import re
import textwrap
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from itertools import groupby
from typing import Literal

import gradio as gr
from gradio.components.chatbot import MetadataDict

from generative_ai_toolkit.context import AuthContext
from generative_ai_toolkit.evaluate.evaluate import ConversationMeasurements
from generative_ai_toolkit.metrics.measurement import Measurement, Unit
from generative_ai_toolkit.tracer.trace import Trace
from generative_ai_toolkit.utils.json import DefaultJsonEncoder


@dataclass
class TraceSummary:
    trace_id: str
    span_id: str
    started_at: datetime
    ended_at: datetime | None
    duration_ms: int | None
    conversation_id: str
    auth_context: AuthContext = field(default_factory=lambda: {"principal_id": None})
    user_input: str | None = None
    agent_cycle_traces: dict[str, Trace] = field(default_factory=dict)
    all_traces: list[Trace] = field(default_factory=list)
    measurements_per_trace: dict[tuple[str, str], list[Measurement]] = field(
        default_factory=dict
    )


def get_summaries_for_traces(traces: Sequence[Trace]):
    trace_summaries: list[TraceSummary] = []
    by_start_date = sorted(traces, key=lambda t: t.started_at)
    by_trace_id = sorted(by_start_date, key=lambda t: t.trace_id)
    for trace_id, traces_for_trace_id_iter in groupby(
        by_trace_id, key=lambda t: t.trace_id
    ):
        traces_for_trace_id = list(traces_for_trace_id_iter)
        root_agent_traces = [
            trace
            for trace in traces_for_trace_id
            if "ai.agent.hierarchy.parent.span.id" not in trace.attributes
        ]
        root_trace = traces_for_trace_id[0]
        summary = TraceSummary(
            conversation_id=root_trace.attributes["ai.conversation.id"],
            auth_context=root_trace.attributes["ai.auth.context"],
            trace_id=trace_id,
            span_id=root_trace.span_id,
            duration_ms=root_trace.ended_at and root_trace.duration_ms,
            started_at=root_trace.started_at,
            ended_at=root_trace.ended_at,
            all_traces=traces_for_trace_id,
            agent_cycle_traces={
                trace.span_id: trace
                for trace in traces_for_trace_id
                if trace.attributes.get("ai.trace.type") == "cycle"
            },
        )

        # Find (first) user input:
        for trace in root_agent_traces:
            if not summary.user_input and "ai.user.input" in trace.attributes:
                summary.user_input = trace.attributes["ai.user.input"]

        trace_summaries.append(summary)
    return sorted(trace_summaries, key=lambda t: t.started_at)


def get_summaries_for_conversation_measurements(
    conv_measurements: ConversationMeasurements,
):
    summaries = get_summaries_for_traces([t.trace for t in conv_measurements.traces])
    for summary in summaries:
        summary.measurements_per_trace = {
            (m.trace.trace_id, m.trace.span_id): m.measurements[:]
            for m in conv_measurements.traces
        }
    return summaries


def get_markdown_for_subagent_error(tool_trace: Trace):
    attributes = dict(tool_trace.attributes)
    tool_error = attributes.pop("ai.tool.error")
    tool_error_traceback = attributes.pop("ai.tool.error.traceback", None)
    res = textwrap.dedent(
        """
        ##### Error

        ~~~
        {tool_error_text}
        ~~~
        """
    ).format(tool_error_text=tool_error_traceback or str(tool_error))
    return EscapeHtml.escape_html_except_code(res, code_fence_style="tilde")


def get_markdown_for_tool_invocation(tool_trace: Trace):
    attributes = dict(tool_trace.attributes)
    tool_input = attributes.pop("ai.tool.input")
    tool_output = attributes.pop("ai.tool.output", None)
    tool_error = attributes.pop("ai.tool.error", None)
    tool_error_traceback = attributes.pop("ai.tool.error.traceback", None)
    res = (
        textwrap.dedent(
            """
            ##### Input

            ~~~json
            {tool_input_json}
            ~~~
            """
        )
        .lstrip()
        .format(
            tool_input_json=json.dumps(tool_input, indent=2, cls=DefaultJsonEncoder)
        )
    )
    if tool_output:
        res += textwrap.dedent(
            """
            ##### Output

            """
        ).lstrip()
        if isinstance(tool_output, str | float | int | bool):
            res += textwrap.dedent(
                """
                ~~~
                {tool_output_txt}
                ~~~
                """
            ).format(tool_output_txt=tool_output)
        else:
            res += textwrap.dedent(
                """
                ~~~json
                {tool_output_json}
                ~~~
                """
            ).format(
                tool_output_json=json.dumps(
                    tool_output, indent=2, cls=DefaultJsonEncoder
                )
            )
    if tool_error or tool_error_traceback:
        res += textwrap.dedent(
            """
            ##### Error

            ~~~
            {tool_error_text}
            ~~~
            """
        ).format(tool_error_text=tool_error_traceback or str(tool_error))
    rest_attributes = without(
        attributes,
        ["ai.conversation.id", "ai.trace.type", "ai.auth.context", "peer.service"],
    )
    if rest_attributes:
        res += textwrap.dedent(
            """
            ##### Other attributes

            ~~~json
            {rest_attributes_json}
            ~~~
            """
        ).format(
            rest_attributes_json=json.dumps(
                rest_attributes, indent=2, cls=DefaultJsonEncoder
            )
        )
    return EscapeHtml.escape_html_except_code(res, code_fence_style="tilde")


def get_markdown_for_llm_invocation(llm_trace: Trace):
    attributes = dict(llm_trace.attributes)
    messages = attributes.pop("ai.llm.request.messages")
    model_id = attributes.pop("ai.llm.request.model.id")
    system_prompt = attributes.pop("ai.llm.request.system", None)
    tool_config = attributes.pop("ai.llm.request.tool.config", None)
    inference_config = attributes.pop("ai.llm.request.inference.config", None)
    output = attributes.pop("ai.llm.response.output", None)
    error = attributes.pop("ai.llm.response.error", None)
    res = ""
    if error:
        res += textwrap.dedent(
            """
            **Error**
            {error}
            """
        ).format(
            error=error,
        )

    res += textwrap.dedent(
        """
        **Inference Config**
        {inference_config}

        **Model ID**
        {model_id}

        **System Prompt**
        {system_prompt}

        **Tool Config**
        {tool_config}

        **Messages**
        {messages}
        """
    ).format(
        inference_config=inference_config,
        model_id=model_id,
        system_prompt=system_prompt,
        tool_config=tool_config,
        messages=messages,
    )
    if output:
        stop_reason = attributes.pop("ai.llm.response.stop.reason", None)
        usage = attributes.pop("ai.llm.response.usage", None)
        metrics = attributes.pop("ai.llm.response.metrics", None)
        res += textwrap.dedent(
            """
            **Output**
            {output}

            **Stop Reason**
            {stop_reason}

            **Usage**
            {usage}

            **Metrics**
            {metrics}
            """
        ).format(
            output=output,
            stop_reason=stop_reason,
            usage=usage,
            metrics=metrics,
        )

    rest_attributes = without(
        attributes,
        ["ai.conversation.id", "ai.trace.type", "ai.auth.context", "peer.service"],
    )
    if rest_attributes:
        res += textwrap.dedent(
            """
            **Attributes**
            {rest_attributes_json}
            """
        ).format(
            rest_attributes_json=json.dumps(rest_attributes, cls=DefaultJsonEncoder)
        )
    return EscapeHtml.escape_html_except_code(res, code_fence_style="tilde")


def without(d: Mapping, keys: Sequence[str]):
    return {k: v for k, v in d.items() if k not in keys}


def get_markdown_generic(trace: Trace):
    res = textwrap.dedent(
        """
        **Trace type**
        {ai_trace_type}

        **Span kind**
        {trace_span_kind}

        **Attributes**
        {trace_attributes}
        """
    ).format(
        ai_trace_type=trace.attributes.get("ai.trace.type"),
        trace_span_kind=trace.span_kind,
        trace_attributes=json.dumps(
            without(
                trace.attributes,
                [
                    "ai.conversation.id",
                    "ai.trace.type",
                    "ai.auth.context",
                    "peer.service",
                ],
            ),
            cls=DefaultJsonEncoder,
        ),
    )
    return EscapeHtml.escape_html_except_code(res, code_fence_style="tilde")


def get_markdown_for_measurement(measurement: Measurement):
    res = textwrap.dedent(
        """
        **{measurement_name}**
        {measurement_value}
        """
    ).format(
        measurement_name=measurement.name,
        measurement_value=f"{measurement.value}{f" ({measurement.unit})" if measurement.unit is not Unit.None_ else ""}",
    )
    if measurement.additional_info:
        res += textwrap.dedent(
            """
            **Additional Info**
            {additional_info}
            """
        ).format(
            additional_info=json.dumps(
                measurement.additional_info, cls=DefaultJsonEncoder
            )
        )
    if measurement.dimensions:
        res += textwrap.dedent(
            """
            **Dimensions**
            {dimensions}
            """
        ).format(dimensions=json.dumps(measurement.dimensions, cls=DefaultJsonEncoder))

    return EscapeHtml.escape_html_except_code(res, code_fence_style="tilde")


def repr_value(v):
    if isinstance(v, str) and (v.startswith("https://") or v.startswith("http://")):
        return f"<a href={v} target='_blank' rel='noopener noreferrer'>{v}</a>"
    else:
        return repr(v)


def get_metadata(trace: Trace):
    # Populate Metadata
    metadata: MetadataDict = {
        "title": trace.attributes.get("peer.service", trace.span_name),
        "id": trace.span_id,
        "status": "done",  # Else message will show expanded
    }
    if trace.ended_at:
        metadata["duration"] = trace.duration_ms / 1000
    if "exception.message" in trace.attributes:
        metadata.pop("status", None)
    if "ai.agent.hierarchy.parent.span.id" in trace.attributes:
        metadata["parent_id"] = trace.attributes["ai.agent.hierarchy.parent.span.id"]
    return metadata


def chat_messages_from_trace_summary(
    summary: TraceSummary,
    *,
    include_traces: Literal["ALL", "CORE", "CONVERSATION_ONLY"] = "CORE",
    include_measurements=False,
):
    chat_messages: list[gr.ChatMessage] = []
    summary_duration: MetadataDict = (
        {"duration": summary.duration_ms / 1000}
        if summary.duration_ms is not None
        else {}
    )
    if summary.user_input:
        chat_messages.append(
            gr.ChatMessage(
                role="user",
                content=EscapeHtml.escape_html_except_code(
                    summary.user_input, code_fence_style="backtick"
                ),
                metadata={"title": "User", **summary_duration},
            ),
        )
    subagent_errors: list[Trace] = []
    if include_traces != "CONVERSATION_ONLY":
        for trace in summary.all_traces:
            metadata = get_metadata(trace)

            ####
            # Chat Messages
            ####

            # Subagent input:
            if (
                trace.attributes.get("ai.trace.type") in {"converse", "converse-stream"}
                and "ai.agent.hierarchy.parent.span.id" in trace.attributes
                and "ai.user.input" in trace.attributes
            ):
                metadata["title"] = "Input"
                metadata.pop("status", None)
                chat_messages.append(
                    gr.ChatMessage(
                        role="assistant",
                        content=trace.attributes["ai.user.input"],
                        metadata=metadata,
                    )
                )

            # Tool invocations
            elif trace.attributes.get("ai.trace.type") == "tool-invocation":
                if "ai.tool.error" in trace.attributes:
                    metadata.pop("status", None)
                if "ai.tool.subagent.subcontext.id" in trace.attributes:
                    metadata["title"] = (
                        f"subagent:{trace.attributes['ai.tool.name']}[subcontext={trace.attributes["ai.tool.subagent.subcontext.id"]}]"
                    )
                    if not trace.ended_at:
                        metadata["status"] = "pending"
                else:
                    tool_input_str = (
                        " ".join(
                            f"{k}={repr_value(v)}"
                            for k, v in trace.attributes.get(
                                "ai.tool.input", {}
                            ).items()
                        )
                        if trace.ended_at
                        else trace.attributes.get("ai.tool.input", "")
                    )
                    if len(tool_input_str) > 300:
                        tool_input_str = tool_input_str[:297] + "..."
                    metadata["title"] += f" [{tool_input_str}]"
                chat_messages.append(
                    gr.ChatMessage(
                        role="assistant",
                        content=(
                            get_markdown_for_tool_invocation(trace)
                            if "ai.tool.subagent.subcontext.id" not in trace.attributes
                            else ""  # subagent messages show inline nested (through metadata parent_id)
                        ),
                        metadata=metadata,
                    )
                )
                if (
                    "ai.tool.subagent.subcontext.id" in trace.attributes
                    and "ai.tool.error" in trace.attributes
                ):
                    subagent_errors.append(trace)

            # LLM invocations
            elif trace.attributes.get("ai.trace.type") == "llm-invocation":
                if "ai.llm.response.stream.events" in trace.attributes:
                    nr_stream_events = trace.attributes["ai.llm.response.stream.events"]
                    title_texts = [f"{nr_stream_events}"]
                    if (
                        "ai.llm.response.output" in trace.attributes
                        and not trace.ended_at
                    ):
                        llm_response = trace.attributes["ai.llm.response.output"]
                        content_blocks = llm_response.get("message", {}).get("content")
                        last_content_block = (
                            list(content_blocks)[-1] if content_blocks else None
                        )
                        if last_content_block:
                            if "toolUse" in last_content_block:
                                title_texts.append(
                                    f"tool:{last_content_block["toolUse"]["name"]}"
                                )
                            else:
                                title_texts.append(
                                    next(iter(last_content_block.keys()))
                                )
                    metadata["title"] += f"[{':'.join(title_texts)}]"
                if "ai.llm.response.error" in trace.attributes:
                    # Fold open
                    metadata.pop("status", None)
                chat_messages.append(
                    gr.ChatMessage(
                        role="assistant",
                        content=get_markdown_for_llm_invocation(trace),
                        metadata=metadata,
                    )
                )
                cycle_response = next(
                    (
                        summary.agent_cycle_traces[parent_trace.span_id].attributes.get(
                            "ai.agent.cycle.response"
                        )
                        for parent_trace in trace.parents
                        if parent_trace.span_id in summary.agent_cycle_traces
                    ),
                    None,
                )
                if cycle_response:
                    metadata = metadata.copy()
                    if not trace.ended_at:
                        metadata["status"] = "pending"
                    elif metadata.get("status") == "done":
                        # Always fold open
                        metadata.pop("status")
                    chat_messages.append(
                        gr.ChatMessage(
                            role="assistant",
                            content=EscapeHtml.escape_html_except_code(
                                cycle_response, code_fence_style="backtick"
                            ),
                            metadata={
                                **metadata,
                                "title": "Assistant",
                            },
                        )
                    )

            # Other trace types
            elif include_traces == "ALL":
                chat_messages.append(
                    gr.ChatMessage(
                        role="assistant",
                        content=get_markdown_generic(trace),
                        metadata=metadata,
                    )
                )
            else:
                continue  # skip including measurements for traces we don't show

            if not include_measurements:
                continue
            for measurement in summary.measurements_per_trace.get(
                (trace.trace_id, trace.span_id), []
            ):
                metadata: MetadataDict = {
                    "title": f"Measurement: {measurement.name}{" [NOK]" if measurement.validation_passed is False else ""}",
                    "parent_id": trace.span_id,
                }
                if measurement.validation_passed is not False:
                    metadata["status"] = "done"
                chat_messages.append(
                    gr.ChatMessage(
                        role="assistant",
                        content=get_markdown_for_measurement(measurement),
                        metadata=metadata,
                    )
                )
        for trace in subagent_errors:
            chat_messages.append(
                gr.ChatMessage(
                    role="assistant",
                    content=get_markdown_for_subagent_error(trace),
                    metadata={
                        "title": "subagent:exception",
                        "parent_id": trace.span_id,
                    },
                )
            )
    else:
        for trace in summary.agent_cycle_traces.values():
            if "ai.agent.hierarchy.parent.span.id" in trace.attributes:
                continue  # Skip responses from subagents
            agent_response = trace.attributes.get("ai.agent.cycle.response")
            if agent_response:
                metadata = get_metadata(trace)
                if not trace.ended_at:
                    metadata["status"] = "pending"
                elif metadata.get("status") == "done":
                    # Always fold open
                    metadata.pop("status")
                chat_messages.append(
                    gr.ChatMessage(
                        role="assistant",
                        content=EscapeHtml.escape_html_except_code(
                            agent_response, code_fence_style="backtick"
                        ),
                        metadata={
                            **metadata,
                            "title": "Assistant",
                        },
                    )
                )
    return chat_messages


@dataclass
class ChatMessages:
    conversation_id: str
    principal_id: str | None
    messages: Sequence[gr.ChatMessage]
    assistant_busy: bool


def chat_messages_from_traces(
    traces: Iterable[Trace],
    show_traces: Literal["ALL", "CORE", "CONVERSATION_ONLY"] = "CORE",
):
    traces = list(traces)
    if not traces:
        return ChatMessages("", None, [], False)
    summaries = get_summaries_for_traces(traces)
    conversations = {
        (s.conversation_id, s.auth_context["principal_id"]) for s in summaries
    }
    if len(conversations) > 1:
        raise ValueError("More than one conversation id found")
    conversation_id, principal_id = conversations.pop()
    assistant_busy = not bool(summaries and summaries[-1].ended_at)
    messages = [
        msg
        for summary in summaries
        for msg in chat_messages_from_trace_summary(
            summary,
            include_traces=show_traces,
        )
    ]
    return ChatMessages(conversation_id, principal_id, messages, assistant_busy)


def chat_messages_from_conversation_measurements(
    conv_measurements: ConversationMeasurements,
    show_traces: Literal["ALL", "CORE", "CONVERSATION_ONLY"] = "CORE",
    show_measurements=False,
):
    summaries = get_summaries_for_conversation_measurements(conv_measurements)
    if not summaries:
        return None, None, []
    conversations = {
        (s.conversation_id, s.auth_context["principal_id"]) for s in summaries
    }
    if len(conversations) > 1:
        raise ValueError("More than one conversation id found")
    conversation_id, auth_context = conversations.pop()
    messages = [
        msg
        for summary in summaries
        for msg in chat_messages_from_trace_summary(
            summary,
            include_traces=show_traces,
            include_measurements=show_measurements,
        )
    ]
    if show_measurements:
        last_summary = summaries[-1]
        for measurement in conv_measurements.measurements:
            metadata: MetadataDict = {
                "title": f"Measurement: {measurement.name}{" [NOK]" if measurement.validation_passed is False else ""}",
                "parent_id": last_summary.span_id,
            }
            if measurement.validation_passed is not False:
                metadata["status"] = "done"
            messages.append(
                gr.ChatMessage(
                    role="assistant",
                    content=get_markdown_for_measurement(measurement),
                    metadata=metadata,
                )
            )
    return conversation_id, auth_context, messages


def ensure_running_event_loop():
    """
    Work-around for https://github.com/gradio-app/gradio/issues/11280
    """
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)


class EscapeHtml:

    CODE_REGEXP_BACKTICK = re.compile(r"^```[\s\S]*?^```|`[^`]*`", re.MULTILINE)
    CODE_REGEXP_TILDE = re.compile(r"^~~~[\s\S]*?^~~~|~[^~]*~", re.MULTILINE)
    CODE_FENCE_REGEX_MAP = {
        "backtick": CODE_REGEXP_BACKTICK,
        "tilde": CODE_REGEXP_TILDE,
    }

    @classmethod
    def escape_html_except_code(
        cls,
        text: str,
        *,
        code_fence_style: Literal["backtick", "tilde"],
    ) -> str:
        """
        Escape HTML characters in the given text, except for code blocks (denoted by ```),
        and inline code snippets (denoted by `), because gradio already escapes those.
        """
        result = []
        last_end = 0

        for m in cls.CODE_FENCE_REGEX_MAP[code_fence_style].finditer(text):
            result.append(html.escape(text[last_end : m.start()]))
            result.append(m.group(0))
            last_end = m.end()
        result.append(html.escape(text[last_end:]))
        return "".join(result)


def format_date(dt: datetime):
    now = datetime.now(UTC)
    diff_days = (now.date() - dt.date()).days

    # Show in local timezone
    dt = dt.astimezone()

    if diff_days >= 7:
        return dt.strftime("%B %d, %Y")  # "September 4, 2025"

    day_text = (
        "Today"
        if diff_days == 0
        else "Yesterday" if diff_days == 1 else dt.strftime("%A")
    )  # "Today" / "Yesterday" / "Monday"

    return f"{day_text} at {dt.strftime("%X")}"


def find_nearest_folded_open_message(messages: Sequence[gr.ChatMessage]):
    search_from = 0
    message = messages[-1]
    while message:
        message_parent_id = message.metadata.get("parent_id")
        if message.metadata.get("status") != "done":  # Folded open!
            return message.metadata.get("id")
        elif message_parent_id:
            offset, message = next(
                (
                    enumerate(
                        msg
                        for msg in reversed(messages[: len(messages) - search_from])
                        if msg.metadata.get("id") == message_parent_id
                    )
                ),
                (-1, None),
            )
            search_from += offset
            continue
        return
