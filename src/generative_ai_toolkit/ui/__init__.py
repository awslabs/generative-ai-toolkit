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
import datetime
import functools
import html
import json
import re
import textwrap
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from itertools import groupby
from threading import Event
from typing import Literal

import gradio as gr
from gradio.components.chatbot import MetadataDict

from generative_ai_toolkit.agent import Agent
from generative_ai_toolkit.context import AuthContext
from generative_ai_toolkit.evaluate.evaluate import ConversationMeasurements
from generative_ai_toolkit.metrics.measurement import Measurement, Unit
from generative_ai_toolkit.tracer.trace import Trace
from generative_ai_toolkit.utils.json import DefaultJsonEncoder


def chat_ui(
    agent: Agent,
    show_traces_drop_down=True,
    show_traces: Literal["ALL", "CORE", "CONVERSATION_ONLY"] = "CORE",
):

    ensure_running_event_loop()

    def user_submit(user_input: str):
        return (
            gr.update(
                value="", interactive=False, submit_btn=False, stop_btn=True
            ),  # clear textbox
            user_input,
            Event(),
        )

    def user_stop(stop_event: Event | None):
        if stop_event:
            stop_event.set()
        return gr.update(stop_btn=False)

    def assistant_stream(user_input: str, stop_event: Event | None):
        if not user_input:
            yield agent.traces
            return
        traces: dict[str, Trace] = {trace.span_id: trace for trace in agent.traces}
        for trace in agent.converse_stream(
            user_input, stream="traces", stop_event=stop_event
        ):
            traces[trace.span_id] = trace
            yield list(traces.values())

    def traces_state_change(
        traces: Iterable[Trace],
        show_traces: Literal["ALL", "CORE", "CONVERSATION_ONLY"] = "CORE",
    ):
        *_, messages = chat_messages_from_traces(traces, show_traces=show_traces)
        return messages

    def reset_agent():
        agent.reset()
        return (
            gr.update(value=[], label=f"Conversation {agent.conversation_id}"),
            [],
        )

    with gr.Blocks(
        theme="origin", fill_width=True, title="Generative AI Toolkit"
    ) as demo:

        show_traces_state = gr.State(value=show_traces)
        traces_state = gr.State(value=[])
        stop_event = gr.State(value=None)
        last_user_input = gr.State("")

        with gr.Row(visible=show_traces_drop_down):
            gr.Markdown("")  # functions as spacer
            trace_visibility_drop_down = gr.Dropdown(
                choices=[
                    ("Show conversation only", "CONVERSATION_ONLY"),
                    ("Show core traces", "CORE"),
                    ("Show all traces", "ALL"),
                ],
                value=show_traces,
                label="Show traces",
                filterable=False,
                container=False,
                scale=0,
                min_width=250,
            )

        chatbot = gr.Chatbot(
            type="messages",
            height="75vh" if show_traces_drop_down else "80vh",
            label=f"Conversation {agent.conversation_id}",
        )

        trace_visibility_drop_down.select(
            lambda show_traces: show_traces,
            inputs=[trace_visibility_drop_down],
            outputs=[show_traces_state],
        )

        show_traces_state.change(
            traces_state_change,
            inputs=[traces_state, show_traces_state],
            outputs=[chatbot],
            show_progress="hidden",
            show_progress_on=[],
        )

        msg = gr.Textbox(
            placeholder="Type your message ...",
            submit_btn=True,
            autofocus=True,
            show_label=False,
            elem_id="user-input",
        )

        msg.submit(
            user_submit,
            inputs=[msg],
            outputs=[msg, last_user_input, stop_event],
        ).then(
            assistant_stream,
            inputs=[last_user_input, stop_event],
            outputs=[traces_state],
            show_progress="full",
            show_progress_on=[chatbot],
        ).then(
            lambda: gr.update(interactive=True, submit_btn=True, stop_btn=False),
            outputs=[msg],
        )

        msg.stop(user_stop, inputs=[stop_event], outputs=[msg])

        traces_state.change(
            traces_state_change,
            inputs=[traces_state, show_traces_state],
            outputs=[chatbot],
            show_progress="hidden",
            show_progress_on=[],
            queue=True,
        )

        chatbot.clear(reset_agent, outputs=[chatbot, traces_state])

        demo.load(lambda: agent.traces, outputs=[traces_state])

        return demo


@dataclass
class TraceSummary:
    trace_id: str
    span_id: str
    started_at: datetime.datetime
    duration_ms: int | None
    conversation_id: str
    auth_context: AuthContext = field(default_factory=lambda: {"principal_id": None})
    user_input: str = ""
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
    if tool_error_traceback:
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
        {"duration": summary.duration_ms / 1000} if summary.duration_ms else {}
    )
    chat_messages.append(
        gr.ChatMessage(
            role="user",
            content=EscapeHtml.escape_html_except_code(
                summary.user_input, code_fence_style="backtick"
            ),
            metadata={"title": "User", **summary_duration},
        ),
    )
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
                if "ai.tool.subagent" in trace.attributes:
                    metadata["title"] = f"subagent:{trace.attributes['ai.tool.name']}"
                    if not trace.ended_at:
                        metadata["status"] = "pending"
                else:
                    tool_input_str = " ".join(
                        f"{k}={repr_value(v)}"
                        for k, v in trace.attributes.get("ai.tool.input", {}).items()
                    )
                    if len(tool_input_str) > 300:
                        tool_input_str = tool_input_str[:297] + "..."
                    metadata["title"] += f" [{tool_input_str}]"
                chat_messages.append(
                    gr.ChatMessage(
                        role="assistant",
                        content=(
                            get_markdown_for_tool_invocation(trace)
                            if "ai.tool.subagent" not in trace.attributes
                            else ""
                        ),
                        metadata=metadata,
                    )
                )

            # LLM invocations
            elif trace.attributes.get("ai.trace.type") == "llm-invocation":
                if "ai.llm.response.error" in trace.attributes:
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
                    metadata.pop("status", None)
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
    else:
        for trace in summary.agent_cycle_traces.values():
            if "ai.agent.hierarchy.parent.span.id" in trace.attributes:
                continue  # Skip responses from subagents
            agent_response = trace.attributes.get("ai.agent.cycle.response")
            if agent_response:
                metadata = get_metadata(trace)
                metadata.pop("status", None)
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


def chat_messages_from_traces(
    traces: Iterable[Trace],
    show_traces: Literal["ALL", "CORE", "CONVERSATION_ONLY"] = "CORE",
):
    traces = list(traces)
    if not traces:
        return None, None, []
    summaries = get_summaries_for_traces(traces)
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
        )
    ]
    return conversation_id, auth_context, messages


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


def traces_ui(
    traces: Iterable[Trace],
):
    conversation_id, auth_context, messages = chat_messages_from_traces(
        traces,
    )

    ensure_running_event_loop()

    with gr.Blocks(
        theme="origin", fill_width=True, title="Generative AI Toolkit"
    ) as demo:
        with gr.Row():
            gr.Markdown("")  # functions as spacer
            toggle_all_traces = gr.Button(
                "Show all traces", scale=0, min_width=200, size="sm"
            )
        chatbot = gr.Chatbot(
            type="messages",
            height="full",
            label=f"Conversation {conversation_id}",
            value=messages,  # type: ignore
        )

        def do_toggle_all_traces(state):
            new_state = not state
            new_label = "Hide internal traces" if new_state else "Show all traces"
            *_, messages = chat_messages_from_traces(
                traces,
                show_traces="ALL" if new_state else "CORE",
            )
            return gr.update(value=new_label), new_state, messages

        show_all_traces_toggle_state = gr.State(value=False)

        toggle_all_traces.click(
            fn=do_toggle_all_traces,
            inputs=[show_all_traces_toggle_state],
            outputs=[toggle_all_traces, show_all_traces_toggle_state, chatbot],
        )
    return demo


def measurements_ui(
    measurements: Iterable[ConversationMeasurements],
):
    def measurements_sort_key(m: ConversationMeasurements):
        return m.case_nr, m.permutation_nr, m.run_nr, m.traces[0].trace.started_at

    all_measurements = sorted(measurements, key=measurements_sort_key)

    def show_conversation(
        conversation_index: int,
        show_all_traces: bool,
        show_measurements: bool,
    ):
        conv_measurements = all_measurements[conversation_index]

        conversation_id, auth_context, messages = (
            chat_messages_from_conversation_measurements(
                conv_measurements,
                show_traces="ALL" if show_all_traces else "CORE",
                show_measurements=show_measurements,
            )
        )
        return (
            gr.update(value=messages, label=f"Conversation {conversation_id}"),
            gr.update(visible=False),
            gr.update(visible=True),
        )

    def go_back():
        return gr.update(visible=True), gr.update(visible=False)

    css = """
    :root {
        --block-border-width: 0;
    }

    .genaitk-header textarea {
        font-weight: bold;
    }

    .genaitk-nowrap-row {
        flex-wrap: nowrap;
    }

    .genaitk-scroll-column {
        overflow-x: auto;
    }

    .genaitk-validation-ok textarea {
        background-color: lightgreen;
        text-align: center;
        border-radius: 20px;
    }

    .genaitk-validation-nok textarea {
        background-color: red;
        text-align: center;
        border-radius: 20px;
    }
    """

    ensure_running_event_loop()

    with gr.Blocks(
        theme="origin", css=css, fill_width=True, title="Generative AI Toolkit"
    ) as demo:
        with gr.Column(
            visible=True, elem_classes="genaitk-scroll-column"
        ) as parent_page:
            gr.Markdown("## Measurements Overview")
            with gr.Row(elem_classes="genaitk-header genaitk-nowrap-row"):
                gr.Textbox(
                    "Conversation ID",
                    scale=10,
                    show_label=False,
                    interactive=False,
                    container=False,
                )
                gr.Textbox(
                    "Case Name",
                    scale=10,
                    show_label=False,
                    interactive=False,
                    container=False,
                )
                gr.Textbox(
                    "Case Nr",
                    scale=4,
                    show_label=False,
                    interactive=False,
                    container=False,
                    min_width=80,
                )
                gr.Textbox(
                    "Permutation Nr",
                    scale=4,
                    show_label=False,
                    interactive=False,
                    container=False,
                    min_width=120,
                )
                gr.Textbox(
                    "Run Nr",
                    scale=4,
                    show_label=False,
                    interactive=False,
                    container=False,
                    min_width=80,
                )
                gr.Textbox(
                    "Duration",
                    scale=6,
                    show_label=False,
                    interactive=False,
                    container=False,
                    min_width=100,
                )
                gr.Textbox(
                    "Nr Traces",
                    scale=4,
                    show_label=False,
                    interactive=False,
                    container=False,
                    min_width=80,
                )
                gr.Textbox(
                    "Nr Measurements",
                    scale=4,
                    show_label=False,
                    interactive=False,
                    container=False,
                    min_width=120,
                )
                gr.Textbox(
                    "Validation",
                    scale=4,
                    show_label=False,
                    interactive=False,
                    container=False,
                    min_width=90,
                )
                gr.Textbox(
                    "Action",
                    scale=4,
                    show_label=False,
                    interactive=False,
                    container=False,
                )
            conversation_buttons: list[tuple[gr.Button, int]] = []

            for index, conv_measurements in enumerate(all_measurements):
                case = conv_measurements.case
                case_name = case.name if case else "-"
                case_nr = (
                    str(conv_measurements.case_nr + 1)
                    if conv_measurements.case_nr is not None
                    else "-"
                )
                permutation_nr = (
                    str(conv_measurements.permutation_nr + 1)
                    if conv_measurements.permutation_nr is not None
                    else "-"
                )
                run_nr = (
                    str(conv_measurements.run_nr + 1)
                    if conv_measurements.run_nr is not None
                    else "-"
                )
                first_trace = conv_measurements.traces[0].trace
                last_trace = conv_measurements.traces[-1].trace
                validation_ok = all(
                    m.validation_passed is not False
                    for m in conv_measurements.measurements
                ) and all(
                    m.validation_passed is not False
                    for t in conv_measurements.traces
                    for m in t.measurements
                )
                nr_measurements = len(conv_measurements.measurements) + sum(
                    len(t.measurements) for t in conv_measurements.traces
                )

                with gr.Row(elem_classes="genaitk-nowrap-row"):
                    gr.Textbox(
                        conv_measurements.conversation_id,
                        scale=10,
                        show_label=False,
                        interactive=False,
                        container=False,
                    )
                    gr.Textbox(
                        case_name,
                        scale=10,
                        show_label=False,
                        interactive=False,
                        container=False,
                    )
                    gr.Textbox(
                        case_nr,
                        scale=4,
                        show_label=False,
                        interactive=False,
                        container=False,
                        min_width=80,
                    )
                    gr.Textbox(
                        permutation_nr,
                        scale=4,
                        show_label=False,
                        interactive=False,
                        container=False,
                        min_width=120,
                    )
                    gr.Textbox(
                        run_nr,
                        scale=4,
                        show_label=False,
                        interactive=False,
                        container=False,
                        min_width=80,
                    )
                    gr.Textbox(
                        str(last_trace.started_at - first_trace.started_at)[:-3],
                        scale=6,
                        show_label=False,
                        interactive=False,
                        container=False,
                        min_width=100,
                    )
                    gr.Textbox(
                        str(len(conv_measurements.traces)),
                        scale=4,
                        show_label=False,
                        interactive=False,
                        container=False,
                        min_width=80,
                    )
                    gr.Textbox(
                        str(nr_measurements),
                        scale=4,
                        show_label=False,
                        interactive=False,
                        container=False,
                        min_width=120,
                    )
                    gr.Textbox(
                        ("OK" if validation_ok else "NOK"),
                        scale=4,
                        show_label=False,
                        interactive=False,
                        container=False,
                        min_width=90,
                        elem_classes=(
                            "genaitk-validation-ok"
                            if validation_ok
                            else "genaitk-validation-nok"
                        ),
                    )
                    button = gr.Button("View", scale=4)
                    conversation_buttons.append((button, index))

        with gr.Column(visible=False) as child_page:
            with gr.Row():
                gr.Markdown("")  # functions as spacer
                toggle_all_traces = gr.Button(
                    "Show all traces", scale=0, min_width=200, size="sm"
                )
                toggle_measurements = gr.Button(
                    "Hide measurements", scale=0, min_width=200, size="sm"
                )
                back_button = gr.Button("Back", scale=0, min_width=200, size="sm")

            chatbot = gr.Chatbot(
                type="messages",
                height="full",
                label="Conversation",
            )

        current_conversation_index = gr.State()
        show_all_traces_toggle_state = gr.State(value=False)
        show_measurements_toggle_state = gr.State(value=True)

        for btn, index in conversation_buttons:
            btn.click(
                fn=functools.partial(lambda i: i, index),
                inputs=[],
                outputs=[current_conversation_index],
            ).then(
                fn=show_conversation,
                inputs=[
                    current_conversation_index,
                    show_all_traces_toggle_state,
                    show_measurements_toggle_state,
                ],
                outputs=[chatbot, parent_page, child_page],
                queue=False,
            )

        def do_toggle_all_traces(state):
            new_state = not state
            new_label = "Hide internal traces" if new_state else "Show all traces"
            return gr.update(value=new_label), new_state

        def do_toggle_measurements(state):
            new_state = not state
            new_label = "Hide measurements" if new_state else "Show measurements"
            return gr.update(value=new_label), new_state

        toggle_all_traces.click(
            fn=do_toggle_all_traces,
            inputs=[show_all_traces_toggle_state],
            outputs=[toggle_all_traces, show_all_traces_toggle_state],
        ).then(
            fn=show_conversation,
            inputs=[
                current_conversation_index,
                show_all_traces_toggle_state,
                show_measurements_toggle_state,
            ],
            outputs=[chatbot, parent_page, child_page],
            queue=False,
        )

        toggle_measurements.click(
            fn=do_toggle_measurements,
            inputs=[show_measurements_toggle_state],
            outputs=[toggle_measurements, show_measurements_toggle_state],
        ).then(
            fn=show_conversation,
            inputs=[
                current_conversation_index,
                show_all_traces_toggle_state,
                show_measurements_toggle_state,
            ],
            outputs=[chatbot, parent_page, child_page],
            queue=False,
        )

        back_button.click(fn=go_back, inputs=[], outputs=[parent_page, child_page])

    return demo


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
