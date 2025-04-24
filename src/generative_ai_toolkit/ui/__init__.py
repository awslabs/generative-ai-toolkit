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

import datetime
import functools
import html
import json
import textwrap
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from itertools import groupby

import gradio as gr
from gradio.components.chatbot import MetadataDict

from generative_ai_toolkit.evaluate.evaluate import ConversationMeasurements
from generative_ai_toolkit.metrics.measurement import Measurement, Unit
from generative_ai_toolkit.tracer.trace import Trace


@dataclass
class TraceSummary:
    trace_id: str
    span_id: str
    started_at: datetime.datetime
    duration_ms: int
    conversation_id: str
    auth_context: str | None = None
    user_input: str = ""
    agent_response: str = ""
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
        root_trace = traces_for_trace_id[0]
        assert root_trace.parent_span is None
        summary = TraceSummary(
            conversation_id=root_trace.attributes["ai.conversation.id"],
            auth_context=root_trace.attributes.get("ai.auth.context"),
            trace_id=trace_id,
            span_id=root_trace.span_id,
            duration_ms=root_trace.duration_ms,
            started_at=root_trace.started_at,
            all_traces=traces_for_trace_id,
        )

        # Find (first) user input:
        for trace in traces_for_trace_id:
            if not summary.user_input and "ai.user.input" in trace.attributes:
                summary.user_input = trace.attributes["ai.user.input"]

        # Find (last) agent response:
        for trace in reversed(traces_for_trace_id):
            if not summary.agent_response and "ai.agent.response" in trace.attributes:
                summary.agent_response = trace.attributes["ai.agent.response"]

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
    res = textwrap.dedent(
        f"""
        **Input**
        {tool_input}
        """
    )
    if tool_output:
        res += textwrap.dedent(
            f"""
            **Output**
            {tool_output}
            """
        )
    if tool_error:
        res += textwrap.dedent(
            f"""
            **Error**
            {tool_error}
            """
        )
    rest_attributes = without(
        attributes,
        ["ai.conversation.id", "ai.trace.type", "ai.auth.context", "peer.service"],
    )
    if rest_attributes:
        res += textwrap.dedent(
            f"""
            **Attributes**
            {json.dumps(rest_attributes)}
            """
        )
    return html.escape(res)


def get_markdown_for_llm_invocation(llm_trace: Trace):
    attributes = dict(llm_trace.attributes)
    messages = attributes.pop("ai.llm.request.messages")
    model_id = attributes.pop("ai.llm.request.model.id")
    system_prompt = attributes.pop("ai.llm.request.system", None)
    tool_config = attributes.pop("ai.llm.request.tool.config", None)
    inference_config = attributes.pop("ai.llm.request.inference.config", None)
    output = attributes.pop("ai.llm.response.output", None)
    res = textwrap.dedent(
        f"""
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
    )
    if output:
        stop_reason = attributes.pop("ai.llm.response.stop.reason", None)
        usage = attributes.pop("ai.llm.response.usage", None)
        metrics = attributes.pop("ai.llm.response.metrics", None)
        res += textwrap.dedent(
            f"""
            **Output**
            {output}

            **Stop Reason**
            {stop_reason}

            **Usage**
            {usage}

            **Metrics**
            {metrics}
            """
        )

    error = attributes.pop("ai.llm.response.error", None)
    if error:
        res += textwrap.dedent(
            f"""
            **Error**
            {error}
            """
        )
    rest_attributes = without(
        attributes,
        ["ai.conversation.id", "ai.trace.type", "ai.auth.context", "peer.service"],
    )
    if rest_attributes:
        res += textwrap.dedent(
            f"""
            **Attributes**
            {json.dumps(rest_attributes)}
            """
        )
    return html.escape(res)


def without(d: Mapping, keys: Sequence[str]):
    return {k: v for k, v in d.items() if k not in keys}


def get_markdown_generic(trace: Trace):
    res = textwrap.dedent(
        f"""
        **Trace type**
        {trace.attributes.get("ai.trace.type")}

        **Span kind**
        {trace.span_kind}

        **Attributes**
        {json.dumps(without(trace.attributes, ["ai.conversation.id", "ai.trace.type", "ai.auth.context", "peer.service"]))}
        """
    )
    return html.escape(res)


def get_markdown_for_measurement(measurement: Measurement):
    res = textwrap.dedent(
        f"""
        **{measurement.name}**
        {measurement.value}{f" ({measurement.unit})" if measurement.unit is not Unit.None_ else ""}
        """
    )
    if measurement.additional_info:
        res += textwrap.dedent(
            f"""
            **Additional Info**
            {json.dumps(measurement.additional_info)}
            """
        )
    if measurement.dimensions:
        res += textwrap.dedent(
            f"""
            **Dimensions**
            {json.dumps(measurement.dimensions)}
            """
        )

    return html.escape(res)


def chat_messages_from_trace_summary(
    summary: TraceSummary,
    *,
    include_all_traces=False,
    include_measurements=False,
):
    chat_messages: list[gr.ChatMessage] = []
    chat_messages.append(
        gr.ChatMessage(
            role="user",
            content=summary.user_input,
            metadata={"title": "User", "duration": summary.duration_ms / 1000},
        ),
    )
    for trace in summary.all_traces:
        metadata: MetadataDict = {
            "title": trace.attributes.get("peer.service", trace.span_name),
            "duration": trace.duration_ms / 1000,
            "id": trace.span_id,
            "status": "done",
        }
        if "exception.message" in trace.attributes:
            metadata.pop("status", None)
        if trace.attributes.get("ai.trace.type") == "tool-invocation":
            if "ai.tool.error" in trace.attributes:
                metadata.pop("status", None)
            chat_messages.append(
                gr.ChatMessage(
                    role="assistant",
                    content=get_markdown_for_tool_invocation(trace),
                    metadata=metadata,
                )
            )
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
        elif include_all_traces:
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
    chat_messages.append(
        gr.ChatMessage(
            role="assistant",
            content=summary.agent_response,
            metadata={
                "title": "Assistant",
                "duration": summary.duration_ms / 1000,
                "id": summary.span_id,
            },
        )
    )
    return chat_messages


def chat_messages_from_traces(
    traces: Iterable[Trace],
    show_all_traces=False,
):
    traces = list(traces)
    if not traces:
        return None, None, []
    summaries = get_summaries_for_traces(traces)
    conversations = {(s.conversation_id, s.auth_context) for s in summaries}
    if len(conversations) > 1:
        raise ValueError("More than one conversation id found")
    conversation_id, auth_context = conversations.pop()
    messages = [
        msg
        for summary in summaries
        for msg in chat_messages_from_trace_summary(
            summary,
            include_all_traces=show_all_traces,
        )
    ]
    return conversation_id, auth_context, messages


def chat_messages_from_conversation_measurements(
    conv_measurements: ConversationMeasurements,
    show_all_traces=False,
    show_measurements=False,
):
    summaries = get_summaries_for_conversation_measurements(conv_measurements)
    if not summaries:
        return None, None, []
    conversations = {(s.conversation_id, s.auth_context) for s in summaries}
    if len(conversations) > 1:
        raise ValueError("More than one conversation id found")
    conversation_id, auth_context = conversations.pop()
    messages = [
        msg
        for summary in summaries
        for msg in chat_messages_from_trace_summary(
            summary,
            include_all_traces=show_all_traces,
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

    with gr.Blocks(fill_width=True, title="Generative AI Toolkit") as demo:
        with gr.Row():
            gr.Markdown(
                f"**Conversation ID**: {conversation_id} | **Auth context**: {auth_context}"
            )
            toggle_all_traces = gr.Button("Show all traces", scale=0, min_width=200)
        chatbot = gr.Chatbot(
            type="messages",
            height="full",
            label="Conversation",
            value=messages,  # type: ignore
        )

        def do_toggle_all_traces(state):
            new_state = not state
            new_label = "Hide internal traces" if new_state else "Show all traces"
            *_, messages = chat_messages_from_traces(
                traces,
                show_all_traces=new_state,
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
        show_all_traces=False,
        show_measurements=False,
    ):
        conv_measurements = all_measurements[conversation_index]

        conversation_id, auth_context, messages = (
            chat_messages_from_conversation_measurements(
                conv_measurements,
                show_all_traces=show_all_traces,
                show_measurements=show_measurements,
            )
        )
        header_text = (
            f"**Conversation ID**: {conversation_id} | **Auth context**: {auth_context}"
        )
        return header_text, messages, gr.update(visible=False), gr.update(visible=True)

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

    with gr.Blocks(css=css, fill_width=True, title="Generative AI Toolkit") as demo:
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
                header = gr.Markdown("Conversation:")
                toggle_all_traces = gr.Button("Show all traces", scale=0, min_width=200)
                toggle_measurements = gr.Button(
                    "Hide measurements", scale=0, min_width=200
                )
                back_button = gr.Button("Back", scale=0, min_width=200)

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
                outputs=[header, chatbot, parent_page, child_page],
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
            outputs=[header, chatbot, parent_page, child_page],
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
            outputs=[header, chatbot, parent_page, child_page],
            queue=False,
        )

        back_button.click(fn=go_back, inputs=[], outputs=[parent_page, child_page])

    return demo
