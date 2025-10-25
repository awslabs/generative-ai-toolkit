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

import functools
import json
import signal
import textwrap
import time
from collections.abc import Callable, Iterable, Sequence
from threading import Event, Lock
from typing import TYPE_CHECKING, Any, Literal, cast

if TYPE_CHECKING:
    from mypy_boto3_bedrock_runtime.type_defs import MessageUnionTypeDef

import gradio as gr

from generative_ai_toolkit.agent import Agent
from generative_ai_toolkit.evaluate.evaluate import ConversationMeasurements
from generative_ai_toolkit.tracer.trace import Trace
from generative_ai_toolkit.ui.conversation_list import (
    ConversationList,
    ConversationPage,
)
from generative_ai_toolkit.ui.conversation_list.conversation_list import Conversation
from generative_ai_toolkit.ui.lib import (
    chat_messages_from_conversation_measurements,
    chat_messages_from_traces,
    ensure_running_event_loop,
    find_nearest_folded_open_message,
    format_date,
)


def chat_ui(
    agent: Agent | Callable[[], Agent],
    *,
    show_top_bar=True,
    show_traces: Literal["ALL", "CORE", "CONVERSATION_ONLY"] = "CORE",
    conversation_list: ConversationList | Callable[[], ConversationList] | None = None,
    title="Generative AI Toolkit",
):

    _stop_events: set[Event] = set()
    _stop_events_lock = Lock()

    def ensure_agent_instance(conversation_id: str | None = None):
        if isinstance(agent, Agent):
            agent_instance = agent
        else:
            agent_instance = agent()

        if conversation_id is not None:
            agent_instance.set_conversation_id(conversation_id)

        return agent_instance

    def conversation_list_instance():
        if not conversation_list:
            return None
        agent_instance = ensure_agent_instance()
        if not isinstance(conversation_list, ConversationList):
            conv_list = conversation_list()
        else:
            conv_list = conversation_list
        conv_list.set_auth_context(**agent_instance.auth_context)
        return conv_list

    ensure_running_event_loop()

    def user_submit(user_input: str | None, messages: list[gr.MessageDict]):
        if user_input:
            messages = [
                *messages,
                gr.MessageDict(
                    role="user", content=user_input, metadata={"title": "User"}
                ),
            ]

        stop_event = Event()
        with _stop_events_lock:
            _stop_events.add(stop_event)

        return (
            gr.update(
                value="", interactive=False, submit_btn=False, stop_btn=True
            ),  # clear textbox
            user_input or None,  # Prefer None over empty string
            stop_event,
            messages,
        )

    def describe_conversation(conversation_id: str, chatbot: list[gr.MessageDict]):
        conv_list = conversation_list_instance()
        if not conv_list:
            return gr.update()

        messages_to_describe = [
            {"role": msg["role"], "content": [{"text": msg["content"]}]}
            for msg in chatbot
            if msg["role"] in {"user", "assistant"}
            and msg.get("metadata", {}).get("title") in {"User", "Assistant"}
            and isinstance(msg["content"], str)
        ]

        conversation = conv_list.add_conversation(
            conversation_id,
            cast("Sequence[MessageUnionTypeDef]", messages_to_describe),
        )
        return conversation.description

    def cleanup_stop_event(stop_event: Event | None):
        """Remove stop event from tracking set after response completes"""
        if stop_event:
            with _stop_events_lock:
                _stop_events.discard(stop_event)

    def user_stop(stop_event: Event | None):
        if stop_event:
            stop_event.set()
        return gr.update(stop_btn=False)

    def assistant_stream(
        conversation_id: str,
        traces_state: list[Trace],
        user_input: str | None,
        stop_event: Event | None,
        yield_snapshot_every=0.25,
    ):
        current_agent_instance = ensure_agent_instance(conversation_id)

        last_snapshot_yielded = 0.0
        traces: dict[str, Trace] = {trace.span_id: trace for trace in traces_state}
        for trace in current_agent_instance.converse_stream(
            user_input, stream="traces", stop_event=stop_event
        ):
            trace_update = trace.span_id in traces
            traces[trace.span_id] = trace
            if (
                trace_update
                and not trace.ended_at
                and trace.attributes.get("ai.trace.type") == "llm-invocation"
            ):
                # Throttle display of snapshots for LLM invocations only.
                # LLM invocations typically generate many intermediate traces during
                # token streaming, but tool invocations and other operations benefit
                # from immediate display to show responsive agent behavior.
                now = time.monotonic()
                if last_snapshot_yielded + yield_snapshot_every > now:
                    continue
                last_snapshot_yielded = now
            yield list(traces.values())

    def traces_state_change(
        traces: Iterable[Trace],
        show_traces: Literal["ALL", "CORE", "CONVERSATION_ONLY"] = "CORE",
    ):
        chat_messages = chat_messages_from_traces(traces, show_traces=show_traces)
        messages = list(chat_messages.messages)[:]
        if (
            messages
            and chat_messages.assistant_busy
            and messages[-1].metadata.get("title") != "Assistant"
        ):
            # Gradio currently (Oct 2025) doesn't allow you to mark a chat message as pending
            # (to show the spinner in the title) without folding it open at the same time––which I don't want,
            # because folding open LLM-invocation messages will show **a lot** of data and is meant for debugging.
            # Nor does gradio make it easy to control progress indicators on our chatbot from trigger code.
            # Therefore, we'll add an empty but pending chat message with spinner to show
            # clearly that the assistant is busy.
            progress_message = gr.ChatMessage(
                role="assistant",
                content="",
                metadata={
                    "title": "Assistant",
                    "status": "pending",  # This makes the message have a spinner
                },
            )
            # Try to show the spinner message under the nearest folded open message:
            parent_id = find_nearest_folded_open_message(messages)
            if parent_id:
                progress_message.metadata["parent_id"] = parent_id
            messages.append(progress_message)
        return messages, gr.update(interactive=bool(messages))

    def reset_agent(stop_event: Event | None):
        if stop_event:
            stop_event.set()
        current_agent_instance = ensure_agent_instance()
        current_agent_instance.reset()
        return (
            gr.update(
                value=[], label=f"Conversation {current_agent_instance.conversation_id}"
            ),
            [],
            current_agent_instance.conversation_id,
            current_agent_instance.conversation_id,
            "",
        )

    def load_conversation_id(conversation_id: str) -> tuple[Sequence[Trace], str]:
        current_agent_instance = ensure_agent_instance(conversation_id)
        conv_list = conversation_list_instance()
        conversation: Conversation | None = None
        if conv_list:
            conversation = conv_list.get_conversation(conversation_id)
        return current_agent_instance.traces, (
            conversation.description if conversation else ""
        )

    def load_page(
        current_page_token: Any,
        previous_page_tokens: list[Any],
        requested_page_token: Any,
    ):
        conv_list = conversation_list_instance()
        if not conv_list:
            return [None, [], ConversationPage()]

        if requested_page_token != current_page_token:
            try:
                index = previous_page_tokens.index(requested_page_token)
            except ValueError:
                # Going forward
                previous_page_tokens = previous_page_tokens[:] + [current_page_token]
            else:
                # Going backward
                previous_page_tokens = previous_page_tokens[:index]

        return [
            requested_page_token,
            previous_page_tokens,
            conv_list.get_conversations(next_page_token=requested_page_token),
        ]

    def get_remove_conversation_list_item(
        conversation_id: str,
    ):
        def bound(
            current_page_token: Any,
            previous_page_tokens: list[Any],
            requested_page_token: Any,
        ):
            conv_list = conversation_list_instance()
            if conv_list:
                conv_list.remove_conversation(conversation_id)
            return load_page(
                current_page_token, previous_page_tokens, requested_page_token
            )

        return bound

    def set_page_size(page_size: int):
        conv_list = conversation_list_instance()
        if not conv_list:
            return ConversationPage(), None
        conv_list.set_page_size(page_size)
        return conv_list.get_conversations(), None

    js_set_conversation_id_as_query_param = textwrap.dedent(
        """
        function (conversationId) {
            if (conversationId) {
              const params=new URLSearchParams(window.location.search);
              params.set('conversation-id', conversationId);
              history.replaceState(null, '', '?' + params);
            }
        }
        """
    )

    js_set_conversation_description_as_title = textwrap.dedent(
        """
        function (conversationDescription) {{
            if (conversationDescription) {{
                document.title = conversationDescription;
            }} else {{
                document.title = {title};
            }}
        }}
        """
    ).format(title=json.dumps(title))

    with gr.Blocks(
        theme="origin",
        fill_width=True,
        title=title,
        css="""
            .centered-row {
              align-items: center !important;
            }
            """,
    ) as demo:

        show_traces_state = gr.State(value=show_traces)
        traces_state = gr.State(value=[])
        stop_event = gr.State(value=None)
        last_user_input = gr.State("")
        last_conversation_id = gr.BrowserState(
            None, storage_key="generative-ai-toolkit.ui.conversation-id"
        )
        conversation_list_current_page = gr.State(ConversationPage())
        conversation_list_previous_page_tokens = gr.State([])
        conversation_list_current_page_token = gr.State(None)
        conversation_list_requested_page_token = gr.State(None)

        # Use Textbox instead of state so JS can see the value:
        conversation_id = gr.Textbox(visible=False)
        conversation_description = gr.Textbox(visible=False)

        with gr.Column(visible=True) as chat_view:
            with gr.Row(visible=show_top_bar):
                conversation_list_btn = gr.Button(
                    "Conversation List",
                    scale=0,
                    min_width=250,
                    visible=conversation_list is not None,
                    interactive=False,
                )
                new_chat_btn = gr.Button("New Chat ↗️", scale=0, interactive=False)
                new_chat_btn.click(
                    fn=None,
                    js="() => { window.open('/?conversation-id=', '_blank') }",
                )
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
                height="75vh" if show_top_bar else "80vh",
            )

            msg = gr.Textbox(
                placeholder="Type your message ...",
                submit_btn=True,
                autofocus=True,
                show_label=False,
                elem_id="user-input",
                interactive=False,
            )

        with gr.Column(visible=False) as conversation_list_view:

            with gr.Row(elem_classes="centered-row"):
                gr.Markdown("## Your previous conversations")
                with gr.Column(scale=0, min_width=60):
                    gr.Markdown("Page size:")
                page_size_drop_down = gr.Dropdown(
                    choices=[(str(psize), psize) for psize in sorted({20, 50, 100})],
                    value=20,
                    label="Page size",
                    filterable=False,
                    container=False,
                    scale=0,
                    min_width=100,
                )

                page_size_drop_down.select(
                    set_page_size,
                    inputs=[page_size_drop_down],
                    outputs=[
                        conversation_list_current_page,
                        conversation_list_requested_page_token,
                    ],
                )

            @gr.render(
                inputs=[
                    conversation_list_current_page,
                    conversation_list_previous_page_tokens,
                    conversation_id,
                ]
            )
            def conversation_list_ui(
                page: ConversationPage,
                previous_page_tokens: list[Any],
                current_conversation_id: str,
            ):
                for conv in page.conversations:
                    with gr.Row():
                        gr.Textbox(
                            format_date(conv.updated_at),
                            scale=2,
                            show_label=False,
                            interactive=False,
                            container=False,
                            max_lines=1,
                        )
                        view_button = gr.Button(
                            "View",
                            link=(
                                f"/?conversation-id={conv.conversation_id}"
                                if conv.conversation_id != current_conversation_id
                                else None  # Behave like Cancel, so we don't reload
                            ),
                            scale=0,
                        )
                        if conv.conversation_id == current_conversation_id:
                            # Behave like Cancel, so we don't reload
                            view_button.click(
                                lambda: [
                                    gr.update(visible=True),
                                    gr.update(visible=False),
                                    None,
                                ],
                                outputs=[
                                    chat_view,
                                    conversation_list_view,
                                    conversation_list_requested_page_token,
                                ],
                            )
                        gr.Textbox(
                            conv.description,
                            scale=16,
                            show_label=False,
                            interactive=False,
                            container=False,
                            max_lines=1,
                        )
                        revert_timer = gr.Timer(
                            value=5.0,
                            active=False,
                        )
                        remove_button = gr.Button(
                            "Remove", scale=0, variant="secondary"
                        )
                        confirm_button = gr.Button(
                            "Confirm Remove",
                            visible=False,
                            variant="stop",
                            scale=0,
                        )
                        remove_button.click(
                            lambda: [
                                gr.update(visible=False),
                                gr.update(visible=True),
                                gr.update(active=True),
                            ],
                            outputs=[remove_button, confirm_button, revert_timer],
                        )
                        confirm_button.click(
                            get_remove_conversation_list_item(
                                conv.conversation_id,
                            ),
                            inputs=[
                                conversation_list_current_page_token,
                                conversation_list_previous_page_tokens,
                                conversation_list_requested_page_token,
                            ],
                            outputs=[
                                conversation_list_current_page_token,
                                conversation_list_previous_page_tokens,
                                conversation_list_current_page,
                            ],
                        )
                        revert_timer.tick(
                            lambda: [
                                gr.update(visible=True),
                                gr.update(visible=False),
                                gr.update(active=False),
                            ],
                            outputs=[remove_button, confirm_button, revert_timer],
                        )
                gr.Markdown("---")
                with gr.Row():
                    prev_page_button = gr.Button(
                        "Previous page",
                        interactive=bool(previous_page_tokens),
                    )
                    cancel_button = gr.Button("Cancel")
                    next_page_button = gr.Button(
                        "Next page", interactive=bool(page.next_page_token)
                    )

                    prev_page_button.click(
                        lambda previous_page_tokens: [
                            previous_page_tokens[-1] if previous_page_tokens else None,
                            gr.update(interactive=False),
                        ],
                        inputs=[conversation_list_previous_page_tokens],
                        outputs=[
                            conversation_list_requested_page_token,
                            prev_page_button,
                        ],
                    )
                    cancel_button.click(
                        lambda: [
                            gr.update(visible=True),
                            gr.update(visible=False),
                            None,
                        ],
                        outputs=[
                            chat_view,
                            conversation_list_view,
                            conversation_list_requested_page_token,
                        ],
                    )
                    next_page_button.click(
                        lambda: [
                            page.next_page_token,
                            gr.update(interactive=True),
                            gr.update(interactive=False),
                        ],
                        outputs=[
                            conversation_list_requested_page_token,
                            prev_page_button,
                            next_page_button,
                        ],
                    )

        conversation_list_requested_page_token.change(
            load_page,
            inputs=[
                conversation_list_current_page_token,
                conversation_list_previous_page_tokens,
                conversation_list_requested_page_token,
            ],
            outputs=[
                conversation_list_current_page_token,
                conversation_list_previous_page_tokens,
                conversation_list_current_page,
            ],
            queue=True,
        )

        conversation_id.change(
            load_conversation_id,
            inputs=[conversation_id],
            outputs=[traces_state, conversation_description],
            queue=True,
            show_progress="full",
            show_progress_on=[msg, chatbot],
        )

        conversation_description.change(
            None,
            inputs=[conversation_description],
            js=js_set_conversation_description_as_title,
        )

        conversation_list_btn.click(
            lambda: [gr.update(visible=False), gr.update(visible=True)],
            outputs=[chat_view, conversation_list_view],
        )

        trace_visibility_drop_down.select(
            lambda show_traces: show_traces,
            inputs=[trace_visibility_drop_down],
            outputs=[show_traces_state],
        )

        show_traces_state.change(
            traces_state_change,
            inputs=[traces_state, show_traces_state],
            outputs=[chatbot, new_chat_btn],
            show_progress="hidden",
            show_progress_on=[],
        )

        conversation_list_current_page.change(
            lambda state: gr.update(interactive=bool(state.conversations)),
            inputs=[conversation_list_current_page],
            outputs=[conversation_list_btn],
        )

        def get_conversation_list() -> ConversationPage:
            conv_list = conversation_list_instance()
            if not conv_list:
                return ConversationPage()
            return conv_list.get_conversations()

        msg_submitted = msg.submit(
            user_submit,
            inputs=[msg, chatbot],
            outputs=[
                msg,
                last_user_input,
                stop_event,
                chatbot,
            ],
        )

        msg_submitted.then(
            describe_conversation,
            inputs=[conversation_id, chatbot],
            outputs=[conversation_description],
            queue=True,
        )

        msg_submitted.then(
            assistant_stream,
            inputs=[conversation_id, traces_state, last_user_input, stop_event],
            outputs=[traces_state],
            show_progress="full",
            show_progress_on=[chatbot],
            queue=True,
        ).then(
            lambda: gr.update(interactive=True, submit_btn=True, stop_btn=False),
            outputs=[msg],
        ).then(
            describe_conversation,
            inputs=[conversation_id, chatbot],
            outputs=[conversation_description],
            queue=True,
        ).then(
            get_conversation_list,
            outputs=[conversation_list_current_page],
            queue=True,
        ).then(
            cleanup_stop_event, inputs=[stop_event], outputs=[stop_event]
        )

        msg.stop(user_stop, inputs=[stop_event], outputs=[msg])

        traces_state.change(
            traces_state_change,
            inputs=[traces_state, show_traces_state],
            outputs=[chatbot, new_chat_btn],
            show_progress="hidden",
            show_progress_on=[],
            queue=True,
        )

        chatbot.clear(
            reset_agent,
            inputs=[stop_event],
            outputs=[
                chatbot,
                traces_state,
                conversation_id,
                last_conversation_id,
                conversation_description,
            ],
        ).then(
            None,
            inputs=[conversation_id],
            js=js_set_conversation_id_as_query_param,
        )

        def load(request: gr.Request, last_conversation_id: str):
            conv_list = conversation_list_instance()

            # Determine the conversation id the user wants to use.
            # This can be either a query param (?conversation-id=ABC),
            # or a value from local storage (the last conversation id).
            # The query param takes precedence:
            conversation_id_to_set = None
            if "conversation-id" in request.query_params and isinstance(
                request.query_params["conversation-id"], str
            ):
                if request.query_params["conversation-id"]:
                    conversation_id_to_set = request.query_params["conversation-id"]
            elif last_conversation_id:
                conversation_id_to_set = last_conversation_id

            current_agent_instance = ensure_agent_instance(conversation_id_to_set)

            return (
                current_agent_instance.traces,
                gr.update(
                    label=f"Conversation: {current_agent_instance.conversation_id}"
                ),
                gr.update(interactive=True),
                current_agent_instance.conversation_id,
                current_agent_instance.conversation_id,
                (conv_list.get_conversations() if conv_list else ConversationPage()),
            )

        demo.load(
            load,
            inputs=[last_conversation_id],
            outputs=[
                traces_state,
                chatbot,
                msg,
                conversation_id,
                last_conversation_id,
                conversation_list_current_page,
            ],
        ).then(
            None,
            inputs=[conversation_id],
            js=js_set_conversation_id_as_query_param,
        )

        def cleanup_on_server_shutdown(signum, frame):
            """Set all stop events when server receives SIGINT (CTRL-C)"""
            # Clean up all in-flight requests
            with _stop_events_lock:
                for event in _stop_events:
                    event.set()
                _stop_events.clear()

            # Restore default SIGINT handler and re-raise to allow normal shutdown
            signal.signal(signal.SIGINT, signal.SIG_DFL)
            raise KeyboardInterrupt

        # Handle server shutdown (CTRL-C)
        signal.signal(signal.SIGINT, cleanup_on_server_shutdown)

        return demo


def traces_ui(
    traces: Iterable[Trace],
):
    chat_messages = chat_messages_from_traces(
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
            label=f"Conversation {chat_messages.conversation_id}",
            value=chat_messages.messages,  # type: ignore
        )

        def do_toggle_all_traces(state):
            new_state = not state
            new_label = "Hide internal traces" if new_state else "Show all traces"
            chat_messages_ = chat_messages_from_traces(
                traces,
                show_traces="ALL" if new_state else "CORE",
            )
            return gr.update(value=new_label), new_state, chat_messages_.messages

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
