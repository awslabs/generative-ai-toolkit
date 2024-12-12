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

from typing import Sequence
from generative_ai_toolkit.metrics import BaseMetric, Measurement, Unit
from generative_ai_toolkit.tracer import Trace
from generative_ai_toolkit.test import CaseTrace


class TemplateMetric(BaseMetric):
    """
    TemplateMetric class for users to create custom metrics for evaluating LLM responses.

    Custom metrics work on traces, and should return zero, one or more measurements based on these traces:

      Trace(s) --> Custom Metric (your code) --> Measurement(s)

    Traces are created by the agent as it operates, i.e. as it invokes LLM and tools, and capture .
    Custom metrics can evaluate traces and thus offer you the possibility to measure your agent's performance,
    both during development and in production.

    This template provides an explanation of the trace data model and demonstrates how to evaluate
    a trace to produce measurements. Users can define their own evaluation logic based on the content of the trace.

    Trace Data Model
    ================
    A trace is a data structure that captures what the agent "does" in full detail.

    There are 2 types of traces, LLM traces and Tool traces.

    - An LLM trace contains the interaction between the agent and the LLM model.
      It includes the request from the user, the response from the LLM, and relevant metadata.
    - A Tool trace contains the interaction between the agent and a tool, i.e. a tool invocation.
      It includes the input to the tool, the output from the tool, and relevant metadata.

    LLM Trace Structure
    -------------------
    {
        "to": "LLM",
        "conversation_id": "unique-conversation-id",
        "trace_id": "unique-trace-id",
        "created_at": "timestamp",
        "request": {
            "modelId": "model-name",
            "inferenceConfig": {
                "temperature": 0.7
            },
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "text": "User's first input text message."
                        }
                    ]
                },
                {
                    "role": "assistant",
                    "content": [
                        {
                            "text": "Assistant's first response text message (that was the output of a prior LLM call)"
                        }
                    ]
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "text": "User's second input text message."
                        }
                    ]
                },
            ],
            "system": [
                {
                    "text": "System instructions provided to the assistant."
                }
            ],
            "toolConfig": {
                "tools": [
                    {
                        "toolSpec": {
                            "name": "tool_name",
                            "description": "Description of the tool available to the LLM.",
                            "inputSchema": {
                                "json": {
                                    "type": "object",
                                    "properties": {}
                                }
                            }
                        }
                    }
                ]
            }
        },
        "response": {
            "output": {
                "message": {
                    "role": "assistant",
                    "content": [
                        {
                            "text": "The newly generated response text from the assistant."
                        }
                    ]
                }
            },
            "usage": {
                "inputTokens": 1040,
                "outputTokens": 50,
                "totalTokens": 1090
            },
            "metrics": {
                "latencyMs": 2000
            }
        },
    }

    Key Components:
    1. **conversation_id**: Unique ID of the conversation.
    2. **trace_id**: Unique ID for this trace in the conversation.
    3. **created_at**: Timestamp when the trace was created.
    4. **request**: Contains the user's message, system instructions, and tool configurations.
        - **modelId**: The model ID used for the inference.
        - **inferenceConfig**: Parameters used for the model inference (e.g., temperature, a float).
        - **messages**: A list of messages exchanged, including both user input and assistant responses.
        - **system**: Any system-level instructions provided to the assistant.
        - **toolConfig**: Configuration of tools available to the assistant during inference.
    5. **response**: Contains the assistant’s response message and metrics.
        - **output**: The assistant's output message.
        - **usage**: Contains information about token usage (input, output, total).
            - **inputTokens**, **outputTokens**, **totalTokens**: Integers representing token counts.
        - **metrics**: Latency and other response-related metrics.
            - **latencyMs**: Response latency in milliseconds (an integer).

    Tool Trace Structure
    --------------------
    {
      "to": "TOOL",
      "conversation_id": "unique-conversation-id",
      "trace_id": "unique-trace-id",
      "created_at": "timestamp",
      "request": {
        "tool_input": {
          "param": "value"
        },
        "tool_name": "the name of the tool",
        "tool_use_id": "unique ID"
      },
      "response": {
        "latency_ms": 123,
        "tool_response": {
          "key": "value"
        }
      },
    }

    Case Traces
    -----------
    You can use Cases to test your agent. Traces that the agent generates as it is running a case will have a top-level `case` property that links to the Case object.
    This allows you to e.g. access expectations or other attributes that you added to the case. Some out-of-the-box metrics work like this,
    e.g. the SimilarityMetric looks at the cosine similarity between the agent's answers and the expected answers that you provided in the case.

    ...
    "case": Case(
      "name": "Name of the case",
      "user_inputs": ["Make me a coffee",  "hot, no sugar or milk"],
      "overall_expectations": "The agent asks if the user wants sugar or milk",
    ),
    ...

    Creating your own Custom Metric
    -------------------------------
    Modify either the `evaluate_trace` method, or the `evaluate_conversation` method, to define how the trace should be evaluated.
    - Modify `evaluate_trace` if your metric is based solely on individual traces, where you don't need to look across other traces in the conversation.
      For example: to determine latency of individual LLM ot Tool invocations.
    - Modify `evaluate_conversation` if your metric requires analyzing multiple traces in a conversation at once.
      For example: to determine the total wall clock time of the conversation.

    Return zero, one or more `Measurement` objects with details about the evaluation.
    """

    def evaluate_trace(
        self, trace: Trace, **kwargs
    ) -> Measurement | Sequence[Measurement] | None:
        """
        Evaluate the trace using a custom metric.

        Args:
            trace (Trace): The trace object containing the request and response to the LLM or tool.
            **kwargs: Additional keyword arguments for customization (optional).

        Returns:
            Measurement | Sequence[Measurement] | None: One or more Measurement objects containing the evaluation results. If no meaningful evaluation
                                                        can be performed, it should return None.
        """

        # Below is a sample implementation that demonstrates how to implement custom metrics.

        measurements: list[Measurement] = []

        if trace.to != "LLM":
            # Many metrics you write would focus on either LLM traces or Tool traces.
            # For example, the sample implementation below is for LLM traces.
            # Therefore, we don't look at Tool traces in this Custom Metric, and simply return:
            return

        # 1. Accessing the conversation with the user
        # For LLM traces, the `trace.user_conversation` contains a list of text messages exchanged between the user and the assistant so far.
        # This works, because all past messages must be included upon invoking the LLM as part of a conversation.
        # This is a convenience method; you can also access the messages directly in `trace.request["messages"]` and `trace.response["output"]["message"]`,
        # however that may also contain messages related to tool usage.

        # The last message from the user:
        user_messages = [
            msg["text"] for msg in trace.user_conversation if msg["role"] == "user"
        ]
        last_user_message = user_messages[-1]

        # The last message from the agent:
        agent_messages = [
            msg["text"] for msg in trace.user_conversation if msg["role"] == "assistant"
        ]
        last_agent_message = agent_messages[-1]

        # Example: measure the length of the user's request (number of characters)
        measurements.append(
            Measurement(
                name="UserRequestLength", value=len(last_user_message), unit=Unit.Count
            )
        )

        # 2. Accessing the tool invocations
        # For LLM traces, the `trace.invocations` contains a list of tool invocations that have taken place so far.
        # This works, because all past tool invocations must be included upon invoking the LLM as part of a conversation.
        # This is a convenience method; you can also access the tool invocations directly in `trace.request["messages"]`,
        # however that may also contain text messages from and to the user.
        # Here is a simple metric that measures how many unique tools where used:
        tools_used = set()
        for invocation in trace.tool_invocations:
            tools_used.add(invocation["tool_name"])
        measurements.append(
            Measurement(
                name="UniqueToolsUsedInConversation",
                value=len(tools_used),
                unit=Unit.Count,
            )
        )

        # 3. You can mark measurements as "failing"; during eval() they will then be reported about accordingly.
        # To mark a measurement as failing, set `validation_passed` to False in the Measurement object.
        # For example, you could measure the length of the agent's request, and "fail" if it's overly short, say less than 10.
        # In this case, it may be helpful to add the agent's actual response (that was too short) as additional info,
        # so that you can readily see it if you access logged measurements later:
        validation_passed = len(last_agent_message) < 10
        additional_info = None
        if not validation_passed:
            additional_info = {"last_agent_message": last_agent_message}

        measurements.append(
            Measurement(
                name="AgentResponseLength",
                value=len(last_agent_message),
                unit=Unit.Count,
                validation_passed=validation_passed,
                additional_info=additional_info,
            )
        )

        # 4. For LLM traces, the `trace.response` contains details about the assistant's response, including the message content,
        # token usage, stop reasons, and other metadata. You can evaluate this section to test the output data.
        # For example, it is likely you want to measure tokens and latency:
        input_tokens = trace.response["usage"]["inputTokens"]
        measurements.append(
            Measurement(
                name="InputTokens",
                value=float(input_tokens),
            )
        )

        output_tokens = trace.response["usage"]["outputTokens"]
        measurements.append(
            Measurement(
                name="OutputTokens",
                value=float(output_tokens),
            )
        )

        measurements.append(
            Measurement(
                name="LlmLatencyMs",
                value=float(trace.response["metrics"]["latencyMs"]),
            )
        )

        # 5. For traces that the agent generates when it runs a case, you can access the case object like so:
        if isinstance(trace, CaseTrace):
            case_ = trace.case  # Note: This attribute is only there, when you run cases through the agent (e.g. while developing and testing).

            # Let's pretend that based on the expectations from the case, we want to measure the quality of the conversation,
            # and we'll do so by measuring the nr of similar words between the last user message and the expectation:
            if case_.overall_expectations:
                words1 = set(case_.overall_expectations.split())
                words2 = set(last_user_message.split())
                common_words = words1.intersection(words2)
                conversation_quality = len(common_words) / len(words1)

                measurements.append(
                    Measurement(
                        name="ConversationQuality",
                        value=conversation_quality,
                    )
                )

        # 6. Finally return the measurements.
        # You can also return just 1 measurement by itself (not in a list), or None if you don't want to report any measurement
        return measurements

    def evaluate_conversation(
        self, conversation_traces: Sequence[Trace], **kwargs
    ) -> Measurement | Sequence[Measurement] | None:
        """
        Evaluate the trace using a custom metric.

        The difference with `evaluate_trace` is that `evaluate_conversation` can look at all traces of the conversation at once,
        and can thus measure things as:

        - There was at least one tool invocation in the conversation
        - The total time taken by the conversation
        - The total cost of the conversation
        - The first response from the agent to the user contains a question

        Args:
            conversation (Sequence[Trace]): The sequence of trace objects that contain the requests and responses to the LLM and tools.
            **kwargs: Additional keyword arguments for customization (optional).

        Returns:
            Measurement | Sequence[Measurement] | None: One or more Measurement objects containing the evaluation results. If no meaningful evaluation
                                                        can be performed, it should return None.
        """

        # Example: count the total number of tools used in the conversation:
        nr_of_tool_invocations = 0
        for trace in conversation_traces:
            if trace.to == "TOOL":
                nr_of_tool_invocations += 1

        return Measurement(
            "NrOfToolInvocations", value=nr_of_tool_invocations, unit=Unit.Count
        )
