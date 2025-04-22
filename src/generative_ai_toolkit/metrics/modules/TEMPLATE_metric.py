# noqa: N999
# # Copyright 2024 Amazon.com, Inc. and its affiliates. All Rights Reserved.
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

from collections.abc import Sequence

from generative_ai_toolkit.metrics import BaseMetric, Measurement, Unit
from generative_ai_toolkit.test import CaseTrace, user_conversation_from_trace
from generative_ai_toolkit.tracer import Trace


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

    Generative AI Toolkit uses the OpenTelemetry span model for traces, however we chose to keep the name "trace". So, a trace in Generative AI Toolkit is what
    OpenTelemetry calls a span.

    Traces in Generative AI Toolkit have the following data model (closely aligned with OpenTelemetry):

    - span_name (str)
    - span_kind (str, possible values: "INTERNAL", "SERVER", "CLIENT")
    - started_at (datetime)
    - ended_date (datetime)
    - trace_id (str)
    - span_id (str)
    - parent_span (Trace, a pointer to the parent Trace)
    - span_status (str, possible values "UNSET", "OK", "ERROR")
    - resource_attributes (dict, e.g. the standard "service.name" field would be in here)
    - scope (TraceScope, which is a name and a version)
    - attributes (dict)

    Many details of traces are stored as attributes. For example, a trace of an LLM invocation would store its details as attributes, see next.

    LLM invocation Trace attributes example
    ---------------------------------------
    {
      'peer.service': 'llm:claude-3-sonnet',
      'ai.trace.type': 'llm-invocation',
      'ai.llm.request.inference.config': {},
      'ai.llm.request.messages': [{'role': 'user', 'content': [{'text': "What's the capital of France?"}]}],
      'ai.llm.request.model.id': 'anthropic.claude-3-sonnet-20240229-v1:0',
      'ai.llm.request.system': None,
      'ai.llm.request.tool.config': None,
      'ai.llm.response.output': {'message': {'role': 'assistant', 'content': [{'text': 'The capital of France is Paris.'}]}},
      'ai.llm.response.stop.reason': 'end_turn',
      'ai.llm.response.usage': {'inputTokens': 14, 'outputTokens': 10, 'totalTokens': 24},
      'ai.llm.response.metrics': {'latencyMs': 350},
      'ai.conversation.id': '01JRXF2JHXACD860A6P7N0MXER',
      'ai.auth.context': None
    }

    Tool invocation attributes example
    ----------------------------------
    {
      'peer.service': 'tool:weather_inquiry',
      'ai.trace.type': 'tool-invocation',
      'ai.tool.name': 'weather_inquiry',
      'ai.tool.use.id': 'tooluse_DGBOHaddTh2Rvg9QhVXgDg',
      'ai.tool.input': {'latitude_longitude_list': [[52.00867, 4.3525599999999995]]},
      'ai.tool.output': {'forecast': [{'latitude': 52.00867, 'longitude': 4.3525599999999995, 'temperature': 13, 'precipitation_chance': 0}]},
      'ai.conversation.id': '01JS1JHTE7R4QWHDXP28C0J6RB',
      'ai.auth.context': None
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

        if trace.attributes.get("ai.trace.type") != "llm-invocation":
            # Many metrics you write would focus on either LLM traces or Tool traces.
            # For example, the sample implementation below is for LLM traces.
            # Therefore, we don't look at other traces in this Custom Metric, and simply return:
            return

        # Accessing the conversation with the user can be done with the following helper utility.
        # This just gets the messages from the attributes ai.llm.request.messages and ai.llm.response.output,
        # and filters the tool uses and results out, so you're left with just the "spoken" conversation between agent and user.
        # Note that this works at single trace level, because each LLM invocation would successively include prior messages from the conversation.
        user_conversation = user_conversation_from_trace(trace)

        # The last message from the user:
        user_messages = [
            msg["text"] for msg in user_conversation if msg["role"] == "user"
        ]
        last_user_message = user_messages[-1]

        # The last message from the agent:
        agent_messages = [
            msg["text"] for msg in user_conversation if msg["role"] == "assistant"
        ]
        last_agent_message = agent_messages[-1]

        # Example: measure the length of the user's request (number of characters)
        measurements.append(
            Measurement(
                name="UserRequestLength", value=len(last_user_message), unit=Unit.Count
            )
        )

        # 2. Accessing the tool invocations
        # Here is a simple metric that measures how many unique tools where used (as captured in the LLM trace).
        # Note that this works at single trace level, because each LLM invocation would successively include prior messages from the conversation.
        tools_used = set()
        for msg in trace.attributes.get("ai.llm.request.messages", []):
            if msg["role"] == "assistant":
                for content in msg["content"]:
                    if "toolUse" in content:
                        tools_used.add(content["toolUse"]["name"])

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

        # 4. For LLM traces, the 'ai.llm.response.*' attributes contains details about the assistant's response, including the message content,
        # token usage, stop reasons, and other metadata. You can evaluate this section to test the output data.
        # For example, it is likely you want to measure tokens and latency:
        input_tokens = trace.attributes["ai.llm.response.usage"]["inputTokens"]
        measurements.append(
            Measurement(
                name="InputTokens",
                value=float(input_tokens),
            )
        )

        output_tokens = trace.attributes["ai.llm.response.usage"]["outputTokens"]
        measurements.append(
            Measurement(
                name="OutputTokens",
                value=float(output_tokens),
            )
        )

        measurements.append(
            Measurement(
                name="LlmLatencyMs",
                value=float(trace.attributes["ai.llm.response.metrics"]["latencyMs"]),
            )
        )

        # 5. For traces that the agent generates when it runs a case, you can access the case object like so:
        if isinstance(trace, CaseTrace):
            case_ = (
                trace.case
            )  # Note: This attribute is only there, when you run cases through the agent (e.g. while developing and testing).

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
            conversation (Sequence[Trace]): The sequence of trace objects that contain e.g.the requests and responses to the LLM and tools.
            **kwargs: Additional keyword arguments for customization (optional).

        Returns:
            Measurement | Sequence[Measurement] | None: One or more Measurement objects containing the evaluation results. If no meaningful evaluation
                                                        can be performed, it should return None.
        """

        # Example: count the total number of tools used in the conversation:
        return Measurement(
            "NrOfToolInvocations",
            value=len(
                [
                    trace
                    for trace in conversation_traces
                    if trace.attributes.get("ai.trace.type") == "tool-invocation"
                ]
            ),
            unit=Unit.Count,
        )
