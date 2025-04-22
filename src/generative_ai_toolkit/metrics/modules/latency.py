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

from generative_ai_toolkit.metrics import BaseMetric, Measurement, Unit
from generative_ai_toolkit.utils.logging import logger


class LatencyMetric(BaseMetric):
    """
    LatencyMetric class for measuring the latency of model invocations and other actions the agent takes.
    """

    def evaluate_trace(self, trace, **kwargs):
        """
        Return the stored latency.

        :return: A dictionary with the latency result in milliseconds.
        """

        dimensions = []
        trace_type = trace.attributes.get("ai.trace.type")
        if trace_type == "tool-invocation":
            dimensions.append({"ToolName": trace.attributes["ai.tool.name"]})
        elif trace_type == "llm-invocation":
            dimensions.append(
                {"ModelName": trace.attributes["ai.llm.request.model.id"]}
            )
        elif trace_type == "conversation-history-list":
            dimensions.append({"ConversationHistory": "list-messages"})
        elif trace_type == "conversation-history-add":
            dimensions.append({"ConversationHistory": "add-message"})
        elif trace_type in {"converse", "converse-stream"}:
            dimensions.append({"Converse": trace_type})
        else:
            logger.warn("Unknown trace type", trace_type=trace_type)

        return Measurement(
            name="Latency",
            value=trace.duration_ms,
            unit=Unit.Milliseconds,
            dimensions=dimensions,
        )
