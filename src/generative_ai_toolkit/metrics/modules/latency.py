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
from generative_ai_toolkit.tracer import ToolTrace, LlmTrace


class LatencyMetric(BaseMetric):
    """
    LatencyMetric class for measuring the latency of model invocations.

    This metric measures the time taken for a provided callback to execute and stores
    the latency in milliseconds.

    Attributes:
        latency: The measured latency in milliseconds.
    """

    def evaluate_trace(self, trace, **kwargs):
        """
        Return the stored latency.

        :return: A dictionary with the latency result in milliseconds.
        """

        dimensions = [{"To": trace.to}]
        if isinstance(trace, ToolTrace):
            dimensions.append({"ToolName": trace.request["tool_name"]})

        latency_value = int(
            trace.response["metrics"]["latencyMs"]
            if isinstance(trace, LlmTrace)
            else trace.response["latency_ms"]
        )

        return Measurement(
            name="Latency",
            value=latency_value,
            unit=Unit.Milliseconds,
            dimensions=dimensions,
            validation_passed=True if latency_value < 12000 else False,
        )
