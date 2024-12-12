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

from generative_ai_toolkit.metrics import BaseMetric, Measurement
from generative_ai_toolkit.tracer import LlmTrace


class TokensMetric(BaseMetric):
    def evaluate_trace(self, trace, **kwargs):
        if not isinstance(trace, LlmTrace):
            return

        input_tokens = trace.response["usage"]["inputTokens"]
        output_tokens = trace.response["usage"]["outputTokens"]

        return [
            Measurement(
                name="TotalTokens",
                value=float(input_tokens + output_tokens),
            ),
            Measurement(
                name="InputTokens",
                value=float(input_tokens),
            ),
            Measurement(
                name="OutputTokens",
                value=float(output_tokens),
            ),
        ]
