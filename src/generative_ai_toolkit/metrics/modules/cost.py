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

from decimal import Decimal, ROUND_HALF_UP
from generative_ai_toolkit.metrics import BaseMetric, Measurement
from generative_ai_toolkit.tracer import LlmTrace


class CostMetric(BaseMetric):
    """
    CostMetric class for measuring the cost of model invocations.

    This metric calculates the cost based on the number of input and output tokens used by the model.
    """

    def __init__(self, pricing_config, cost_threshold=None, cost_comparator="<="):
        """
        Initialize the CostMetric with pricing configuration and optional cost threshold.

        :param pricing_config: Dictionary containing pricing details for input and output tokens.
        :param cost_threshold: Optional float value to set a cost threshold for evaluation.
        :param cost_comparator: Optional string to set the comparator for the cost threshold ('<=', '>=', etc.).
        """
        super().__init__()
        self.pricing_config = pricing_config
        self.cost = None

    def evaluate_trace(self, trace, **kwargs):
        """
        Evaluate the cost using the provided trace, which includes user input, system prompt, and response.

        This method calculates the cost based on the number of input and output tokens used by the model
        and compares it against a predefined cost threshold if provided.

        Evaluate the model using the provided prompt and response.

        :param trace: Trace object that contains the request and response to the LLM
        :return: A dictionary with the evaluation results including cost, cost threshold, comparator, and cost difference.
        """

        if not isinstance(trace, LlmTrace):
            return

        try:
            input_tokens = trace.response["usage"]["inputTokens"]
            output_tokens = trace.response["usage"]["outputTokens"]
            model_id = trace.request["modelId"]

            # Calculate cost based on tokens used and pricing configuration
            per_token = self.pricing_config[model_id]["per_token"]
            input_cost = (Decimal(input_tokens) / Decimal(per_token)) * Decimal(
                self.pricing_config[model_id]["input_cost"]
            )
            output_cost = (Decimal(output_tokens) / Decimal(per_token)) * Decimal(
                self.pricing_config[model_id]["output_cost"]
            )

            input_cost = input_cost.quantize(
                Decimal(".00000001"), rounding=ROUND_HALF_UP
            )
            output_cost = output_cost.quantize(
                Decimal(".00000001"), rounding=ROUND_HALF_UP
            )

            self.cost = input_cost + output_cost

            return Measurement(
                name="Cost",
                value=float(self.cost),
                validation_passed=True if self.cost < 0.5 else False,
            )

        except Exception as e:
            print(f"Error in cost calculation: {e}")
