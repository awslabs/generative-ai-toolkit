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

from decimal import Decimal

from generative_ai_toolkit.metrics import BaseMetric, Measurement


class CostMetric(BaseMetric):
    """
    CostMetric class for measuring the cost of model invocations.

    This metric calculates the cost based on the number of input and output tokens used by the model.
    """

    def __init__(self, pricing_config, cost_threshold=None):
        """
        Initialize the CostMetric with pricing configuration and optional cost threshold.

        :param pricing_config: Dictionary containing pricing details for input and output tokens.
        :param cost_threshold: Optional float value to set a cost threshold for evaluation.
        :param cost_comparator: Optional string to set the comparator for the cost threshold ('<=', '>=', etc.).
        """
        super().__init__()
        self.pricing_config = pricing_config
        self.cost_threshold = cost_threshold

    def evaluate_trace(self, trace, **kwargs):
        """
        Evaluate the cost using the provided trace, which includes user input, system prompt, and response.

        This method calculates the cost based on the number of input and output tokens used by the model
        and compares it against a predefined cost threshold if provided.

        Evaluate the model using the provided prompt and response.

        :param trace: Trace object that contains the request and response to the LLM
        :return: A dictionary with the evaluation results including cost, cost threshold, comparator, and cost difference.
        """

        if trace.attributes.get("ai.trace.type") != "llm-invocation":
            return

        input_tokens = trace.attributes["ai.llm.response.usage"]["inputTokens"]
        output_tokens = trace.attributes["ai.llm.response.usage"]["outputTokens"]
        model_id = trace.attributes["ai.llm.request.model.id"]

        # Calculate cost based on tokens used and pricing configuration
        per_token = self.pricing_config[model_id]["per_token"]
        input_cost = (Decimal(input_tokens) / Decimal(per_token)) * Decimal(
            self.pricing_config[model_id]["input_cost"]
        )
        output_cost = (Decimal(output_tokens) / Decimal(per_token)) * Decimal(
            self.pricing_config[model_id]["output_cost"]
        )

        cost = float(input_cost + output_cost)

        return Measurement(
            name="Cost",
            value=cost,
            validation_passed=(
                None if self.cost_threshold is None else cost <= self.cost_threshold
            ),
            additional_info={
                "cost_threshold": self.cost_threshold,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "per_token": per_token,
            },
        )
