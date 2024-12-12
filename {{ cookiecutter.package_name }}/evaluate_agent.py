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

###########
# NOTICE
#
# This is a sample file with some metrics that you could run as part of an integration test suite
#
# Note: any metrics that should be calculated against the deployed agent, should also be added in: lib/evaluation/measure.py
###########

from generative_ai_toolkit.evaluate.interactive import GenerativeAIToolkit
from generative_ai_toolkit.metrics import BaseMetric, Measurement
from generative_ai_toolkit.test import Case
from generative_ai_toolkit.metrics.modules.latency import LatencyMetric
from generative_ai_toolkit.metrics.modules.conciseness import AgentResponseConcisenessMetric
from generative_ai_toolkit.metrics.modules.conversation import ConversationExpectationMetric

from lib.agent.agent import {{ cookiecutter.agent_name }}


class MyCustomConcisenessMetric(BaseMetric):
    """
    A simple metric that measures the length of the LLM output
    """

    def evaluate_trace(self, trace, **kwargs):
        if trace.to != "LLM":
            return  # Only collect this metric for LLM traces
        response_text_length = 0
        has_text = False
        assistant_msg = trace.response["output"]["message"]
        for content in assistant_msg["content"]:
            if "text" in content:
                response_text_length += len(content["text"])
                has_text = True
        if has_text:
            return Measurement(
                name="ResponseNrOfCharacters", value=response_text_length
            )


metrics = [
    LatencyMetric(),
    AgentResponseConcisenessMetric(),
    ConversationExpectationMetric(),
    MyCustomConcisenessMetric(),
]

case1 = Case(
    name="user wants to know the capital of France",
    user_inputs=[
        "What is the capital of France?",
        "What are some touristic highlights there?",
    ],
    overall_expectations="The agent mentions the Eiffel tower",
)
cases = [case1]

traces = GenerativeAIToolkit.generate_traces(
    cases=cases,
    agent_factory={{ cookiecutter.agent_name }},
    nr_runs_per_case=3,
    agent_parameters={
        "model_id": "anthropic.claude-3-haiku-20240307-v1:0",
    },
)

if __name__ == "__main__":
    measurements = GenerativeAIToolkit.eval(traces=traces, metrics=metrics)

    measurements.summary() # This prints a summary table to stdout

    # TODO
    # Add conditions, e.g. if the conciseness metric is below a certain threshold, exit with status 1
