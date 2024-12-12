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

import time
from generative_ai_toolkit.evaluate.interactive import GenerativeAIToolkit, Permute
from generative_ai_toolkit.agent import BedrockConverseAgent
from generative_ai_toolkit.test import Case
from generative_ai_toolkit.metrics.modules import (
    conciseness,
    conversation,
    latency,
    token,
    sentiment,
)


cases = [
    Case(
        f"User inquires capital of {country}",
        user_inputs=[
            f"What is the capital of {country}",
            "What are the touristic highlights there?",
        ],
        overall_expectations="Agent correctly identifies the country's capital and provides highlights to the user",
    )
    for country in ["France", "Germany", "The Netherlands"]
]

traces = GenerativeAIToolkit.generate_traces(
    cases=cases,
    agent_factory=BedrockConverseAgent,
    agent_parameters={
        "system_prompt": Permute(
            [
                "You are a helpful assistant, and only give succinct and to-the-point answers",
                "You are a helpful assistant, and are very friendly in your communication",
            ]
        ),
        "model_id": "anthropic.claude-3-haiku-20240307-v1:0",
    },
)

results = GenerativeAIToolkit.eval(
    metrics=[
        conciseness.AgentResponseConcisenessMetric(),
        conversation.ConversationExpectationMetric(),
        latency.LatencyMetric(),
        token.TokensMetric(),
        sentiment.SentimentMetric(),
    ],
    traces=traces,
)

try:
    results.start_ui()
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    print("Stopping UI")
results.stop_ui()
