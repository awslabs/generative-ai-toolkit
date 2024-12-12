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

import json
import textwrap
from concurrent.futures import Executor
from typing import cast

import boto3

from generative_ai_toolkit.metrics import BaseMetric, Measurement
from generative_ai_toolkit.tracer import LlmTrace
from generative_ai_toolkit.test import LlmCaseTrace
from generative_ai_toolkit.utils.typings import NonStreamingResponse
from generative_ai_toolkit.utils.llm_response import json_parse


class ConversationExpectationMetric(BaseMetric):
    """
    This metric measures whether the conversation with the agent goes as intended.
    """

    def __init__(
        self,
        *,
        model_id="anthropic.claude-3-sonnet-20240229-v1:0",
        expectations: str | None = None,
    ):
        super().__init__()
        self.model_id = model_id
        self.bedrock_client = boto3.client("bedrock-runtime")
        self.expectations = expectations

    def evaluate_conversation(
        self, conversation_traces, *, executor: Executor, **kwargs
    ):
        trace = conversation_traces[
            -1
        ]  # Use the conversation as captured in the last trace

        if not isinstance(trace, LlmTrace):
            return

        expectations = self.expectations
        if isinstance(trace, LlmCaseTrace) and trace.case.overall_expectations:
            expectations = trace.case.overall_expectations
        if not expectations:
            return

        fut = executor.submit(
            self.bedrock_client.converse,
            modelId=self.model_id,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "text": textwrap.dedent(
                                """
                                You will be given two inputs:
                                1. Conversation between a car driver and an LLM agent.
                                2. Developer provided expectations for the LLM agent's responses.

                                You will compare the actual conversation against the provided expectations.

                                Here is the conversation:
                                <conversation>
                                {conversation}
                                </conversation>

                                Here are the expectations:
                                <expectations>
                                {expectations}
                                </expectations>

                                Your output should be a JSON object with an overall score between 1 and 10, where 10 would indicate that the conversation fully aligns with the expectations. Also provide your reasoning.

                                Example output:
                                {{ "score": 9, "reasoning": "The agent succeeded in helping the user as expected"}}

                                Only return the JSON object.
                                """
                            )
                            .format(
                                conversation=json.dumps(trace.user_conversation),
                                expectations=expectations,
                            )
                            .strip()
                        }
                    ],
                }
            ],
            system=[
                {
                    "text": textwrap.dedent(
                        """
                        You are an expert at judging the effectiveness of LLM agents.
                        """
                    ).strip()
                }
            ],
        )
        response = json_parse(cast(NonStreamingResponse, fut.result()))

        return Measurement(
            name="Correctness",
            value=response["score"],
            additional_info=response["reasoning"],
        )
