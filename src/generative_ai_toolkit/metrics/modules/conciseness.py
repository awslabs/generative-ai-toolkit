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

import boto3

from generative_ai_toolkit.metrics import BaseMetric, Measurement
from generative_ai_toolkit.test import user_conversation_from_trace
from generative_ai_toolkit.utils.llm_response import json_parse


class AgentResponseConcisenessMetric(BaseMetric):
    """
    This metric measures whether the agent is concise and does not ramble.
    """

    def __init__(self, model_id="anthropic.claude-3-sonnet-20240229-v1:0"):
        super().__init__()
        self.model_id = model_id
        self.bedrock_client = boto3.client("bedrock-runtime")

    def evaluate_conversation(self, conversation_traces, **kwargs):
        for trace in reversed(conversation_traces):
            if trace.attributes.get("ai.trace.type") == "llm-invocation":
                break
        else:
            return

        user_conversation = user_conversation_from_trace(trace)

        text_response = self.bedrock_client.converse(
            modelId=self.model_id,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "text": textwrap.dedent(
                                """
                                You will be given a conversation between a user and an LLM agent, and you will score the LLM agent's responses on these points:

                                1. Are the agents responses concise?
                                2. Does the agent provide superfluous examples when it asks questions?
                                3. Does the agent utter useless encouragements to the user, such as "Sure thing!", "Great!", "Awesome!"?
                                4. Does the agent mention how it works to the user, without this being solicited?
                                5. When the user provides information, the agent should not reiterate that back to the user. E.g. saying "Got it, you want to ..." is unnecessary.

                                Here is the conversation:
                                <conversation>
                                {conversation}
                                </conversation>

                                Your output should be a (valid!) JSON object with an overall score between 1 and 10, where 10 is the most concise and clear response. Also provide your reasoning.

                                Example output:
                                {{ "score": 9, "reasoning": "The agent's responses are concise, and it does not provide superfluous examples or useless encouragements."}}

                                Only return the valid JSON object.
                                """
                            )
                            .format(conversation=json.dumps(user_conversation))
                            .strip()
                        }
                    ],
                }
            ],
            system=[
                {
                    "text": textwrap.dedent(
                        """
                        You are an expert at judging the brevity and conciseness of LLM agents. These LLM agents are supposed to be very to the point.
                        """
                    ).strip()
                }
            ],
        )
        response = json_parse(text_response)

        return Measurement(
            name="Conciseness",
            value=response["score"],
            additional_info={
                "reasoning": response["reasoning"],
                "conversation": user_conversation,
            },
        )
