# Copyright 2025 Amazon.com, Inc. and its affiliates. All Rights Reserved.
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

from generative_ai_toolkit.agent import BedrockConverseAgent
from generative_ai_toolkit.conversation_history import DynamoDbConversationHistory
from generative_ai_toolkit.test import Expect
from generative_ai_toolkit.tracer.dynamodb import DynamoDbTracer


def test_image(
    dynamodb_traces_table_name, dynamodb_conversation_history_table_name, invention_png
):
    identifier = "integration-test:test_image"
    agent = BedrockConverseAgent(
        model_id="eu.anthropic.claude-3-7-sonnet-20250219-v1:0",
        conversation_history=DynamoDbConversationHistory(
            table_name=dynamodb_conversation_history_table_name, identifier=identifier
        ),
        temperature=0.0,
        tracer=DynamoDbTracer(
            table_name=dynamodb_traces_table_name, identifier=identifier
        ),
    )

    agent.conversation_history.add_message(
        {
            "role": "user",
            "content": [
                {"image": {"format": "png", "source": {"bytes": invention_png}}}
            ],
        }
    )

    agent.converse("Describe the image please")

    Expect(agent.traces).agent_text_response.to_include(
        "Invention comes in many forms and at many scales"
    )
    Expect(agent.traces).agent_text_response.to_include("Amazon")
