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

from typing import TYPE_CHECKING

from generative_ai_toolkit.agent import BedrockConverseAgent
from generative_ai_toolkit.conversation_history import DynamoDbConversationHistory
from generative_ai_toolkit.test import Expect
from generative_ai_toolkit.tracer.dynamodb import DynamoDbTracer

if TYPE_CHECKING:
    from mypy_boto3_bedrock_runtime.type_defs import ToolResultContentBlockUnionTypeDef


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

    def get_random_image() -> "list[ToolResultContentBlockUnionTypeDef]":
        """
        Generate a random image for the user
        """
        return [{"image": {"format": "png", "source": {"bytes": invention_png}}}]

    agent.register_tool(get_random_image)

    agent.converse(
        "Generate a random image for me please, and then describe the image to me"
    )

    Expect(agent.traces).agent_text_response.to_include(
        "Invention comes in many forms and at many scales"
    )
    Expect(agent.traces).agent_text_response.to_include("Amazon")


def test_document_word(
    dynamodb_traces_table_name,
    dynamodb_conversation_history_table_name,
    sample_document_docx,
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

    def get_random_document() -> "list[ToolResultContentBlockUnionTypeDef]":
        """
        Generate a random document for the user
        """
        return [
            {
                "document": {
                    "format": "docx",
                    "name": "Sample document",
                    "source": {"bytes": sample_document_docx},
                }
            }
        ]

    agent.register_tool(get_random_document)

    agent.converse(
        "Generate a random document for me please, and then describe what it says to me"
    )

    Expect(agent.traces).agent_text_response.with_fn(lambda x: x.lower()).to_include(
        "hello, world"
    )
    Expect(agent.traces).agent_text_response.to_include("toolkit")
    Expect(agent.traces).agent_text_response.to_include("agent")
    Expect(agent.traces).agent_text_response.to_include("test")


def test_document_pdf(
    dynamodb_traces_table_name,
    dynamodb_conversation_history_table_name,
    sample_document_pdf,
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

    def get_random_document() -> "list[ToolResultContentBlockUnionTypeDef]":
        """
        Generate a random document for the user
        """
        return [
            {
                "document": {
                    "format": "pdf",
                    "name": "Sample document",
                    "source": {"bytes": sample_document_pdf},
                }
            }
        ]

    agent.register_tool(get_random_document)

    agent.converse(
        "Generate a random document for me please, and then describe what it says to me"
    )

    Expect(agent.traces).agent_text_response.with_fn(lambda x: x.lower()).to_include(
        "hello, world"
    )
    Expect(agent.traces).agent_text_response.to_include("toolkit")
    Expect(agent.traces).agent_text_response.to_include("agent")
    Expect(agent.traces).agent_text_response.to_include("test")
