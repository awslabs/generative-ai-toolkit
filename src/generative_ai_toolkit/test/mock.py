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

from collections.abc import Sequence
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, Literal, NotRequired, TypedDict, Unpack
from unittest.mock import Mock
from uuid import uuid4

import boto3.session

from generative_ai_toolkit.tracer import BaseTracer
from generative_ai_toolkit.tracer.trace import Trace

if TYPE_CHECKING:
    from mypy_boto3_bedrock_runtime.type_defs import (
        ContentBlockOutputTypeDef,
        ConverseRequestTypeDef,
        ConverseResponseTypeDef,
    )


class ToolUseOutput(TypedDict):
    name: str
    input: dict[str, Any]
    toolUseId: NotRequired[str]


RealResponse = Literal["RealResponse"]


class MockBedrockConverse:
    def __init__(self, session: boto3.session.Session | None = None) -> None:
        self.real_client = (session or boto3).client("bedrock-runtime")
        self.mock_responses: list[ConverseResponseTypeDef | RealResponse] = []

    def reset(self):
        self.mock_responses = []

    def _converse(
        self, **kwargs: Unpack["ConverseRequestTypeDef"]
    ) -> "ConverseResponseTypeDef":
        if len(self.mock_responses) == 0:
            raise RuntimeError(
                f"Exhausted all mock responses, but need to reply to message: {kwargs.get("messages", [])[-1]}"
            )
        response, *self.mock_responses = self.mock_responses
        if response == "RealResponse":
            response = self.real_client.converse(**kwargs)
        return response

    def client(self):
        mock_client = Mock(name="MockClient")
        mock_client.converse = self._converse
        return mock_client

    def session(self):
        mock_session = Mock(name="MockSession")
        mock_session.client.return_value = self.client()
        return mock_session

    def add_raw_response(self, response: "ConverseResponseTypeDef"):
        self.mock_responses.append(response)

    def _get_raw_response(self, message_content: list["ContentBlockOutputTypeDef"]):
        if not message_content:
            raise Exception("No message content provided")
        has_tool_output = any("toolUse" in message for message in message_content)
        request_id = uuid4()
        response: ConverseResponseTypeDef = {
            "ResponseMetadata": {
                "RequestId": request_id.hex,
                "HTTPStatusCode": 200,
                "HTTPHeaders": {
                    "date": datetime.now(UTC).strftime(
                        "%a, %d %b %Y %H:%M:%S GMT"  # Tue, 11 Mar 2025 13:58:48 GMT
                    ),
                    "content-type": "application/json",
                    "content-length": "359",
                    "connection": "keep-alive",
                    "x-amzn-requestid": request_id.hex,
                    "x-mocked-response": "true",
                },
                "RetryAttempts": 0,
            },
            "output": {
                "message": {
                    "role": "assistant",
                    "content": message_content,
                }
            },
            "stopReason": "tool_use" if has_tool_output else "end_turn",
            "usage": {"inputTokens": 1485, "outputTokens": 70, "totalTokens": 1555},
            "metrics": {"latencyMs": 2468},
            "additionalModelResponseFields": {},
            "performanceConfig": {},
            "trace": {},
        }
        return response

    def add_output(
        self,
        tool_use_output: Sequence[ToolUseOutput] = (),
        text_output: Sequence[str] = (),
    ):
        tool_uses: list[ContentBlockOutputTypeDef] = [
            {"toolUse": {"toolUseId": uuid4().hex, **t}} for t in tool_use_output
        ]
        texts: list[ContentBlockOutputTypeDef] = [{"text": t} for t in text_output]
        self.mock_responses.append(self._get_raw_response([*tool_uses, *texts]))

    def add_real_response(self):
        self.mock_responses.append("RealResponse")


class LlmInvocationTracer(BaseTracer):
    """
    Use this Tracer with real LLM invocations, to help you write MockBedrockConverse.add_output() statements
    """

    def persist(self, trace: Trace):
        if trace.attributes.get("ai.trace.type") == "llm-invocation":
            output = trace.attributes.get("ai.llm.response.output")
            if not output:
                return
            texts = []
            tool_uses = []
            for msg in output["message"]["content"]:
                if "toolUse" in msg:
                    tool_uses.append(
                        {
                            "name": msg["toolUse"]["name"],
                            "input": msg["toolUse"]["input"],
                        }
                    )
                elif "text" in msg:
                    texts.append(msg["text"])
            print(f"add_output(text_output={texts},tool_use_output={tool_uses})")
