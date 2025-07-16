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

import hashlib
import json
import time
from collections.abc import Callable, Mapping, Sequence
from datetime import UTC, datetime
from functools import cached_property
from typing import (
    TYPE_CHECKING,
    Any,
    Literal,
    NotRequired,
    TypedDict,
    Unpack,
    cast,
)
from unittest.mock import Mock
from uuid import uuid4

import boto3.session

from generative_ai_toolkit.tracer import BaseTracer
from generative_ai_toolkit.tracer.trace import Trace

if TYPE_CHECKING:
    from botocore.eventstream import EventStream
    from mypy_boto3_bedrock_runtime.type_defs import (
        ContentBlockOutputTypeDef,
        ConverseRequestTypeDef,
        ConverseResponseTypeDef,
        ConverseStreamOutputTypeDef,
        ConverseStreamRequestTypeDef,
        ConverseStreamResponseTypeDef,
    )


class ToolUseOutput(TypedDict):
    name: str
    input: dict[str, Any]
    toolUseId: NotRequired[str]


RealResponse = Literal["RealResponse"]


class MockBedrockConverse:
    def __init__(
        self,
        session: boto3.session.Session | None = None,
        response_generator: (
            Callable[
                [
                    "MockBedrockConverse",
                    "ConverseRequestTypeDef | ConverseStreamRequestTypeDef",
                ],
                None,
            ]
            | None
        ) = None,
        stream_delay_between_tokens: int | float | None = None,
    ) -> None:
        self._session = session
        self._mock_responses: list[ConverseResponseTypeDef | RealResponse] = []
        self.response_generator = response_generator
        self.stream_delay_between_tokens = stream_delay_between_tokens

    @property
    def mock_responses(self) -> "list[ConverseResponseTypeDef | RealResponse]":
        return self._mock_responses

    @cached_property
    def real_client(self):
        return (self._session or boto3).client("bedrock-runtime")

    def reset(self):
        self._mock_responses.clear()

    def _converse(
        self, **kwargs: Unpack["ConverseRequestTypeDef"]
    ) -> "ConverseResponseTypeDef":
        if len(self.mock_responses) == 0 and self.response_generator:
            self.response_generator(self, kwargs)
        if len(self.mock_responses) == 0:
            raise RuntimeError(
                f"Exhausted all mock responses, but need to reply to message: {kwargs.get('messages', [])[-1]}"
            )
        response, *self._mock_responses = self.mock_responses
        if response == "RealResponse":
            response = self.real_client.converse(**kwargs)
        return response

    def _converse_stream(self, **kwargs: Unpack["ConverseStreamRequestTypeDef"]):
        if len(self.mock_responses) == 0 and self.response_generator:
            self.response_generator(self, kwargs)
        if len(self.mock_responses) == 0:
            raise RuntimeError(
                f"Exhausted all mock responses, but need to reply to message: {kwargs.get('messages', [])[-1]}"
            )
        response, *self._mock_responses = self.mock_responses
        if response == "RealResponse":
            response = self.real_client.converse_stream(**kwargs)
        else:
            response = self._response_as_stream(response)
        return response

    def _response_as_stream(self, response: "ConverseResponseTypeDef"):
        def event_stream():
            has_tool_output = False
            output = response["output"]
            if output and "message" in output:
                for index, content in enumerate(output["message"]["content"]):
                    if "toolUse" in content:
                        has_tool_output = True
                        yield {
                            "contentBlockStart": {
                                "start": {
                                    "toolUse": {
                                        "toolUseId": content["toolUse"].pop(
                                            "toolUseId"
                                        ),
                                        "name": content["toolUse"].pop("name"),
                                    }
                                },
                                "contentBlockIndex": index,
                            }
                        }
                        content["toolUse"]["input"] = json.dumps(content["toolUse"]["input"])  # type: ignore
                    elif "reasoningContent" in content:
                        content["reasoningContent"] = content["reasoningContent"]["reasoningText"]  # type: ignore
                    yield {"messageStart": {"role": "assistant"}}
                    if "text" in content:
                        for chunk in self._chunk_text(content["text"]):
                            if self.stream_delay_between_tokens is not None:
                                time.sleep(self.stream_delay_between_tokens)
                            yield {
                                "contentBlockDelta": {
                                    "delta": {"text": chunk},
                                    "contentBlockIndex": index,
                                }
                            }
                    else:
                        yield {
                            "contentBlockDelta": {
                                "delta": content,
                                "contentBlockIndex": index,
                            }
                        }
                    yield {"contentBlockStop": {"contentBlockIndex": index}}
            yield {
                "messageStop": {
                    "stopReason": "tool_use" if has_tool_output else "end_turn"
                }
            }
            yield {
                "metadata": {
                    "usage": {
                        "inputTokens": 431,
                        "outputTokens": 168,
                        "totalTokens": 599,
                    },
                    "metrics": {"latencyMs": 4600},
                }
            }

        request_id = uuid4()
        stream_response: ConverseStreamResponseTypeDef = {
            "ResponseMetadata": {
                "RequestId": request_id.hex,
                "HTTPStatusCode": 200,
                "HTTPHeaders": {
                    "date": datetime.now(UTC).strftime(
                        "%a, %d %b %Y %H:%M:%S GMT"  # Tue, 11 Mar 2025 13:58:48 GMT
                    ),
                    "content-type": "application/vnd.amazon.eventstream",
                    "transfer-encoding": "chunked",
                    "connection": "keep-alive",
                    "x-amzn-requestid": request_id.hex,
                    "x-mocked-response": "true",
                },
                "RetryAttempts": 0,
            },
            "stream": cast("EventStream[ConverseStreamOutputTypeDef]", event_stream()),
        }
        return stream_response

    def client(self):
        mock_client = Mock(name="MockClient")
        mock_client.converse = self._converse
        mock_client.converse_stream = self._converse_stream
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
        text_output: str | Sequence[str] | None = None,
        *,
        tool_use_output: ToolUseOutput | Sequence[ToolUseOutput] | None = None,
        reasoning_output: str | Sequence[str] | None = None,
    ):
        if tool_use_output is not None and isinstance(tool_use_output, Mapping):
            tool_use_output = [tool_use_output]
        if text_output is not None and isinstance(text_output, str):
            text_output = [text_output]
        tool_uses: list[ContentBlockOutputTypeDef] = [
            {"toolUse": {"toolUseId": uuid4().hex, "input": {}, **t}}
            for t in (tool_use_output or [])
        ]
        texts: list[ContentBlockOutputTypeDef] = [
            {"text": t} for t in (text_output or [])
        ]
        reasoning_outputs: list[ContentBlockOutputTypeDef] = [
            {
                "reasoningContent": {
                    "reasoningText": {
                        "text": t,
                        "signature": hashlib.sha256(t.encode()).hexdigest(),
                    }
                }
            }
            for t in (reasoning_output or [])
        ]
        self.mock_responses.append(
            self._get_raw_response(
                [
                    *reasoning_outputs,
                    *texts,
                    *tool_uses,
                ]
            )
        )

    def add_real_response(self):
        self.mock_responses.append("RealResponse")

    @staticmethod
    def _chunk_text(t: str, chunk_size=5):
        while t:
            yield t[:chunk_size]
            t = t[chunk_size:]


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
            reasoning_texts = []
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
                elif "reasoningContent" in msg:
                    if "reasoningText" in msg["reasoningContent"]:
                        reasoning_texts.append(msg["reasoningContent"]["reasoningText"])
            print(
                f"add_output(text_output={texts},tool_use_output={tool_uses}),reasoning_output={reasoning_texts}"
            )
