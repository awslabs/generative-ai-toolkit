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

from typing import (
    Iterator,
    NotRequired,
    TypedDict,
    Literal,
    Union,
    Dict,
    List,
    Any,
)


class ToolUse(TypedDict):
    toolUseId: str
    name: str


class ToolUseInput(TypedDict):
    input: Any


class TextDelta(TypedDict):
    text: str


class ToolUseDelta(TypedDict):
    toolUse: ToolUseInput


class MessageStart(TypedDict):
    role: Literal["user", "assistant"]


class MessageStartEvent(TypedDict):
    messageStart: MessageStart


class ContentBlockStartStart(TypedDict):
    toolUse: ToolUse


class ContentBlockStart(TypedDict):
    start: ContentBlockStartStart
    contentBlockIndex: int


class ContentBlockStartEvent(TypedDict):
    contentBlockStart: ContentBlockStart


Delta = Union[TextDelta, ToolUseDelta]


class ContentBlockDelta(TypedDict):
    delta: Delta
    contentBlockIndex: int


class ContentBlockDeltaEvent(TypedDict):
    contentBlockDelta: ContentBlockDelta


class ContentBlockStop(TypedDict):
    contentBlockIndex: int


class ContentBlockStopEvent(TypedDict):
    contentBlockStop: ContentBlockStop


class MessageStop(TypedDict):
    stopReason: Literal[
        "end_turn",
        "tool_use",
        "max_tokens",
        "stop_sequence",
        "guardrail_intervened",
        "content_filtered",
    ]
    additionalModelResponseFields: Union[
        Dict[str, Any], List[Any], int, float, str, bool, None
    ]


class MessageStopEvent(TypedDict):
    messageStop: MessageStop


class Usage(TypedDict):
    inputTokens: int
    outputTokens: int
    totalTokens: int


class Latency(TypedDict):
    latencyMs: int


class Topic(TypedDict):
    name: str
    type: Literal["DENY"]
    action: Literal["BLOCKED"]


class Filter(TypedDict):
    type: Literal[
        "INSULTS", "HATE", "SEXUAL", "VIOLENCE", "MISCONDUCT", "PROMPT_ATTACK"
    ]
    confidence: Literal["NONE", "LOW", "MEDIUM", "HIGH"]
    action: Literal["BLOCKED"]


class CustomWord(TypedDict):
    match: str
    action: Literal["BLOCKED"]


class ManagedWordList(TypedDict):
    match: str
    type: Literal["PROFANITY"]
    action: Literal["BLOCKED"]


class PiiEntity(TypedDict):
    match: str
    type: Literal[
        "ADDRESS",
        "AGE",
        "AWS_ACCESS_KEY",
        "AWS_SECRET_KEY",
        "CA_HEALTH_NUMBER",
        "CA_SOCIAL_INSURANCE_NUMBER",
        "CREDIT_DEBIT_CARD_CVV",
        "CREDIT_DEBIT_CARD_EXPIRY",
        "CREDIT_DEBIT_CARD_NUMBER",
        "DRIVER_ID",
        "EMAIL",
        "INTERNATIONAL_BANK_ACCOUNT_NUMBER",
        "IP_ADDRESS",
        "LICENSE_PLATE",
        "MAC_ADDRESS",
        "NAME",
        "PASSWORD",
        "PHONE",
        "PIN",
        "SWIFT_CODE",
        "UK_NATIONAL_HEALTH_SERVICE_NUMBER",
        "UK_NATIONAL_INSURANCE_NUMBER",
        "UK_UNIQUE_TAXPAYER_REFERENCE_NUMBER",
        "URL",
        "USERNAME",
        "US_BANK_ACCOUNT_NUMBER",
        "US_BANK_ROUTING_NUMBER",
        "US_INDIVIDUAL_TAX_IDENTIFICATION_NUMBER",
        "US_PASSPORT_NUMBER",
        "US_SOCIAL_SECURITY_NUMBER",
        "VEHICLE_IDENTIFICATION_NUMBER",
    ]
    action: Literal["ANONYMIZED", "BLOCKED"]


class Regex(TypedDict):
    name: str
    match: str
    regex: str
    action: Literal["ANONYMIZED", "BLOCKED"]


class InputAssessment(TypedDict):
    topicPolicy: Dict[str, List[Topic]]
    contentPolicy: Dict[str, List[Filter]]
    wordPolicy: Dict[str, Union[List[CustomWord], List[ManagedWordList]]]
    sensitiveInformationPolicy: Dict[str, Union[List[PiiEntity], List[Regex]]]


class Guardrail(TypedDict):
    modelOutput: List[str]
    inputAssessment: Dict[str, InputAssessment]
    outputAssessments: Dict[str, List[InputAssessment]]


class Trace(TypedDict):
    guardrail: Guardrail


class Metrics(TypedDict):
    latencyMs: int


class Metadata(TypedDict):
    usage: Usage
    metrics: Metrics
    trace: Trace


class MetadataEvent(TypedDict):
    metadata: Metadata


class InternalServerException(TypedDict):
    message: str


class InternalServerExceptionEvent(TypedDict):
    internalServerException: InternalServerException


class ModelStreamErrorException(TypedDict):
    message: str
    originalStatusCode: int
    originalMessage: str


class ModelStreamErrorExceptionEvent(TypedDict):
    modelStreamErrorException: ModelStreamErrorException


class ValidationException(TypedDict):
    message: str


class ValidationExceptionEvent(TypedDict):
    validationException: ValidationException


class ThrottlingException(TypedDict):
    message: str


class ThrottlingExceptionEvent(TypedDict):
    throttlingException: ThrottlingException


class StreamingResponse(TypedDict):
    ResponseMetadata: "ResponseMetadata"
    stream: Iterator[
        Union[
            MessageStartEvent,
            ContentBlockStartEvent,
            ContentBlockDeltaEvent,
            ContentBlockStopEvent,
            MessageStopEvent,
            MetadataEvent,
            InternalServerExceptionEvent,
            ModelStreamErrorExceptionEvent,
            ValidationExceptionEvent,
            ThrottlingExceptionEvent,
        ]
    ]


##################
# Non streaming
##################


class ResponseMetadata(TypedDict):
    RequestId: str
    HTTPStatusCode: int
    HTTPHeaders: Dict[str, str]
    RetryAttempts: int


class ToolUseWithInput(TypedDict):
    toolUseId: str
    name: str
    input: Dict[str, Any]


class TextContent(TypedDict):
    text: str


class ToolUseContent(TypedDict):
    toolUse: ToolUseWithInput


class ToolResultJsonContent(TypedDict):
    json: dict[str, Any]


class ToolResultTextContent(TypedDict):
    text: str


ToolUseResultStatus = Literal["success", "error"]

ToolUseResultContent = List[Union[ToolResultJsonContent, ToolResultTextContent]]


class ToolResult(TypedDict):
    toolUseId: str
    status: ToolUseResultStatus
    content: ToolUseResultContent


class ToolResultContent(TypedDict):
    toolResult: ToolResult


class EmptyContent(TypedDict):
    pass


MessageContent = Union[TextContent, ToolUseContent, ToolResultContent, EmptyContent]


class Message(TypedDict):
    role: Literal["user", "assistant"]
    content: List[MessageContent]


class Output(TypedDict):
    message: Message


class NonStreamingResponse(TypedDict):
    ResponseMetadata: ResponseMetadata
    output: Output
    stopReason: Literal[
        "end_turn",
        "tool_use",
        "max_tokens",
        "stop_sequence",
        "guardrail_intervened",
        "content_filtered",
    ]
    usage: Usage
    metrics: Latency


class LlmRequest(TypedDict):
    modelId: str
    inferenceConfig: NotRequired[dict]
    messages: list[Message]
    system: NotRequired[list[TextContent]]
    toolConfig: NotRequired[dict]


##################
# Toolspec
##################


class InputSchemaProperty(TypedDict):
    type: str
    description: str


class InputSchemaJson(TypedDict):
    type: Literal["object"]
    properties: dict[str, InputSchemaProperty]
    required: NotRequired[List[str]]


class InputSchema(TypedDict):
    json: InputSchemaJson


class ToolSpecFields(TypedDict):
    name: str
    description: str
    inputSchema: InputSchema


class ToolSpec(TypedDict):
    toolSpec: ToolSpecFields
