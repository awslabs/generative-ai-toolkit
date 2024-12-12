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

from dataclasses import dataclass, field
from datetime import datetime, timezone
import sys
import json
import textwrap
from typing import (
    Any,
    Iterable,
    Literal,
    Mapping,
    Protocol,
    Sequence,
    TypeVar,
    TypedDict,
    TypeGuard,
)
from collections import defaultdict

import boto3.session
import boto3
from boto3.dynamodb.conditions import Key

from generative_ai_toolkit.utils.dynamodb import DynamoDbMapper
from generative_ai_toolkit.utils.typings import (
    LlmRequest,
    NonStreamingResponse,
    ToolUseResultStatus,
    ToolUseResultContent,
    Message,
)
from generative_ai_toolkit.utils.ulid import Ulid


class ToolRequestTrace(TypedDict):
    tool_name: str
    tool_input: dict
    tool_use_id: str


class ToolResponseTrace(TypedDict):
    tool_response: Any
    latency_ms: int


RequestTrace = LlmRequest | ToolRequestTrace
ResponseTrace = NonStreamingResponse | ToolResponseTrace


class ConversationMessage(TypedDict):
    role: Literal["user", "assistant"]
    text: str


class ToolUse(TypedDict):
    tool_name: str
    input: dict[str, Any]


class ToolUseWithOutput(ToolUse):
    status: ToolUseResultStatus
    output: ToolUseResultContent


@dataclass(slots=True, frozen=True, kw_only=True)
class BaseTrace:
    conversation_id: str
    trace_id: Ulid
    auth_context: str | None = None
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    additional_info: dict[str, Any] = field(default_factory=dict)

    def __repr__(self) -> str:
        return f"Trace(to={getattr(self, "to", "??")}, conversation_id={self.conversation_id}, trace_id={self.trace_id})"

    def __str__(self) -> str:
        return textwrap.dedent(
            f"""
            ======================================
            {getattr(self, "to", "??")} TRACE ({self.__class__.__name__})
            ======================================
            To:              {getattr(self, "to", "??")}
            Trace ID:        {self.trace_id}
            Conversation ID: {self.conversation_id}
            Auth context:    {self.auth_context}
            Created at:      {self.created_at}
            Additional info:
              {json.dumps(self.additional_info, separators=(",", ":"))}
            """
        ).strip()


def user_conversation_from_messages(messages: Iterable[Message]):
    user_conversation: list[ConversationMessage] = []

    for msg in messages:
        texts: list[str] = []
        for part in msg["content"]:
            if "text" not in part:
                continue
            texts.append(part["text"])
        if texts:
            text = "\n".join(texts)
            if user_conversation and user_conversation[-1]["role"] == msg["role"]:
                user_conversation[-1]["text"] += "\n" + text
            else:
                user_conversation.append({"role": msg["role"], "text": text})

    return user_conversation


def tool_invocations_from_messages(messages: Iterable[Message]):
    tool_invocations: dict[str, ToolUse | ToolUseWithOutput] = {}

    for msg in messages:
        if msg["role"] == "assistant":
            for item in msg["content"]:
                if "toolUse" in item:
                    tool_use = item["toolUse"]
                    tool_invocations[tool_use["toolUseId"]] = ToolUse(
                        tool_name=tool_use["name"],
                        input=tool_use["input"],
                    )
        if msg["role"] == "user":
            for item in msg["content"]:
                if "toolResult" in item:
                    tool_result = item["toolResult"]
                    tool_use = tool_invocations[tool_result["toolUseId"]]
                    tool_invocations[tool_result["toolUseId"]] = ToolUseWithOutput(
                        tool_name=tool_use["tool_name"],
                        input=tool_use["input"],
                        status=tool_result["status"],
                        output=tool_result["content"],
                    )

    return list(tool_invocations.values())


@dataclass(slots=True, frozen=True, repr=False)
class LlmTrace(BaseTrace):
    request: LlmRequest
    response: NonStreamingResponse
    to: Literal["LLM"] = "LLM"

    @property
    def user_conversation(self):
        """
        All that was said between agent and user, as captured by this trace
        """

        return user_conversation_from_messages(
            (*self.request["messages"], self.response["output"]["message"])
        )

    @property
    def tool_invocations(self):
        """
        All tool invocations, as captured by this trace
        """

        return tool_invocations_from_messages(
            (*self.request["messages"], self.response["output"]["message"])
        )

    def __str__(self) -> str:
        return (
            textwrap.dedent(
                """
                {base}
                Request messages:
                  {req}
                Response message:
                  {res}
                Request (full):
                  {req_full}
                Response (full):
                  {res_full}
                """
            )
            .format(
                to=self.to,
                base=BaseTrace.__str__(self),
                req=self.request["messages"][-1]["content"][0],
                req_full=json.dumps(self.request, separators=(", ", ":")),
                res=self.response["output"]["message"]["content"],
                res_full=json.dumps(self.response, separators=(", ", ":")),
            )
            .strip()
        )


@dataclass(slots=True, frozen=True, repr=False)
class ToolTrace(BaseTrace):
    request: ToolRequestTrace
    response: ToolResponseTrace
    to: Literal["TOOL"] = "TOOL"

    def __str__(self) -> str:
        return (
            textwrap.dedent(
                """
                {base}
                Request:
                  {req}
                Response:
                  {res}
                """
            )
            .format(
                to=self.to,
                base=BaseTrace.__str__(self),
                req=json.dumps(self.request, separators=(", ", ":")),
                res=json.dumps(self.response, separators=(", ", ":")),
            )
            .strip()
        )


Trace = LlmTrace | ToolTrace


def is_llm_request(req: RequestTrace) -> TypeGuard[LlmRequest]:
    return "messages" in req


def is_tool_request(req: RequestTrace) -> TypeGuard[ToolRequestTrace]:
    return "tool_name" in req


def is_llm_response(res: ResponseTrace) -> TypeGuard[NonStreamingResponse]:
    return "output" in res


def is_tool_response(res: ResponseTrace) -> TypeGuard[ToolResponseTrace]:
    return "tool_response" in res


class AgentTracer(Protocol):
    def trace(
        self,
        conversation_id: str,
        to: Literal["LLM", "TOOL"],
        req: RequestTrace,
        res: ResponseTrace,
        auth_context: str | None = None,
    ) -> None: ...

    def get_traces(
        self, conversation_id: str, auth_context: str | None
    ) -> Sequence[Trace]: ...

    def set_additional_info(self, additional_info: Mapping[str, Any]): ...


class InMemoryAgentTracer(AgentTracer):
    _traces: dict[str | None, dict[str, list[Trace]]]
    additional_info: dict[str, Any]

    def __init__(self) -> None:
        self._traces = defaultdict(lambda: defaultdict(list))
        self.additional_info = {}

    def trace(
        self,
        conversation_id: str,
        to: Literal["LLM", "TOOL"],
        req: RequestTrace,
        res: ResponseTrace,
        auth_context: str | None = None,
    ):
        ulid = Ulid()
        request = deepcopy(req)
        response = deepcopy(res)
        trace: Trace
        if to == "LLM" and is_llm_request(request) and is_llm_response(response):
            trace = LlmTrace(
                conversation_id=conversation_id,
                created_at=ulid.timestamp,
                trace_id=ulid,
                request=request,
                response=response,
                additional_info=self.additional_info,
            )
        elif to == "TOOL" and is_tool_request(request) and is_tool_response(response):
            trace = ToolTrace(
                conversation_id=conversation_id,
                created_at=ulid.timestamp,
                trace_id=ulid,
                request=request,
                response=response,
                additional_info=self.additional_info,
            )
        else:
            raise ValueError("Invalid trace")
        self._traces[auth_context][conversation_id].append(trace)

    def get_traces(self, conversation_id: str, auth_context: str | None = None):
        return self._traces.get(auth_context, {}).get(conversation_id, [])

    def set_additional_info(self, additional_info: Mapping[str, Any]):
        self.additional_info = dict(deepcopy(additional_info))


class SingleConversationTracer(InMemoryAgentTracer):
    conversation_id: str | None

    def __init__(self) -> None:
        super().__init__()
        self.conversation_id = None

    def trace(
        self,
        conversation_id: str,
        to: Literal["LLM", "TOOL"],
        req: RequestTrace,
        res: ResponseTrace,
        auth_context: str | None = None,
    ):
        if conversation_id != self.conversation_id:
            self._traces = defaultdict(
                lambda: defaultdict(list)
            )  # clear traces from prior conversation_id
            self.conversation_id = conversation_id
        super().trace(
            conversation_id=conversation_id,
            to=to,
            req=req,
            res=res,
            auth_context=auth_context,
        )


class NoopAgentTracer(AgentTracer):
    def trace(
        self,
        conversation_id: str,
        to: Literal["LLM", "TOOL"],
        req: RequestTrace,
        res: ResponseTrace,
        auth_context: str | None = None,
    ):
        pass

    def get_traces(self, conversation_id: str, auth_context: str | None = None):
        raise NotImplementedError(
            f"You're using the {self.__class__.__name__} which doesn't support retrieving past traces"
        )

    def set_additional_info(self, additional_info: Mapping[str, Any]):
        pass


class DynamoDbAgentTracer(AgentTracer):
    additional_info: dict[str, Any]

    def __init__(
        self, table_name: str, session: boto3.session.Session | None = None
    ) -> None:
        self.table = (session or boto3).resource("dynamodb").Table(table_name)
        self.additional_info = {}

    def trace(
        self,
        conversation_id: str,
        to: Literal["LLM", "TOOL"],
        req: RequestTrace,
        res: ResponseTrace,
        auth_context: str | None = None,
    ):
        now = datetime.now(timezone.utc)
        ulid = Ulid()
        trace: Trace
        if to == "LLM" and is_llm_request(req) and is_llm_response(res):
            trace = LlmTrace(
                conversation_id=conversation_id,
                created_at=now,
                trace_id=ulid,
                request=DynamoDbMapper.to_dynamo(req),
                response=DynamoDbMapper.to_dynamo(res),
                additional_info=self.additional_info,
                auth_context=auth_context,
            )
        elif to == "TOOL" and is_tool_request(req) and is_tool_response(res):
            trace = ToolTrace(
                conversation_id=conversation_id,
                created_at=now,
                trace_id=ulid,
                request=DynamoDbMapper.to_dynamo(req),
                response=DynamoDbMapper.to_dynamo(res),
                additional_info=self.additional_info,
                auth_context=auth_context,
            )
        else:
            raise ValueError("Invalid trace")
        try:
            self.table.put_item(
                Item={
                    "pk": f"CONV#{auth_context or '_'}#{conversation_id}",
                    "sk": f"TRACE#{trace.trace_id}",
                    **DynamoDbMapper.to_dynamo(
                        {
                            "created_at": trace.created_at,
                            "conversation_id": trace.conversation_id,
                            "trace_id": trace.trace_id.ulid,
                            "to": trace.to,
                            "request": trace.request,
                            "response": trace.response,
                            "additional_info": trace.additional_info,
                            "auth_context": trace.auth_context,
                        }
                    ),
                },
                ConditionExpression="attribute_not_exists(pk) AND attribute_not_exists(sk)",
            )
        except self.table.meta.client.exceptions.ResourceNotFoundException as e:
            raise ValueError(f"Table {self.table.name} does not exist") from e
        except self.table.meta.client.exceptions.ConditionalCheckFailedException as e:
            raise ValueError(f"Trace {trace.trace_id} already exists") from e

    def get_traces(self, conversation_id: str, auth_context: str | None = None):
        items = []
        last_evaluated_key_param: dict[str, Any] = {}
        while True:
            try:
                response = self.table.query(
                    KeyConditionExpression=Key("pk").eq(
                        f"CONV#{auth_context or '_'}#{conversation_id}"
                    )
                    & Key("sk").begins_with("TRACE#"),
                    **last_evaluated_key_param,
                )

            except self.table.meta.client.exceptions.ResourceNotFoundException as e:
                raise ValueError(f"Table {self.table.name} does not exist") from e
            items.extend(response["Items"])
            if "LastEvaluatedKey" not in response:
                break
            last_evaluated_key_param = {
                "ExclusiveStartKey": response["LastEvaluatedKey"]
            }
        return [self.item_to_trace(item) for item in DynamoDbMapper.from_dynamo(items)]

    @staticmethod
    def item_to_trace(item: dict[str, Any]):
        params = {
            "conversation_id": str(item["conversation_id"]),
            "trace_id": Ulid(item["trace_id"]),
            "to": item["to"],
            "request": item["request"],
            "response": item["response"],
            "created_at": datetime.fromisoformat(item["created_at"]),
            "additional_info": item["additional_info"],
            "auth_context": item.get("auth_context"),
        }
        return LlmTrace(**params) if item["to"] == "LLM" else ToolTrace(**params)

    def set_additional_info(self, additional_info: Mapping[str, Any]):
        self.additional_info = dict(deepcopy(additional_info))


class StderrAgentTracer(AgentTracer):
    additional_info: dict[str, Any]

    def __init__(self) -> None:
        self.additional_info = {}

    def trace(
        self,
        conversation_id: str,
        to: Literal["LLM", "TOOL"],
        req,
        res,
        auth_context: str | None = None,
    ):
        print(
            f"TRACE {conversation_id}:{auth_context or '_'}:{to}",
            json.dumps(req, default=str),
            json.dumps(res, default=str),
            json.dumps(self.additional_info, default=str),
            file=sys.stderr,
        )

    def get_traces(self, conversation_id: str, auth_context: str | None = None):
        raise NotImplementedError

    def set_additional_info(self, additional_info: Mapping[str, Any]):
        self.additional_info = dict(deepcopy(additional_info))


T = TypeVar("T", bound=Mapping)


def deepcopy(d: T) -> T:
    """
    A dumb but threadsafe alternative to copy.deepcopy()
    """
    return json.loads(json.dumps(d))
