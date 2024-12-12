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

from typing import Any, Protocol, Sequence
from datetime import datetime, timezone

import boto3.session
import boto3
from boto3.dynamodb.conditions import Key

from generative_ai_toolkit.utils.typings import Message
from generative_ai_toolkit.utils.dynamodb import DynamoDbMapper
from generative_ai_toolkit.utils.ulid import Ulid


class ConversationHistory(Protocol):
    @property
    def conversation_id(self) -> str:
        """
        The current conversation id
        """
        ...

    def set_conversation_id(self, conversation_id: str) -> None:
        """
        Set the current conversation id
        """
        ...

    @property
    def auth_context(self) -> str | None:
        """
        The current auth context
        """
        ...

    def set_auth_context(self, auth_context: str | None) -> None:
        """
        Set the current auth context
        """
        ...

    @property
    def messages(self) -> Sequence[Message]:
        """
        All messages of the current conversation
        """
        ...

    def add_message(self, msg: Message) -> None:
        """
        Add a message to the conversation history
        """
        ...

    def reset(self) -> None:
        """
        Change the conversation id to a new one, and start a new conversation with empty history
        """
        ...


class InMemoryConversationHistory(ConversationHistory):
    _conversation_id: str
    _message_cache: dict[str | None, dict[str, list[Message]]]
    _auth_context: str | None

    def __init__(
        self,
    ) -> None:
        self._conversation_id = Ulid().ulid
        self._auth_context = None
        self._message_cache = {None: {self._conversation_id: []}}

    @property
    def conversation_id(self):
        return self._conversation_id

    def set_conversation_id(self, conversation_id: str):
        self._conversation_id = conversation_id

    @property
    def auth_context(self) -> str | None:
        return self._auth_context

    def set_auth_context(self, auth_context: str | None):
        self._auth_context = auth_context

    def add_message(self, msg: Message) -> None:
        self._message_cache[self._auth_context][self._conversation_id].append(msg)

    @property
    def messages(self) -> Sequence[Message]:
        return self._message_cache[self._auth_context][self._conversation_id]

    def reset(self) -> None:
        self._conversation_id = Ulid().ulid
        self._message_cache = {self._auth_context: {self._conversation_id: []}}


class DynamoDbConversationHistory(ConversationHistory):
    _conversation_id: str
    _auth_context: str | None
    _message_cache: list[Message] | None

    def __init__(
        self,
        table_name: str,
        session: boto3.session.Session | None = None,
    ) -> None:
        self.table = (session or boto3).resource("dynamodb").Table(table_name)
        self._conversation_id = Ulid().ulid
        self._auth_context = None
        self._message_cache = None

    @property
    def conversation_id(self) -> str:
        return self._conversation_id

    def set_conversation_id(self, conversation_id: str):
        self._conversation_id = conversation_id
        self._message_cache = None

    @property
    def auth_context(self) -> str | None:
        return self._auth_context

    def set_auth_context(self, auth_context: str | None):
        self._auth_context = auth_context
        self._message_cache = None

    def add_message(self, msg: Message) -> None:
        try:
            self.table.put_item(
                Item={
                    "pk": f"CONV#{self._auth_context or "_"}#{self.conversation_id}",
                    "sk": f"MSG#{str(len(self.messages) + 1).zfill(3)}",
                    "created_at": datetime.now(timezone.utc).isoformat(),
                    "conversation_id": self.conversation_id,
                    "msg_nr": len(self.messages) + 1,
                    "role": msg["role"],
                    "content": DynamoDbMapper.to_dynamo(msg["content"]),
                    "auth_context": self._auth_context,
                },
                ConditionExpression="attribute_not_exists(pk) AND attribute_not_exists(sk)",
            )
            if not self._message_cache:
                self._message_cache = []
            self._message_cache.append(msg)
        except self.table.meta.client.exceptions.ResourceNotFoundException as e:
            raise ValueError(f"Table {self.table.name} does not exist") from e

    @property
    def messages(self) -> Sequence[Message]:
        if self._message_cache is not None:
            return self._message_cache
        self._message_cache = []
        last_evaluated_key_param: dict[str, Any] = {}
        while True:
            try:
                response = self.table.query(
                    KeyConditionExpression=Key("pk").eq(
                        f"CONV#{self._auth_context or "_"}#{self.conversation_id}"
                    )
                    & Key("sk").begins_with("MSG#"),
                    **last_evaluated_key_param,
                    ConsistentRead=True,
                )
            except self.table.meta.client.exceptions.ResourceNotFoundException as e:
                raise ValueError(f"Table {self.table.name} does not exist") from e
            self._message_cache.extend(
                [
                    Message(
                        role=item["role"],
                        content=DynamoDbMapper.from_dynamo(item["content"]),
                    )
                    for item in DynamoDbMapper.from_dynamo(response["Items"])
                ]
            )
            if "LastEvaluatedKey" not in response:
                return self._message_cache
            last_evaluated_key_param = {
                "ExclusiveStartKey": response["LastEvaluatedKey"]
            }

    def reset(self) -> None:
        self._conversation_id = Ulid().ulid
        self._message_cache = None
