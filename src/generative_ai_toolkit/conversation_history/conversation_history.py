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
from typing import TYPE_CHECKING, Any, Protocol

import boto3
import boto3.session
from boto3.dynamodb.conditions import Key

if TYPE_CHECKING:
    from mypy_boto3_bedrock_runtime.type_defs import MessageUnionTypeDef

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
    def messages(self) -> Sequence["MessageUnionTypeDef"]:
        """
        All messages of the current conversation
        """
        ...

    def add_message(self, msg: "MessageUnionTypeDef") -> None:
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
    _message_cache: dict[str | None, dict[str, list["MessageUnionTypeDef"]]]
    _auth_context: str | None

    def __init__(
        self,
    ) -> None:
        self._conversation_id = Ulid().ulid
        self._auth_context = None
        self._message_cache = {}

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

    def add_message(self, msg: "MessageUnionTypeDef") -> None:
        self._message_cache.setdefault(self._auth_context, {}).setdefault(
            self._conversation_id, []
        ).append(msg)

    @property
    def messages(self) -> Sequence["MessageUnionTypeDef"]:
        return self._message_cache.get(self._auth_context, {}).get(
            self._conversation_id, []
        )

    def reset(self) -> None:
        self._conversation_id = Ulid().ulid


class DynamoDbConversationHistory(ConversationHistory):
    _conversation_id: str
    _auth_context: str | None

    def __init__(
        self,
        table_name: str,
        *,
        session: boto3.session.Session | None = None,
        identifier: str | None = None,
    ) -> None:
        self.table = (session or boto3).resource("dynamodb").Table(table_name)
        self._conversation_id = Ulid().ulid
        self._auth_context = None
        self.identifier = identifier

    def __repr__(self) -> str:
        return f"DynamoDbConversationHistory(table_name={self.table.name}, identifier={self.identifier})"

    @property
    def conversation_id(self) -> str:
        return self._conversation_id

    def set_conversation_id(self, conversation_id: str):
        self._conversation_id = conversation_id

    @property
    def auth_context(self) -> str | None:
        return self._auth_context

    def set_auth_context(self, auth_context: str | None):
        self._auth_context = auth_context

    def add_message(self, msg: "MessageUnionTypeDef") -> None:
        now = datetime.now(UTC)
        item = {
            "pk": f"CONV#{self._auth_context or "_"}#{self.conversation_id}",
            "sk": f"MSG#{self.identifier or "_"}#{int(now.timestamp() * 1000):014x}",
            "created_at": now.isoformat(),
            "conversation_id": self.conversation_id,
            "role": msg["role"],
            "content": DynamoDbMapper.to_dynamo(msg["content"]),
            "auth_context": self._auth_context,
            "identifier": self.identifier,
        }
        try:
            self.table.put_item(
                Item=item,
                ConditionExpression="attribute_not_exists(pk) AND attribute_not_exists(sk)",
            )
        except self.table.meta.client.exceptions.ConditionalCheckFailedException as e:
            raise ValueError(
                f"The message with pk {item['pk']} and sk {item['sk']} already exists!"
            ) from e
        except self.table.meta.client.exceptions.ResourceNotFoundException as e:
            raise ValueError(f"Table {self.table.name} does not exist") from e

    @property
    def messages(self) -> Sequence["MessageUnionTypeDef"]:
        collected = []
        last_evaluated_key_param: dict[str, Any] = {}
        while True:
            try:
                response = self.table.query(
                    KeyConditionExpression=Key("pk").eq(
                        f"CONV#{self._auth_context or "_"}#{self.conversation_id}"
                    )
                    & Key("sk").begins_with(f"MSG#{self.identifier or "_"}#"),
                    **last_evaluated_key_param,
                    ConsistentRead=True,
                )
            except self.table.meta.client.exceptions.ResourceNotFoundException as e:
                raise ValueError(f"Table {self.table.name} does not exist") from e
            collected.extend(
                [
                    {
                        "role": item["role"],
                        "content": DynamoDbMapper.from_dynamo(item["content"]),
                    }
                    for item in DynamoDbMapper.from_dynamo(response["Items"])
                ]
            )
            if "LastEvaluatedKey" not in response:
                return collected
            last_evaluated_key_param = {
                "ExclusiveStartKey": response["LastEvaluatedKey"]
            }

    def reset(self) -> None:
        self._conversation_id = Ulid().ulid
