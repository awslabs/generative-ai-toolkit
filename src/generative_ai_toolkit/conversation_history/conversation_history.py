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

import os
import sqlite3
import threading
from collections.abc import Hashable, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, Unpack

import boto3
import boto3.session
from boto3.dynamodb.conditions import Key

if TYPE_CHECKING:
    from mypy_boto3_bedrock_runtime.type_defs import MessageUnionTypeDef

from generative_ai_toolkit.context import AuthContext
from generative_ai_toolkit.utils.dynamodb import DynamoDbMapper
from generative_ai_toolkit.utils.json import JsonBytes
from generative_ai_toolkit.utils.ulid import Ulid


class ConversationHistory(Protocol):

    @property
    def conversation_id(self) -> str:
        """
        The current conversation id
        """
        ...

    @property
    def subcontext_id(self) -> str | None:
        """
        The current subcontext id (if any).

        Within an end-to-end conversation with a user (identified by the externally known conversation id), a supervisor agent may initiate sub-conversations with other agents.
        Those sub-conversations share the same conversation id, but must be distinguished from each other (see example below). For for this, the subcontext_id is used.

        Consider the example of a supervisor agent that uses a web research agent. The supervisor may wish to execute multiple distinct web research activities.
        Thus, it may invoke the web research agent multiple times. Each invocation must have its own conversational memory, because:

        - The invocations may be in parallel, and sharing conversational memory would lead to a mix up of messages
        - Even if not parallel, subsequent invocations shouldn't (per se) be burdened by carrying the conversational history of prior invocations.

        By using a unique subcontext_id for each subagent invocation, the conversational messages (between supervisor and subagent) of that invocation can be kept separate.
        This does mean that subagents must return the subcontext_id as metadata together with their response, so that the supervisor agent can provide it alongside follow up questions/instructions.
        Such follow up questions/instructions **should** of course use the conversational history of the prior invocation (or the subagent wouldn't understand them).
        """
        ...

    def set_conversation_id(
        self, conversation_id: str, *, subcontext_id: str | None = None
    ) -> None:
        """
        Set the current conversation id and subcontext_id (if any).
        """
        ...

    @property
    def auth_context(self) -> AuthContext:
        """
        The current auth context
        """
        ...

    def set_auth_context(self, **auth_context: Unpack[AuthContext]) -> None:
        """
        Set the current auth context
        """
        ...

    @property
    def context_key(self) -> Hashable:
        """
        Unique reference to the current context that consists of the auth_context, conversation_id, subcontext_id (if any),
        and any other identifiers the implementation wishes to add to be able to uniquely reference a context
        across all instances of the ConversationHistory implementation.
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


class BaseConversationHistory(ConversationHistory):

    def __init__(self) -> None:
        super().__init__()
        self._conversation_id = Ulid().ulid
        self._subcontext_id = None
        self._auth_context: AuthContext = {"principal_id": None}

    @property
    def conversation_id(self) -> str:
        return self._conversation_id

    @property
    def subcontext_id(self) -> str | None:
        return self._subcontext_id

    def set_conversation_id(
        self, conversation_id: str, *, subcontext_id: str | None = None
    ) -> None:
        self._conversation_id = conversation_id
        self._subcontext_id = subcontext_id

    @property
    def auth_context(self) -> AuthContext:
        return self._auth_context

    def set_auth_context(self, **auth_context: Unpack[AuthContext]) -> None:
        self._auth_context = auth_context

    @property
    def context_key(self) -> Hashable:
        return (
            self._auth_context["principal_id"],
            self._conversation_id,
            self._subcontext_id,
        )

    @property
    def messages(self) -> Sequence["MessageUnionTypeDef"]:
        raise NotImplementedError

    def add_message(self, msg: "MessageUnionTypeDef") -> None:
        raise NotImplementedError

    def reset(self) -> None:
        self._conversation_id = Ulid().ulid
        self._subcontext_id = None


class InMemoryConversationHistory(BaseConversationHistory):
    _message_cache: dict[
        str | None, dict[str, dict[str | None, list["MessageUnionTypeDef"]]]
    ]

    def __init__(
        self,
    ) -> None:
        super().__init__()
        self._message_cache = {}

    @property
    def context_key(self) -> Hashable:
        return (id(self), super().context_key)

    def add_message(self, msg: "MessageUnionTypeDef") -> None:
        self._message_cache.setdefault(
            self._auth_context["principal_id"], {}
        ).setdefault(self._conversation_id, {}).setdefault(
            self._subcontext_id, []
        ).append(
            msg
        )

    @property
    def messages(self) -> Sequence["MessageUnionTypeDef"]:
        return (
            self._message_cache.get(self._auth_context["principal_id"], {})
            .get(self._conversation_id, {})
            .get(self._subcontext_id, [])
        )


class SqliteConversationHistory(BaseConversationHistory):

    def __init__(
        self,
        *,
        db_path: str | Path | None = None,
        identifier: str | None = None,
        create_tables: bool = True,
    ):
        super().__init__()
        self.db_path = (
            Path(db_path)
            if db_path is not None
            else Path(os.getcwd()) / "conversations.db"
        )
        self.identifier = identifier
        if create_tables:
            self._create_tables()

    def _create_tables(self) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    conversation_id TEXT NOT NULL,
                    identifier TEXT,
                    subcontext_id TEXT,
                    timestamp_ms INTEGER NOT NULL,
                    created_at TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content BLOB NOT NULL
                )
                """
            )

            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_messages_conversation
                ON messages (conversation_id, identifier, subcontext_id, timestamp_ms)
                """
            )

            conn.commit()

    def __repr__(self) -> str:
        return f"SqliteConversationHistory(db_path={self.db_path}, identifier={self.identifier})"

    @property
    def context_key(self) -> Hashable:
        return (self.db_path, super().context_key)

    def add_message(self, msg: "MessageUnionTypeDef") -> None:
        now = datetime.now(UTC)
        timestamp_ms = int(now.timestamp() * 1000)

        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT INTO messages
                (conversation_id, identifier, subcontext_id, timestamp_ms, created_at, role, content)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    self._conversation_id,
                    self.identifier,
                    self._subcontext_id,
                    timestamp_ms,
                    now.isoformat(),
                    msg["role"],
                    JsonBytes.dumps(msg["content"]),
                ),
            )
            conn.commit()

    @property
    def messages(self) -> Sequence["MessageUnionTypeDef"]:
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row

            cursor = conn.execute(
                """
                SELECT role, content
                FROM messages
                WHERE conversation_id = ?
                    AND (identifier IS ? OR (identifier IS NULL AND ? IS NULL))
                    AND (subcontext_id IS ? OR (subcontext_id IS NULL AND ? IS NULL))
                ORDER BY timestamp_ms ASC
                """,
                (
                    self._conversation_id,
                    self.identifier,
                    self.identifier,
                    self._subcontext_id,
                    self._subcontext_id,
                ),
            )

            messages = []
            for row in cursor.fetchall():

                messages.append(
                    {
                        "role": row["role"],
                        "content": JsonBytes.loads(row["content"]),
                    }
                )

            return messages


class DynamoDbConversationHistory(BaseConversationHistory):

    def __init__(
        self,
        table_name: str,
        *,
        session: boto3.session.Session | None = None,
        identifier: str | None = None,
    ) -> None:
        super().__init__()
        self.table_name = table_name
        self.session = session
        self._locals = threading.local()
        if identifier is not None and "#" in identifier:
            raise ValueError(
                "DynamoDbConversationHistory cannot use # character in identifier"
            )
        self.identifier = identifier

    @property
    def table(self):
        """Thread-local Table resource for thread safety"""
        if not hasattr(self._locals, "table"):
            self._locals.table = (
                (self.session or boto3).resource("dynamodb").Table(self.table_name)
            )
        return self._locals.table

    def __repr__(self) -> str:
        return f"DynamoDbConversationHistory(table_name={self.table_name}, identifier={self.identifier})"

    def set_auth_context(self, **auth_context: Unpack[AuthContext]) -> None:
        principal_id = auth_context.get("principal_id")
        if principal_id and isinstance(principal_id, str) and "#" in principal_id:
            raise ValueError(
                "DynamoDbConversationHistory cannot use # character in principal_id"
            )
        super().set_auth_context(**auth_context)

    def set_conversation_id(
        self, conversation_id: str, *, subcontext_id: str | None = None
    ) -> None:
        if "#" in conversation_id or (
            subcontext_id is not None and "#" in subcontext_id
        ):
            raise ValueError(
                "DynamoDbConversationHistory cannot use # character in conversation_id or subcontext_id"
            )
        super().set_conversation_id(conversation_id, subcontext_id=subcontext_id)

    @property
    def context_key(self) -> Hashable:
        return (self.table.table_arn, self.identifier, super().context_key)

    def add_message(self, msg: "MessageUnionTypeDef") -> None:
        now = datetime.now(UTC)
        item = {
            "pk": f"CONV#{self._auth_context["principal_id"] or "_"}#{self._conversation_id}",
            "sk": f"MSG#{self.identifier or "_"}#{self._subcontext_id or "_"}#{int(now.timestamp() * 1000):014x}",
            "created_at": now.isoformat(),
            "conversation_id": self._conversation_id,
            "subcontext_id": self._subcontext_id,
            "role": msg["role"],
            "content": msg["content"],
            "auth_context": self._auth_context,
            "identifier": self.identifier,
        }
        try:
            self.table.put_item(
                Item=DynamoDbMapper.serialize(item),
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
                        f"CONV#{self._auth_context["principal_id"] or "_"}#{self._conversation_id}"
                    )
                    & Key("sk").begins_with(
                        f"MSG#{self.identifier or "_"}#{self._subcontext_id or "_"}#"
                    ),
                    **last_evaluated_key_param,
                    ConsistentRead=True,
                )
            except self.table.meta.client.exceptions.ResourceNotFoundException as e:
                raise ValueError(f"Table {self.table.name} does not exist") from e
            collected.extend(
                [
                    {
                        "role": item["role"],
                        "content": DynamoDbMapper.deserialize(item["content"]),
                    }
                    for item in response["Items"]
                ]
            )
            if "LastEvaluatedKey" not in response:
                return collected
            last_evaluated_key_param = {
                "ExclusiveStartKey": response["LastEvaluatedKey"]
            }
