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

import datetime
import os
import sqlite3
import textwrap
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, Unpack, runtime_checkable

import boto3
import boto3.session
from botocore.config import Config

from generative_ai_toolkit.context import AuthContext
from generative_ai_toolkit.test import user_conversation_from_messages
from generative_ai_toolkit.utils.llm_response import get_text

if TYPE_CHECKING:
    from mypy_boto3_bedrock_runtime import BedrockRuntimeClient
    from mypy_boto3_bedrock_runtime.type_defs import MessageUnionTypeDef


class ConversationDescriber(Protocol):
    def __call__(self, messages: "Sequence[MessageUnionTypeDef]") -> str: ...


@dataclass
class Conversation:
    conversation_id: str
    description: str
    updated_at: datetime.datetime


@dataclass
class ConversationPage:
    conversations: Sequence[Conversation] = ()
    next_page_token: Any | None = None


@runtime_checkable
class ConversationList(Protocol):

    @property
    def page_size(self) -> int:
        """
        Get the page size
        """
        ...

    def set_page_size(self, page_size: int) -> None:
        """
        Set the page size
        """
        ...

    def add_conversation(
        self, conversation_id: str, messages: "Sequence[MessageUnionTypeDef]"
    ) -> Conversation:
        """
        Describe and add a conversation to the conversation list

        If the given conversation_id is already in the list, it will be overwritten.
        """
        ...

    def get_conversation(self, conversation_id: str) -> Conversation | None:
        """
        Get a conversation from the conversation list

        Returns None if the conversation does not exist
        """
        ...

    def remove_conversation(self, conversation_id: str) -> None:
        """
        Remove a conversation from the conversation list
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

    def get_conversations(self, next_page_token: Any | None = None) -> ConversationPage:
        """
        Fetch all conversations (paginated)
        """
        ...


class SqliteConversationList(ConversationList):
    """
    SQLite-based implementation of ConversationList for local single-user development.

    This implementation stores only conversation descriptions in a single SQLite table
    without any user/principal isolation.
    """

    def __init__(
        self,
        *,
        describer: ConversationDescriber,
        db_path: str | Path | None = None,
        page_size: int = 20,
        create_tables: bool = True,
    ):
        self.db_path = (
            Path(db_path)
            if db_path is not None
            else Path(os.getcwd()) / "conversations.db"
        )
        self.describer = describer
        self._page_size = page_size
        self._auth_context: AuthContext = {"principal_id": None}

        if create_tables:
            self._create_tables()

    def _create_tables(self) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS conversations (
                    conversation_id TEXT PRIMARY KEY,
                    description TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )

            # Create index for better query performance
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_conversations_updated
                ON conversations (updated_at DESC)
                """
            )

            conn.commit()

    @property
    def page_size(self) -> int:
        """
        Get the page size
        """
        return self._page_size

    def set_page_size(self, page_size: int) -> None:
        """
        Set the page size
        """
        self._page_size = page_size

    @property
    def auth_context(self) -> AuthContext:
        """The current auth context."""
        return self._auth_context

    def set_auth_context(self, **auth_context: Unpack[AuthContext]) -> None:
        """Set the current auth context."""
        self._auth_context = auth_context

    def add_conversation(
        self, conversation_id: str, messages: Sequence["MessageUnionTypeDef"]
    ) -> Conversation:
        if not messages:
            raise ValueError("Cannot add conversation with empty messages list")

        # Generate description using the provided describer
        description = self.describer(messages)
        now = datetime.datetime.now(datetime.UTC)

        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO conversations
                (conversation_id, description, updated_at)
                VALUES (?, ?, ?)
                """,
                (conversation_id, description, now.isoformat()),
            )

            conn.commit()

        return Conversation(
            conversation_id=conversation_id, description=description, updated_at=now
        )

    def get_conversation(self, conversation_id: str) -> Conversation | None:
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row

            cursor = conn.execute(
                """
                SELECT conversation_id, description, updated_at
                FROM conversations
                WHERE conversation_id = ?
                """,
                (conversation_id,),
            )

            row = cursor.fetchone()
            if not row:
                return None

            return Conversation(
                conversation_id=row["conversation_id"],
                description=row["description"],
                updated_at=datetime.datetime.fromisoformat(row["updated_at"]),
            )

    def remove_conversation(self, conversation_id: str) -> None:
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                """
                DELETE FROM conversations WHERE conversation_id = ?
                """,
                (conversation_id,),
            )

            if cursor.rowcount == 0:
                raise ValueError(f"Conversation {conversation_id} not found")

            conn.commit()

    def get_conversations(self, next_page_token: Any | None = None) -> ConversationPage:
        page_nr = int(next_page_token) if next_page_token else 0
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row

            cursor = conn.execute(
                """
                SELECT conversation_id, description, updated_at
                FROM conversations
                ORDER BY updated_at DESC
                LIMIT ? OFFSET ?
                """,
                (self.page_size + 1, page_nr * self.page_size),
            )

            rows = cursor.fetchall()
            has_next_page = len(rows) > self.page_size

            conversations = [
                Conversation(
                    conversation_id=row["conversation_id"],
                    description=row["description"],
                    updated_at=datetime.datetime.fromisoformat(row["updated_at"]),
                )
                for row in rows[: self.page_size]
            ]

            return ConversationPage(
                conversations=conversations,
                next_page_token=page_nr + 1 if has_next_page else None,
            )


class BedrockConverseConversationDescriber(ConversationDescriber):
    def __init__(
        self,
        *,
        model_id: str,
        max_nr_of_characters=70,
        session: boto3.session.Session | None = None,
        bedrock_client: "BedrockRuntimeClient | None" = None,
    ) -> None:
        self.model_id = model_id
        self.bedrock_client: BedrockRuntimeClient = bedrock_client or (
            session or boto3
        ).client(
            "bedrock-runtime",
            config=Config(
                tcp_keepalive=True,
            ),
        )
        self.system_prompt = textwrap.dedent(
            """
            You are an expert at creating concise, informative descriptions of conversations between an AI-agent and a user.
            You will be given a conversation, between <conversation> tags. You will describe the topic(s) of that conversation.
            Your description will be displayed in a UI and used to select the right conversation, from a list of conversations.
            Your description is maximally {max_nr_of_characters} characters long.
            Only return the description, without preamble or label,
            """.strip().format(
                max_nr_of_characters=max_nr_of_characters
            )
        )

        self.user_prompt = textwrap.dedent(
            """
            <conversation>
            {conversation_text}
            </conversation>
            """
        ).strip()

    @staticmethod
    def get_conversation_text(messages: "Sequence[MessageUnionTypeDef]"):
        conversation = user_conversation_from_messages(messages)
        return "\n\n".join(
            f"<{message["role"]}>\n{message["text"]}\n</{message["role"]}>"
            for message in conversation
            if message["role"] == "user"
        )

    def __call__(self, messages: "Sequence[MessageUnionTypeDef]") -> str:
        conversation_text = self.get_conversation_text(messages)
        text = self.user_prompt.format(conversation_text=conversation_text)
        response = self.bedrock_client.converse(
            modelId=self.model_id,
            inferenceConfig={"temperature": 0.0},
            system=[
                {
                    "text": self.system_prompt,
                }
            ],
            messages=[
                {
                    "role": "user",
                    "content": [{"text": text}],
                }
            ],
        )
        return get_text(response).strip(". '\"")
