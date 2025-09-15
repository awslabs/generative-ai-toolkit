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
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, Unpack

import boto3
import boto3.session
from boto3.dynamodb.conditions import Key

from generative_ai_toolkit.context import AuthContext

if TYPE_CHECKING:
    from mypy_boto3_bedrock_runtime.type_defs import MessageUnionTypeDef

from generative_ai_toolkit.ui.conversation_list.conversation_list import (
    Conversation,
    ConversationDescriber,
    ConversationList,
    ConversationPage,
)


class DynamoDbConversationList(ConversationList):

    def __init__(
        self,
        *,
        describer: ConversationDescriber,
        table_name: str,
        updated_at_gsi_name: str = "by_updated_at",
        page_size: int = 20,
        session: boto3.session.Session | None = None,
    ):
        self.describer = describer
        self.table_name = table_name
        self.updated_at_gsi_name = updated_at_gsi_name
        self._page_size = page_size
        self._auth_context: AuthContext = {"principal_id": None}
        self.table = (session or boto3).resource("dynamodb").Table(table_name)

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
        self.table.put_item(
            Item={
                "pk": f"LIST#{self._auth_context["principal_id"] or "_"}",
                "sk": f"CONV#{conversation_id}#",
                "conversation_id": conversation_id,
                "description": description,
                "updated_at": now.isoformat(),
                "auth_context": self._auth_context,
            },
        )

        return Conversation(
            conversation_id=conversation_id, description=description, updated_at=now
        )

    def remove_conversation(self, conversation_id: str) -> None:
        self.table.delete_item(
            Key={
                "pk": f"LIST#{self._auth_context["principal_id"] or "_"}",
                "sk": f"CONV#{conversation_id}#",
            },
        )

    def get_conversations(self, next_page_token: Any | None = None) -> ConversationPage:
        params = {
            "IndexName": self.updated_at_gsi_name,
            "KeyConditionExpression": Key("pk").eq(
                f"LIST#{self._auth_context["principal_id"] or "_"}"
            ),
            "Limit": self.page_size,
            "ScanIndexForward": False,
        }

        if next_page_token:
            params["ExclusiveStartKey"] = next_page_token

        response = self.table.query(**params)

        rows = response["Items"]
        last_evaluated_key = response.get("LastEvaluatedKey")

        conversations = [
            Conversation(
                conversation_id=str(row["conversation_id"]),
                description=str(row["description"]),
                updated_at=datetime.datetime.fromisoformat(str(row["updated_at"])),
            )
            for row in rows
        ]

        return ConversationPage(
            conversations=conversations,
            next_page_token=last_evaluated_key,
        )
