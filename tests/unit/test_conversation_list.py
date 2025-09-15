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
import tempfile
from unittest.mock import MagicMock

import pytest
from botocore.config import Config

from generative_ai_toolkit.ui.conversation_list import (
    BedrockConverseConversationDescriber,
    Conversation,
    ConversationPage,
    SqliteConversationList,
)


class TestSqliteConversationList:
    """Test cases for SqliteConversationList class."""

    @pytest.fixture
    def temp_db_path(self):
        """Create a temporary database file for testing."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name
        yield db_path
        # Cleanup
        if os.path.exists(db_path):
            os.unlink(db_path)

    @pytest.fixture
    def mock_describer(self):
        """Create a mock conversation describer."""
        mock = MagicMock()
        mock.return_value = "Test conversation description"
        return mock

    @pytest.fixture
    def sample_messages(self):
        """Create sample messages for testing."""
        return [
            {"role": "user", "content": [{"text": "Hello, how are you?"}]},
            {"role": "assistant", "content": [{"text": "I'm doing well, thank you!"}]},
        ]

    def test_create_tables(self, mock_describer, temp_db_path):
        """Test database table creation."""
        SqliteConversationList(
            describer=mock_describer, db_path=temp_db_path, create_tables=True
        )

        # Verify tables were created
        with sqlite3.connect(temp_db_path) as conn:
            cursor = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='conversations'"
            )
            assert cursor.fetchone() is not None

            # Verify index was created
            cursor = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='index' AND name='idx_conversations_updated'"
            )
            assert cursor.fetchone() is not None

    def test_page_size_property_and_setter(self, mock_describer, temp_db_path):
        """Test page size property getter and setter."""
        conv_list = SqliteConversationList(
            describer=mock_describer, db_path=temp_db_path, create_tables=False
        )

        assert conv_list.page_size == 20

        conv_list.set_page_size(50)
        assert conv_list.page_size == 50

    def test_auth_context_property_and_setter(self, mock_describer, temp_db_path):
        """Test auth context property getter and setter."""
        conv_list = SqliteConversationList(
            describer=mock_describer, db_path=temp_db_path, create_tables=False
        )

        assert conv_list.auth_context == {"principal_id": None}

        conv_list.set_auth_context(principal_id="user123")
        assert conv_list.auth_context == {"principal_id": "user123"}

    def test_add_conversation_success(
        self, mock_describer, temp_db_path, sample_messages
    ):
        """Test successfully adding a conversation."""
        conv_list = SqliteConversationList(
            describer=mock_describer, db_path=temp_db_path, create_tables=True
        )

        mock_describer.return_value = "Weather discussion"

        result = conv_list.add_conversation("conv-123", sample_messages)

        assert isinstance(result, Conversation)
        assert result.conversation_id == "conv-123"
        assert result.description == "Weather discussion"
        assert isinstance(result.updated_at, datetime.datetime)

        # Verify describer was called with messages
        mock_describer.assert_called_once_with(sample_messages)

        # Verify data was stored in database
        with sqlite3.connect(temp_db_path) as conn:
            cursor = conn.execute(
                "SELECT conversation_id, description FROM conversations WHERE conversation_id = ?",
                ("conv-123",),
            )
            row = cursor.fetchone()
            assert row is not None
            assert row[0] == "conv-123"
            assert row[1] == "Weather discussion"

    def test_add_conversation_empty_messages(self, mock_describer, temp_db_path):
        """Test adding conversation with empty messages raises ValueError."""
        conv_list = SqliteConversationList(
            describer=mock_describer, db_path=temp_db_path, create_tables=True
        )

        with pytest.raises(
            ValueError, match="Cannot add conversation with empty messages list"
        ):
            conv_list.add_conversation("conv-123", [])

    def test_add_conversation_overwrite_existing(
        self, mock_describer, temp_db_path, sample_messages
    ):
        """Test overwriting an existing conversation."""
        conv_list = SqliteConversationList(
            describer=mock_describer, db_path=temp_db_path, create_tables=True
        )

        # Add first conversation
        mock_describer.return_value = "First description"
        conv_list.add_conversation("conv-123", sample_messages)

        # Overwrite with new description
        mock_describer.return_value = "Updated description"
        result = conv_list.add_conversation("conv-123", sample_messages)

        assert result.description == "Updated description"

        # Verify only one record exists in database
        with sqlite3.connect(temp_db_path) as conn:
            cursor = conn.execute(
                "SELECT COUNT(*) FROM conversations WHERE conversation_id = ?",
                ("conv-123",),
            )
            count = cursor.fetchone()[0]
            assert count == 1

    def test_remove_conversation_success(
        self, mock_describer, temp_db_path, sample_messages
    ):
        """Test successfully removing a conversation."""
        conv_list = SqliteConversationList(
            describer=mock_describer, db_path=temp_db_path, create_tables=True
        )

        # Add conversation first
        conv_list.add_conversation("conv-123", sample_messages)

        # Remove conversation
        conv_list.remove_conversation("conv-123")

        # Verify conversation was removed from database
        with sqlite3.connect(temp_db_path) as conn:
            cursor = conn.execute(
                "SELECT COUNT(*) FROM conversations WHERE conversation_id = ?",
                ("conv-123",),
            )
            count = cursor.fetchone()[0]
            assert count == 0

    def test_remove_conversation_not_found(self, mock_describer, temp_db_path):
        """Test removing non-existent conversation raises ValueError."""
        conv_list = SqliteConversationList(
            describer=mock_describer, db_path=temp_db_path, create_tables=True
        )

        with pytest.raises(ValueError, match="Conversation non-existent not found"):
            conv_list.remove_conversation("non-existent")

    def test_get_conversations_empty_database(self, mock_describer, temp_db_path):
        """Test getting conversations from empty database."""
        conv_list = SqliteConversationList(
            describer=mock_describer, db_path=temp_db_path, create_tables=True
        )

        result = conv_list.get_conversations()

        assert isinstance(result, ConversationPage)
        assert len(result.conversations) == 0
        assert result.next_page_token is None

    def test_get_conversations_single_page(
        self, mock_describer, temp_db_path, sample_messages
    ):
        """Test getting conversations that fit in a single page."""
        conv_list = SqliteConversationList(
            describer=mock_describer,
            db_path=temp_db_path,
            page_size=10,
            create_tables=True,
        )

        # Add a few conversations
        for i in range(3):
            mock_describer.return_value = f"Description {i}"
            conv_list.add_conversation(f"conv-{i}", sample_messages)

        result = conv_list.get_conversations()

        assert len(result.conversations) == 3
        assert result.next_page_token is None

        # Verify conversations are ordered by updated_at DESC
        assert result.conversations[0].conversation_id == "conv-2"  # Most recent
        assert result.conversations[2].conversation_id == "conv-0"  # Oldest

    def test_get_conversations_multiple_pages(
        self, mock_describer, temp_db_path, sample_messages
    ):
        """Test getting conversations with pagination."""
        conv_list = SqliteConversationList(
            describer=mock_describer,
            db_path=temp_db_path,
            page_size=2,
            create_tables=True,
        )

        # Add more conversations than page size
        for i in range(5):
            mock_describer.return_value = f"Description {i}"
            conv_list.add_conversation(f"conv-{i}", sample_messages)

        # Get first page
        result = conv_list.get_conversations()

        assert len(result.conversations) == 2
        assert result.next_page_token == 1
        assert result.conversations[0].conversation_id == "conv-4"  # Most recent
        assert result.conversations[1].conversation_id == "conv-3"

        # Get second page
        result = conv_list.get_conversations(next_page_token=1)

        assert len(result.conversations) == 2
        assert result.next_page_token == 2
        assert result.conversations[0].conversation_id == "conv-2"
        assert result.conversations[1].conversation_id == "conv-1"

        # Get third page (last page)
        result = conv_list.get_conversations(next_page_token=2)

        assert len(result.conversations) == 1
        assert result.next_page_token is None
        assert result.conversations[0].conversation_id == "conv-0"  # Oldest

    def test_get_conversations_with_string_token(
        self, mock_describer, temp_db_path, sample_messages
    ):
        """Test getting conversations with string page token."""
        conv_list = SqliteConversationList(
            describer=mock_describer,
            db_path=temp_db_path,
            page_size=1,
            create_tables=True,
        )

        # Add conversations
        for i in range(3):
            mock_describer.return_value = f"Description {i}"
            conv_list.add_conversation(f"conv-{i}", sample_messages)

        # Get second page using string token
        result = conv_list.get_conversations(next_page_token="1")

        assert len(result.conversations) == 1
        assert result.conversations[0].conversation_id == "conv-1"


class TestBedrockConverseConversationDescriber:
    """Test cases for BedrockConverseConversationDescriber class."""

    @pytest.fixture
    def mock_bedrock_client(self):
        """Create a mock Bedrock client."""
        mock_client = MagicMock()
        mock_response = {
            "output": {
                "message": {"content": [{"text": "Weather and travel discussion"}]}
            }
        }
        mock_client.converse.return_value = mock_response
        return mock_client

    @pytest.fixture
    def sample_messages(self):
        """Create sample messages for testing."""
        return [
            {
                "role": "user",
                "content": [{"text": "What's the weather like in Paris?"}],
            },
            {
                "role": "assistant",
                "content": [{"text": "The weather in Paris is sunny today."}],
            },
            {
                "role": "user",
                "content": [{"text": "Great! Any good restaurants you'd recommend?"}],
            },
        ]

    def test_call_method_success(self, mock_bedrock_client, sample_messages):
        """Test successful conversation description generation."""
        describer = BedrockConverseConversationDescriber(
            model_id="claude-v1", bedrock_client=mock_bedrock_client
        )

        result = describer(sample_messages)

        assert result == "Weather and travel discussion"

        # Verify Bedrock client was called with correct parameters
        mock_bedrock_client.converse.assert_called_once()
        call_args = mock_bedrock_client.converse.call_args

        assert call_args[1]["modelId"] == "claude-v1"
        assert call_args[1]["inferenceConfig"]["temperature"] == 0.0
        assert len(call_args[1]["system"]) == 1
        assert "maximally 70 characters long" in call_args[1]["system"][0]["text"]
        assert len(call_args[1]["messages"]) == 1
        assert call_args[1]["messages"][0]["role"] == "user"

    def test_call_method_with_session_parameter(self):
        """Test initialization with session parameter."""
        mock_session = MagicMock()
        mock_client = MagicMock()
        mock_session.client.return_value = mock_client

        describer = BedrockConverseConversationDescriber(
            model_id="claude-v1", session=mock_session
        )

        assert describer.bedrock_client == mock_client

        # Verify session.client was called once
        mock_session.client.assert_called_once()

        # Check the call arguments separately
        call_args = mock_session.client.call_args
        assert call_args[0] == ("bedrock-runtime",)
        assert "config" in call_args[1]
        config = call_args[1]["config"]
        assert isinstance(config, Config)

    def test_conversation_text_formatting_in_prompt(
        self, mock_bedrock_client, sample_messages
    ):
        """Test that conversation text is properly formatted in the user prompt."""
        describer = BedrockConverseConversationDescriber(
            model_id="claude-v1", bedrock_client=mock_bedrock_client
        )

        describer(sample_messages)

        # Verify the conversation text was included in the prompt
        call_args = mock_bedrock_client.converse.call_args
        user_message_text = call_args[1]["messages"][0]["content"][0]["text"]

        assert "<conversation>" in user_message_text
        assert "</conversation>" in user_message_text
        assert "What's the weather like in Paris?" in user_message_text
        assert "Great! Any good restaurants you'd recommend?" in user_message_text

    def test_system_prompt_customization(self, mock_bedrock_client):
        """Test that max_nr_of_characters customizes the system prompt."""
        describer = BedrockConverseConversationDescriber(
            model_id="claude-v1",
            max_nr_of_characters=150,
            bedrock_client=mock_bedrock_client,
        )

        assert "maximally 150 characters long" in describer.system_prompt
