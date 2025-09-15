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
from unittest.mock import MagicMock, patch

import pytest
from boto3.dynamodb.conditions import Key

from generative_ai_toolkit.ui.conversation_list import Conversation, ConversationPage
from generative_ai_toolkit.ui.conversation_list.dynamodb import DynamoDbConversationList


class TestDynamoDbConversationList:
    """Test suite for the DynamoDbConversationList class covering core functionality."""

    @pytest.fixture
    def mock_describer(self):
        """Create a mock conversation describer."""
        mock = MagicMock()
        mock.return_value = "Test conversation description"
        return mock

    @pytest.fixture
    def mock_table(self):
        """Create a mock DynamoDB table."""
        mock_table = MagicMock()
        return mock_table

    @pytest.fixture
    def mock_session(self, mock_table):
        """Create a mock boto3 session."""
        mock_session = MagicMock()
        mock_resource = MagicMock()
        mock_resource.Table.return_value = mock_table
        mock_session.resource.return_value = mock_resource
        return mock_session

    @pytest.fixture
    def sample_messages(self):
        """Create sample messages for testing."""
        return [
            {"role": "user", "content": [{"text": "Hello, how are you?"}]},
            {"role": "assistant", "content": [{"text": "I'm doing well, thank you!"}]},
        ]

    @pytest.fixture
    def conversation_list(self, mock_describer, mock_session):
        """Create a DynamoDbConversationList instance for testing."""
        return DynamoDbConversationList(
            describer=mock_describer,
            table_name="test-conversations",
            updated_at_gsi_name="by_updated_at",
            page_size=20,
            session=mock_session,
        )

    def test_init_with_default_parameters(self, mock_describer):
        """Test initialization with default parameters."""
        with patch("boto3.resource") as mock_resource:
            mock_table = MagicMock()
            mock_resource.return_value.Table.return_value = mock_table

            conv_list = DynamoDbConversationList(
                describer=mock_describer, table_name="test-table"
            )

            assert conv_list.describer == mock_describer
            assert conv_list.table_name == "test-table"
            assert conv_list.updated_at_gsi_name == "by_updated_at"
            assert conv_list.page_size == 20
            assert conv_list.auth_context == {"principal_id": None}
            assert conv_list.table == mock_table

            # Verify boto3.resource was called correctly
            mock_resource.assert_called_once_with("dynamodb")

    def test_init_with_custom_parameters(self, mock_describer, mock_session):
        """Test initialization with custom parameters."""
        conv_list = DynamoDbConversationList(
            describer=mock_describer,
            table_name="custom-table",
            updated_at_gsi_name="custom_gsi",
            page_size=50,
            session=mock_session,
        )

        assert conv_list.table_name == "custom-table"
        assert conv_list.updated_at_gsi_name == "custom_gsi"
        assert conv_list.page_size == 50

    def test_page_size_property_and_setter(self, conversation_list):
        """Test page size property getter and setter."""
        assert conversation_list.page_size == 20

        conversation_list.set_page_size(100)
        assert conversation_list.page_size == 100

    def test_auth_context_property_and_setter(self, conversation_list):
        """Test auth context property getter and setter."""
        assert conversation_list.auth_context == {"principal_id": None}

        conversation_list.set_auth_context(principal_id="user123")
        assert conversation_list.auth_context == {"principal_id": "user123"}

    def test_add_conversation_success(
        self, conversation_list, mock_table, mock_describer, sample_messages
    ):
        """Test successfully adding a conversation."""
        mock_describer.return_value = "Weather discussion"
        conversation_list.set_auth_context(principal_id="user123")

        with patch("datetime.datetime") as mock_datetime:
            mock_now = datetime.datetime(2023, 1, 1, 12, 0, 0, tzinfo=datetime.UTC)
            mock_datetime.now.return_value = mock_now
            mock_datetime.UTC = datetime.UTC

            result = conversation_list.add_conversation("conv-123", sample_messages)

        # Verify result
        assert isinstance(result, Conversation)
        assert result.conversation_id == "conv-123"
        assert result.description == "Weather discussion"
        assert result.updated_at == mock_now

        # Verify describer was called
        mock_describer.assert_called_once_with(sample_messages)

        # Verify DynamoDB put_item was called correctly
        mock_table.put_item.assert_called_once_with(
            Item={
                "pk": "LIST#user123",
                "sk": "CONV#conv-123#",
                "conversation_id": "conv-123",
                "description": "Weather discussion",
                "updated_at": mock_now.isoformat(),
                "auth_context": {"principal_id": "user123"},
            }
        )

    def test_add_conversation_with_none_principal_id(
        self, conversation_list, mock_table, mock_describer, sample_messages
    ):
        """Test adding conversation with None principal_id uses '_' as fallback."""
        mock_describer.return_value = "Test description"
        # auth_context principal_id is None by default

        with patch("datetime.datetime") as mock_datetime:
            mock_now = datetime.datetime(2023, 1, 1, 12, 0, 0, tzinfo=datetime.UTC)
            mock_datetime.now.return_value = mock_now
            mock_datetime.UTC = datetime.UTC

            conversation_list.add_conversation("conv-456", sample_messages)

        # Verify the pk uses '_' when principal_id is None
        mock_table.put_item.assert_called_once_with(
            Item={
                "pk": "LIST#_",
                "sk": "CONV#conv-456#",
                "conversation_id": "conv-456",
                "description": "Test description",
                "updated_at": mock_now.isoformat(),
                "auth_context": {"principal_id": None},
            }
        )

    def test_add_conversation_empty_messages(self, conversation_list):
        """Test adding conversation with empty messages raises ValueError."""
        with pytest.raises(
            ValueError, match="Cannot add conversation with empty messages list"
        ):
            conversation_list.add_conversation("conv-123", [])

    def test_remove_conversation_success(self, conversation_list, mock_table):
        """Test successfully removing a conversation."""
        conversation_list.set_auth_context(principal_id="user123")

        conversation_list.remove_conversation("conv-123")

        # Verify DynamoDB delete_item was called correctly
        mock_table.delete_item.assert_called_once_with(
            Key={
                "pk": "LIST#user123",
                "sk": "CONV#conv-123#",
            }
        )

    def test_remove_conversation_with_none_principal_id(
        self, conversation_list, mock_table
    ):
        """Test removing conversation with None principal_id uses '_' as fallback."""
        # auth_context principal_id is None by default

        conversation_list.remove_conversation("conv-456")

        # Verify the key uses '_' when principal_id is None
        mock_table.delete_item.assert_called_once_with(
            Key={
                "pk": "LIST#_",
                "sk": "CONV#conv-456#",
            }
        )

    def test_get_conversations_empty_table(self, conversation_list, mock_table):
        """Test getting conversations from empty table."""
        conversation_list.set_auth_context(principal_id="user123")

        # Mock empty response
        mock_table.query.return_value = {
            "Items": [],
            "LastEvaluatedKey": None,
        }

        result = conversation_list.get_conversations()

        # Verify result
        assert isinstance(result, ConversationPage)
        assert len(result.conversations) == 0
        assert result.next_page_token is None

        # Verify query parameters
        mock_table.query.assert_called_once_with(
            IndexName="by_updated_at",
            KeyConditionExpression=Key("pk").eq("LIST#user123"),
            Limit=20,
            ScanIndexForward=False,
        )

    def test_get_conversations_with_data(self, conversation_list, mock_table):
        """Test getting conversations with data."""
        conversation_list.set_auth_context(principal_id="user123")

        # Mock response with data
        mock_items = [
            {
                "conversation_id": "conv-2",
                "description": "Recent conversation",
                "updated_at": "2023-01-02T12:00:00+00:00",
            },
            {
                "conversation_id": "conv-1",
                "description": "Older conversation",
                "updated_at": "2023-01-01T12:00:00+00:00",
            },
        ]
        mock_table.query.return_value = {
            "Items": mock_items,
            "LastEvaluatedKey": None,
        }

        result = conversation_list.get_conversations()

        # Verify result
        assert len(result.conversations) == 2
        assert result.next_page_token is None

        # Verify first conversation (most recent)
        conv1 = result.conversations[0]
        assert conv1.conversation_id == "conv-2"
        assert conv1.description == "Recent conversation"
        assert conv1.updated_at == datetime.datetime(
            2023, 1, 2, 12, 0, 0, tzinfo=datetime.UTC
        )

        # Verify second conversation
        conv2 = result.conversations[1]
        assert conv2.conversation_id == "conv-1"
        assert conv2.description == "Older conversation"
        assert conv2.updated_at == datetime.datetime(
            2023, 1, 1, 12, 0, 0, tzinfo=datetime.UTC
        )

    def test_get_conversations_with_pagination(self, conversation_list, mock_table):
        """Test getting conversations with pagination."""
        conversation_list.set_auth_context(principal_id="user123")

        # Mock response with pagination
        mock_items = [
            {
                "conversation_id": "conv-1",
                "description": "Conversation 1",
                "updated_at": "2023-01-01T12:00:00+00:00",
            },
        ]
        last_evaluated_key = {"pk": "LIST#user123", "sk": "CONV#conv-1#"}
        mock_table.query.return_value = {
            "Items": mock_items,
            "LastEvaluatedKey": last_evaluated_key,
        }

        result = conversation_list.get_conversations()

        # Verify pagination token is returned
        assert result.next_page_token == last_evaluated_key

    def test_get_conversations_with_next_page_token(
        self, conversation_list, mock_table
    ):
        """Test getting conversations with next page token."""
        conversation_list.set_auth_context(principal_id="user123")

        # Mock response
        mock_table.query.return_value = {
            "Items": [],
            "LastEvaluatedKey": None,
        }

        # Call with next page token
        start_key = {"pk": "LIST#user123", "sk": "CONV#conv-1#"}
        conversation_list.get_conversations(next_page_token=start_key)

        # Verify ExclusiveStartKey was passed
        mock_table.query.assert_called_once_with(
            IndexName="by_updated_at",
            KeyConditionExpression=Key("pk").eq("LIST#user123"),
            Limit=20,
            ScanIndexForward=False,
            ExclusiveStartKey=start_key,
        )

    def test_get_conversations_with_custom_page_size(
        self, conversation_list, mock_table
    ):
        """Test getting conversations with custom page size."""
        conversation_list.set_auth_context(principal_id="user123")
        conversation_list.set_page_size(50)

        # Mock response
        mock_table.query.return_value = {
            "Items": [],
            "LastEvaluatedKey": None,
        }

        conversation_list.get_conversations()

        # Verify custom page size was used
        mock_table.query.assert_called_once_with(
            IndexName="by_updated_at",
            KeyConditionExpression=Key("pk").eq("LIST#user123"),
            Limit=50,
            ScanIndexForward=False,
        )

    def test_get_conversations_with_none_principal_id(
        self, conversation_list, mock_table
    ):
        """Test getting conversations with None principal_id uses '_' as fallback."""
        # auth_context principal_id is None by default

        # Mock response
        mock_table.query.return_value = {
            "Items": [],
            "LastEvaluatedKey": None,
        }

        conversation_list.get_conversations()

        # Verify the query uses '_' when principal_id is None
        mock_table.query.assert_called_once_with(
            IndexName="by_updated_at",
            KeyConditionExpression=Key("pk").eq("LIST#_"),
            Limit=20,
            ScanIndexForward=False,
        )

    def test_query_ordering(self, conversation_list, mock_table):
        """Test that conversations are queried in descending order by updated_at."""
        conversation_list.set_auth_context(principal_id="user123")

        # Mock response
        mock_table.query.return_value = {
            "Items": [],
            "LastEvaluatedKey": None,
        }

        conversation_list.get_conversations()

        # Verify ScanIndexForward=False for descending order
        call_args = mock_table.query.call_args[1]
        assert call_args["ScanIndexForward"] is False

    def test_data_type_conversion(self, conversation_list, mock_table):
        """Test that data types are properly converted from DynamoDB response."""
        conversation_list.set_auth_context(principal_id="user123")

        # Mock response with various data types that might come from DynamoDB
        mock_items = [
            {
                "conversation_id": 12345,  # Number that should be converted to string
                "description": 67890,  # Number that should be converted to string
                "updated_at": "2023-01-01T12:00:00+00:00",
            },
        ]
        mock_table.query.return_value = {
            "Items": mock_items,
            "LastEvaluatedKey": None,
        }

        result = conversation_list.get_conversations()

        # Verify data type conversion
        conv = result.conversations[0]
        assert conv.conversation_id == "12345"
        assert conv.description == "67890"
        assert isinstance(conv.updated_at, datetime.datetime)

    def test_custom_gsi_name(self, mock_describer, mock_session, mock_table):
        """Test using custom GSI name."""
        conversation_list = DynamoDbConversationList(
            describer=mock_describer,
            table_name="test-conversations",
            updated_at_gsi_name="custom_index",
            session=mock_session,
        )
        conversation_list.set_auth_context(principal_id="user123")

        # Mock response
        mock_table.query.return_value = {
            "Items": [],
            "LastEvaluatedKey": None,
        }

        conversation_list.get_conversations()

        # Verify custom GSI name was used
        call_args = mock_table.query.call_args[1]
        assert call_args["IndexName"] == "custom_index"

    def test_datetime_parsing(self, conversation_list, mock_table):
        """Test that datetime strings are properly parsed."""
        conversation_list.set_auth_context(principal_id="user123")

        # Mock response with ISO datetime string
        mock_items = [
            {
                "conversation_id": "conv-1",
                "description": "Test conversation",
                "updated_at": "2023-12-25T15:30:45.123456+00:00",  # ISO format with microseconds
            },
        ]
        mock_table.query.return_value = {
            "Items": mock_items,
            "LastEvaluatedKey": None,
        }

        result = conversation_list.get_conversations()

        # Verify datetime parsing
        conv = result.conversations[0]
        expected_dt = datetime.datetime(
            2023, 12, 25, 15, 30, 45, 123456, tzinfo=datetime.UTC
        )
        assert conv.updated_at == expected_dt

    def test_table_resource_initialization(self, mock_describer):
        """Test that DynamoDB table resource is properly initialized."""
        with patch("boto3.resource") as mock_resource:
            mock_dynamodb_resource = MagicMock()
            mock_table = MagicMock()
            mock_resource.return_value = mock_dynamodb_resource
            mock_dynamodb_resource.Table.return_value = mock_table

            conversation_list = DynamoDbConversationList(
                describer=mock_describer, table_name="my-table"
            )

            # Verify boto3.resource was called correctly
            mock_resource.assert_called_once_with("dynamodb")

            # Verify Table was called with correct table name
            mock_dynamodb_resource.Table.assert_called_once_with("my-table")

            # Verify table is set correctly
            assert conversation_list.table == mock_table

    def test_auth_context_in_stored_item(
        self, conversation_list, mock_table, mock_describer, sample_messages
    ):
        """Test that auth_context is properly stored in DynamoDB item."""
        auth_context = {"principal_id": "user456"}
        conversation_list.set_auth_context(**auth_context)
        mock_describer.return_value = "Test description"

        with patch("datetime.datetime") as mock_datetime:
            mock_now = datetime.datetime(2023, 1, 1, 12, 0, 0, tzinfo=datetime.UTC)
            mock_datetime.now.return_value = mock_now
            mock_datetime.UTC = datetime.UTC

            conversation_list.add_conversation("conv-789", sample_messages)

        # Verify that the full auth_context is stored in the item
        call_args = mock_table.put_item.call_args[1]
        stored_item = call_args["Item"]
        assert stored_item["auth_context"] == auth_context

    @patch("boto3.resource")
    def test_session_parameter_usage(self, mock_resource, mock_describer):
        """Test that custom session is used when provided."""
        custom_session = MagicMock()
        mock_dynamodb_resource = MagicMock()
        custom_session.resource.return_value = mock_dynamodb_resource

        DynamoDbConversationList(
            describer=mock_describer, table_name="test-table", session=custom_session
        )

        # Verify custom session was used instead of boto3
        custom_session.resource.assert_called_once_with("dynamodb")
        mock_resource.assert_not_called()

    def test_iso_datetime_format_consistency(
        self, conversation_list, mock_table, mock_describer, sample_messages
    ):
        """Test that datetime is stored in ISO format consistently."""
        conversation_list.set_auth_context(principal_id="user123")
        mock_describer.return_value = "Test description"

        # Create a specific datetime to test ISO formatting
        test_datetime = datetime.datetime(
            2023, 6, 15, 14, 30, 45, 123456, tzinfo=datetime.UTC
        )

        with patch("datetime.datetime") as mock_datetime:
            mock_datetime.now.return_value = test_datetime
            mock_datetime.UTC = datetime.UTC

            conversation_list.add_conversation("conv-iso", sample_messages)

        # Verify ISO format is used
        call_args = mock_table.put_item.call_args[1]
        stored_item = call_args["Item"]
        assert stored_item["updated_at"] == "2023-06-15T14:30:45.123456+00:00"
