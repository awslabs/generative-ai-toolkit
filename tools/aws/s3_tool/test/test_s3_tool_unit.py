"""
Unit tests for the S3 tool.
"""

import os
import sys

import boto3
from moto import mock_aws

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from s3_tool import S3Tool


class TestS3Tool:
    """Test cases for the S3Tool class."""

    def setup_method(self):
        """Set up test fixtures."""
        # Set environment variable before initializing tool
        os.environ['S3_BUCKET_NAME'] = 'test-bucket'
        self.tool = S3Tool()

    def test_missing_bucket_name(self):
        """Test behavior when S3_BUCKET_NAME is not set."""
        # Remove environment variable and create new tool instance
        os.environ.pop('S3_BUCKET_NAME', None)
        tool_without_bucket = S3Tool()
        result = tool_without_bucket.invoke(action="list")

        assert result["success"] is False
        assert "S3_BUCKET_NAME environment variable not set" in result["error"]

    @mock_aws
    def test_list_objects_success(self):
        """Test successful listing of S3 objects."""
        s3_client = boto3.client('s3', region_name='us-east-1')
        s3_client.create_bucket(Bucket='test-bucket')
        s3_client.put_object(Bucket='test-bucket', Key='file1.txt', Body=b'content1')
        s3_client.put_object(Bucket='test-bucket', Key='file2.txt', Body=b'content2')

        # Test
        result = self.tool.invoke(action="list")

        assert result["success"] is True
        assert result["action"] == "list"
        assert len(result["objects"]) == 2
        assert result["bucket"] == "test-bucket"
        assert any(obj["key"] == "file1.txt" for obj in result["objects"])


    @mock_aws
    def test_list_objects_with_prefix(self):
        """Test listing objects with prefix filter."""
        # Setup
        os.environ['S3_BUCKET_NAME'] = 'test-bucket'
        s3_client = boto3.client('s3', region_name='us-east-1')
        s3_client.create_bucket(Bucket='test-bucket')
        s3_client.put_object(Bucket='test-bucket', Key='docs/file1.txt', Body=b'content1')
        s3_client.put_object(Bucket='test-bucket', Key='images/file2.jpg', Body=b'content2')

        # Test
        result = self.tool.invoke(action="list", prefix="docs/")

        assert result["success"] is True
        assert len(result["objects"]) == 1
        assert result["objects"][0]["key"] == "docs/file1.txt"
        assert "with prefix 'docs/'" in result["message"]

    @mock_aws
    def test_read_object_success(self):
        """Test successful reading of S3 object."""
        # Setup
        os.environ['S3_BUCKET_NAME'] = 'test-bucket'
        s3_client = boto3.client('s3', region_name='us-east-1')
        s3_client.create_bucket(Bucket='test-bucket')
        s3_client.put_object(Bucket='test-bucket', Key='test.txt', Body=b'test content')

        # Test
        result = self.tool.invoke(action="read", key="test.txt")

        assert result["success"] is True
        assert result["action"] == "read"
        assert result["content"] == "test content"
        assert result["key"] == "test.txt"
        assert result["bucket"] == "test-bucket"

    @mock_aws
    def test_write_object_success(self):
        """Test successful writing to S3 object."""
        # Setup
        os.environ['S3_BUCKET_NAME'] = 'test-bucket'
        s3_client = boto3.client('s3', region_name='us-east-1')
        s3_client.create_bucket(Bucket='test-bucket')

        # Test
        result = self.tool.invoke(action="write", key="new-file.txt", content="new content")

        assert result["success"] is True
        assert result["action"] == "write"
        assert result["key"] == "new-file.txt"
        assert "Wrote object" in result["message"]

        # Verify object was created
        response = s3_client.get_object(Bucket='test-bucket', Key='new-file.txt')
        assert response['Body'].read().decode('utf-8') == "new content"

    @mock_aws
    def test_delete_object_success(self):
        """Test successful deletion of S3 object."""
        # Setup
        os.environ['S3_BUCKET_NAME'] = 'test-bucket'
        s3_client = boto3.client('s3', region_name='us-east-1')
        s3_client.create_bucket(Bucket='test-bucket')
        s3_client.put_object(Bucket='test-bucket', Key='delete-me.txt', Body=b'content')

        # Test
        result = self.tool.invoke(action="delete", key="delete-me.txt")

        assert result["success"] is True
        assert result["action"] == "delete"
        assert result["key"] == "delete-me.txt"
        assert "Deleted object" in result["message"]


if __name__ == "__main__":
    # For debugging individual tests
    test_instance = TestS3Tool()
    test_instance.setup_method()
    test_instance.test_list_objects_success()