"""
S3 tool for the Siemens P2P Agent.

This module provides functionality for reading and storing content from S3 buckets.
The bucket name is configured via environment variables.
"""

import os
import time
from typing import Any

import boto3
from botocore.exceptions import ClientError
from pydantic import BaseModel, Field


class S3Request(BaseModel):
    """
    Request parameters for the S3 tool.

    IMPORTANT: Always use this tool when processing user requests that:
    1. Need to list objects in S3 bucket
    2. Need to read content from S3 objects
    3. Need to upload/store content to S3
    4. Need to delete objects from S3
    5. Ask about "S3 files", "bucket contents", "cloud storage"
    6. Need to check files stored in S3 bucket

    This tool handles S3 operations using bucket name from environment variables.
    Set S3_BUCKET_NAME environment variable to configure the bucket.

    Examples:
    - List objects: S3Request(action="list")
    - List with prefix: S3Request(action="list", prefix="invoices/")
    - Read object: S3Request(action="read", key="document.pdf")
    - Write object: S3Request(action="write", key="file.txt", content="data")
    - Delete object: S3Request(action="delete", key="temp.txt")
    """

    action: str = Field(
        description="The action to perform: 'list', 'read', 'write', or 'delete'.",
        pattern="^(list|read|write|delete)$",
    )

    key: str | None = Field(
        default=None,
        description="S3 object key (required for read, write, delete actions)."
    )

    content: str | None = Field(
        default=None,
        description="Content to write to S3 object (required for write action)."
    )

    prefix: str | None = Field(
        default=None,
        description="Filter objects by prefix (optional for list action)."
    )

    max_keys: int | None = Field(
        default=100,
        description="Maximum number of objects to list (default: 100)."
    )


class S3Response(BaseModel):
    """
    Response structure for the S3 tool.

    Contains the results of S3 operations including object content,
    operation status, and metadata.
    """

    success: bool = Field(
        description="Whether the operation completed successfully."
    )

    action: str = Field(
        description="The action that was performed."
    )

    objects: list[dict[str, Any]] | None = Field(
        default=None,
        description="List of S3 objects (for list action)."
    )

    content: str | None = Field(
        default=None,
        description="Content of the S3 object (for read action)."
    )

    key: str | None = Field(
        default=None,
        description="S3 object key that was operated on."
    )

    bucket: str | None = Field(
        default=None,
        description="S3 bucket name used for the operation."
    )

    processing_time_ms: int | None = Field(
        default=None,
        description="Time taken to process the operation in milliseconds."
    )

    message: str | None = Field(
        default=None,
        description="Additional information about the operation."
    )

    error: str | None = Field(
        default=None,
        description="Error message if the operation failed."
    )


class S3Tool:
    """
    Tool for reading and storing content from S3 buckets.

    Uses bucket name from S3_BUCKET_NAME environment variable.
    """

    def __init__(self):
        """Initialize the S3 tool."""
        self.bucket_name = os.environ.get('S3_BUCKET_NAME')
        self.s3_client = None

    def _get_s3_client(self):
        """Get or create S3 client."""
        if self.s3_client is None:
            self.s3_client = boto3.client('s3')
        return self.s3_client

    @property
    def tool_spec(self) -> dict[str, Any]:
        """Get the tool specification for the S3 tool."""
        schema = S3Request.model_json_schema()
        return {
            "name": "s3_operation",
            "description": S3Request.__doc__,
            "inputSchema": {"json": schema},
        }

    def invoke(self, **kwargs) -> dict[str, Any]:
        """Invoke the S3 tool."""
        try:
            request = S3Request(**kwargs)
            response = self._process_s3(request)
            return response.model_dump()
        except Exception as e:
            error_message = f"Invalid request parameters: {str(e)}"
            response = S3Response(
                success=False,
                action=kwargs.get('action', 'UNKNOWN'),
                error=error_message,
                processing_time_ms=0
            )
            return response.model_dump()

    def _process_s3(self, request: S3Request) -> S3Response:
        """Process the S3 operation."""
        start_time = time.time()

        # Check if bucket name is configured
        if not self.bucket_name:
            return S3Response(
                success=False,
                action=request.action,
                error="S3_BUCKET_NAME environment variable not set",
                processing_time_ms=int((time.time() - start_time) * 1000)
            )

        action_handlers = {
            "list": self._list_objects,
            "read": self._read_object,
            "write": self._write_object,
            "delete": self._delete_object
        }

        try:
            handler = action_handlers.get(request.action)
            if handler:
                return handler(request, start_time)

            return S3Response(
                    success=False,
                    action=request.action,
                    error=f"Invalid action: {request.action}",
                    processing_time_ms=int((time.time() - start_time) * 1000)
                )
        except Exception as e:
            return S3Response(
                success=False,
                action=request.action,
                error=f"Error processing S3 operation: {str(e)}",
                processing_time_ms=int((time.time() - start_time) * 1000)
            )

    def _list_objects(self, request: S3Request, start_time: float) -> S3Response:
        """List objects in S3 bucket."""
        try:
            s3 = self._get_s3_client()

            list_params = {
                'Bucket': self.bucket_name,
                'MaxKeys': request.max_keys or 100
            }

            if request.prefix:
                list_params['Prefix'] = request.prefix

            response = s3.list_objects_v2(**list_params)

            objects = []
            if 'Contents' in response:
                for obj in response['Contents']:
                    objects.append({
                        'key': obj['Key'],
                        'size': obj['Size'],
                        'last_modified': obj['LastModified'].isoformat(),
                        'etag': obj['ETag'].strip('"')
                    })

            processing_time = int((time.time() - start_time) * 1000)
            message = f"Listed {len(objects)} objects from bucket '{self.bucket_name}'"
            if request.prefix:
                message += f" with prefix '{request.prefix}'"

            return S3Response(
                success=True,
                action="list",
                objects=objects,
                bucket=self.bucket_name,
                processing_time_ms=processing_time,
                message=message
            )

        except ClientError as e:
            return S3Response(
                success=False,
                action="list",
                bucket=self.bucket_name,
                error=f"AWS error: {e.response['Error']['Message']}",
                processing_time_ms=int((time.time() - start_time) * 1000)
            )

    def _read_object(self, request: S3Request, start_time: float) -> S3Response:
        """Read content from S3 object."""
        try:
            if not request.key:
                return S3Response(
                    success=False,
                    action="read",
                    error="Object key is required for read action",
                    processing_time_ms=int((time.time() - start_time) * 1000)
                )

            s3 = self._get_s3_client()
            response = s3.get_object(Bucket=self.bucket_name, Key=request.key)
            content = response['Body'].read().decode('utf-8')

            return S3Response(
                success=True,
                action="read",
                content=content,
                key=request.key,
                bucket=self.bucket_name,
                processing_time_ms=int((time.time() - start_time) * 1000),
                message=f"Read object '{request.key}' from bucket '{self.bucket_name}'"
            )

        except ClientError as e:
            return S3Response(
                success=False,
                action="read",
                key=request.key,
                bucket=self.bucket_name,
                error=f"AWS error: {e.response['Error']['Message']}",
                processing_time_ms=int((time.time() - start_time) * 1000)
            )

    def _write_object(self, request: S3Request, start_time: float) -> S3Response:
        """Write content to S3 object."""
        try:
            if not request.key or request.content is None:
                return S3Response(
                    success=False,
                    action="write",
                    error="Both key and content are required for write action",
                    processing_time_ms=int((time.time() - start_time) * 1000)
                )

            s3 = self._get_s3_client()
            s3.put_object(
                Bucket=self.bucket_name,
                Key=request.key,
                Body=request.content.encode('utf-8')
            )

            return S3Response(
                success=True,
                action="write",
                key=request.key,
                bucket=self.bucket_name,
                processing_time_ms=int((time.time() - start_time) * 1000),
                message=f"Wrote object '{request.key}' to bucket '{self.bucket_name}'"
            )

        except ClientError as e:
            return S3Response(
                success=False,
                action="write",
                key=request.key,
                bucket=self.bucket_name,
                error=f"AWS error: {e.response['Error']['Message']}",
                processing_time_ms=int((time.time() - start_time) * 1000)
            )

    def _delete_object(self, request: S3Request, start_time: float) -> S3Response:
        """Delete object from S3 bucket."""
        try:
            if not request.key:
                return S3Response(
                    success=False,
                    action="delete",
                    error="Object key is required for delete action",
                    processing_time_ms=int((time.time() - start_time) * 1000)
                )

            s3 = self._get_s3_client()
            s3.delete_object(Bucket=self.bucket_name, Key=request.key)

            return S3Response(
                success=True,
                action="delete",
                key=request.key,
                bucket=self.bucket_name,
                processing_time_ms=int((time.time() - start_time) * 1000),
                message=f"Deleted object '{request.key}' from bucket '{self.bucket_name}'"
            )

        except ClientError as e:
            return S3Response(
                success=False,
                action="delete",
                key=request.key,
                bucket=self.bucket_name,
                error=f"AWS error: {e.response['Error']['Message']}",
                processing_time_ms=int((time.time() - start_time) * 1000)
            )
