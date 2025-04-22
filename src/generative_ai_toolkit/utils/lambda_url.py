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

import http.client
import json
import urllib.request
from codecs import iterdecode
from collections.abc import Iterator, Mapping
from dataclasses import dataclass

import boto3
import boto3.session
from botocore.auth import SigV4Auth
from botocore.awsrequest import AWSRequest


@dataclass
class ConverseStreamResponse:
    output_tokens: Iterator[str]
    """
    The response body, as an iterable of strings
    """
    conversation_id: str
    """
    The conversation ID (from the x-conversation-id header)
    """

    def __iter__(self):
        yield from self.output_tokens


class IamAuthInvoker:
    """
    Invoke a Lambda function URL with IAM authentication.
    """

    def __init__(
        self, lambda_function_url: str, session: boto3.session.Session | None = None
    ) -> None:
        # Ensure HTTPS
        if not lambda_function_url.startswith("https://"):
            raise Exception("Lambda function URL must start with https://")
        self.lambda_function_url = lambda_function_url
        self.session = session or boto3.session.Session()

    def _invoke_signed(
        self,
        headers: Mapping[str, str] | None = None,
        data: bytes | None = None,
        method="POST",
    ):
        credentials = self.session.get_credentials()
        if not credentials:
            raise Exception("Cannot locate valid AWS credentials")

        request = AWSRequest(
            method=method,
            url=self.lambda_function_url,
            data=data,
            headers=headers,
        )
        SigV4Auth(credentials, "lambda", self.session.region_name).add_auth(request)

        request = urllib.request.Request(
            self.lambda_function_url,
            data=data,
            headers=dict(request.headers),
            method=method,
        )

        # nosemgrep: python.lang.security.audit.dynamic-urllib-use-detected
        return urllib.request.urlopen(request, timeout=30)

    def _response_body_iterator(self, response: http.client.HTTPResponse, chunk_size=4):
        """
        Iterate over the response body, one chunk of size chunk_size at a time
        """
        with response:
            while bytes_ := response.read(chunk_size):
                yield bytes_

    def converse_stream(
        self,
        user_input: str,
        conversation_id: str | None = None,
    ):
        """
        Invoke the Lambda function with IAM authentication.
        """

        headers = {"Content-Type": "application/json"}
        if conversation_id:
            headers["x-conversation-id"] = conversation_id

        response = self._invoke_signed(
            headers=headers,
            data=json.dumps({"user_input": user_input}).encode(),
        )

        if response.status != 200:
            raise Exception(f"Error: HTTP {response.status} - {response.reason}")

        return ConverseStreamResponse(
            output_tokens=iterdecode(self._response_body_iterator(response), "utf8"),
            conversation_id=response.headers["x-conversation-id"],
        )
