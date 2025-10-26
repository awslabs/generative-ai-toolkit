"""
Test JWT authentication for AgentCore runtime.

This test module demonstrates and validates JWT bearer token authentication
for invoking the AgentCore runtime using Cognito User Pool credentials.

Test Flow:
1. Retrieve client user credentials from AWS Secrets Manager
2. Authenticate with Cognito to get a JWT access token
3. Invoke the agent runtime using the JWT bearer token
4. Validate the response

Environment Variables Required:
    - AWS_REGION: AWS region where resources are deployed
    - CLIENT_USER_CREDENTIALS_SECRET_NAME: Secret name for client user credentials
    - OAUTH_USER_POOL_ID: Cognito User Pool ID
    - OAUTH_USER_POOL_CLIENT_ID: Cognito User Pool Client ID
    - AGENT_RUNTIME_ARN: Agent runtime ARN to invoke
"""

import base64
import json
import os
import urllib.parse
import uuid
from typing import Any

import pytest
import requests


class TestJWTAuthentication:
    """Test class for JWT authentication with AgentCore runtime."""

    def test_client_credentials_retrieval(self, client_credentials: dict[str, str]):
        """Test that client credentials can be retrieved from Secrets Manager."""
        assert "username" in client_credentials
        assert "password" in client_credentials
        assert client_credentials["username"], "Username should not be empty"
        assert client_credentials["password"], "Password should not be empty"

    def test_jwt_token_generation(self, jwt_token: str):
        """Test that JWT token can be generated from Cognito."""
        assert jwt_token, "JWT token should not be empty"
        assert len(jwt_token) > 100, "JWT token should be a reasonable length"

        # JWT tokens should have 3 parts separated by dots
        parts = jwt_token.split(".")
        assert len(parts) == 3, (
            "JWT token should have 3 parts (header.payload.signature)"
        )

        # Decode and inspect JWT claims (for debugging)

        # Decode payload (add padding if needed)
        payload_b64 = parts[1]
        # Add padding if needed
        payload_b64 += "=" * (4 - len(payload_b64) % 4)
        payload_bytes = base64.urlsafe_b64decode(payload_b64)
        claims = json.loads(payload_bytes)

        print(f"JWT Claims: {json.dumps(claims, indent=2)}")

        # Verify expected claims are present
        assert "sub" in claims, "JWT should contain 'sub' (subject) claim for user ID"
        assert "client_id" in claims, "JWT should contain 'client_id' claim"
        assert "username" in claims, "JWT should contain 'username' claim"

    def test_agent_invocation_with_jwt_weather_prompt(self, jwt_token: str):
        """Test agent invocation with JWT token using a weather-related prompt."""
        prompt = "What is the weather like in Seattle?"
        response = self._invoke_agent_with_jwt(jwt_token, prompt)

        assert response is not None
        assert isinstance(response, dict)
        # Since this is a weather agent, it should be able to handle weather queries

    def test_agent_invocation_with_invalid_token(self):
        """Test that agent invocation fails with an invalid JWT token."""
        invalid_token = "invalid.jwt.token"

        with pytest.raises(Exception) as exc_info:
            self._invoke_agent_with_jwt(invalid_token, "Hello!")

        # Should fail with authentication error
        assert "401" in str(exc_info.value) or "403" in str(exc_info.value)

    def test_agent_invocation_with_expired_token(self):
        """Test that agent invocation fails with an expired JWT token."""
        # This is a known expired JWT token (expired in the past)
        expired_token = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiIxMjM0NTY3ODkwIiwibmFtZSI6IkpvaG4gRG9lIiwiaWF0IjoxNTE2MjM5MDIyLCJleHAiOjE1MTYyMzkwMjJ9.invalid"

        with pytest.raises(Exception) as exc_info:
            self._invoke_agent_with_jwt(expired_token, "Hello!")

        # Should fail with authentication error
        assert "401" in str(exc_info.value) or "403" in str(exc_info.value)

    def _invoke_agent_with_jwt(self, access_token: str, prompt: str) -> dict[str, Any]:
        """Helper method to invoke the agent runtime using JWT bearer token."""
        agent_runtime_arn = os.environ["AGENT_RUNTIME_ARN"]
        region = os.environ["AWS_REGION"]

        # URL encode the agent ARN
        escaped_agent_arn = urllib.parse.quote(agent_runtime_arn, safe="")

        # Construct the URL
        url = f"https://bedrock-agentcore.{region}.amazonaws.com/runtimes/{escaped_agent_arn}/invocations?qualifier=DEFAULT"

        # Set up headers
        session_id = f"test-session-{uuid.uuid4().hex}"  # Generates 44 character string
        trace_id = f"test-trace-{uuid.uuid4().hex[:16]}"  # Shorter trace ID

        headers = {
            "Authorization": f"Bearer {access_token}",
            "X-Amzn-Trace-Id": trace_id,
            "Content-Type": "application/json",
            "X-Amzn-Bedrock-AgentCore-Runtime-Session-Id": session_id,
        }

        # Prepare payload in the correct format expected by AgentCore
        payload = {"input": {"prompt": prompt}}

        response = requests.post(
            url, headers=headers, data=json.dumps(payload), timeout=60
        )

        if response.status_code == 200:
            return response.json()
        else:
            # Try to get error details
            try:
                error_data = response.json()
                error_msg = f"Agent invocation failed with status {response.status_code}: {json.dumps(error_data)}"
            except json.JSONDecodeError:
                error_msg = f"Agent invocation failed with status {response.status_code}: {response.text}"

            raise Exception(error_msg)
