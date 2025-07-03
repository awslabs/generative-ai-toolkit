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


from generative_ai_toolkit.context import AuthContext
from generative_ai_toolkit.run.agent import Runner


def test_runner_happy_flow(mock_agent_1, mock_bedrock_converse):
    # Configure mock responses for a weather conversation
    mock_bedrock_converse.add_output(
        text_output=["Let me check the weather for you."],
        tool_use_output=[{"name": "weather_tool", "input": {"city": "Paris"}}],
    )
    mock_bedrock_converse.add_output(
        text_output=["The weather in Paris is 20 degrees celsius. It's quite pleasant!"]
    )
    mock_bedrock_converse.add_output(text_output=["20 degrees celsius."])

    # Custom auth context function that reads x-user-id header
    def test_auth_context_fn(request) -> AuthContext:
        user_id = request.headers.get("x-user-id")
        if not user_id:
            raise ValueError("Missing x-user-id header")
        return {"principal_id": user_id}

    # Configure Runner with agent factory and custom auth
    Runner.configure(
        agent=mock_agent_1,
        auth_context_fn=test_auth_context_fn,
    )

    # Test initial request
    with Runner().test_client() as client:
        response = client.post(
            "/",
            json={"user_input": "What's the weather like in Paris?"},
            headers={
                "Content-Type": "application/json",
                "x-user-id": "test-user-123",  # Set user ID in header
            },
        )

        # Assert response properties
        assert response.status_code == 200
        assert response.content_type == "text/plain; charset=utf-8"
        assert "x-conversation-id" in response.headers

        # Collect streaming response
        full_response = b"".join(response.response).decode()  # type: ignore
        assert "20 degrees celsius" in full_response

        conversation_id = response.headers["x-conversation-id"]

        # Test conversation continuation
        response2 = client.post(
            "/",
            json={"user_input": "What was that temperature again?"},
            headers={
                "Content-Type": "application/json",
                "x-conversation-id": conversation_id,
                "x-user-id": "test-user-123",  # Same user continues conversation
            },
        )

        assert response2.status_code == 200
        assert (
            response2.headers["x-conversation-id"] == conversation_id
        )  # Same conversation
        full_response2 = b"".join(response2.response).decode()  # type: ignore
        assert "20 degrees celsius" in full_response2
