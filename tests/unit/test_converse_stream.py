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

from generative_ai_toolkit.test import Expect


class TestConverseStream:
    @staticmethod
    def response_generator(
        mock,
        request,
    ):
        if "toolConfig" in request and "tools" in request["toolConfig"]:
            mock.add_output(
                tool_use_output=[
                    {
                        "name": tool["toolSpec"]["name"],
                        "input": {"test_input": "foobar"},
                    }
                    for tool in request["toolConfig"]["tools"]
                    if "toolSpec" in tool
                ]
            )
        if "messages" in request:
            last_msg = request["messages"][-1]["content"][0].get("text")
            mock.add_output(f"You said: {last_msg}")

    def test_converse_stream_text_by_default(self, mock_agent_1, mock_bedrock_converse):
        mock_bedrock_converse.response_generator = TestConverseStream.response_generator

        text = "".join(
            mock_agent_1.converse_stream(
                "Hello",
            )
        )

        assert text.strip() == "You said: Hello"
        Expect(mock_agent_1.traces).agent_text_response.with_fn(
            lambda r: r.strip()
        ).to_equal("You said: Hello")

    def test_converse_stream_text(self, mock_agent_1, mock_bedrock_converse):
        mock_bedrock_converse.response_generator = TestConverseStream.response_generator

        text = "".join(
            mock_agent_1.converse_stream(
                "Hello",
                stream="text",
            )
        )

        assert text.strip() == "You said: Hello"
        Expect(mock_agent_1.traces).agent_text_response.with_fn(
            lambda r: r.strip()
        ).to_equal("You said: Hello")

    def test_converse_stream_traces(self, mock_agent_1, mock_bedrock_converse):
        mock_bedrock_converse.response_generator = TestConverseStream.response_generator

        traces1 = [
            trace
            for trace in mock_agent_1.converse_stream(
                "Hello 1",
                stream="traces",
            )
            if trace.ended_at is not None
        ]

        Expect(traces1).agent_text_response.with_fn(lambda r: r.strip()).to_equal(
            "You said: Hello 1"
        )

        assert mock_agent_1.traces == sorted(traces1, key=lambda t: t.started_at)

        traces2 = [
            trace
            for trace in mock_agent_1.converse_stream(
                "Hello 2",
                stream="traces",
            )
            if trace.ended_at is not None
        ]

        Expect(traces2).agent_text_response.with_fn(lambda r: r.strip()).to_equal(
            "You said: Hello 2"
        )

        assert mock_agent_1.traces == sorted(
            traces1 + traces2, key=lambda t: t.started_at
        )
