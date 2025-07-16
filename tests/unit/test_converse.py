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


class TestConverse:
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

    def test_converse(self, mock_agent_1, mock_bedrock_converse):
        mock_bedrock_converse.response_generator = TestConverse.response_generator

        text = mock_agent_1.converse(
            "Hello",
        )

        assert text.strip() == "You said: Hello"
        Expect(mock_agent_1.traces).agent_text_response.with_fn(
            lambda r: r.strip()
        ).to_equal("You said: Hello")

        cycles = [
            trace
            for trace in mock_agent_1.traces
            if trace.attributes.get("ai.trace.type") == "cycle"
        ]
        assert len(cycles) == 2

        for index, cycle in enumerate(cycles):
            assert cycle.attributes.get("ai.agent.cycle.nr") == index

        assert not cycles[0].attributes.get("ai.agent.cycle.response")
        assert cycles[1].attributes.get("ai.agent.cycle.response") == "You said: Hello"

    def test_converse_with_converse_stream_backend(
        self, mock_agent_1, mock_bedrock_converse
    ):
        mock_agent_1.converse_stream_backend = "converse_stream"
        mock_bedrock_converse.response_generator = TestConverse.response_generator

        text = mock_agent_1.converse(
            "Hello",
        )

        assert text.strip() == "You said: Hello"
        Expect(mock_agent_1.traces).agent_text_response.with_fn(
            lambda r: r.strip()
        ).to_equal("You said: Hello")

        cycles = [
            trace
            for trace in mock_agent_1.traces
            if trace.attributes.get("ai.trace.type") == "cycle"
        ]
        assert len(cycles) == 2

        assert not cycles[0].attributes.get("ai.agent.cycle.response")
        assert cycles[1].attributes.get("ai.agent.cycle.response") == "You said: Hello"

        for index, cycle in enumerate(cycles):
            assert cycle.attributes.get("ai.agent.cycle.nr") == index
