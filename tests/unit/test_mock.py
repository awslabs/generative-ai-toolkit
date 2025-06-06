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


import pytest


def test_mock_bedrock_converse(mock_bedrock_converse):
    mock_bedrock_converse.add_output("test1")
    mock_bedrock_converse.add_output(["test2"])
    mock_bedrock_converse.add_output(["test3", "test4"])
    assert len(mock_bedrock_converse.mock_responses) == 3
    client = mock_bedrock_converse.client()
    response1 = client.converse(messages=["dummy_input"])
    assert response1["output"] == {
        "message": {"role": "assistant", "content": [{"text": "test1"}]}
    }
    assert len(mock_bedrock_converse.mock_responses) == 2
    response2 = client.converse(messages=["dummy_input"])
    assert response2["output"] == {
        "message": {"role": "assistant", "content": [{"text": "test2"}]}
    }
    response3 = client.converse(messages=["dummy_input"])
    assert response3["output"] == {
        "message": {
            "role": "assistant",
            "content": [{"text": "test3"}, {"text": "test4"}],
        }
    }
    assert len(mock_bedrock_converse.mock_responses) == 0
    with pytest.raises(RuntimeError, match="Exhausted"):
        client.converse(messages=["dummy_input"])
