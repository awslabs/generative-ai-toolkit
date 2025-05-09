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

from unittest.mock import MagicMock

import pytest
from sample_agent_1 import sample_agent_1
from sample_agent_2 import sample_agent_2

from generative_ai_toolkit.test.mock import MockBedrockConverse


@pytest.fixture
def mock_bedrock_converse():
    mock = MockBedrockConverse()
    yield mock
    if mock.mock_responses:
        raise Exception("Still have unconsumed mock responses")


@pytest.fixture
def mock_agent_1(mock_bedrock_converse):
    yield sample_agent_1(session=mock_bedrock_converse.session())


@pytest.fixture
def mock_agent_2(mock_bedrock_converse):
    yield sample_agent_2(session=mock_bedrock_converse.session())


@pytest.fixture
def mock_bedrock_converse_stream_spaghetti_recipe_session():
    mock_client = MagicMock(name="MockClient")
    mock_session = MagicMock(name="MockSession")
    mock_session.client.return_value = mock_client

    def response_stream_1():
        yield from [
            {"messageStart": {"role": "assistant"}},
            {
                "contentBlockDelta": {
                    "delta": {"reasoningContent": {"text": "The"}},
                    "contentBlockIndex": 0,
                }
            },
            {
                "contentBlockDelta": {
                    "delta": {"reasoningContent": {"text": " user"}},
                    "contentBlockIndex": 0,
                }
            },
            {
                "contentBlockDelta": {
                    "delta": {"reasoningContent": {"text": " is"}},
                    "contentBlockIndex": 0,
                }
            },
            {
                "contentBlockDelta": {
                    "delta": {"reasoningContent": {"text": " asking"}},
                    "contentBlockIndex": 0,
                }
            },
            {
                "contentBlockDelta": {
                    "delta": {"reasoningContent": {"text": " for"}},
                    "contentBlockIndex": 0,
                }
            },
            {
                "contentBlockDelta": {
                    "delta": {"reasoningContent": {"text": " a"}},
                    "contentBlockIndex": 0,
                }
            },
            {
                "contentBlockDelta": {
                    "delta": {"reasoningContent": {"text": " recipe for"}},
                    "contentBlockIndex": 0,
                }
            },
            {
                "contentBlockDelta": {
                    "delta": {"reasoningContent": {"text": " Spagh"}},
                    "contentBlockIndex": 0,
                }
            },
            {
                "contentBlockDelta": {
                    "delta": {"reasoningContent": {"text": "etti Carbonara"}},
                    "contentBlockIndex": 0,
                }
            },
            {
                "contentBlockDelta": {
                    "delta": {"reasoningContent": {"text": ". I"}},
                    "contentBlockIndex": 0,
                }
            },
            {
                "contentBlockDelta": {
                    "delta": {"reasoningContent": {"text": " have"}},
                    "contentBlockIndex": 0,
                }
            },
            {
                "contentBlockDelta": {
                    "delta": {"reasoningContent": {"text": " a"}},
                    "contentBlockIndex": 0,
                }
            },
            {
                "contentBlockDelta": {
                    "delta": {"reasoningContent": {"text": " tool"}},
                    "contentBlockIndex": 0,
                }
            },
            {
                "contentBlockDelta": {
                    "delta": {"reasoningContent": {"text": " available"}},
                    "contentBlockIndex": 0,
                }
            },
            {
                "contentBlockDelta": {
                    "delta": {"reasoningContent": {"text": " calle"}},
                    "contentBlockIndex": 0,
                }
            },
            {
                "contentBlockDelta": {
                    "delta": {"reasoningContent": {"text": "d `"}},
                    "contentBlockIndex": 0,
                }
            },
            {
                "contentBlockDelta": {
                    "delta": {"reasoningContent": {"text": "get"}},
                    "contentBlockIndex": 0,
                }
            },
            {
                "contentBlockDelta": {
                    "delta": {"reasoningContent": {"text": "_recipe` that"}},
                    "contentBlockIndex": 0,
                }
            },
            {
                "contentBlockDelta": {
                    "delta": {"reasoningContent": {"text": " can provide recipes"}},
                    "contentBlockIndex": 0,
                }
            },
            {
                "contentBlockDelta": {
                    "delta": {"reasoningContent": {"text": "."}},
                    "contentBlockIndex": 0,
                }
            },
            {
                "contentBlockDelta": {
                    "delta": {"reasoningContent": {"text": "\n\nThe"}},
                    "contentBlockIndex": 0,
                }
            },
            {
                "contentBlockDelta": {
                    "delta": {"reasoningContent": {"text": " require"}},
                    "contentBlockIndex": 0,
                }
            },
            {
                "contentBlockDelta": {
                    "delta": {"reasoningContent": {"text": "d parameter"}},
                    "contentBlockIndex": 0,
                }
            },
            {
                "contentBlockDelta": {
                    "delta": {"reasoningContent": {"text": " for"}},
                    "contentBlockIndex": 0,
                }
            },
            {
                "contentBlockDelta": {
                    "delta": {"reasoningContent": {"text": " this function"}},
                    "contentBlockIndex": 0,
                }
            },
            {
                "contentBlockDelta": {
                    "delta": {"reasoningContent": {"text": " is:"}},
                    "contentBlockIndex": 0,
                }
            },
            {
                "contentBlockDelta": {
                    "delta": {"reasoningContent": {"text": "\n-"}},
                    "contentBlockIndex": 0,
                }
            },
            {
                "contentBlockDelta": {
                    "delta": {"reasoningContent": {"text": " dish: The name"}},
                    "contentBlockIndex": 0,
                }
            },
            {
                "contentBlockDelta": {
                    "delta": {"reasoningContent": {"text": " of the dish to"}},
                    "contentBlockIndex": 0,
                }
            },
            {
                "contentBlockDelta": {
                    "delta": {"reasoningContent": {"text": " get a recipe for"}},
                    "contentBlockIndex": 0,
                }
            },
            {
                "contentBlockDelta": {
                    "delta": {"reasoningContent": {"text": "\n\nIn"}},
                    "contentBlockIndex": 0,
                }
            },
            {
                "contentBlockDelta": {
                    "delta": {"reasoningContent": {"text": " this"}},
                    "contentBlockIndex": 0,
                }
            },
            {
                "contentBlockDelta": {
                    "delta": {"reasoningContent": {"text": " case, the dish"}},
                    "contentBlockIndex": 0,
                }
            },
            {
                "contentBlockDelta": {
                    "delta": {"reasoningContent": {"text": ' is "S'}},
                    "contentBlockIndex": 0,
                }
            },
            {
                "contentBlockDelta": {
                    "delta": {"reasoningContent": {"text": "paghetti"}},
                    "contentBlockIndex": 0,
                }
            },
            {
                "contentBlockDelta": {
                    "delta": {"reasoningContent": {"text": ' Carbonara". This'}},
                    "contentBlockIndex": 0,
                }
            },
            {
                "contentBlockDelta": {
                    "delta": {"reasoningContent": {"text": " is clearly"}},
                    "contentBlockIndex": 0,
                }
            },
            {
                "contentBlockDelta": {
                    "delta": {"reasoningContent": {"text": " stated in the user"}},
                    "contentBlockIndex": 0,
                }
            },
            {
                "contentBlockDelta": {
                    "delta": {"reasoningContent": {"text": "'s request,"}},
                    "contentBlockIndex": 0,
                }
            },
            {
                "contentBlockDelta": {
                    "delta": {"reasoningContent": {"text": " so I"}},
                    "contentBlockIndex": 0,
                }
            },
            {
                "contentBlockDelta": {
                    "delta": {"reasoningContent": {"text": " can call"}},
                    "contentBlockIndex": 0,
                }
            },
            {
                "contentBlockDelta": {
                    "delta": {"reasoningContent": {"text": " the function"}},
                    "contentBlockIndex": 0,
                }
            },
            {
                "contentBlockDelta": {
                    "delta": {"reasoningContent": {"text": " with this"}},
                    "contentBlockIndex": 0,
                }
            },
            {
                "contentBlockDelta": {
                    "delta": {"reasoningContent": {"text": " parameter."}},
                    "contentBlockIndex": 0,
                }
            },
            {
                "contentBlockDelta": {
                    "delta": {
                        "reasoningContent": {
                            "signature": "ErcBCkgIAxABGAIiQGe2vlYmGavCTzCi6P+Ur23sKQwAgQ1SXGV+Qo++w+vr2Rwn1mCyu706GfqjmKp7lx6CnFfHBJj3fFp7kTlJB6MSDGLA/3wQ6NXSnYVtPxoM1TWYaoh7l8fbOTRHIjCQkiTU4Lz3b2j/VCc0jUJoY6kAr0XPv4HYlrof3Xtx9Bc46guGHv/H35kJVfQwhowqHRuvQD9PFtZHu35tnx3VNZagLfRzPeK+MFxzER+y"
                        }
                    },
                    "contentBlockIndex": 0,
                }
            },
            {"contentBlockStop": {"contentBlockIndex": 0}},
            {"contentBlockDelta": {"delta": {"text": "I can"}, "contentBlockIndex": 1}},
            {"contentBlockDelta": {"delta": {"text": " help"}, "contentBlockIndex": 1}},
            {"contentBlockDelta": {"delta": {"text": " you"}, "contentBlockIndex": 1}},
            {"contentBlockDelta": {"delta": {"text": " with"}, "contentBlockIndex": 1}},
            {"contentBlockDelta": {"delta": {"text": " a"}, "contentBlockIndex": 1}},
            {
                "contentBlockDelta": {
                    "delta": {"text": " recipe for S"},
                    "contentBlockIndex": 1,
                }
            },
            {
                "contentBlockDelta": {
                    "delta": {"text": "paghetti"},
                    "contentBlockIndex": 1,
                }
            },
            {
                "contentBlockDelta": {
                    "delta": {"text": " Carbonara!"},
                    "contentBlockIndex": 1,
                }
            },
            {
                "contentBlockDelta": {
                    "delta": {"text": " Let me get"},
                    "contentBlockIndex": 1,
                }
            },
            {"contentBlockDelta": {"delta": {"text": " that"}, "contentBlockIndex": 1}},
            {
                "contentBlockDelta": {
                    "delta": {"text": " for you."},
                    "contentBlockIndex": 1,
                }
            },
            {"contentBlockStop": {"contentBlockIndex": 1}},
            {
                "contentBlockStart": {
                    "start": {
                        "toolUse": {
                            "toolUseId": "tooluse_W0JfmJb6Si61QMXXy0ynYw",
                            "name": "get_recipe",
                        }
                    },
                    "contentBlockIndex": 2,
                }
            },
            {
                "contentBlockDelta": {
                    "delta": {"toolUse": {"input": ""}},
                    "contentBlockIndex": 2,
                }
            },
            {
                "contentBlockDelta": {
                    "delta": {"toolUse": {"input": '{"dish"'}},
                    "contentBlockIndex": 2,
                }
            },
            {
                "contentBlockDelta": {
                    "delta": {"toolUse": {"input": ': "Spaghet'}},
                    "contentBlockIndex": 2,
                }
            },
            {
                "contentBlockDelta": {
                    "delta": {"toolUse": {"input": "ti Car"}},
                    "contentBlockIndex": 2,
                }
            },
            {
                "contentBlockDelta": {
                    "delta": {"toolUse": {"input": 'bonara"}'}},
                    "contentBlockIndex": 2,
                }
            },
            {"contentBlockStop": {"contentBlockIndex": 2}},
            {"messageStop": {"stopReason": "tool_use"}},
            {
                "metadata": {
                    "usage": {
                        "inputTokens": 431,
                        "outputTokens": 168,
                        "totalTokens": 599,
                    },
                    "metrics": {"latencyMs": 4600},
                }
            },
        ]

    def response_stream_2():
        yield from [
            {"messageStart": {"role": "assistant"}},
            {
                "contentBlockDelta": {
                    "delta": {"text": "Here is the recipe bla bla"},
                    "contentBlockIndex": 0,
                }
            },
            {"contentBlockStop": {"contentBlockIndex": 0}},
            {"messageStop": {"stopReason": "end_turn"}},
            {
                "metadata": {
                    "usage": {
                        "inputTokens": 431,
                        "outputTokens": 168,
                        "totalTokens": 599,
                    },
                    "metrics": {"latencyMs": 4600},
                }
            },
        ]

    response_streams = [response_stream_1(), response_stream_2()]

    def converse_stream(*args, **kwargs):
        return {
            "ResponseMetadata": {
                "RequestId": "16642fc8-2ac3-43fd-bcea-5779911b54fd",
                "HTTPStatusCode": 200,
                "HTTPHeaders": {
                    "date": "Mon, 05 May 2025 10:49:44 GMT",
                    "content-type": "application/vnd.amazon.eventstream",
                    "transfer-encoding": "chunked",
                    "connection": "keep-alive",
                    "x-amzn-requestid": "16642fc8-2ac3-43fd-bcea-5779911b54fd",
                },
                "RetryAttempts": 0,
            },
            "stream": response_streams.pop(0),
        }

    mock_client.converse_stream = converse_stream
    yield mock_session
    if response_streams:
        raise Exception(f"Unconsumed response streams left! {response_streams}")
