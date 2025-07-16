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

from generative_ai_toolkit.ui import chat_ui


@pytest.fixture
def mock_agent1_chat_ui(mock_agent_1):
    demo = chat_ui(mock_agent_1)
    try:
        _, url, _ = demo.launch(prevent_thread_lock=True, quiet=True)
        yield url
    finally:
        demo.close()


def test_gradio_ui(page, mock_bedrock_converse, mock_agent1_chat_ui):
    mock_bedrock_converse.add_output("Hello, human!")
    page.goto(mock_agent1_chat_ui)
    page.fill("#user-input textarea", "Hello, assistant!")
    page.press("#user-input textarea", "Enter")
    page.wait_for_selector('.bot .message .message-content p:has-text("Hello, human")')
