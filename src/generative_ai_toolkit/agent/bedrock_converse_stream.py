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

import json
from typing import (
    TYPE_CHECKING,
    TypedDict,
    cast,
)

if TYPE_CHECKING:
    from mypy_boto3_bedrock_runtime.type_defs import (
        ContentBlockOutputTypeDef,
        ConverseStreamOutputTypeDef,
        MessageOutputTypeDef,
    )


class ScratchpadToolUse(TypedDict):
    name: str
    toolUseId: str
    input: str


class ScratchpadReasoningContent(TypedDict):
    text: str
    signature: str
    redactedContent: bytes


class Scratchpad(TypedDict):
    text: str
    toolUse: ScratchpadToolUse
    reasoningContent: ScratchpadReasoningContent


class BedrockConverseStreamEventContentBlockHandler:

    def __init__(self) -> None:
        self.finalized_blocks: list[ContentBlockOutputTypeDef] = []
        self.scratchpad: dict[int, Scratchpad] = {}
        self.nr_of_events_handled = 0

    def ensure_scratchpad(self, index: int):
        if index not in self.scratchpad:
            self.scratchpad[index] = {
                "text": "",
                "toolUse": {
                    "name": "",
                    "toolUseId": "",
                    "input": "",
                },
                "reasoningContent": {
                    "text": "",
                    "signature": "",
                    "redactedContent": b"",
                },
            }
        return self.scratchpad[index]

    def process_stream_event(self, stream_event: "ConverseStreamOutputTypeDef"):
        if "contentBlockStart" in stream_event:
            index = stream_event["contentBlockStart"]["contentBlockIndex"]
            current_block = self.ensure_scratchpad(index)
            tool_use = stream_event["contentBlockStart"]["start"].get("toolUse")
            if tool_use:
                current_block["toolUse"]["name"] = tool_use["name"]
                current_block["toolUse"]["toolUseId"] = tool_use["toolUseId"]
        elif "contentBlockDelta" in stream_event:
            index = stream_event["contentBlockDelta"]["contentBlockIndex"]
            current_block = self.ensure_scratchpad(index)
            delta = stream_event["contentBlockDelta"]["delta"]
            if "text" in delta:
                current_block["text"] += delta["text"]
            elif "toolUse" in delta and "input" in delta["toolUse"]:
                current_block["toolUse"]["input"] = (
                    current_block["toolUse"].get("input", "")
                    + delta["toolUse"]["input"]
                )
            elif "reasoningContent" in delta:
                if "text" in delta["reasoningContent"]:
                    current_block["reasoningContent"]["text"] += delta[
                        "reasoningContent"
                    ]["text"]
                if "signature" in delta["reasoningContent"]:
                    current_block["reasoningContent"]["signature"] = delta[
                        "reasoningContent"
                    ]["signature"]
                if "redactedContent" in delta["reasoningContent"]:
                    current_block["reasoningContent"]["redactedContent"] = delta[
                        "reasoningContent"
                    ]["redactedContent"]
        elif "contentBlockStop" in stream_event:
            index = stream_event["contentBlockStop"]["contentBlockIndex"]
            self.finalize_block(index)
        self.nr_of_events_handled += 1

    def finalize_block(self, index: int):
        current_block = self.scratchpad[index]
        content_block: ContentBlockOutputTypeDef = {}
        if current_block["toolUse"]["toolUseId"]:
            try:
                content_block["toolUse"] = {
                    "name": current_block["toolUse"]["name"],
                    "toolUseId": current_block["toolUse"]["toolUseId"],
                    "input": (
                        json.loads(current_block["toolUse"]["input"])
                        if current_block["toolUse"].get("input")
                        else {}
                    ),
                }
            except json.decoder.JSONDecodeError as err:
                raise ValueError(
                    f"Failed to JSON parse '{current_block['toolUse']['input']}' because: {err}"
                ) from err
        if current_block["text"]:
            content_block["text"] = current_block["text"]

        if current_block["reasoningContent"]["text"]:
            content_block["reasoningContent"] = {
                "reasoningText": {
                    "text": current_block["reasoningContent"]["text"],
                    "signature": current_block["reasoningContent"]["signature"],
                },
            }
            redacted_content = current_block["reasoningContent"].get("redactedContent")
            if redacted_content:
                content_block["reasoningContent"]["redactedContent"] = redacted_content

        self.finalized_blocks.append(content_block)

    def get_message(self, provisional=False) -> "MessageOutputTypeDef":
        content_blocks = self.finalized_blocks[:]
        if provisional and len(self.scratchpad) > len(self.finalized_blocks):
            for scratchpad_block in list(self.scratchpad.values())[
                len(self.finalized_blocks) :
            ]:
                provisional_block = {}
                if scratchpad_block.get("text"):
                    provisional_block["text"] = scratchpad_block["text"]
                if scratchpad_block["toolUse"].get("toolUseId"):
                    provisional_block["toolUse"] = scratchpad_block["toolUse"]
                if scratchpad_block["reasoningContent"].get("text"):
                    provisional_block["reasoningContent"] = scratchpad_block[
                        "reasoningContent"
                    ]
                content_blocks.append(
                    cast("ContentBlockOutputTypeDef", provisional_block)
                )
        return {"role": "assistant", "content": content_blocks}
