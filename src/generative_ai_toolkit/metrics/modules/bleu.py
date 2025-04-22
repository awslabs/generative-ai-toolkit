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

from typing import cast

from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu

from generative_ai_toolkit.metrics import BaseMetric, Measurement
from generative_ai_toolkit.test import CaseTrace, user_conversation_from_trace


class BleuMetric(BaseMetric):
    """
    This metric measures whether a conversation runs as intended,
    by comparing everything that was said by user and agent in a predefined test case.
    """

    def evaluate_conversation(self, conversation_traces, **kwargs):

        for trace in reversed(conversation_traces):
            if not isinstance(trace, CaseTrace):
                continue
            if trace.attributes.get("ai.trace.type") == "llm-invocation":
                break
        else:
            return

        case = trace.case
        if not case.expected_agent_responses_per_turn:
            return

        user_conversation = user_conversation_from_trace(trace)

        min_bleu = 1
        actual_responses = [
            msg["text"] for msg in user_conversation if msg["role"] == "assistant"
        ]

        for index, turn in enumerate(case.expected_agent_responses_per_turn):
            try:
                actual_response = actual_responses[index]
            except IndexError as e:
                raise Exception(
                    f"Not enough actual responses ({len(actual_responses)}) to compare against the expectations ({len(case.expected_agent_responses_per_turn)}) at index {index}"
                ) from e
            references = [clean_and_split(expected) for expected in turn]
            hypothesis = clean_and_split(actual_response)
            bleu_score = float(
                cast(
                    float,
                    sentence_bleu(
                        references,
                        hypothesis,
                        smoothing_function=SmoothingFunction().method1,
                    ),
                ),
            )
            min_bleu = min(min_bleu, bleu_score)

        return Measurement(
            name="BleuScore",
            value=min_bleu,
            additional_info={
                "conversation": user_conversation,
                "expected_responses": case.expected_agent_responses_per_turn,
            },
        )


def clean_and_split(s: str):
    return s.replace("\n", " ").strip().split()
