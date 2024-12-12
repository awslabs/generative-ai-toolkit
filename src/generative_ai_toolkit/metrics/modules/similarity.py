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

import json
from threading import Lock
from concurrent.futures import ThreadPoolExecutor, Future

import boto3.session
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from generative_ai_toolkit.metrics import BaseMetric, Measurement
from generative_ai_toolkit.test import Case, LlmCaseTrace


class AgentResponseSimilarityMetric(BaseMetric):
    """
    This metric measures whether a conversation runs as intended,
    by comparing everything that was said by user and agent in a predefined test case.
    """

    expected_embeddings: dict[Case, list[list[np.ndarray]]]
    preparing: dict[Case, Future]

    def __init__(
        self,
        session: boto3.session.Session | None = None,
        embeddings_model_id="amazon.titan-embed-text-v2:0",
    ):
        super().__init__()
        self.bedrock_client = (session or boto3).client("bedrock-runtime")
        self.embeddings_model_id = embeddings_model_id
        self.expected_embeddings = {}
        self.lock = Lock()

    def prepare(self, case: Case):
        with self.lock:
            if case in self.expected_embeddings:
                return
            with ThreadPoolExecutor() as executor:
                self.expected_embeddings[case] = []
                futures: list[list[Future]] = []
                for turn in case.expected_agent_responses_per_turn:
                    turn_futures: list[Future] = []
                    futures.append(turn_futures)
                    for text in turn:
                        turn_futures.append(executor.submit(self.get_embedding, text))
                for turn in futures:
                    self.expected_embeddings[case].append(
                        [future.result() for future in turn]
                    )
                assert (
                    len(self.expected_embeddings[case])
                    == len(case.expected_agent_responses_per_turn)
                ), f"{len(self.expected_embeddings[case])} != {len(case.expected_agent_responses_per_turn)}"

    def get_embedding(self, text: str):
        response = self.bedrock_client.invoke_model(
            body=json.dumps({"inputText": text, "dimensions": 256, "normalize": True}),
            modelId=self.embeddings_model_id,
            accept="application/json",
            contentType="application/json",
        )
        response_body = json.loads(response.get("body").read())
        return np.array(response_body["embedding"]).reshape(1, -1)

    def evaluate_conversation(self, conversation_traces, **kwargs):
        trace = conversation_traces[
            -1
        ]  # Use the conversation as captured in the last trace
        if not isinstance(trace, LlmCaseTrace):
            return

        case = trace.case
        if not case.expected_agent_responses_per_turn:
            return

        self.prepare(case)

        min_similarity = 1
        actual_responses = [
            msg["text"] for msg in trace.user_conversation if msg["role"] == "assistant"
        ]

        for index, turn in enumerate(self.expected_embeddings[case]):
            try:
                actual_response = actual_responses[index]
            except IndexError as e:
                raise Exception(
                    f"Not enough actual responses ({len(actual_responses)}) to compare against the expectations ({len(case.expected_agent_responses_per_turn)}) at index {index}"
                ) from e
            actual = self.get_embedding(actual_response)
            turn_max_similarity = -1
            for expected_embedding in turn:
                similarity = cosine_similarity(expected_embedding, actual)[0][0]
                if similarity > turn_max_similarity:
                    turn_max_similarity = similarity
            if turn_max_similarity < min_similarity:
                min_similarity = turn_max_similarity

        return Measurement(
            name="CosineSimilarity",
            value=min_similarity,
        )
