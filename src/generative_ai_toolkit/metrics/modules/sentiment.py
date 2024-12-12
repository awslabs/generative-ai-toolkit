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

import boto3

from generative_ai_toolkit.metrics import BaseMetric, Measurement
from generative_ai_toolkit.tracer import LlmTrace


class SentimentMetric(BaseMetric):
    """
    SentimentMetric class for measuring the sentiment in model outputs using Amazon Comprehend.

    This metric evaluates text output from a model, assigns a sentiment score to each message, and appends the sentiment
    score to the message as 'extra'. The validation passes only if all messages have a non-negative sentiment score.
    """

    def __init__(self):
        """
        Initialize the SentimentMetric.

        This constructor initializes the SentimentMetric instance by calling the parent class constructor.
        """
        super().__init__()
        self.comprehend_client = boto3.client("comprehend")

    def evaluate_trace(self, trace, **kwargs) -> Measurement | None:
        """
        Evaluate the trace for sentiment using Amazon Comprehend.

        This method processes each message in both the trace's request and response content, evaluates it using Amazon
        Comprehend's sentiment analysis API, and appends the sentiment score as 'extra' (as a JSON string) to each
        message. If any message has a negative sentiment score, the sentiment compound value is reflected in the final
        measurement, otherwise the value is set to 0.

        Args:
            trace (LlmTrace): The trace object containing the request and response to the LLM.
            **kwargs: Additional keyword arguments (not used in this implementation).

        Returns:
            Measurement | None: A Measurement object with the evaluation results, including sentiment information.
                                If the trace is not an instance of LlmTrace, it returns None.

        Raises:
            Exception: If an error occurs during the evaluation, it prints an error message.
        """

        if not isinstance(trace, LlmTrace):
            return None

        # The last message from the user:
        user_messages = [
            msg["text"] for msg in trace.user_conversation if msg["role"] == "user"
        ]
        last_user_message = user_messages[-1]

        # The last message from the agent:
        agent_messages = [
            msg["text"] for msg in trace.user_conversation if msg["role"] == "assistant"
        ]
        last_agent_message = agent_messages[-1]

        # Initialize variables
        validation_passed = True
        additional_info = {}
        overall_value = 0.0

        #####
        # Evaluate the request message
        #####

        result = self.comprehend_client.detect_sentiment(
            Text=last_user_message, LanguageCode="en"
        )
        # Simplify the result to only include Sentiment and SentimentScore
        sentiment = {
            "Sentiment": result["Sentiment"],
            "SentimentScore": result["SentimentScore"],
        }
        additional_info["request_sentiment"] = sentiment

        if result["Sentiment"] == "NEGATIVE":
            overall_value += result["SentimentScore"]["Negative"]
            validation_passed = False

        #####
        # Evaluate the response messages
        #####

        result = self.comprehend_client.detect_sentiment(
            Text=last_agent_message, LanguageCode="en"
        )

        # Simplify the result to only include Sentiment and SentimentScore
        sentiment = {
            "Sentiment": result["Sentiment"],
            "SentimentScore": result["SentimentScore"],
        }
        additional_info["response_sentiment"] = sentiment

        if result["Sentiment"] == "NEGATIVE":
            overall_value += result["SentimentScore"]["Negative"]
            validation_passed = False

        # Return the Measurement object after all messages have been evaluated
        return Measurement(
            name="Sentiment",
            value=overall_value,  # Reflect the negative sentiment value
            validation_passed=validation_passed,
            additional_info=additional_info,
        )
