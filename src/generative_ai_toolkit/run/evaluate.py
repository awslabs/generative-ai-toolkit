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

import os
import random
import time
from collections.abc import Sequence
from itertools import groupby

import boto3
from boto3.dynamodb.types import TypeDeserializer

from generative_ai_toolkit.evaluate import GenerativeAIToolkit
from generative_ai_toolkit.metrics import BaseMetric
from generative_ai_toolkit.tracer import Trace
from generative_ai_toolkit.tracer.dynamodb import DynamoDbTracer
from generative_ai_toolkit.utils.logging import logger


class _AWSLambdaRunner:
    _metrics: Sequence[BaseMetric] = []
    _agent_name: str

    @property
    def metrics(self) -> Sequence[BaseMetric]:
        return self._metrics

    @property
    def agent_name(self) -> str:
        return self._agent_name

    def configure(self, *, metrics: Sequence[BaseMetric], agent_name: str):
        self._metrics = metrics
        self._agent_name = agent_name

    def __call__(self, event, context):
        logger.debug("Received event", event=event)

        traces: list[Trace] = []

        for record in event["Records"]:
            event_name = record["eventName"]
            if event_name == "INSERT":
                new_image = unmarshall(record["dynamodb"].get("NewImage", {}))
                if "trace_id" not in new_image:
                    continue
                trace = DynamoDbTracer.item_to_trace(new_image)
                if "ai.conversation.id" not in trace.attributes:
                    logger.info(
                        "Skipping evaluation of trace without conversation id",
                        trace=trace.as_dict(),
                    )
                    continue
                traces.append(trace)

        traces.sort(key=lambda t: t.started_at)
        traces.sort(key=lambda t: t.attributes["ai.conversation.id"])

        conversations = [
            list(group)
            for _, group in groupby(
                traces, key=lambda t: t.attributes["ai.conversation.id"]
            )
        ]

        evaluation_timeout = (context.get_remaining_time_in_millis() / 1000) - 10
        measure(conversations, evaluation_timeout=evaluation_timeout)


AWSLambdaRunner = _AWSLambdaRunner()

deserializer = TypeDeserializer()

ssm = boto3.client("ssm")


def get_sampling_rate():
    """
    Get the sampling rate from SSM Parameter Store.
    """
    return max(
        0,
        min(
            100,
            int(
                ssm.get_parameter(
                    Name=os.environ["SAMPLING_RATE_PARAM_NAME"],
                )[
                    "Parameter"
                ]["Value"]
            ),
        ),
    )


def measure(
    conversations: Sequence[Sequence[Trace]], evaluation_timeout: float | None = None
):
    sampling_rate = get_sampling_rate()
    logger.info("Sampling rate", sampling_rate=sampling_rate)
    conversations_to_evaluate = [
        conversation
        for conversation in conversations
        if random.randint(1, 99) < sampling_rate
    ]
    logger.debug(
        "Evaluating traces",
        sampled_nr_of_conversations=len(conversations_to_evaluate),
        sampled_nr_of_traces=len(
            [_ for conversation in conversations_to_evaluate for _ in conversation]
        ),
        nr_of_conversations=len(conversations),
        nr_of_traces=len([_ for conversation in conversations for _ in conversation]),
    )

    logger.debug(
        "Starting GenerativeAIToolkit evaluation",
        evaluation_timeout=f"{evaluation_timeout:.1f}",
    )
    start = time.perf_counter()
    try:
        results = GenerativeAIToolkit.eval(
            traces=conversations_to_evaluate,
            metrics=AWSLambdaRunner.metrics,
            timeout=evaluation_timeout,
        )
        for conversation_measurements in results:
            # Emit EMF logs for measurements at conversation level:
            last_trace = conversation_measurements.traces[-1].trace
            for measurement in conversation_measurements.measurements:
                logger.metric(
                    measurement,
                    conversation_id=conversation_measurements.conversation_id,
                    auth_context=last_trace.attributes.get("auth_context"),
                    additional_info=measurement.additional_info,
                    namespace="GenerativeAIToolkit",
                    common_dimensions={
                        "AgentName": AWSLambdaRunner.agent_name,
                    },
                    timestamp=int(last_trace.started_at.timestamp() * 1000),
                )
            # Emit EMF logs for measurements at trace level:
            for conversation_traces in conversation_measurements.traces:
                trace = conversation_traces.trace
                for measurement in conversation_traces.measurements:
                    logger.metric(
                        measurement,
                        conversation_id=conversation_measurements.conversation_id,
                        auth_context=trace.attributes.get("auth_context"),
                        trace_id=trace.trace_id,
                        additional_info=measurement.additional_info,
                        namespace="GenerativeAIToolkit",
                        common_dimensions={
                            "AgentName": AWSLambdaRunner.agent_name,
                        },
                        timestamp=int(trace.started_at.timestamp() * 1000),
                    )
    except TimeoutError:
        logger.error("GenerativeAIToolkit evaluation timed out")
        return
    finally:
        logger.debug(
            "GenerativeAIToolkit evaluation finished",
            seconds_elapsed=f"{(time.perf_counter() - start):.1f}",
        )


def unmarshall(dynamo_obj: dict) -> dict:
    """Convert a DynamoDB dict into a standard dict."""
    return {k: deserializer.deserialize(v) for k, v in dynamo_obj.items()}
