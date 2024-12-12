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

from typing import Sequence

from generative_ai_toolkit.metrics.measurement import Measurement
from generative_ai_toolkit.tracer.tracer import Trace


class BaseMetric:
    """
    Base class for all metrics.

    Developers should subclass this base class, and implement either one of the below methods.
    """

    def evaluate_trace(
        self, trace: Trace, **kwargs
    ) -> Measurement | Sequence[Measurement] | None:
        """
        Evaluate the trace.
        You would use this for measurements that should be made on individual traces (e.g. LLM invocation latency).

        :param trace: Trace object that contains the request and response to the LLM, or the input and output of the Tool
        :return: Zero, one or more Measurement(s)
        """
        raise NotImplementedError(
            "Subclasses should implement either this method, or evaluate_conversation()"
        )

    def evaluate_conversation(
        self, conversation_traces: Sequence[Trace], **kwargs
    ) -> Measurement | Sequence[Measurement] | None:
        """
        Evaluate the conversation: the sequence of traces pertaining to one particular conversation between the user and the agent.
        You would use this for measurements that should be made across the entire conversation and must take all traces into account.

        :param conversation: Sequence of trace objects that contain the request and response to the LLM, or the input and output of the Tool
        :return: Zero, one or more Measurement(s)
        """
        raise NotImplementedError(
            "Subclasses should implement either this method, or evaluate_trace()"
        )
