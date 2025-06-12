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

import queue
import threading
from typing import Literal

from generative_ai_toolkit.agent import Agent
from generative_ai_toolkit.tracer import BaseTracer, ChainableTracer, Trace
from generative_ai_toolkit.tracer.context import TraceContextProvider

TraceItem = tuple[Literal["trace"], Trace]
ErrorItem = tuple[Literal["error"], Exception]
EndItem = tuple[Literal["end"], None]


def converse_stream_with_traces(agent: Agent, user_input: str):
    """
    Stream traces (including previews)
    """

    if not isinstance(agent.tracer, ChainableTracer):
        raise RuntimeError(
            "Agent tracer must be a ChainableTracer (support add_tracer() and remove_tracer()) for use with converse_stream_with_traces"
        )

    item_queue = queue.Queue[TraceItem | ErrorItem | EndItem]()
    tracer = QueueTracer(item_queue)
    agent.tracer.add_tracer(tracer)
    try:

        def produce_traces():
            try:
                for _ in agent.converse_stream(user_input):
                    pass
            except Exception as e:
                item_queue.put(("error", e))
            else:
                item_queue.put(("end", None))

        text_thread = threading.Thread(target=produce_traces, daemon=True)
        text_thread.start()

        while True:
            item = item_queue.get()
            if item[0] == "end":
                break
            elif item[0] == "error":
                raise item[1]
            yield item[1]
    finally:
        agent.tracer.remove_tracer(tracer)
        item_queue.shutdown()


class QueueTracer(BaseTracer):

    def __init__(
        self,
        queue: queue.Queue,
        trace_context_provider: TraceContextProvider | None = None,
    ):
        super().__init__(trace_context_provider)
        self.queue = queue

    def persist(self, trace: Trace):
        self.queue.put(("trace", trace))

    def preview(self, trace: Trace):
        self.queue.put(("trace", trace))
