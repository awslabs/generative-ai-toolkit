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

import time

import pytest

from generative_ai_toolkit.tracer.tracer import InMemoryTracer


class TestInMemoryTracer:
    """Test suite for the InMemoryTracer class covering core functionality."""

    @pytest.fixture
    def tracer(self):
        """Create an InMemoryTracer instance for testing."""
        tracer = InMemoryTracer()
        tracer.set_context(resource_attributes={"service.name": "TestAgent"})
        return tracer

    def test_basic_trace_persistence_and_retrieval(self, tracer):
        """Test persisting and retrieving a basic trace with conversation_id and subcontext_id."""
        conversation_id = "conv-basic-test"
        subcontext_id = "subcontext-basic"
        trace_id = None

        # Create a trace with conversation_id and subcontext_id
        with tracer.trace("test_operation", span_kind="INTERNAL") as trace:
            trace.add_attribute("ai.conversation.id", conversation_id, inheritable=True)
            trace.add_attribute("ai.subcontext.id", subcontext_id, inheritable=True)
            trace.add_attribute("test_attr", "test_value")
            trace_id = trace.trace_id
            time.sleep(0.01)

        # Retrieve by trace_id
        traces = tracer.get_traces(trace_id=trace_id)
        assert len(traces) == 1
        assert traces[0].span_name == "test_operation"
        assert traces[0].attributes["test_attr"] == "test_value"
        assert traces[0].attributes["ai.conversation.id"] == conversation_id
        assert traces[0].attributes["ai.subcontext.id"] == subcontext_id

        # Retrieve by conversation_id and subcontext_id
        traces = tracer.get_traces(
            attribute_filter={
                "ai.conversation.id": conversation_id,
                "ai.subcontext.id": subcontext_id,
            }
        )
        assert len(traces) == 1
        assert traces[0].span_name == "test_operation"
        assert traces[0].attributes["ai.conversation.id"] == conversation_id
        assert traces[0].attributes["ai.subcontext.id"] == subcontext_id

    def test_trace_with_none_subcontext_id(self, tracer):
        """Test handling of NULL/None subcontext_id.

        Note: To match a filter with None value, the trace must explicitly
        have the attribute set to None. Missing attributes won't match.
        """
        conversation_id = "conv-none-subcontext"

        # Create traces with explicit None subcontext_id
        with tracer.trace("operation_no_subcontext") as trace1:
            trace1.add_attribute(
                "ai.conversation.id", conversation_id, inheritable=True
            )
            trace1.add_attribute("ai.subcontext.id", None, inheritable=True)
            trace1.add_attribute("data", "no-subcontext-data")
            time.sleep(0.01)

        # Create another trace with explicit subcontext_id for comparison
        with tracer.trace("operation_with_subcontext") as trace2:
            trace2.add_attribute(
                "ai.conversation.id", conversation_id, inheritable=True
            )
            trace2.add_attribute(
                "ai.subcontext.id", "some-subcontext", inheritable=True
            )
            trace2.add_attribute("data", "with-subcontext-data")
            time.sleep(0.01)

        # Query for None subcontext_id - should only match traces with explicit None
        traces = tracer.get_traces(
            attribute_filter={
                "ai.conversation.id": conversation_id,
                "ai.subcontext.id": None,
            }
        )
        assert len(traces) == 1
        assert traces[0].span_name == "operation_no_subcontext"
        assert traces[0].attributes["data"] == "no-subcontext-data"
        assert traces[0].attributes["ai.subcontext.id"] is None

    def test_subcontext_isolation(self, tracer):
        """Test that different subcontext_ids properly isolate traces."""
        conversation_id = "conv-subcontext-isolation"
        subcontext_1 = "subcontext-alpha"
        subcontext_2 = "subcontext-beta"

        # Create traces with first subcontext
        with tracer.trace("operation_subcontext_1") as trace1:
            trace1.add_attribute(
                "ai.conversation.id", conversation_id, inheritable=True
            )
            trace1.add_attribute("ai.subcontext.id", subcontext_1, inheritable=True)
            trace1.add_attribute("data", "alpha-data")
            time.sleep(0.01)

        # Create traces with second subcontext
        with tracer.trace("operation_subcontext_2") as trace2:
            trace2.add_attribute(
                "ai.conversation.id", conversation_id, inheritable=True
            )
            trace2.add_attribute("ai.subcontext.id", subcontext_2, inheritable=True)
            trace2.add_attribute("data", "beta-data")
            time.sleep(0.01)

        # Create traces with explicit None subcontext
        with tracer.trace("operation_no_subcontext") as trace3:
            trace3.add_attribute(
                "ai.conversation.id", conversation_id, inheritable=True
            )
            trace3.add_attribute("ai.subcontext.id", None, inheritable=True)
            trace3.add_attribute("data", "no-subcontext-data")
            time.sleep(0.01)

        # Query for first subcontext - should only get trace1
        traces_1 = tracer.get_traces(
            attribute_filter={
                "ai.conversation.id": conversation_id,
                "ai.subcontext.id": subcontext_1,
            }
        )
        assert len(traces_1) == 1
        assert traces_1[0].span_name == "operation_subcontext_1"
        assert traces_1[0].attributes["data"] == "alpha-data"
        assert traces_1[0].attributes["ai.subcontext.id"] == subcontext_1

        # Query for second subcontext - should only get trace2
        traces_2 = tracer.get_traces(
            attribute_filter={
                "ai.conversation.id": conversation_id,
                "ai.subcontext.id": subcontext_2,
            }
        )
        assert len(traces_2) == 1
        assert traces_2[0].span_name == "operation_subcontext_2"
        assert traces_2[0].attributes["data"] == "beta-data"
        assert traces_2[0].attributes["ai.subcontext.id"] == subcontext_2

        # Query for NULL subcontext - should only get trace3
        traces_none = tracer.get_traces(
            attribute_filter={
                "ai.conversation.id": conversation_id,
                "ai.subcontext.id": None,
            }
        )
        assert len(traces_none) == 1
        assert traces_none[0].span_name == "operation_no_subcontext"
        assert traces_none[0].attributes["data"] == "no-subcontext-data"
        assert traces_none[0].attributes["ai.subcontext.id"] is None

    def test_nested_traces_with_inheritance(self, tracer):
        """Test that inheritable attributes properly propagate to child spans."""
        conversation_id = "conv-inheritance-test"
        subcontext_id = "subcontext-inheritance"

        with tracer.trace("parent_operation", span_kind="SERVER") as parent:
            parent.add_attribute(
                "ai.conversation.id", conversation_id, inheritable=True
            )
            parent.add_attribute("ai.subcontext.id", subcontext_id, inheritable=True)
            parent.add_attribute("parent_attr", "parent_value")
            time.sleep(0.01)

            with tracer.trace("child_operation") as child:
                child.add_attribute("child_attr", "child_value")
                time.sleep(0.01)

        # Verify inheritance worked correctly
        traces = tracer.get_traces(
            attribute_filter={
                "ai.conversation.id": conversation_id,
                "ai.subcontext.id": subcontext_id,
            }
        )
        assert len(traces) == 2

        child_trace = next(t for t in traces if t.span_name == "child_operation")

        # Child should have all inherited attributes
        assert child_trace.attributes["ai.conversation.id"] == conversation_id
        assert child_trace.attributes["ai.subcontext.id"] == subcontext_id
        assert child_trace.attributes["child_attr"] == "child_value"

    def test_memory_size_limit(self, tracer):
        """Test that the memory deque respects the memory_size limit."""
        # Create a tracer with small memory size
        small_tracer = InMemoryTracer(memory_size=3)
        small_tracer.set_context(resource_attributes={"service.name": "TestAgent"})
        conversation_id = "conv-memory-limit"

        # Create more traces than the memory limit
        for i in range(5):
            with small_tracer.trace(f"operation_{i}") as trace:
                trace.add_attribute(
                    "ai.conversation.id", conversation_id, inheritable=True
                )
                trace.add_attribute("ai.subcontext.id", None, inheritable=True)
                trace.add_attribute("index", i)
                time.sleep(0.01)

        # Should only have the last 3 traces
        traces = small_tracer.get_traces(
            attribute_filter={
                "ai.conversation.id": conversation_id,
                "ai.subcontext.id": None,
            }
        )
        assert len(traces) == 3
        # Should be traces 2, 3, 4 (the last 3)
        assert traces[0].attributes["index"] == 2
        assert traces[1].attributes["index"] == 3
        assert traces[2].attributes["index"] == 4
