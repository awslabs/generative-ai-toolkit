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

import os
import sqlite3
import tempfile
import threading
import time
from pathlib import Path

import pytest

from generative_ai_toolkit.tracer.tracer import SqliteTracer


class TestSqliteTracer:
    """Test suite for the SqliteTracer class covering core functionality."""

    @pytest.fixture
    def temp_db_path(self):
        """Create a temporary database file path for testing."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as temp_file:
            temp_path = temp_file.name
        yield temp_path
        # Cleanup
        if os.path.exists(temp_path):
            os.unlink(temp_path)

    @pytest.fixture
    def tracer(self, temp_db_path):
        """Create a SqliteTracer instance for testing."""
        tracer = SqliteTracer(db_path=temp_db_path)
        tracer.set_context(resource_attributes={"service.name": "TestAgent"})
        return tracer

    def test_init_with_default_db_path(self):
        """Test initialization with default database path."""
        tracer = SqliteTracer()
        assert tracer.db_path == Path(os.getcwd()) / "conversations.db"
        assert tracer.identifier is None
        try:
            os.unlink(tracer.db_path)
        except FileNotFoundError:
            pass

    def test_init_with_custom_db_path(self, temp_db_path):
        """Test initialization with custom database path."""
        tracer = SqliteTracer(db_path=temp_db_path)
        assert tracer.db_path == Path(temp_db_path)

    def test_init_with_identifier(self, temp_db_path):
        """Test initialization with custom identifier."""
        identifier = "test-session-123"
        tracer = SqliteTracer(db_path=temp_db_path, identifier=identifier)
        assert tracer.identifier == identifier

    def test_init_without_table_creation(self, temp_db_path):
        """Test initialization without creating tables."""
        tracer = SqliteTracer(db_path=temp_db_path, create_tables=False)
        # Database file should exist but without tables
        assert tracer.db_path.exists()

        # Check that tables don't exist
        with sqlite3.connect(temp_db_path) as conn:
            cursor = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='traces'"
            )
            assert cursor.fetchone() is None

    def test_table_creation(self, temp_db_path):
        """Test that tables and indexes are created correctly."""
        SqliteTracer(db_path=temp_db_path, create_tables=True)

        with sqlite3.connect(temp_db_path) as conn:
            # Check table exists
            cursor = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='traces'"
            )
            assert cursor.fetchone() is not None

            # Check indexes exist
            cursor = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='index' AND name LIKE 'idx_traces_%'"
            )
            indexes = [row[0] for row in cursor.fetchall()]
            expected_indexes = [
                "idx_traces_trace_id_started_at",
                "idx_traces_conversation_id_started_at",
                "idx_traces_trace_id",
                "idx_traces_span_id",
            ]
            for expected_index in expected_indexes:
                assert expected_index in indexes

    def test_thread_local_connections(self, tracer):
        """Test that different threads get different database connections."""
        connections = []

        def get_connection():
            connections.append(tracer.conn)

        threads = []
        for _ in range(3):
            thread = threading.Thread(target=get_connection)
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()

        # All connections should be different objects
        assert len({id(conn) for conn in connections}) == 3

    def test_persist_basic_trace(self, tracer):
        """Test persisting a basic trace to the database using context manager."""
        trace_id = None

        with tracer.trace("test_operation", span_kind="INTERNAL") as trace:
            trace.add_attribute("test_attr", "test_value")
            trace.add_attribute("operation.type", "read")
            trace_id = trace.trace_id
            time.sleep(0.01)  # Add realistic timing

        # Verify it was stored in database
        with sqlite3.connect(tracer.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(
                "SELECT * FROM traces WHERE trace_id = ?", (trace_id,)
            )
            row = cursor.fetchone()

            assert row is not None
            assert row["span_name"] == "test_operation"
            assert row["span_kind"] == "INTERNAL"
            assert row["trace_id"] == trace_id
            assert row["ended_at"] is not None

        # Verify via tracer's get_traces method (which reconstructs Trace objects)
        traces = tracer.get_traces(trace_id=trace_id)
        assert len(traces) == 1
        persisted_trace = traces[0]

        assert persisted_trace.ended_at is not None
        assert (
            persisted_trace.duration_ms > 0
        )  # duration_ms is computed on Trace object
        assert persisted_trace.attributes["test_attr"] == "test_value"
        assert persisted_trace.attributes["operation.type"] == "read"

    def test_persist_trace_with_parent_child_relationship(self, tracer):
        """Test persisting traces with proper parent-child relationships."""
        conversation_id = "conv-456"
        parent_trace_id = None

        with tracer.trace("parent_operation", span_kind="SERVER") as parent_trace:
            parent_trace.add_attribute(
                "ai.conversation.id", conversation_id, inheritable=True
            )
            parent_trace.add_attribute(
                "ai.auth.context", {"principal_id": "user123"}, inheritable=True
            )
            parent_trace.add_attribute("operation.name", "process_request")
            parent_trace_id = parent_trace.trace_id
            time.sleep(0.01)

            with tracer.trace("child_operation") as child_trace:
                child_trace.add_attribute("ai.trace.type", "tool-invocation")
                child_trace.add_attribute("ai.tool.input", "Hello, world!")
                child_trace.add_attribute("ai.tool.output", "World, hello!")
                time.sleep(0.01)

        # Verify parent relationship is stored correctly in database
        with sqlite3.connect(tracer.db_path) as conn:
            conn.row_factory = sqlite3.Row

            # Check parent trace
            cursor = conn.execute(
                "SELECT * FROM traces WHERE span_id = ?",
                (parent_trace.span_id,),
            )
            parent_row = cursor.fetchone()
            assert parent_row is not None
            assert parent_row["parent_span_id"] is None  # Root span
            assert parent_row["conversation_id"] == conversation_id

            # Check child trace
            cursor = conn.execute(
                "SELECT * FROM traces WHERE span_id = ?",
                (child_trace.span_id,),
            )
            child_row = cursor.fetchone()
            assert child_row is not None
            assert child_row["parent_span_id"] == parent_trace.span_id
            assert child_row["conversation_id"] == conversation_id  # Inherited
            assert child_row["trace_id"] == parent_trace_id  # Same trace

    def test_persist_nested_traces_with_inheritance(self, tracer):
        """Test that inheritable attributes properly propagate to child spans."""
        conversation_id = "conv-inheritance-test"
        subcontext_id = "subcontext-inheritance-1"

        with tracer.trace("grandparent", span_kind="SERVER") as grandparent:
            grandparent.add_attribute(
                "ai.conversation.id", conversation_id, inheritable=True
            )
            grandparent.add_attribute(
                "ai.subcontext.id", subcontext_id, inheritable=True
            )
            grandparent.add_attribute(
                "ai.auth.context", {"principal_id": "user456"}, inheritable=True
            )
            grandparent.add_attribute("request.id", "req-123", inheritable=True)
            time.sleep(0.01)

            with tracer.trace("parent") as parent:
                parent.add_attribute("operation.stage", "processing", inheritable=True)
                time.sleep(0.01)

                with tracer.trace("child") as child:
                    child.add_attribute("ai.trace.type", "llm-invocation")
                    child.add_attribute("specific.attr", "child-only")
                    time.sleep(0.01)

        # Verify inheritance worked correctly
        traces = tracer.get_traces(
            attribute_filter={"ai.conversation.id": conversation_id, "ai.subcontext.id": subcontext_id}
        )
        assert len(traces) == 3

        child_trace = next(t for t in traces if t.span_name == "child")

        # Child should have all inherited attributes
        assert child_trace.attributes["ai.conversation.id"] == conversation_id
        assert child_trace.attributes["ai.subcontext.id"] == subcontext_id
        assert child_trace.attributes["ai.auth.context"]["principal_id"] == "user456"
        assert child_trace.attributes["request.id"] == "req-123"
        assert child_trace.attributes["operation.stage"] == "processing"
        assert child_trace.attributes["specific.attr"] == "child-only"

    def test_get_traces_by_trace_id(self, tracer):
        """Test retrieving traces by trace ID."""
        trace_id_1 = None

        # Create first trace
        with tracer.trace("operation_1") as trace1:
            trace1.add_attribute("operation.type", "read")
            trace_id_1 = trace1.trace_id
            time.sleep(0.01)

        # Create second trace
        with tracer.trace("operation_2") as trace2:
            trace2.add_attribute("operation.type", "write")
            time.sleep(0.01)

        # Retrieve traces by specific trace ID
        traces = tracer.get_traces(trace_id=trace_id_1)

        assert len(traces) == 1
        assert traces[0].trace_id == trace_id_1
        assert traces[0].span_name == "operation_1"
        assert traces[0].attributes["operation.type"] == "read"

    def test_get_traces_by_conversation_id(self, tracer):
        """Test retrieving traces by conversation ID."""
        conversation_id = "conv-search-test"

        # Create trace with conversation ID
        with tracer.trace("conversation_operation", span_kind="SERVER") as trace1:
            trace1.add_attribute(
                "ai.conversation.id", conversation_id, inheritable=True
            )
            trace1.add_attribute("ai.trace.type", "converse")
            trace1.add_attribute("ai.user.input", "What is the weather?")
            time.sleep(0.01)

        # Create trace without conversation ID
        with tracer.trace("other_operation") as trace2:
            trace2.add_attribute("operation.type", "maintenance")
            time.sleep(0.01)

        # Retrieve by conversation ID
        traces = tracer.get_traces(
            attribute_filter={"ai.conversation.id": conversation_id, "ai.subcontext.id": None}
        )

        assert len(traces) == 1
        assert traces[0].span_name == "conversation_operation"
        assert traces[0].attributes["ai.conversation.id"] == conversation_id
        assert traces[0].attributes["ai.trace.type"] == "converse"

    def test_get_traces_invalid_filter(self, tracer):
        """Test that get_traces raises ValueError for invalid filters."""
        with pytest.raises(
            ValueError, match="To use get_traces\\(\\) you must either provide trace_id"
        ):
            tracer.get_traces()

        with pytest.raises(
            ValueError, match="To use get_traces\\(\\) you must either provide trace_id"
        ):
            tracer.get_traces(attribute_filter={"some.other.attr": "value"})

        with pytest.raises(
            ValueError, match="To use get_traces\\(\\) you must either provide trace_id"
        ):
            tracer.get_traces(attribute_filter={"ai.conversation.id": "conv-123"})

    def test_get_traces_with_identifier_filter(self, temp_db_path):
        """Test that traces are filtered by identifier."""
        # Create tracers with different identifiers
        tracer1 = SqliteTracer(db_path=temp_db_path, identifier="session1")
        tracer1.set_context(resource_attributes={"service.name": "Agent1"})

        tracer2 = SqliteTracer(db_path=temp_db_path, identifier="session2")
        tracer2.set_context(resource_attributes={"service.name": "Agent2"})

        conversation_id = "conv-identifier-test"
        subcontext_id = "subcontext-filter-1"

        # Create traces with each tracer
        with tracer1.trace("operation_1") as trace1:
            trace1.add_attribute(
                "ai.conversation.id", conversation_id, inheritable=True
            )
            trace1.add_attribute(
                "ai.subcontext.id", subcontext_id, inheritable=True
            )
            trace1.add_attribute("session.data", "session1_data")
            time.sleep(0.01)

        with tracer2.trace("operation_2") as trace2:
            trace2.add_attribute(
                "ai.conversation.id", conversation_id, inheritable=True
            )
            trace2.add_attribute(
                "ai.subcontext.id", subcontext_id, inheritable=True
            )
            trace2.add_attribute("session.data", "session2_data")
            time.sleep(0.01)

        # Each tracer should only see its own traces
        traces1 = tracer1.get_traces(
            attribute_filter={"ai.conversation.id": conversation_id, "ai.subcontext.id": subcontext_id}
        )
        traces2 = tracer2.get_traces(
            attribute_filter={"ai.conversation.id": conversation_id, "ai.subcontext.id": subcontext_id}
        )

        assert len(traces1) == 1
        assert len(traces2) == 1
        assert traces1[0].span_name == "operation_1"
        assert traces2[0].span_name == "operation_2"
        assert traces1[0].attributes["session.data"] == "session1_data"
        assert traces2[0].attributes["session.data"] == "session2_data"

    def test_get_traces_with_additional_attribute_filter(self, tracer):
        """Test get_traces with additional attribute filtering."""
        conversation_id = "conv-multi-filter"

        # Create traces with same conversation ID but different attributes
        with tracer.trace("read_operation") as trace1:
            trace1.add_attribute(
                "ai.conversation.id", conversation_id, inheritable=True
            )
            trace1.add_attribute("operation.type", "read")
            trace1.add_attribute("ai.trace.type", "tool-invocation")
            time.sleep(0.01)

        with tracer.trace("write_operation") as trace2:
            trace2.add_attribute(
                "ai.conversation.id", conversation_id, inheritable=True
            )
            trace2.add_attribute("operation.type", "write")
            trace2.add_attribute("ai.trace.type", "tool-invocation")
            time.sleep(0.01)

        # Filter by conversation ID and additional attribute
        traces = tracer.get_traces(
            attribute_filter={
                "ai.conversation.id": conversation_id,
                "ai.subcontext.id": None,
                "operation.type": "read",
            }
        )

        assert len(traces) == 1
        assert traces[0].span_name == "read_operation"
        assert traces[0].attributes["operation.type"] == "read"

    def test_get_traces_ordered_by_started_at(self, tracer):
        """Test that retrieved traces are ordered by started_at."""
        conversation_id = "conv-ordering-test"

        # Create traces with deliberate timing gaps
        with tracer.trace("second_operation") as trace1:
            trace1.add_attribute(
                "ai.conversation.id", conversation_id, inheritable=True
            )
            trace1.add_attribute("sequence", 2)
            time.sleep(0.01)

        time.sleep(0.02)  # Ensure different timestamps

        with tracer.trace("first_operation") as trace2:
            trace2.add_attribute(
                "ai.conversation.id", conversation_id, inheritable=True
            )
            trace2.add_attribute("sequence", 1)
            time.sleep(0.01)

        time.sleep(0.02)  # Ensure different timestamps

        with tracer.trace("third_operation") as trace3:
            trace3.add_attribute(
                "ai.conversation.id", conversation_id, inheritable=True
            )
            trace3.add_attribute("sequence", 3)
            time.sleep(0.01)

        # Retrieve and verify order (should be chronological by started_at)
        traces = tracer.get_traces(
            attribute_filter={"ai.conversation.id": conversation_id, "ai.subcontext.id": None}
        )

        assert len(traces) == 3
        # Verify chronological ordering
        assert traces[0].attributes["sequence"] == 2  # First created
        assert traces[1].attributes["sequence"] == 1  # Second created
        assert traces[2].attributes["sequence"] == 3  # Third created

    def test_trace_context_manager_lifecycle(self, tracer):
        """Test the complete lifecycle of trace context manager."""
        trace_id = None

        with tracer.trace("lifecycle_test", span_kind="SERVER") as trace:
            trace_id = trace.trace_id
            assert trace.span_name == "lifecycle_test"
            assert trace.span_kind == "SERVER"
            assert trace.started_at is not None
            assert trace.ended_at is None  # Should not be ended yet

            trace.add_attribute("ai.conversation.id", "conv-lifecycle")
            trace.add_attribute("step", "processing")
            time.sleep(0.02)  # Add measurable duration

        # After exiting context, trace should be persisted and ended
        traces = tracer.get_traces(trace_id=trace_id)
        assert len(traces) == 1
        persisted_trace = traces[0]

        assert persisted_trace.ended_at is not None
        assert persisted_trace.duration_ms > 0  # duration_ms computed on Trace object
        assert persisted_trace.attributes["ai.conversation.id"] == "conv-lifecycle"
        assert persisted_trace.attributes["step"] == "processing"

    def test_concurrent_trace_creation(self, tracer):
        """Test concurrent access to the tracer from multiple threads."""
        num_threads = 5
        traces_per_thread = 3
        conversation_id = "conv-concurrent-test"
        subcontext_id = "subcontext-concurrent-1"
        all_trace_ids = []

        def create_traces(thread_id):
            thread_trace_ids = []
            for i in range(traces_per_thread):
                with tracer.trace(f"thread_{thread_id}_span_{i}") as trace:
                    trace.add_attribute(
                        "ai.conversation.id", conversation_id, inheritable=True
                    )
                    trace.add_attribute(
                        "ai.subcontext.id", subcontext_id, inheritable=True
                    )
                    trace.add_attribute("thread_id", thread_id)
                    trace.add_attribute("span_index", i)
                    trace.add_attribute("ai.trace.type", "concurrent-test")
                    thread_trace_ids.append(trace.trace_id)
                    time.sleep(0.01)
            all_trace_ids.extend(thread_trace_ids)

        # Create and start threads
        threads = []
        for i in range(num_threads):
            thread = threading.Thread(target=create_traces, args=(i,))
            threads.append(thread)
            thread.start()

        # Wait for all threads to complete
        for thread in threads:
            thread.join()

        # Verify all traces were persisted
        total_expected = num_threads * traces_per_thread
        traces = tracer.get_traces(
            attribute_filter={"ai.conversation.id": conversation_id, "ai.subcontext.id": subcontext_id}
        )

        assert len(traces) == total_expected

        # Verify all traces have proper attributes and are completed
        for trace in traces:
            assert trace.ended_at is not None
            assert trace.duration_ms is not None  # duration_ms computed on Trace object
            assert "thread_id" in trace.attributes
            assert "span_index" in trace.attributes
            assert trace.attributes["ai.trace.type"] == "concurrent-test"
            assert trace.attributes["ai.subcontext.id"] == subcontext_id

    def test_trace_with_error_handling(self, tracer):
        """Test trace behavior when exceptions occur within context."""
        trace_id = None

        with pytest.raises(ValueError, match="Simulated error"):
            with tracer.trace("error_operation") as trace:
                trace.add_attribute("ai.conversation.id", "conv-error-test")
                trace.add_attribute("status", "processing")
                trace_id = trace.trace_id
                time.sleep(0.01)
                raise ValueError("Simulated error")

        # Trace should still be persisted even when exception occurred
        traces = tracer.get_traces(trace_id=trace_id)
        assert len(traces) == 1
        error_trace = traces[0]

        assert error_trace.ended_at is not None
        assert error_trace.attributes["status"] == "processing"
        assert error_trace.attributes["ai.conversation.id"] == "conv-error-test"

    def test_complex_conversation_flow(self, tracer):
        """Test a complex conversation flow with multiple nested operations."""
        conversation_id = "conv-complex-flow"
        user_id = "user789"

        with tracer.trace("user_request", span_kind="SERVER") as request_trace:
            request_trace.add_attribute(
                "ai.conversation.id", conversation_id, inheritable=True
            )
            request_trace.add_attribute(
                "ai.auth.context", {"principal_id": user_id}, inheritable=True
            )
            request_trace.add_attribute("ai.trace.type", "converse")
            request_trace.add_attribute(
                "ai.user.input", "What's the weather in Seattle?"
            )
            time.sleep(0.01)

            with tracer.trace("intent_recognition") as intent_trace:
                intent_trace.add_attribute("ai.trace.type", "llm-invocation")
                intent_trace.add_attribute("ai.llm.request.model", "claude-3")
                intent_trace.add_attribute("intent.detected", "weather_query")
                time.sleep(0.01)

            with tracer.trace("weather_tool_call") as tool_trace:
                tool_trace.add_attribute("ai.trace.type", "tool-invocation")
                tool_trace.add_attribute("ai.tool.name", "get_weather")
                tool_trace.add_attribute("ai.tool.input", {"location": "Seattle"})
                tool_trace.add_attribute(
                    "ai.tool.output", {"temperature": "72F", "condition": "sunny"}
                )
                time.sleep(0.01)

            with tracer.trace("response_generation") as response_trace:
                response_trace.add_attribute("ai.trace.type", "llm-invocation")
                response_trace.add_attribute("ai.llm.request.model", "claude-3")
                response_trace.add_attribute(
                    "ai.agent.response", "The weather in Seattle is 72F and sunny."
                )
                time.sleep(0.01)

        # Verify the complete conversation flow
        traces = tracer.get_traces(
            attribute_filter={"ai.conversation.id": conversation_id, "ai.subcontext.id": None}
        )
        assert len(traces) == 4

        # Verify all traces share the conversation context
        for trace in traces:
            assert trace.attributes["ai.conversation.id"] == conversation_id
            assert trace.attributes["ai.auth.context"]["principal_id"] == user_id

        # Verify specific trace types
        trace_types = {
            trace.span_name: trace.attributes.get("ai.trace.type") for trace in traces
        }
        assert trace_types["user_request"] == "converse"
        assert trace_types["intent_recognition"] == "llm-invocation"
        assert trace_types["weather_tool_call"] == "tool-invocation"
        assert trace_types["response_generation"] == "llm-invocation"

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
            trace1.add_attribute(
                "ai.subcontext.id", subcontext_1, inheritable=True
            )
            trace1.add_attribute("data", "alpha-data")
            time.sleep(0.01)

        # Create traces with second subcontext
        with tracer.trace("operation_subcontext_2") as trace2:
            trace2.add_attribute(
                "ai.conversation.id", conversation_id, inheritable=True
            )
            trace2.add_attribute(
                "ai.subcontext.id", subcontext_2, inheritable=True
            )
            trace2.add_attribute("data", "beta-data")
            time.sleep(0.01)

        # Create traces with no subcontext (NULL)
        with tracer.trace("operation_no_subcontext") as trace3:
            trace3.add_attribute(
                "ai.conversation.id", conversation_id, inheritable=True
            )
            trace3.add_attribute("data", "no-subcontext-data")
            time.sleep(0.01)

        # Query for first subcontext - should only get trace1
        traces_1 = tracer.get_traces(
            attribute_filter={"ai.conversation.id": conversation_id, "ai.subcontext.id": subcontext_1}
        )
        assert len(traces_1) == 1
        assert traces_1[0].span_name == "operation_subcontext_1"
        assert traces_1[0].attributes["data"] == "alpha-data"
        assert traces_1[0].attributes["ai.subcontext.id"] == subcontext_1

        # Query for second subcontext - should only get trace2
        traces_2 = tracer.get_traces(
            attribute_filter={"ai.conversation.id": conversation_id, "ai.subcontext.id": subcontext_2}
        )
        assert len(traces_2) == 1
        assert traces_2[0].span_name == "operation_subcontext_2"
        assert traces_2[0].attributes["data"] == "beta-data"
        assert traces_2[0].attributes["ai.subcontext.id"] == subcontext_2

        # Query for NULL subcontext - should only get trace3
        traces_none = tracer.get_traces(
            attribute_filter={"ai.conversation.id": conversation_id, "ai.subcontext.id": None}
        )
        assert len(traces_none) == 1
        assert traces_none[0].span_name == "operation_no_subcontext"
        assert traces_none[0].attributes["data"] == "no-subcontext-data"
        assert "ai.subcontext.id" not in traces_none[0].attributes
