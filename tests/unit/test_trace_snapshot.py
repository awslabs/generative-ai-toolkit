import threading
import time
from unittest.mock import MagicMock

from generative_ai_toolkit.tracer.trace import Trace
from generative_ai_toolkit.tracer.tracer import (
    InMemoryTracer,
    IterableTracer,
    SnapshotCapableTracer,
    TeeTracer,
)


class SnapshotInMemoryTracer(InMemoryTracer):
    """An InMemoryTracer that also handles snapshots."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.snapshots = []
        self.snapshot_enabled = True

    def persist_snapshot(self, trace: Trace):
        self.snapshots.append(trace)


class TestTraceSnapshotFunctionality:
    """Tests for the trace snapshot functionality."""

    def test_trace_emit_snapshot(self):
        """Test that emit_snapshot calls the snapshot handler with a clone of the trace."""
        # Setup
        snapshot_handler_mock = MagicMock()
        trace = Trace(
            span_name="test_span",
            snapshot_handler=snapshot_handler_mock,
        )

        # Act
        trace.emit_snapshot()

        # Assert
        snapshot_handler_mock.assert_called_once()
        # The handler should be called with a clone of the trace
        actual_trace = snapshot_handler_mock.call_args[0][0]
        assert actual_trace is not trace  # Should be a different object
        assert actual_trace.span_name == trace.span_name  # But with same attributes
        assert actual_trace.span_id == trace.span_id
        assert actual_trace.trace_id == trace.trace_id

    def test_snapshot_capable_tracer_protocol(self):
        """Test that TeeTracer correctly implements the SnapshotCapableTracer protocol."""
        # Setup
        tee_tracer = TeeTracer()

        # Assert
        assert isinstance(tee_tracer, SnapshotCapableTracer)
        assert hasattr(tee_tracer, "snapshot_enabled")
        assert hasattr(tee_tracer, "persist_snapshot")

    def test_tee_tracer_snapshot_capabilities(self):
        """Test that TeeTracer correctly forwards snapshots to capable tracers."""
        # Setup
        mock_snapshot_tracer = MagicMock(spec=SnapshotCapableTracer)
        mock_snapshot_tracer.snapshot_enabled = True

        regular_tracer = MagicMock()

        tee_tracer = TeeTracer()
        tee_tracer.add_tracer(mock_snapshot_tracer)
        tee_tracer.add_tracer(regular_tracer)

        trace = Trace(span_name="test_span")

        # Act
        tee_tracer.persist_snapshot(trace)

        # Assert
        mock_snapshot_tracer.persist_snapshot.assert_called_once_with(trace)
        # Ensure regular tracer doesn't have persist_snapshot called
        assert (
            not hasattr(regular_tracer, "persist_snapshot")
            or not regular_tracer.persist_snapshot.called
        )

    def test_snapshot_enabled_propagation_in_tee_tracer(self):
        """Test that adding a SnapshotCapableTracer correctly enables snapshots in TeeTracer."""
        # Setup
        tee_tracer = TeeTracer()
        assert not tee_tracer.snapshot_enabled  # Initially disabled

        mock_snapshot_tracer = MagicMock(spec=SnapshotCapableTracer)
        mock_snapshot_tracer.snapshot_enabled = True

        # Act
        tee_tracer.add_tracer(mock_snapshot_tracer)

        # Assert
        assert tee_tracer.snapshot_enabled

    def test_in_memory_tracer_snapshot_implementation(self):
        """Test that InMemoryTracer can implement SnapshotCapableTracer."""

        tracer = SnapshotInMemoryTracer()
        trace = Trace(span_name="test_span")

        tracer.persist_snapshot(trace)
        traces = tracer.get_traces()
        assert len(traces) == 0

        snapshots = tracer.snapshots
        assert len(snapshots) == 1

    def test_iterable_tracer_snapshots(self):
        """Test that IterableTracer correctly handles snapshots."""
        # Setup
        tracer = IterableTracer()
        trace = Trace(span_name="test_span")

        # Create a thread to consume from the tracer
        traces = []

        def consume_traces():
            for t in tracer:
                traces.append(t)
                if len(traces) == 1:  # Stop after receiving one trace
                    break

        consumer_thread = threading.Thread(target=consume_traces)
        consumer_thread.start()

        # Act
        tracer.persist_snapshot(trace)

        # Wait for consumer to finish
        consumer_thread.join(timeout=1.0)
        tracer.shutdown()  # Ensure cleanup

        # Assert
        assert len(traces) == 1
        assert traces[0].span_name == "test_span"

    def test_full_snapshot_flow(self):
        """Test a complete snapshot flow from trace creation to consumption."""
        # Setup - create a tracer that can handle snapshots
        tracer = SnapshotInMemoryTracer()

        # Start a trace
        with tracer.trace("parent_operation") as parent:
            # Add some attributes
            parent.add_attribute("stage", "setup")

            # Emit a snapshot of the initial state
            parent.emit_snapshot()

            # Do some work and update attributes
            parent.add_attribute("stage", "processing")
            parent.emit_snapshot()

            # Create a child span
            with tracer.trace("child_operation") as child:
                child.add_attribute("sub_task", "data_loading")

                # Both the parent and child can emit snapshots
                parent.emit_snapshot()
                child.emit_snapshot()

                # Update child and emit again
                child.add_attribute("sub_task", "data_processing")
                child.emit_snapshot()

            # Final updates to parent
            parent.add_attribute("stage", "cleanup")
            parent.emit_snapshot()

        # Assert - check the snapshots
        assert len(tracer.snapshots) == 6  # We emitted 6 snapshots

        # Check snapshot progression for parent
        parent_snapshots = [
            s for s in tracer.snapshots if s.span_name == "parent_operation"
        ]
        assert len(parent_snapshots) == 4
        assert parent_snapshots[0].attributes["stage"] == "setup"
        assert parent_snapshots[1].attributes["stage"] == "processing"
        assert parent_snapshots[2].attributes["stage"] == "processing"
        assert parent_snapshots[3].attributes["stage"] == "cleanup"

        # Check child snapshots
        child_snapshots = [
            s for s in tracer.snapshots if s.span_name == "child_operation"
        ]
        assert len(child_snapshots) == 2
        assert child_snapshots[0].attributes["sub_task"] == "data_loading"
        assert child_snapshots[1].attributes["sub_task"] == "data_processing"

        # Check final persisted traces
        persisted = tracer.get_traces()
        assert len(persisted) == 2  # Parent and child

        # Ensure complete traces are different from snapshots (they're completed)
        parent_trace = next(t for t in persisted if t.span_name == "parent_operation")
        assert parent_trace.attributes["stage"] == "cleanup"
        assert parent_trace.ended_at is not None  # Should be completed

        child_trace = next(t for t in persisted if t.span_name == "child_operation")
        assert child_trace.attributes["sub_task"] == "data_processing"
        assert child_trace.ended_at is not None  # Should be completed

    def test_iterable_real_time_consumption(self):
        """Test consuming snapshots in real-time using IterableTracer."""
        # Setup - create an IterableTracer
        tracer = IterableTracer()

        # Track received traces
        received_traces = []

        # Function to consume traces in a separate thread
        def consume_traces():
            for trace in tracer:
                received_traces.append((trace.clone(), time.time()))
                # Stop after receiving 4 traces
                if len(received_traces) >= 4:
                    break

        # Start consumer thread
        consumer_thread = threading.Thread(target=consume_traces)
        consumer_thread.start()

        # Allow consumer to start
        time.sleep(0.1)

        # Create and emit trace snapshots
        with tracer.trace("progress_operation") as trace:
            # Initial state
            trace.add_attribute("progress", "0%")
            trace.emit_snapshot()
            time.sleep(0.1)  # Simulate work

            # Update progress
            trace.add_attribute("progress", "33%")
            trace.emit_snapshot()
            time.sleep(0.1)  # Simulate work

            # Update again
            trace.add_attribute("progress", "66%")
            trace.emit_snapshot()
            time.sleep(0.1)  # Simulate work

            # Will be emitted when the context exits
            trace.add_attribute("progress", "100%")

        # Shutdown and wait for consumer
        tracer.shutdown()
        consumer_thread.join(timeout=1.0)

        # Assert
        assert len(received_traces) == 4

        # Verify the progression of snapshots
        assert received_traces[0][0].attributes["progress"] == "0%"
        assert received_traces[1][0].attributes["progress"] == "33%"
        assert received_traces[2][0].attributes["progress"] == "66%"
        assert received_traces[3][0].attributes["progress"] == "100%"

        # Verify timing - ensure they were received in order with reasonable timing
        for i in range(1, 4):
            assert received_traces[i][1] > received_traces[i - 1][1]

    def test_nested_span_snapshots(self):
        """Test snapshots with nested spans to ensure correct parent relationships."""
        # Setup
        tracer = SnapshotInMemoryTracer()

        # Create nested spans
        with tracer.trace("outer") as outer:
            outer.add_attribute("level", "outer")
            outer.emit_snapshot()

            with tracer.trace("middle") as middle:
                middle.add_attribute("level", "middle")
                middle.emit_snapshot()

                with tracer.trace("inner") as inner:
                    inner.add_attribute("level", "inner")
                    inner.emit_snapshot()

        # Assert
        assert len(tracer.snapshots) == 3

        # Check outer span snapshot
        outer_snapshot = next(s for s in tracer.snapshots if s.span_name == "outer")
        assert outer_snapshot.attributes["level"] == "outer"
        assert outer_snapshot.parent_span is None  # Outer has no parent

        # Check middle span snapshot
        middle_snapshot = next(s for s in tracer.snapshots if s.span_name == "middle")
        assert middle_snapshot.attributes["level"] == "middle"
        assert middle_snapshot.parent_span.span_name == "outer"

        # Check inner span snapshot
        inner_snapshot = next(s for s in tracer.snapshots if s.span_name == "inner")
        assert inner_snapshot.attributes["level"] == "inner"
        assert inner_snapshot.parent_span.span_name == "middle"

    def test_snapshot_only_called_if_enabled(self):
        """Test that emit_snapshot is only called if snapshot_enabled is True."""
        # Setup
        tracer = SnapshotInMemoryTracer()
        tracer.snapshot_enabled = False

        # Act
        with tracer.trace("test_span") as span:
            span.emit_snapshot()

        # Assert
        assert len(tracer.snapshots) == 0
