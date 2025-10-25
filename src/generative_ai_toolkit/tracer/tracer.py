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

import inspect
import os
import shutil
import sqlite3
import sys
import threading
import time
import traceback
from collections import deque
from collections.abc import Callable, Iterable, Mapping, Sequence
from contextlib import AbstractContextManager, contextmanager
from datetime import UTC, datetime
from functools import wraps
from pathlib import Path
from queue import Queue, ShutDown
from typing import (
    Any,
    Literal,
    Protocol,
    TextIO,
    TypeVar,
    Unpack,
    cast,
    overload,
    runtime_checkable,
)

from generative_ai_toolkit.tracer.context import (
    ContextVarTraceContextProvider,
    TraceContext,
    TraceContextProvider,
    TraceContextUpdate,
)
from generative_ai_toolkit.tracer.trace import Trace, TraceScope
from generative_ai_toolkit.utils.json import JsonBytes
from generative_ai_toolkit.utils.logging import SimpleLogger


@runtime_checkable
class Tracer(Protocol):

    @property
    def context(self) -> TraceContext: ...

    def set_context(
        self, **update: Unpack[TraceContextUpdate]
    ) -> Callable[[], None]: ...

    @property
    def current_trace(self) -> Trace: ...

    def trace(
        self,
        span_name: str,
        *,
        parent_span: Trace | None = None,
        scope: TraceScope | None = None,
        resource_attributes: Mapping[str, Any] | None = None,
        span_kind: Literal["INTERNAL", "SERVER", "CLIENT"] = "INTERNAL",
    ) -> AbstractContextManager[Trace]: ...

    def get_traces(
        self,
        trace_id: str | None = None,
        attribute_filter: Mapping[str, Any] | None = None,
    ) -> Sequence[Trace]: ...

    def persist(self, trace: Trace): ...


@runtime_checkable
class SnapshotCapableTracer(Protocol):

    snapshot_enabled: bool

    def persist_snapshot(self, trace: Trace): ...


@runtime_checkable
class HasTracer(Protocol):

    @property
    def tracer(self) -> Tracer: ...


F = TypeVar("F", bound=Callable)


@overload
def traced[F: Callable](arg: F) -> F: ...


@overload
def traced(
    arg: str,
    *,
    scope: TraceScope | None = None,
    span_kind: Literal["INTERNAL", "SERVER", "CLIENT"] = "INTERNAL",
    tracer: Tracer | None = None,
) -> Callable[[F], F]: ...


def traced[F: Callable](
    arg: F | str,
    *,
    scope: TraceScope | None = None,
    span_kind: Literal["INTERNAL", "SERVER", "CLIENT"] = "INTERNAL",
    tracer: Tracer | None = None,
) -> F | Callable[[F], F]:
    if callable(arg):
        func = arg
        span_name = func.__name__
        return _create_decorator(
            func, span_name, scope=scope, span_kind=span_kind, tracer=tracer
        )

    span_name = arg

    def decorator(func: F) -> F:
        return _create_decorator(
            func, span_name, scope=scope, span_kind=span_kind, tracer=tracer
        )

    return decorator


def _create_decorator[F: Callable](
    func: F,
    span_name: str,
    *,
    parent_span: Trace | None = None,
    scope: TraceScope | None = None,
    resource_attributes: Mapping[str, Any] | None = None,
    span_kind: Literal["INTERNAL", "SERVER", "CLIENT"] = "INTERNAL",
    tracer: Tracer | None = None,
) -> F:
    @wraps(func)
    def generator_wrapper(*args, **kwargs):
        if tracer:
            resolved_tracer = tracer
        elif args and isinstance(args[0], HasTracer):
            resolved_tracer = args[0].tracer
        else:
            raise ValueError(
                f"Function {func.__name__} is not compatible with the @traced decorator."
                " Use the decorator on a method of a class that satisfies the HasTracer protocol (i.e. has a .tracer property)."
                " Alternatively, pass a tracer explicitly: @traced('span-name', tracer=mytracer)"
            )
        with resolved_tracer.trace(
            span_name,
            span_kind=span_kind,
            parent_span=parent_span,
            scope=scope,
            resource_attributes=resource_attributes,
        ):
            try:
                res = func(*args, **kwargs)
                if inspect.isgenerator(res):
                    return (yield from res)
                elif inspect.iscoroutine(res) or inspect.isasyncgen(res):
                    raise RuntimeError(
                        "The @traced decorator can only be used for sync functions and generators, not async ones."
                    )
                yield res
            except GeneratorExit:
                pass

    if inspect.isgeneratorfunction(func):
        return cast(F, generator_wrapper)
    else:

        @wraps(func)
        def non_generator_wrapper(*args, **kwargs):
            for res in generator_wrapper(*args, **kwargs):
                return res

        return cast(F, non_generator_wrapper)


class ContextAwareSpanPersistor:
    span_name: str
    span_kind: Literal["INTERNAL", "SERVER", "CLIENT"]
    trace: Trace
    persistor: Callable[["Trace"], None]
    snapshot_handler: Callable[["Trace"], None] | None
    trace_context: TraceContextProvider
    _reset: Callable[[], None]
    parent_span: Trace | None
    scope: TraceScope | None
    resource_attributes: Mapping[str, Any] | None

    def __init__(
        self,
        span_name: str,
        *,
        parent_span: Trace | None = None,
        scope: TraceScope | None = None,
        resource_attributes: Mapping[str, Any] | None = None,
        span_kind: Literal["INTERNAL", "SERVER", "CLIENT"] = "INTERNAL",
        persistor: Callable[["Trace"], None],
        snapshot_handler: Callable[["Trace"], None] | None = None,
        trace_context: TraceContextProvider,
    ) -> None:
        self.span_name = span_name
        self.span_kind = span_kind
        self.persistor = persistor
        self.snapshot_handler = snapshot_handler
        self.trace_context = trace_context
        self.parent_span = parent_span
        self.scope = scope
        self.resource_attributes = resource_attributes

    def __enter__(self):
        started_at = datetime.now(UTC)
        context = self.trace_context.context
        self.trace = Trace(
            self.span_name,
            trace_id=context.span.trace_id if context.span else None,
            span_kind=self.span_kind,
            started_at=started_at,
            parent_span=self.parent_span or context.span,
            scope=self.scope or context.scope,
            resource_attributes=self.resource_attributes or context.resource_attributes,
            snapshot_handler=self.snapshot_handler,
        )
        for k, v in context.span_attributes.items():
            self.trace.add_attribute(k, v)
        self._reset = self.trace_context.set_context(span=self.trace)
        return self.trace

    def __exit__(self, exc_type, exc_value, _traceback):
        if exc_type:
            self.trace.span_status = "ERROR"
            self.trace.add_attribute("exception.type", exc_type.__name__)
            self.trace.add_attribute("exception.message", str(exc_value))
            self.trace.add_attribute(
                "exception.traceback", "".join(traceback.format_tb(_traceback))
            )
        self.trace.ended_at = datetime.now(UTC)
        self._reset()
        self.persistor(self.trace)


class BaseTracer(Tracer, SnapshotCapableTracer):

    trace_context_provider: TraceContextProvider

    def __init__(self, trace_context_provider: TraceContextProvider | None = None):
        self.trace_context_provider = (
            trace_context_provider or ContextVarTraceContextProvider()
        )
        self.lock = threading.Lock()
        self.snapshot_enabled = False

    @property
    def context(self) -> TraceContext:
        return self.trace_context_provider.context

    def set_context(self, **update: Unpack[TraceContextUpdate]) -> Callable[[], None]:
        return self.trace_context_provider.set_context(**update)

    @property
    def current_trace(self) -> Trace:
        context = self.trace_context_provider.context
        if not context.span:
            raise ValueError("No active trace in context")
        return context.span

    def trace(
        self,
        span_name: str,
        *,
        parent_span: Trace | None = None,
        scope: TraceScope | None = None,
        resource_attributes: Mapping[str, Any] | None = None,
        span_kind: Literal["INTERNAL", "SERVER", "CLIENT"] = "INTERNAL",
    ):
        return ContextAwareSpanPersistor(
            span_name,
            span_kind=span_kind,
            parent_span=parent_span,
            scope=scope,
            resource_attributes=resource_attributes,
            persistor=self.persist,
            snapshot_handler=(
                self.persist_snapshot
                if isinstance(self, SnapshotCapableTracer) and self.snapshot_enabled
                else None
            ),
            trace_context=self,
        )

    def get_traces(
        self,
        trace_id: str | None = None,
        attribute_filter: Mapping[str, Any] | None = None,
    ) -> Sequence[Trace]:
        raise NotImplementedError(
            f"You're using the {self.__class__.__name__} tracer, that doesn't support get_traces(). "
            "Use another tracer, such as the InMemoryTracer, SqliteTracer or the DynamoDBTracer. "
        )

    def persist(self, trace: Trace):
        raise NotImplementedError(
            f"You're using the {self.__class__.__name__} tracer, that doesn't support persist()."
        )

    def persist_snapshot(self, trace: Trace):
        pass

    @staticmethod
    def apply_attribute_filter(
        traces: Iterable[Trace],
        trace_id: str | None = None,
        attribute_filter: Mapping[str, Any] | None = None,
    ):
        return sorted(
            filter(
                lambda trace: (trace_id is None or trace.trace_id == trace_id)
                and (
                    attribute_filter is None
                    or all(
                        k in trace.attributes and trace.attributes[k] == v
                        for k, v in attribute_filter.items()
                    )
                ),
                traces,
            ),
            key=lambda t: t.started_at,
        )


class NoopTracer(BaseTracer):

    def persist(self, trace: Trace):
        pass


class StreamTracer(BaseTracer, SnapshotCapableTracer):

    def __init__(
        self,
        *,
        stream: TextIO | None = None,
        trace_context_provider: TraceContextProvider | None = None,
        snapshot_enabled=False,
    ):
        super().__init__(trace_context_provider)
        self._stream = stream or sys.stdout
        self.snapshot_enabled = snapshot_enabled

    def persist(self, trace: Trace):
        raise NotImplementedError

    def persist_snapshot(self, trace: Trace):
        if self.snapshot_enabled:
            return self.persist(trace)


class StructuredLogsTracer(StreamTracer):

    def __init__(
        self,
        *,
        stream: TextIO | None = None,
        trace_context_provider: TraceContextProvider | None = None,
    ):
        super().__init__(stream=stream, trace_context_provider=trace_context_provider)
        self.logger = SimpleLogger("TraceLogger", stream=self._stream)

    def persist(self, trace: Trace):
        with self.lock:
            self.logger.info("Trace", trace=trace.as_dict())


class HumanReadableTracer(StreamTracer):

    def __init__(
        self,
        *,
        max_length: int | None = None,
        max_lines: int | None = None,
        snapshot_trace_types: tuple[str, ...] = ("llm-invocation", "tool-invocation"),
        stream: TextIO | None = None,
        trace_context_provider: TraceContextProvider | None = None,
        print_snapshot_every: float | None = None,
    ):
        super().__init__(
            stream=stream,
            trace_context_provider=trace_context_provider,
        )
        self.snapshot_enabled = bool(snapshot_trace_types)
        self.max_length = (
            max_length
            if max_length is not None
            else (max(shutil.get_terminal_size().columns - 40, 80))
        )
        self.max_lines = max_lines if max_lines is not None else 4
        self.snapshot_trace_types = snapshot_trace_types
        self.last_snapshot_printed = 0.0
        self.print_snapshot_every = (
            print_snapshot_every if print_snapshot_every is not None else 0.3
        )

    def persist(self, trace: Trace):
        human_readable_trace = trace.as_human_readable(
            max_length=self.max_length, max_lines=self.max_lines
        )
        with self.lock:
            print(human_readable_trace, file=self._stream)

    def persist_snapshot(self, trace: Trace):
        if (
            not self.snapshot_enabled
            or "ai.trace.type" not in trace.attributes
            or trace.attributes["ai.trace.type"] not in self.snapshot_trace_types
        ):
            return
        with self.lock:
            now = time.monotonic()
            if (now - self.last_snapshot_printed) > self.print_snapshot_every:
                human_readable_trace = trace.as_human_readable(
                    max_length=self.max_length, max_lines=self.max_lines
                )
                print(human_readable_trace, file=self._stream)
                self.last_snapshot_printed = now


class InMemoryTracer(BaseTracer):

    def __init__(
        self,
        memory_size=1000,
        trace_context_provider: TraceContextProvider | None = None,
    ) -> None:
        super().__init__(trace_context_provider=trace_context_provider)
        self._memory: deque[Trace] = deque(maxlen=memory_size)

    def persist(self, trace: Trace):
        with self.lock:
            self._memory.append(trace)

    def get_traces(
        self,
        trace_id: str | None = None,
        attribute_filter: Mapping[str, Any] | None = None,
    ) -> Sequence[Trace]:
        with self.lock:
            # Create a shallow copy of the deque to safely iterate
            memory_snapshot = list(self._memory)
        return self.apply_attribute_filter(
            memory_snapshot,
            trace_id=trace_id,
            attribute_filter=attribute_filter,
        )


class SqliteTracer(BaseTracer):

    def __init__(
        self,
        *,
        db_path: str | Path | None = None,
        identifier: str | None = None,
        create_tables: bool = True,
        trace_context_provider: TraceContextProvider | None = None,
    ):
        super().__init__(trace_context_provider=trace_context_provider)
        self.db_path = (
            Path(db_path)
            if db_path is not None
            else Path(os.getcwd()) / "conversations.db"
        )
        self.identifier = identifier

        if create_tables:
            self._create_tables()

    @property
    def conn(self) -> sqlite3.Connection:
        if not hasattr(self, "_locals"):
            self._locals = threading.local()
        if not hasattr(self._locals, "conn"):
            self._locals.conn = sqlite3.connect(self.db_path)
            self._locals.conn.row_factory = sqlite3.Row
        return self._locals.conn

    def _create_tables(self) -> None:
        with self.lock, self.conn as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS traces (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    trace_id TEXT NOT NULL,
                    span_id TEXT NOT NULL,
                    span_kind TEXT NOT NULL,
                    span_name TEXT NOT NULL,
                    span_status TEXT NOT NULL,
                    scope_name TEXT,
                    scope_version TEXT,
                    resource_attributes BLOB,
                    parent_span_id TEXT,
                    started_at TEXT NOT NULL,
                    ended_at TEXT,
                    attributes BLOB,
                    identifier TEXT,
                    conversation_id TEXT,
                    subcontext_id TEXT
                )
                """
            )

            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_traces_trace_id_started_at
                ON traces (trace_id, started_at)
                """
            )

            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_traces_conversation_id_started_at
                ON traces (conversation_id, started_at)
                """
            )

            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_traces_conversation_id_subcontext_id_started_at
                ON traces (conversation_id, subcontext_id, started_at)
                """
            )

            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_traces_trace_id
                ON traces (trace_id)
                """
            )

            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_traces_span_id
                ON traces (span_id)
                """
            )

            conn.commit()

    def __repr__(self) -> str:
        return f"SqliteTracer(db_path={self.db_path}, identifier={self.identifier})"

    def persist(self, trace: Trace) -> None:
        with self.lock, self.conn as conn:
            conn.execute(
                """
                INSERT INTO traces
                (trace_id,
                 span_id,
                 span_kind,
                 span_name,
                 span_status,
                 scope_name,
                 scope_version,
                 resource_attributes,
                 parent_span_id,
                 started_at,
                 ended_at,
                 attributes,
                 identifier,
                 conversation_id,
                 subcontext_id)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    trace.trace_id,
                    trace.span_id,
                    trace.span_kind,
                    trace.span_name,
                    trace.span_status,
                    trace.scope.name,
                    trace.scope.version,
                    JsonBytes.dumps(dict(trace.resource_attributes)),
                    trace.parent_span.span_id if trace.parent_span else None,
                    trace.started_at.isoformat(),
                    trace.ended_at.isoformat() if trace.ended_at else None,
                    JsonBytes.dumps(dict(trace.attributes)),
                    self.identifier,
                    trace.attributes.get("ai.conversation.id") or None,
                    trace.attributes.get("ai.subcontext.id") or None,
                ),
            )
            conn.commit()

    def get_traces(
        self,
        trace_id: str | None = None,
        attribute_filter: Mapping[str, Any] | None = None,
    ) -> Sequence[Trace]:
        query_parts = [
            """
            SELECT trace_id,
                    span_id,
                    span_kind,
                    span_name,
                    span_status,
                    scope_name,
                    scope_version,
                    resource_attributes,
                    parent_span_id,
                    started_at,
                    ended_at,
                    attributes,
                    identifier,
                    conversation_id,
                    subcontext_id
            FROM traces
            WHERE (identifier IS ? OR (identifier IS NULL AND ? IS NULL))
            """
        ]
        params = [self.identifier, self.identifier]
        # Create shallow copy:
        attribute_filter = dict(attribute_filter or {})
        if trace_id:
            query_parts.append("AND trace_id = ?")
            params.append(trace_id)
        elif (
            attribute_filter
            and "ai.conversation.id" in attribute_filter
            and "ai.subcontext.id" in attribute_filter
        ):
            conversation_id = attribute_filter.pop("ai.conversation.id")
            subcontext_id = attribute_filter.pop("ai.subcontext.id")
            query_parts.append("AND conversation_id = ?")
            params.append(conversation_id)
            # Handle NULL subcontext_id correctly in SQL
            if subcontext_id is None:
                query_parts.append("AND subcontext_id IS NULL")
            else:
                query_parts.append("AND subcontext_id = ?")
                params.append(subcontext_id)
        else:
            raise ValueError(
                "To use get_traces() you must either provide trace_id, or attribute_filter with keys 'ai.conversation.id' and 'ai.subcontext.id'"
            )

        query_parts.append("ORDER BY started_at ASC")
        query = " ".join(query_parts)

        with self.lock, self.conn as conn:
            cursor = conn.execute(query, params)

            traces: dict[str, Trace] = {}
            for row in cursor.fetchall():
                attributes = (
                    JsonBytes.loads(row["attributes"]) if row["attributes"] else {}
                )
                resource_attributes = (
                    JsonBytes.loads(row["resource_attributes"])
                    if row["resource_attributes"]
                    else {}
                )

                started_at = datetime.fromisoformat(row["started_at"])
                ended_at = (
                    datetime.fromisoformat(row["ended_at"]) if row["ended_at"] else None
                )

                trace = Trace(
                    span_name=row["span_name"],
                    span_kind=row["span_kind"],
                    trace_id=row["trace_id"],
                    span_id=row["span_id"],
                    parent_span=(
                        traces.get(row["parent_span_id"])
                        if row["parent_span_id"]
                        else None
                    ),
                    started_at=started_at,
                    ended_at=ended_at,
                    attributes=attributes,
                    span_status=row["span_status"],
                    resource_attributes=resource_attributes,
                    scope=TraceScope(row["scope_name"], row["scope_version"]),
                )

                traces[trace.span_id] = trace

        return self.apply_attribute_filter(
            traces.values(),
            trace_id=trace_id,
            attribute_filter=attribute_filter,
        )


@runtime_checkable
class ChainableTracer(Tracer, Protocol):
    @property
    def tracers(self) -> Sequence[Tracer]: ...

    def add_tracer(self, tracer: Tracer) -> "TeeTracer": ...

    def remove_tracer(self, tracer: Tracer) -> "TeeTracer": ...

    def temporary_tracer(self, tracer: Tracer) -> AbstractContextManager[None]: ...


class TeeTracer(BaseTracer, ChainableTracer):

    _tracers: list[Tracer]

    def __init__(
        self,
        trace_context_provider: TraceContextProvider | None = None,
    ) -> None:
        super().__init__(trace_context_provider=trace_context_provider)
        self._tracers = []

    @property
    def tracers(self) -> Sequence[Tracer]:
        return self._tracers

    def add_tracer(self, tracer: Tracer) -> "TeeTracer":
        with self.lock:
            if not self.snapshot_enabled and isinstance(tracer, SnapshotCapableTracer):
                self.snapshot_enabled = True
            self._tracers.append(tracer)
        return self  # allow chaining add_tracer() calls

    def remove_tracer(self, tracer: Tracer) -> "TeeTracer":
        with self.lock:
            self._tracers.remove(tracer)
            for remaining_tracer in self._tracers:
                if isinstance(remaining_tracer, SnapshotCapableTracer):
                    break
            else:
                self.snapshot_enabled = False

        return self  # allow chaining remove_tracer() calls

    @contextmanager
    def temporary_tracer(self, tracer: Tracer):
        self.add_tracer(tracer)
        try:
            yield
        finally:
            self.remove_tracer(tracer)

    def persist(self, trace: Trace):
        with self.lock:
            for tracer in self._tracers:
                tracer.persist(trace)

    def persist_snapshot(self, trace: Trace):
        with self.lock:
            for tracer in self._tracers:
                if isinstance(tracer, SnapshotCapableTracer):
                    tracer.persist_snapshot(trace)

    def get_traces(
        self,
        trace_id: str | None = None,
        attribute_filter: Mapping[str, Any] | None = None,
    ) -> Sequence[Trace]:
        with self.lock:
            if not self._tracers:
                return []
            first_tracer = self._tracers[0]
        return first_tracer.get_traces(
            trace_id=trace_id, attribute_filter=attribute_filter
        )


class QueueTracer(BaseTracer):
    def __init__(
        self,
        maxsize: int = -1,
        trace_context_provider: TraceContextProvider | None = None,
        queue: Queue | None = None,
    ):
        super().__init__(trace_context_provider)
        self.snapshot_enabled = True
        self.queue = queue or Queue(maxsize)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if not self.queue.is_shutdown:
            self.queue.shutdown()

    def shutdown(self):
        self.queue.shutdown()

    def persist(self, trace: Trace):
        self.queue.put(trace)

    def persist_snapshot(self, trace: Trace):
        self.queue.put(trace)


class IterableTracer(QueueTracer):
    """
    Threadsafe tracer that allows you to iterate over incoming traces.
    The iteration can by stopped by signalling `shutdown()`.
    This tracer can e.g. be used for tracing one "turn", i.e. one invocation of converse_stream()
    """

    def __iter__(self):
        while True:
            try:
                yield self.queue.get()
            except ShutDown:
                break
