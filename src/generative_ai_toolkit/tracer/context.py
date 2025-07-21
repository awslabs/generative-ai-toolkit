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

from collections.abc import Callable, Mapping
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import (
    Any,
    Protocol,
    TypedDict,
    Unpack,
)

from generative_ai_toolkit.tracer.trace import Trace, TraceScope


class TraceContextUpdate(TypedDict, total=False):
    span: Trace | None
    scope: TraceScope
    resource_attributes: Mapping[str, Any]
    span_attributes: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class TraceContext:
    span: Trace | None = None
    scope: TraceScope | None = None
    resource_attributes: Mapping[str, Any] = field(default_factory=dict)
    span_attributes: Mapping[str, Any] = field(default_factory=dict)


class TraceContextProvider(Protocol):

    @property
    def context(self) -> TraceContext: ...

    def set_context(
        self, **update: Unpack[TraceContextUpdate]
    ) -> Callable[[], None]: ...


class ContextVarTraceContextProvider(TraceContextProvider):
    def __init__(self) -> None:
        self._context = ContextVar[TraceContext | None]("trace_context", default=None)

    @property
    def context(self) -> TraceContext:
        trace_context = self._context.get()
        if not trace_context:
            trace_context = TraceContext()
            self._context.set(trace_context)
        return trace_context

    def set_context(self, **update: Unpack[TraceContextUpdate]) -> Callable[[], None]:
        old_context = self.context
        new_context = TraceContext(
            span=update["span"] if "span" in update else old_context.span,
            scope=update["scope"] if "scope" in update else old_context.scope,
            resource_attributes=(
                update["resource_attributes"]
                if "resource_attributes" in update
                else old_context.resource_attributes
            ),
            span_attributes=(
                update["span_attributes"]
                if "span_attributes" in update
                else old_context.span_attributes
            ),
        )
        token = self._context.set(new_context)
        return lambda: self._context.reset(token)
