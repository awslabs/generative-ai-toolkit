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

import http.client
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from itertools import groupby
from typing import Any, Literal

from opentelemetry.proto.collector.trace.v1.trace_service_pb2 import (
    ExportTraceServiceRequest as OtlpExportTraceServiceRequest,
)
from opentelemetry.proto.common.v1.common_pb2 import (
    AnyValue as OtlpAnyValue,
)
from opentelemetry.proto.common.v1.common_pb2 import (
    InstrumentationScope as OtlpInstrumentationScope,
)
from opentelemetry.proto.common.v1.common_pb2 import (
    KeyValue as OtlpKeyValue,
)
from opentelemetry.proto.resource.v1.resource_pb2 import Resource as OtlpResource
from opentelemetry.proto.trace.v1.trace_pb2 import (
    ResourceSpans as OtlpResourceSpans,
)
from opentelemetry.proto.trace.v1.trace_pb2 import (
    ScopeSpans as OtlpScopeSpans,
)
from opentelemetry.proto.trace.v1.trace_pb2 import (
    Span as OtlpSpan,
)
from opentelemetry.proto.trace.v1.trace_pb2 import (
    Status as OtlpStatus,
)

from generative_ai_toolkit.tracer.trace import Trace
from generative_ai_toolkit.tracer.tracer import (
    BaseTracer,
    TraceContextProvider,
    TraceScope,
)
from generative_ai_toolkit.utils.json import DefaultJsonEncoder


@dataclass
class ScopeSpan:
    scope: TraceScope
    spans: list[Trace]


@dataclass
class ResourceSpan:
    attributes: Mapping[str, Any]
    scopes: list[ScopeSpan]


class OtlpBatch:
    def __init__(self, traces: Sequence[Trace]):
        traces = sorted(traces, key=lambda trace: trace.scope)
        traces = sorted(
            traces,
            key=lambda trace: tuple(sorted(trace.resource_attributes.items())),
        )
        self.resource_spans: list[ResourceSpan] = []
        by_resource = groupby(traces, key=lambda trace: trace.resource_attributes)
        for resource_attributes, traces_by_resource in by_resource:
            resource_span = ResourceSpan(attributes=resource_attributes, scopes=[])
            self.resource_spans.append(resource_span)
            by_scope = groupby(traces_by_resource, key=lambda trace: trace.scope)
            for scope, traces_by_scope in by_scope:
                scope_span = ScopeSpan(scope=scope, spans=[])
                resource_span.scopes.append(scope_span)
                for trace in traces_by_scope:
                    scope_span.spans.append(trace)

    def protobuf(self):
        """
        Return a protobuf that can be sent to the ADOT collector
        """
        resource_spans: list[OtlpResourceSpans] = []
        for resource_span in self.resource_spans:
            scope_spans: list[OtlpScopeSpans] = []
            for scope_span in resource_span.scopes:
                otlp_spans: list[OtlpSpan] = []
                for span in scope_span.spans:
                    otlp_spans.append(
                        OtlpSpan(
                            trace_id=bytes.fromhex(span.trace_id),
                            span_id=bytes.fromhex(span.span_id),
                            status=self._otlp_span_status_protobuf(span.span_status),
                            kind=self._otlp_span_kind_protobuf(span.span_kind),
                            name=span.span_name,
                            start_time_unix_nano=int(span.started_at.timestamp() * 1e9),
                            end_time_unix_nano=int(
                                (span.ended_at or span.started_at).timestamp() * 1e9
                            ),
                            attributes=[
                                OtlpKeyValue(key=k, value=self._otlp_protobuf_value(v))
                                for k, v in span.attributes.items()
                            ],
                        )
                    )
                    if span.parent_span:
                        otlp_spans[-1].parent_span_id = bytes.fromhex(
                            span.parent_span.span_id
                        )
                otlp_scope_span = OtlpScopeSpans(
                    scope=OtlpInstrumentationScope(
                        name=scope_span.scope.name, version=scope_span.scope.version
                    ),
                    spans=otlp_spans,
                )
                scope_spans.append(otlp_scope_span)
            resource_spans.append(
                OtlpResourceSpans(
                    resource=OtlpResource(
                        attributes=[
                            OtlpKeyValue(key=k, value=self._otlp_protobuf_value(v))
                            for k, v in resource_span.attributes.items()
                        ]
                    ),
                    scope_spans=scope_spans,
                )
            )
        return OtlpExportTraceServiceRequest(resource_spans=resource_spans)

    @staticmethod
    def _otlp_protobuf_value(value: Any) -> OtlpAnyValue:
        if isinstance(value, str):
            return OtlpAnyValue(string_value=value)
        elif isinstance(value, bool):
            return OtlpAnyValue(bool_value=value)
        elif isinstance(value, int):
            return OtlpAnyValue(int_value=value)
        elif isinstance(value, float):
            return OtlpAnyValue(double_value=value)
        elif value is None:
            return OtlpAnyValue()
        else:
            return OtlpAnyValue(string_value=json.dumps(value, cls=DefaultJsonEncoder))

    SPAN_KIND_PROTOBUF_MAPPING = {
        "SERVER": OtlpSpan.SpanKind.SPAN_KIND_SERVER,
        "CLIENT": OtlpSpan.SpanKind.SPAN_KIND_CLIENT,
    }

    @classmethod
    def _otlp_span_kind_protobuf(cls, kind: Literal["INTERNAL", "SERVER", "CLIENT"]):
        return cls.SPAN_KIND_PROTOBUF_MAPPING.get(
            kind, OtlpSpan.SpanKind.SPAN_KIND_INTERNAL
        )

    SPAN_STATUS_PROTOBUF_MAPPING = {
        "OK": OtlpStatus.STATUS_CODE_OK,
        "ERROR": OtlpStatus.STATUS_CODE_ERROR,
    }

    @classmethod
    def _otlp_span_status_protobuf(cls, status: Literal["OK", "ERROR", "UNSET"]):
        return OtlpStatus(
            code=cls.SPAN_STATUS_PROTOBUF_MAPPING.get(
                status, OtlpStatus.STATUS_CODE_UNSET
            )
        )


class OtlpTracer(BaseTracer):

    def __init__(
        self,
        host="localhost",
        port=4318,
        trace_context_provider: TraceContextProvider | None = None,
    ):
        super().__init__(trace_context_provider)
        self.conn = http.client.HTTPConnection(host, port)

    def persist(self, trace: Trace):
        self._send_protobuf(OtlpBatch([trace]).protobuf().SerializeToString())

    def _send_protobuf(self, body: bytes):
        with self.lock:
            self.conn.request(
                "POST",
                "/v1/traces",
                body=body,
                headers={
                    "Content-Type": "application/x-protobuf",
                },
            )
            response = self.conn.getresponse()

            # Must read response, in order to be able to re-use connection
            # Also, nice for potential error message
            response_body = response.read()

            if response.status != 200:
                raise ValueError(
                    f"Failed to send batch: {response.status} {response.reason} {response_body.decode()}"
                )
