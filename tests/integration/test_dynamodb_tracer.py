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


import pytest

from generative_ai_toolkit.tracer.dynamodb import DynamoDbTracer
from generative_ai_toolkit.tracer.trace import Trace


def test_dynamodb_serialization():
    tracer = DynamoDbTracer("traces")
    with tracer.trace("test-parent") as parent_trace:
        parent_trace.add_attribute("string", "string")
        parent_trace.add_attribute("set", {"a", "b", "c"})
        parent_trace.add_attribute("list", [1, 2, 3])
        parent_trace.add_attribute("bytes", b"bytes")
        parent_trace.add_attribute(
            "exception.message", RuntimeError("exception message")
        )

        with tracer.trace("test-child") as child_trace:
            child_trace.add_attribute("child", "child")

    parent_deserialized, child_deserialized = tracer.get_traces(
        trace_id=parent_trace.trace_id
    )

    # Check each attribute of parent seperately:
    assert parent_deserialized.trace_id == parent_trace.trace_id
    assert parent_deserialized.span_id == parent_trace.span_id
    assert parent_deserialized.span_name == parent_trace.span_name
    assert parent_deserialized.span_kind == parent_trace.span_kind
    assert parent_deserialized.span_status == parent_trace.span_status
    assert parent_deserialized.duration_ms == parent_trace.duration_ms
    assert parent_deserialized.started_at == parent_trace.started_at
    assert parent_deserialized.ended_at == parent_trace.ended_at
    assert parent_deserialized.parents == parent_trace.parents
    assert parent_deserialized.resource_attributes == parent_trace.resource_attributes
    assert parent_deserialized.scope == parent_trace.scope
    for k, v in parent_deserialized.attributes.items():
        assert k in parent_trace.attributes
        if k == "exception.message":
            assert v == repr(parent_trace.attributes[k])
        else:
            assert v == parent_trace.attributes[k]
    assert child_deserialized.as_dict() == child_trace.as_dict()

    error_trace = Trace("bound")
    with pytest.raises(ValueError):
        with tracer.trace("test-exception") as error_trace:
            raise ValueError("Oooops")

    error_deserialized = tracer.get_traces(trace_id=error_trace.trace_id)[0]
    assert error_deserialized.as_dict() == error_trace.as_dict()
