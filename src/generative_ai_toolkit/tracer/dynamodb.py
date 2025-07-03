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

from collections.abc import Mapping
from datetime import datetime
from typing import Any

import boto3
import boto3.session
from boto3.dynamodb.conditions import Key

from generative_ai_toolkit.tracer.tracer import (
    BaseTracer,
    Trace,
    TraceContextProvider,
    TraceScope,
)
from generative_ai_toolkit.utils.dynamodb import DynamoDbMapper


class DynamoDbTracer(BaseTracer):

    def __init__(
        self,
        table_name: str,
        *,
        session: boto3.session.Session | None = None,
        trace_context_provider: TraceContextProvider | None = None,
        identifier: str | None = None,
        ttl: int | None = 60 * 60 * 24 * 30,  # 30 days
        conversation_id_gsi_name="by_conversation_id",
    ) -> None:
        super().__init__(trace_context_provider=trace_context_provider)
        self.table = (session or boto3).resource("dynamodb").Table(table_name)
        self.identifier = identifier
        self.ttl = ttl
        self.conversation_id_gsi_name = conversation_id_gsi_name

    def persist(self, trace: Trace):
        with self.lock:
            item = {
                "pk": f"TRACE#{trace.trace_id}",
                "sk": f"SPAN#{self.identifier or "_"}#{int(trace.started_at.timestamp() * 1_000_000):017x}#{trace.span_id}",
                "trace_id": trace.trace_id,
                "span_id": trace.span_id,
                "span_kind": trace.span_kind,
                "span_name": trace.span_name,
                "scope_name": trace.scope.name,
                "scope_version": trace.scope.version,
                "resource_attributes": trace.resource_attributes,
                "parent_span_id": (
                    trace.parent_span.span_id if trace.parent_span else None
                ),
                "started_at": trace.started_at,
                "ended_at": trace.ended_at,
                "duration_ms": trace.duration_ms,
                "attributes": trace.attributes,
                "identifier": self.identifier,
            }
            if self.ttl is not None:
                item["expire_at"] = int(trace.started_at.timestamp()) + self.ttl

            # Maintain as top level attributes for querying with GSI:
            if "ai.conversation.id" in trace.attributes:
                item["conversation_id"] = trace.attributes["ai.conversation.id"]

            try:
                self.table.put_item(
                    Item=DynamoDbMapper.to_dynamo(item),
                    ConditionExpression="attribute_not_exists(pk) AND attribute_not_exists(sk)",
                )
            except self.table.meta.client.exceptions.ResourceNotFoundException as e:
                raise ValueError(f"Table {self.table.name} does not exist") from e
            except (
                self.table.meta.client.exceptions.ConditionalCheckFailedException
            ) as e:
                raise ValueError(f"Trace {trace.trace_id} already exists") from e

    def get_traces(
        self,
        trace_id: str | None = None,
        attribute_filter: Mapping[str, Any] | None = None,
    ):
        with self.lock:
            params = {}
            if trace_id:
                params["KeyConditionExpression"] = Key("pk").eq(
                    f"TRACE#{trace_id}"
                ) & Key("sk").begins_with(f"SPAN#{self.identifier or "_"}#")
            elif attribute_filter and "ai.conversation.id" in attribute_filter:
                params["KeyConditionExpression"] = Key("conversation_id").eq(
                    attribute_filter["ai.conversation.id"]
                ) & Key("sk").begins_with(f"SPAN#{self.identifier or "_"}#")
                params["IndexName"] = self.conversation_id_gsi_name
            else:
                raise ValueError(
                    "To use get_traces() you must either provide trace_id, or attribute_filter with key 'ai.conversation.id'"
                )

            items = []
            last_evaluated_key: dict[str, Any] = {}
            while True:
                try:
                    response = self.table.query(
                        ScanIndexForward=True,
                        **params,
                        **last_evaluated_key,
                    )

                except self.table.meta.client.exceptions.ResourceNotFoundException as e:
                    raise ValueError(f"Table {self.table.name} does not exist") from e
                items.extend(response["Items"])
                if "LastEvaluatedKey" not in response:
                    break
                last_evaluated_key = {"ExclusiveStartKey": response["LastEvaluatedKey"]}

            traces: dict[str, Trace] = {}
            for item in response["Items"]:
                trace = self.item_to_trace(item, traces)
                # Apply "ai.auth.context" filter:
                if attribute_filter and "ai.auth.context" in attribute_filter:
                    if (
                        trace.attributes.get("ai.auth.context")
                        != attribute_filter["ai.auth.context"]
                    ):
                        continue
                traces[trace.span_id] = trace

        return sorted(traces.values(), key=lambda t: t.started_at)

    @staticmethod
    def item_to_trace(
        item: Mapping[str, Any], parent_traces_lookup: Mapping[str, Trace] = {}
    ):
        parsed: dict = DynamoDbMapper.from_dynamo(item)
        parent_span_id = parsed.pop("parent_span_id", None)
        return Trace(
            parsed["span_name"],
            trace_id=parsed["trace_id"],
            span_id=parsed["span_id"],
            span_kind=parsed["span_kind"],
            scope=TraceScope(parsed["scope_name"], parsed["scope_version"]),
            parent_span=(
                parent_traces_lookup.get(parent_span_id) if parent_span_id else None
            ),
            started_at=datetime.fromisoformat(parsed["started_at"]),
            ended_at=datetime.fromisoformat(parsed["ended_at"]),
            resource_attributes=parsed.get("resource_attributes", {}),
            attributes=parsed["attributes"],
        )
