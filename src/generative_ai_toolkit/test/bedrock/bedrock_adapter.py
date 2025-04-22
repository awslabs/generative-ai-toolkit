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

import time
import uuid
from collections.abc import Iterable
from datetime import datetime, timedelta
from typing import Any, Protocol

import boto3.session

from generative_ai_toolkit.tracer import Trace
from generative_ai_toolkit.tracer.tracer import TraceScope


class BedrockAgentAdapter(Protocol):

    def __init__(self, aget_id: str, aget_alias_id: str) -> None:
        self.__bedrock_agent_runtime = boto3.client("bedrock-agent-runtime")
        self.__conversation_id = uuid.uuid4().hex
        self.__aget_id = aget_id
        self.__aget_alias_id = aget_alias_id
        self.__conversation_id = uuid.uuid4().hex
        self.__spans = []

    def __invoke_agent(
        self,
        question: str,
        aget_id: str,
        aget_alias_id: str,
        conversation_id: str = uuid.uuid4().hex,
    ) -> tuple[list[str], list[str]]:
        agent_response = self.__bedrock_agent_runtime.invoke_agent(
            agentId=aget_id,
            agentAliasId=aget_alias_id,
            sessionId=conversation_id,
            inputText=question,
            enableTrace=True,
        )

        traces = []
        chunks = []
        for event in agent_response.get("completion"):
            time.sleep(0.5)
            if "trace" in event:
                traces.append(event["trace"])
            if "chunk" in event:
                chunks.append(event["chunk"]["bytes"].decode())

        return traces, chunks

    def converse(self, user_input: str) -> Any:

        bedrock_traces, chunks = self.__invoke_agent(
            question=user_input,
            aget_id=self.__aget_id,
            aget_alias_id=self.__aget_alias_id,  # Note: This was using conversation_id before
            conversation_id=self.__conversation_id,
        )

        model_inputs = []
        model_outputs = []
        ag_input = []
        ag_output = []
        # TODO: at the moment we ignore rationale, invocationInput, and some observation node, the test are runig well but we may lose details
        # consider those details in future
        for br_trace in bedrock_traces:
            session_id = br_trace.get("sessionId", None)
            if not session_id:
                raise ValueError("sessionId is not None")
            brt = br_trace.get("trace", None)

            if not brt:
                raise ValueError("trace is not None")
            orchestration_trace = brt.get("orchestrationTrace", {})

            if "modelInvocationInput" in orchestration_trace:
                model_inputs.append(
                    orchestration_trace.get("modelInvocationInput", None)
                )
            if "modelInvocationOutput" in orchestration_trace:
                model_outputs.append(
                    orchestration_trace.get("modelInvocationOutput", None)
                )
            # if "rationale" in orchestration_trace:
            #     rationales.append(orchestration_trace.get("rationale", None))
            # if "invocationInput" in orchestration_trace :
            #     invocation_inputs.append(orchestration_trace.get("invocationInput", None))
            # if "observation" in orchestration_trace:
            #     observations.append(orchestration_trace.get("observation", None))
            if orchestration_trace.get("invocationInput", {}).get(
                "actionGroupInvocationInput", None
            ):
                ag_input.append(orchestration_trace)
            if orchestration_trace.get("observation", {}).get(
                "actionGroupInvocationOutput", None
            ):
                ag_output.append(orchestration_trace.get("observation", {}))

        first_bedrock_trace = bedrock_traces[0]
        last_bedrock_trace = bedrock_traces[-1]

        agent_id = first_bedrock_trace.get("agentId", None)
        agent_version = first_bedrock_trace.get("agentVersion", None)
        session_id = first_bedrock_trace.get("sessionId", None)

        final_response = (
            last_bedrock_trace.get("trace", {})
            .get("orchestrationTrace", {})
            .get("observation")
            .get("finalResponse")
            .get("text", None)
        )
        trace_id = uuid.uuid4().hex

        agent_span = Trace(
            trace_id=trace_id,
            # span_id=uuid.uuid4().hex[:16],
            span_name="agent-trace",
            span_kind="SERVER",
            started_at=datetime.now(),
            ended_at=datetime.now() + timedelta(microseconds=50),
            scope=TraceScope(name="agent-trace-scope", version="1.0.0"),
            resource_attributes={
                "service.name": "agent-trace",
                "session.id": session_id,
            },
            attributes={
                "user-input": user_input,
                "response": final_response,
                "ai.conversation.id": self.__conversation_id,
                "agent.id": agent_id,
                "agent.alias.id": session_id,
                "agent.version": agent_version,
                "ai.trace.type": "converse",
                "ai.agent.response": final_response,
                "duration_ms": 10,
            },
        )
        self.__spans.append(agent_span)

        llm_invocation_spans = [
            self.__create_llm_invocation_span(
                model_input=model_input,
                model_output=model_output,
                parent_span=agent_span,
                agent_id=agent_id,
                session_id=session_id,
                agent_version=agent_version,
                user_input=user_input,
                final_response=final_response,
                conversation_id=self.__conversation_id,
            )
            for model_input, model_output in zip(model_inputs, model_outputs, strict=False)
        ]
        self.__spans.extend(llm_invocation_spans)

        tool_spans = [
            self.__create_tool_span(
                tool_input=tool_input,
                tool_output=tool_output,
                trace_id=trace_id,
                agent_span=agent_span,
                session_id=session_id,
                agent_id=agent_id,
                agent_version=agent_version,
                conversation_id=self.__conversation_id,
            )
            for tool_input, tool_output in zip(ag_input, ag_output, strict=False)
        ]
        self.__spans.extend(tool_spans)

        return "".join(chunks)

    # TODO: make it static
    def __create_llm_invocation_span(
        self,
        model_input: dict[str, Any],
        model_output: dict[str, Any],
        parent_span: Trace,
        agent_id: str,
        session_id: str,
        agent_version: str,
        user_input: str,
        final_response: str,
        conversation_id: str,
    ) -> Trace:

        resource_attributes = {
            "service.name": "agent-tool",
            "agent.id": agent_id,
            "agent.alias.id": session_id,
            "agent.version": agent_version,
            "session.id": session_id,
        }

        span_attributes = {
            "ai.trace.type": "llm-invocation",
            "ai.conversation.id": conversation_id,
            "ai.user.input": user_input,
            "ai.agent.response": final_response,
            "ai.llm.request.inference.config": model_input.get(
                "inferenceConfiguration", {}
            ),
            "ai.llm.request.messages": model_input.get("text", {}),
            "ai.llm.request.model.id": "eu.anthropic.claude-3-sonnet-20240229-v1:0",
            "ai.llm.request.system": model_input.get("text", {}),
            "peer.service": "llm:claude-3-sonnet",
            "ai.llm.response.usage": model_output.get("metadata", {}).get("usage", {}),
            "ai.llm.response.stop.reason": "tool_use",
        }

        start_time = datetime.now()

        return Trace(
            # trace_id=model_input.get("traceId"),
            span_id=uuid.uuid4().hex,
            span_name="llm-invocation",
            span_kind="CLIENT",
            parent_span=parent_span,
            started_at=start_time,
            ended_at=start_time + timedelta(microseconds=50),
            scope=TraceScope(name="agent-tool-scope", version="1.0.0"),
            resource_attributes=resource_attributes,
            attributes=span_attributes,
        )

    # TODO: make it static
    def __create_tool_span(
        self,
        tool_input: dict[str, Any],
        tool_output: dict[str, Any],
        trace_id: str,
        agent_span: Trace,
        session_id: str,
        agent_id: str,
        agent_version: str,
        conversation_id: str,
    ) -> Trace:
        invocation_input = tool_input.get("invocationInput", {})
        action_group_input = invocation_input.get("actionGroupInvocationInput", {})
        tool_api_path = action_group_input.get("apiPath")

        if not tool_api_path:
            raise ValueError(
                "invocationInput.actionGroupInvocationInput.apiPath can not be None"
            )

        current_time = datetime.now()

        resource_attrs = {
            "service.name": "agent-tool",
            "agent.id": agent_id,
            "agent.alias.id": session_id,
            "agent.version": agent_version,
            "session.id": session_id,
        }

        raw_response = tool_output.get("rawResponse", {})
        span_attrs = {
            "ai.trace.type": "tool-invocation",
            "ai.tool.use.id": f"tooluse_{uuid.uuid4().hex}",
            "ai.conversation.id": conversation_id,
            "ai.tool.input": action_group_input,
            "ai.tool.name": tool_api_path,
            "ai.tool.output": raw_response.get("content"),
            "peer.service": f"tool:{tool_api_path[1:]}",
        }

        return Trace(
            trace_id=trace_id,
            span_name="tool-trace",
            span_kind="CLIENT",
            parent_span=agent_span,
            started_at=current_time,
            ended_at=current_time + timedelta(microseconds=50),
            scope=TraceScope(name="agent-tool-scope", version="1.0.0"),
            resource_attributes=resource_attrs,
            attributes=span_attrs,
        )

    @property
    def traces(self) -> Iterable[Trace]:
        return self.__spans

    def reset(self) -> None:
        self.__conversation_id = uuid.uuid4().hex
        self.__spans.clear()


if __name__ == "__main__":
    bedrock_agent_adapter = BedrockAgentAdapter()
    answer = bedrock_agent_adapter.converse("What is the weather in Munich?")
    traces = bedrock_agent_adapter.traces

    # tracer = OtlpTracer()
    # for t in bedrockAgent_adapter.traces:
    #     tracer.persist(t)
    # batch = OtlpBatch([x])
    # print(batch.protobuf())

    print("Done")
