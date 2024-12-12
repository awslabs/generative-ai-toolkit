/**
 * Copyright 2024 Amazon.com, Inc. and its affiliates. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

import { ReactNode } from "react";
import { Node, Edge } from "@xyflow/react";

export type CustomNodeData = {
  label: string;
  content: string;
  additional_information: Record<string, unknown>;
  caseName?: string;
  icon?: ReactNode;
  footer?: string;
  toolInput?: string;
  toolOutput?: string;
  startNode?: boolean;
  stepNode?: boolean;
  conversationId?: string;
  step: number;
  measurements?: Measurement[];
  usage?: Usage;
  modelId?: string;
  nodeType: "step" | "request" | "response" | "tool" | "summary";
};

// Define CustomNode without generics
export type CustomNode = Node & {
  data: CustomNodeData;
};

export type CustomEdge = Edge;

export type EdgeType = "no_label" | "LLM" | "TOOL" | "REQUEST" | "RESPONSE";

export interface ConversationMeasurements {
  conversation_id: string;
  case: Case;
  traces: TraceMeasurements[];
  measurements: Measurement[];
}

interface Case {
  name: string;
}

interface TraceMeasurements {
  trace: LlmTrace | LlmCaseTrace | ToolTrace | ToolCaseTrace;
  measurements: Measurement[];
}

interface Trace {
  conversation_id: string;
  trace_id: string;
  auth_context: unknown;
  created_at: string;
  additional_info: Record<string, any>;
  to: string;
}

export interface CaseTrace extends Trace {
  conversation_id: string;
  trace_id: string;
  auth_context: unknown;
  created_at: string;
  additional_info: Record<string, any>;
  to: string;
  case: Case;
  case_nr: number;
  run_nr: number;
  permutation: Permutation;
}

interface ToolRequest {
  tool_name: string;
  tool_input: Record<string, any>;
  tool_use_id: string;
}

interface ToolResponse {
  tool_response: Record<string, any>;
  latency_ms: number;
}

export interface ToolTrace extends Trace {
  to: "TOOL";
  request: ToolRequest
  response: ToolResponse
}

interface ToolCaseTrace extends CaseTrace {
  to: "TOOL";
  request: ToolRequest
  response: ToolResponse
}

export interface LlmTrace extends Trace {
  to: "LLM";
  request: Request;
  response: Response;
}

interface LlmCaseTrace extends CaseTrace {
  to: "LLM";
  request: Request;
  response: Response;
}

interface Request {
  modelId: string;
  inferenceConfig: Record<string, any>;
  messages: Message[];
  system: SystemMessage[];
}

interface Message {
  role: string;
  content: Content[];
}

interface Content {
  text: string;
}

interface SystemMessage {
  text: string;
}

interface Response {
  ResponseMetadata: unknown;
  output: Output;
  stopReason: string;
  usage: Usage;
  metrics: Metrics;
}

interface Output {
  message: Message;
}

export interface Usage {
  inputTokens: number;
  outputTokens: number;
  totalTokens: number;
}

interface Metrics {
  latencyMs: number;
}

interface Permutation {
  system_prompt: string;
}

export interface Measurement {
  name: string;
  value: number;
  unit: string;
  additional_info: unknown;
  dimensions: Dimension[];
  validation_passed: boolean | null;
}

interface Dimension {
  To?: string;
}
