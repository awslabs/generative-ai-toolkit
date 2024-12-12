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

import React, {useEffect} from 'react';
import {
    ReactFlow,
    useNodesState,
    useEdgesState,
    Position,
    Background,
    Controls,
    MarkerType,
} from '@xyflow/react';
import NodeComponent from '../components/NodeComponent';
import {FiTool, FiUser, FiInfo} from 'react-icons/fi';
import {TbProgressBolt} from "react-icons/tb";
import {RiRobot2Line} from "react-icons/ri";
import '@xyflow/react/dist/style.css';
import '../index.css';
import useSWR from 'swr'

import {CustomNode, CustomEdge, CustomNodeData, EdgeType, ConversationMeasurements, LlmTrace, CaseTrace, Usage } from '../types/types';
import {useLocation} from "react-router-dom";

const createNode = (
    id: string,
    data: CustomNodeData,
    position: { x: number; y: number }
): CustomNode => ({
    id,
    type: 'custom',
    data,
    position,
});

const createEdge = (
    id: string,
    source: string,
    target: string,
    edgeType: EdgeType
): CustomEdge => {
    let label: string | undefined;
    let sourceHandle: Position | undefined;
    let targetHandle: Position | undefined;

    switch (edgeType) {
        case 'LLM':
            label = 'LLM';
            sourceHandle = Position.Bottom;
            targetHandle = Position.Top;
            break;
        case 'TOOL':
            label = 'TOOL';
            sourceHandle = Position.Bottom;
            targetHandle = Position.Top;
            break;
        case 'REQUEST':
            label = 'REQUEST';
            sourceHandle = Position.Right;
            targetHandle = Position.Left;
            break;
        case 'RESPONSE':
            label = 'RESPONSE';
            sourceHandle = Position.Right;
            targetHandle = Position.Left;
            break;
        case 'no_label':
        default:
            label = undefined;
            sourceHandle = Position.Bottom;
            targetHandle = Position.Top;
    }

    return {
        id,
        source,
        target,
        type: 'smoothstep',
        sourceHandle,
        targetHandle,
        markerEnd: {
            type: MarkerType.ArrowClosed,
            width: 22,
            height: 22,
            color: '#1a1a1a',
        },
        style: {
            strokeWidth: 1,
            stroke: '#1a1a1a',
        },
        label,
        labelStyle: {fill: '#000', fontWeight: 700, fontSize: 12},
        labelBgStyle: {fill: '#FFFFFF'},
        labelBgPadding: [8, 4],
        labelBgBorderRadius: 4,
        data: {testId: `edge-${id}`, edgeType},
    };
};

const TestPage: React.FC = () => {
    const [nodes, setNodes, onNodesChange] = useNodesState<CustomNode>([]);
    const [edges, setEdges, onEdgesChange] = useEdgesState<CustomEdge>([]);
    const { data: testData = [], isLoading } = useSWR<ConversationMeasurements[]>('http://127.0.0.1:8000/get_conversation_traces', (url: string) => fetch(url).then(res => res.json()))

    const location = useLocation();
    const queryParams = new URLSearchParams(location.search);
    const index = Number(queryParams.get('index'));

    useEffect(() => {
        if (!Number.isFinite(index)) {
            return;
        }
        const conversationMeasurement = testData.at(index);
        if (!conversationMeasurement) return;

        const printStyledLog = () => {
            console.log(
                // nosemgrep: javascript.lang.security.audit.unsafe-formatstring.unsafe-formatstring
                `%c Test Number: ${index} %c`,
                'color: white; background-color: blue; font-size: 12px; font-weight: bold; padding: 2px 10px; border-radius: 5px;',
                'color: black; background-color: #f0f0f0; font-size: 14px; padding: 2px 10px;'
            );
        };

        printStyledLog();

        const newNodes: CustomNode[] = [];
        const newEdges: CustomEdge[] = [];
        const yOffset = 600;
        // let xOffset = 1200;
        const messageXOffset = 650;
        // let numberOfOutputs = 0;

        let totalUsage: Usage = {inputTokens: 0, outputTokens: 0, totalTokens: 0};
        let totalMeasurements: { [key: string]: number } = {};

        let previousStepNodeId: string | null = null;
            let currentY = 0;
            let stepCount = 0;

            const evaluationNumber = `test ${index}`;
            const modelId = (conversationMeasurement.traces[0].trace as LlmTrace).request.modelId || 'Unknown Model';

            const conversationId = conversationMeasurement.conversation_id || 'Unknown Conversation ID';
            const caseName = (conversationMeasurement.traces[0].trace as CaseTrace).case.name || 'Unknown Case';
            const topNodeId = `trace-${index}-top`;
            let extraYOffset = 0;

            const topNode = createNode(
                topNodeId,
                {
                    label: modelId,
                    startNode: true,
                    stepNode: false,
                    content: '',
                    additional_information: {},
                    conversationId: conversationId,
                    caseName,
                    icon: <FiInfo/>,
                    footer: evaluationNumber,
                    step: 0,
                    modelId,
                    nodeType: 'step',
                },
                {x: 0, y: currentY + extraYOffset}
            );

            newNodes.push(topNode);
            previousStepNodeId = topNodeId;
            currentY += 400 + extraYOffset;

        conversationMeasurement.traces.forEach((traceMeasurement, traceIndex: number) => {

            const printStyledLog = () => {
                console.log(
                    // nosemgrep: javascript.lang.security.audit.unsafe-formatstring.unsafe-formatstring
                    `%c Trace #: ${traceIndex} %c`,
                    'color: white; background-color: purple; font-size: 12px; font-weight: bold; padding: 2px 10px; border-radius: 5px;',
                    'color: black; background-color: #f0f0f0; font-size: 14px; padding: 2px 10px;'
                );
            };

            printStyledLog();
            console.log(traceMeasurement);
            const trace = traceMeasurement.trace;


            const stepNodeId = `trace-${traceIndex}-step-${stepCount}`;

            if (trace.to === "LLM" && trace.response.usage) {
                totalUsage.inputTokens += trace.response.usage.inputTokens || 0;
                totalUsage.outputTokens += trace.response.usage.outputTokens || 0;
                totalUsage.totalTokens += trace.response.usage.totalTokens || 0;
            }

            if (traceMeasurement.measurements) {
                traceMeasurement.measurements.forEach((measurement) => {
                    if (!totalMeasurements[measurement.name]) {
                        totalMeasurements[measurement.name] = 0;
                    }
                    totalMeasurements[measurement.name] += measurement.value;
                });
            }

            const stepNode = createNode(
                stepNodeId,
                {
                    label: `Trace ${stepCount}`,
                    stepNode: true,
                    startNode: false,
                    content: '',
                    additional_information: {},
                    caseName,
                    icon: <FiInfo/>,
                    footer: evaluationNumber,
                    step: stepCount,
                    modelId,
                    measurements: traceMeasurement.measurements,
                    usage: (trace as LlmTrace).response.usage,
                    nodeType: 'step',
                },
                {x: 0, y: currentY}
            );
            stepCount++;
            newNodes.push(stepNode);

            if (previousStepNodeId && trace.to === 'LLM') {
                newEdges.push(
                    createEdge(
                        `edge-step-${traceIndex}-${stepCount - 1}-${stepCount}`,
                        previousStepNodeId,
                        stepNodeId,
                        'LLM'
                    )
                );
            } else if (previousStepNodeId && trace.to === 'TOOL') {
                newEdges.push(
                    createEdge(
                        `edge-step-${traceIndex}-${stepCount - 1}-${stepCount}`,
                        previousStepNodeId,
                        stepNodeId,
                        'TOOL'
                    )
                );
            }

            previousStepNodeId = stepNodeId;

            if (trace.to === 'LLM') {
                // REQUEST
                if (trace.request.messages && trace.request.messages.length > 0) {
                    const message = trace.request.messages[trace.request.messages.length - 1];
                    const messageNodeId = `trace-${traceIndex}-msg-${stepCount}`;
                    let icon = message.role === 'user' ? <FiUser/> : <RiRobot2Line/>;

                    const messageNode = createNode(
                        messageNodeId,
                        {
                            label: `${message.role}[${trace.request.messages.length - 1}]`,
                            startNode: false,
                            stepNode: false,
                            content: message.content[0]?.text || 'No content',
                            additional_information: traceMeasurement.trace.additional_info,
                            caseName,
                            icon,
                            footer: evaluationNumber,
                            step: stepCount,
                            measurements: traceMeasurement.measurements,
                            modelId,
                            nodeType: 'request',
                        },
                        {
                            x: messageXOffset,
                            y: currentY - 160
                        }
                    );

                    newNodes.push(messageNode);
                    newEdges.push(
                        createEdge(
                            `edge-${traceIndex}-step-${stepCount}-msg`,
                            stepNodeId,
                            messageNodeId,
                            'REQUEST'
                        )
                    );
                }

                // RESPONSE
                if (trace.response.output && Array.isArray(trace.response.output.message.content)) {

                    trace.response.output.message.content.forEach((output: any, outputIndex: number) => {

                        // numberOfOutputs = 2;

                        const outputNodeId = `trace-${traceIndex}-output-${stepCount}-${outputIndex}`;
                        let icon = output.role === 'user' ? <FiUser/> : <RiRobot2Line/>;
                        let content = 'No content';

                        if (output["toolUse"] || output["tool_response"]) {
                            icon = <FiTool/>;
                            content = output.toolUse.name;
                        } else {
                            content = output.text;
                        }

                        const xOffsetIndex =
                            outputIndex % 2 === 0
                                ? messageXOffset
                                : messageXOffset + 630;

                        const yOffsetIndex =
                            outputIndex < 2
                                ? currentY + 140
                                : currentY + 140 + 200;

                        const outputNode = createNode(
                            outputNodeId,
                            {
                                label: `${trace.response.output.message.role} [${outputIndex}]`,
                                startNode: false,
                                stepNode: false,
                                content: content,
                                additional_information: trace.additional_info,
                                caseName,
                                icon,
                                footer: evaluationNumber,
                                step: stepCount,
                                measurements: traceMeasurement.measurements,
                                modelId,
                                nodeType: 'response',
                            },
                            {x: xOffsetIndex, y: yOffsetIndex}
                        );

                        newNodes.push(outputNode);

                        if (outputIndex >= 1) {
                            const prevOutputIndex = outputIndex - 1;
                            newEdges.push(
                                createEdge(
                                    `edge-${traceIndex}-output-${stepCount}-${prevOutputIndex}-to-${outputIndex}`,
                                    `trace-${traceIndex}-output-${stepCount}-${prevOutputIndex}`,
                                    outputNodeId,
                                    'RESPONSE'
                                )
                            );
                        } else {
                            newEdges.push(
                                createEdge(
                                    `edge-${traceIndex}-step-${stepCount}-output-${outputIndex}`,
                                    stepNodeId, // Source: Step node
                                    outputNodeId, // Target: Current output node
                                    'RESPONSE'
                                )
                            );
                        }


                    });
                }
            } else if (trace.to === 'TOOL') {
                // Request with Tool Use
                if (trace.request["tool_name"]) {
                    const messageNodeId = `trace-${traceIndex}-msg-${stepCount}`;
                    let icon = <FiTool/>;

                    const messageNode = createNode(
                        messageNodeId,
                        {
                            label: `assistant`,
                            startNode: false,
                            stepNode: false,
                            content: JSON.stringify(trace.request["tool_input"]) || 'No content',
                            additional_information: {},
                            caseName,
                            icon,
                            footer: evaluationNumber,
                            step: stepCount,
                            measurements: traceMeasurement.measurements,
                            modelId,
                            nodeType: 'request',
                        },
                        {
                            x: messageXOffset,
                            y: currentY - 160
                        }
                    );

                    newNodes.push(messageNode);
                    newEdges.push(
                        createEdge(
                            `edge-${traceIndex}-step-${stepCount}-msg`,
                            stepNodeId,
                            messageNodeId,
                            'REQUEST'
                        )
                    );
                }

                const toolNodeId = `trace-${traceIndex}-tool-${stepCount}`;

                    const toolNode = createNode(
                        toolNodeId,
                        {
                            label: 'tool response',
                            startNode: false,
                            stepNode: false,
                            content: JSON.stringify(trace.response.tool_response),
                            additional_information: {},
                            caseName,
                            icon: <TbProgressBolt/>,
                            footer: evaluationNumber,
                            toolInput: JSON.stringify(trace.request, null, 2),
                            toolOutput: JSON.stringify(trace.response, null, 2),
                            step: stepCount,
                            measurements: traceMeasurement.measurements,
                            modelId,
                            nodeType: 'response',
                        },
                        {
                            x: messageXOffset,
                            y: currentY + 140
                        }
                    );

                    newNodes.push(toolNode);
                    newEdges.push(
                        createEdge(
                            `edge-${traceIndex}-step-${stepCount}-tool`,
                            stepNodeId,
                            toolNodeId,
                            'REQUEST'
                        )
                    );

            }

            currentY += yOffset;

            // if (traceIndex >= conversationMeasurement.traces.length - 1) {
            //     const summaryNodeId = `summary-${traceIndex}-node`;
            //
            //     const summaryNode = createNode(
            //         summaryNodeId,
            //         {
            //             label: "Summary",
            //             startNode: false,
            //             stepNode: false,
            //             content: "No content",
            //             additional_information: {},
            //             icon: <FiInfo/>,
            //             nodeType: 'summary',
            //             footer: evaluationNumber,
            //             step: -1,
            //             measurements: conversationMeasurement.measurements,
            //         },
            //         {x: 0, y: currentY}
            //     );
            //
            //
            //     newNodes.push(summaryNode);
            //     newEdges.push(
            //         createEdge(
            //             `edge-${traceIndex}-step-${stepCount}-summary`,
            //             stepNodeId,
            //             summaryNodeId,
            //             'LLM'
            //         )
            //     );
            // }

        });
        setNodes(() => newNodes);
        setEdges(() => newEdges);
    }, [setNodes, setEdges, index, testData]);

    return (
        <div className="w-screen h-screen">
            {isLoading ? (
                <div className="flex justify-center items-center h-full">
                    <div className="bg-white p-4 rounded shadow-lg text-center">
                        <p className="text-lg font-semibold">Loading...</p>
                    </div>
                </div>
            ) : (
                <ReactFlow
                    nodes={nodes}
                    edges={edges}
                    onNodesChange={onNodesChange}
                    onEdgesChange={onEdgesChange}
                    nodesDraggable={false}
                    nodeTypes={{custom: NodeComponent}}
                    className="bg-gray-100"
                    fitView
                    fitViewOptions={{padding: 0.2}}
                    minZoom={0.1}
                    maxZoom={1.5}
                >
                    <Background/>
                    <Controls/>
                </ReactFlow>
            )}
        </div>
    );
};

export default TestPage;
