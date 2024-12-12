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

import React, {memo, useState, useRef, useEffect} from 'react';
import {Handle, Position, NodeProps} from '@xyflow/react';
import {FiChevronDown, FiChevronUp, FiMoreHorizontal} from 'react-icons/fi';
import {CustomNodeData} from '../types/types';

const NodeComponent: React.FC<NodeProps & { data: CustomNodeData }> = ({data, isConnectable}) => {
    const [contentExpanded, setContentExpanded] = useState(false);
    const [measurementsExpanded, setMeasurementsExpanded] = useState(false);
    const [usageExpanded, setUsageExpanded] = useState(false);
    const [showMore, setShowMore] = useState(false);

    const contentRef = useRef<HTMLDivElement>(null);

    useEffect(() => {
        if (contentRef.current) {
            setShowMore(contentRef.current.scrollHeight > contentRef.current.clientHeight);
        }
    }, [data.content, contentExpanded]);

    const toggleContentExpand = () => setContentExpanded(!contentExpanded);
    const toggleMeasurementsExpand = () => setMeasurementsExpanded(!measurementsExpanded);
    const toggleUsageExpand = () => setUsageExpanded(!usageExpanded);

    const renderContent = (text: string, additional_information: string, expanded: boolean) => {
        return (
            <>
                {!!text && (
                    <div
                        ref={contentRef}
                        className={`mt-2 p-2 bg-gray-100 rounded text-sm ${expanded ? 'h-auto' : 'h-20'} overflow-hidden`}
                    > {text}
                    </div>)}
                {!!additional_information && (
                    <div
                        ref={contentRef}
                        className={`mt-2 p-2 bg-gray-100 rounded text-sm ${expanded ? 'h-auto' : 'h-20'} overflow-hidden`}
                    > {additional_information}
                    </div>
                )}
            </>
        );
    };

    const renderMeasurements = () => {
        if (!data.measurements) return null;
        return (
            <div className="mt-2 text-sm">
                {data.measurements.map((m, index) => (
                    <div key={index}>
                        <div className="mb-1">
                            <span className="font-semibold">{m.name}:</span> {m.value.toFixed(6)}
                        </div>
                        {!!m.additional_info && (
                            <div
                                key={index}
                                ref={contentRef}
                                className={`mt-2 p-2 bg-gray-100 rounded text-sm h-auto overflow-hidden`}
                            > {typeof m.additional_info === "string" ? m.additional_info : JSON.stringify(m.additional_info, null, 2)}
                            </div>
                        )}
                    </div>
                ))}
            </div>
        );
    };

    const renderUsage = () => {
        if (!data.usage) return null;
        return (
            <div className="mt-2 text-sm">
                <div className="mb-1">
                    <span className="font-semibold">Input Tokens:</span> {data.usage.inputTokens}
                </div>
                <div className="mb-1">
                    <span className="font-semibold">Output Tokens:</span> {data.usage.outputTokens}
                </div>
                <div className="mb-1">
                    <span className="font-semibold">Total Tokens:</span> {data.usage.totalTokens}
                </div>
            </div>
        );
    };

    const renderHandles = () => {
        if (data.nodeType === 'step') {
            return (
                <>
                    <Handle type="target" position={Position.Top} id="top" isConnectable={isConnectable}/>
                    <Handle type="source" position={Position.Bottom} id="bottom" isConnectable={isConnectable}/>
                    <Handle type="source" position={Position.Right} id="right" isConnectable={isConnectable}/>
                </>
            );
        } else if (data.nodeType === 'response') {
            return (
                <>
                    <Handle type="target" position={Position.Left} id="left" isConnectable={isConnectable}/>
                    <Handle type="source" position={Position.Right} id="right" isConnectable={isConnectable}/>
                </>
            );
        } else if (data.nodeType === 'summary') {
            return (
                <>
                    <Handle type="target" position={Position.Top} id="top" isConnectable={isConnectable}/>
                    <Handle type="source" position={Position.Bottom} id="bottom" isConnectable={isConnectable}/>
                    <Handle type="source" position={Position.Right} id="right" isConnectable={isConnectable}/>
                </>
            );
        } else {
            return (
                <Handle type="target" position={Position.Left} id="left" isConnectable={isConnectable}/>
            );
        }
    };

    // Helper to render validation badges if validation_passed is not null
    const renderValidationBadges = () => {
        if (!data.measurements) return null;

        return (
            <div className="flex gap-2 flex-wrap">
                {data.measurements.map((measurement, index) => {
                    return (
                        <span
                            key={index}
                            className={`inline-block px-2 py-1 text-xs font-semibold rounded-full ${
                                measurement.validation_passed === true ? 'bg-green-200 text-green-700' : measurement.validation_passed === false ? 'bg-red-200 text-red-700' : 'bg-gray-200 text-gray-700'
                            }`}
                        >
                            {measurement.name}
                        </span>
                    );
                })}
            </div>
        );
    };

    return (
        <div className="bg-white w-[32rem] min-h-[16rem] flex flex-col overflow-hidden">
            {renderHandles()}
            <div className="p-4 border-b border-gray-200 flex items-center">
                {data.icon && (
                    <div className="w-8 h-8 rounded-full bg-gray-200 flex items-center justify-center mr-3">
                        {data.icon}
                    </div>
                )}
                <h3 className="text-lg font-semibold truncate">{data.label}</h3>
            </div>

            <div className="p-4 flex-grow">

                {(!data.stepNode && !data.startNode) && (
                    <div>
                        <button onClick={toggleContentExpand}
                                className="text-gray-600 flex items-center justify-between w-full">
                            {data.nodeType === 'request' && ("Request")}
                            {data.nodeType === 'response' && ("Response")}
                            {data.nodeType === 'summary' && ("Summary")}
                            {contentExpanded ? <FiChevronUp/> : <FiChevronDown/>}
                        </button>
                        {contentExpanded && renderContent(data.content, Object.keys(data.additional_information).length ? JSON.stringify(data.additional_information) : "", contentExpanded)}
                    </div>
                )}
                {showMore && (
                    <button
                        onClick={() => setContentExpanded(!contentExpanded)}
                        className="mt-2 text-blue-500 flex items-center"
                    >
                        <FiMoreHorizontal className="mr-1"/>
                        {contentExpanded ? 'Show less' : 'Show more'}
                    </button>
                )}

                {(data.startNode) && (
                    <div className="mb-4">
                        <div className="mb-4"><strong>Conversation ID: </strong>{data.conversationId}</div>
                        <div className="mb-4"><strong>Case Name: </strong>{data.caseName}</div>
                        <div><strong>Model ID: </strong>{data.modelId}</div>
                    </div>
                )}

                {(data.measurements) && (
                    <div className="mb-4">
                        <button onClick={toggleMeasurementsExpand}
                                className="text-gray-600 flex items-center justify-between w-full">
                            Measurements {measurementsExpanded ? <FiChevronUp/> : <FiChevronDown/>}
                        </button>
                        {measurementsExpanded && renderMeasurements()}
                    </div>
                )}


                {(data?.stepNode && data?.usage?.inputTokens && data?.usage?.outputTokens && data?.usage?.totalTokens) && (
                    <div className="mb-4">
                        <button onClick={toggleUsageExpand}
                                className="text-gray-600 flex items-center justify-between w-full">
                            Usage {usageExpanded ? <FiChevronUp/> : <FiChevronDown/>}
                        </button>
                        {usageExpanded && renderUsage()}
                    </div>
                )}

            </div>

            {data.stepNode && (
                (data.footer || data.step !== undefined || data.modelId) && (
                    <>
                        <div
                            className="p-4 border-t border-gray-200 text-sm text-gray-500 fle justify-between">
                            {data.modelId && (
                                <div className="text-right justify-end" title={data.modelId}>
                                    <div><strong>Model ID: </strong>{data.modelId}</div>
                                </div>
                            )}
                        </div>
                        <div
                            className="p-4 border-t border-gray-200 text-sm text-gray-500 flex items-center justify-between">
                            <div className="flex items-center">
                                {renderValidationBadges()}
                            </div>

                        </div>
                    </>
                )
            )}

            {data.nodeType === "summary" &&
                <div
                    className="p-4 border-t border-gray-200 text-sm text-gray-500 flex items-center justify-between">
                    <div className="flex items-center">
                        {renderValidationBadges()}
                    </div>
                </div>
            }

            <div
                className="p-4 border-t border-gray-200 text-sm text-gray-500 flex items-center justify-between">
                <div className="text-right" title={data.footer}>
                    <div
                        className="inline-block px-2 py-1 bg-purple-200 text-purple-700 text-xs font-semibold rounded-full">
                        {data.footer}
                    </div>
                </div>
            </div>
        </div>
    );
};

export default memo(NodeComponent);
