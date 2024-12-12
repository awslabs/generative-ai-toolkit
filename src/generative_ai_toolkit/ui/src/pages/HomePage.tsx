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

import React, {useState} from 'react';
import { ConversationMeasurements, LlmTrace } from "../types/types";
import useSWR from 'swr'

import '@xyflow/react/dist/style.css';
import '../index.css';

const HomePage: React.FC = () => {
    const { data: testData = [], isLoading } = useSWR<ConversationMeasurements[]>('http://127.0.0.1:8000/get_conversation_traces', (url: string) => fetch(url).then(res => res.json()))
    const [selectedValidation, setSelectedValidation] = useState<string>('All');
    const [selectedCaseName, setSelectedCaseName] = useState<string>('All');
    const [selectedModel, setSelectedModel] = useState<string>('All');

    const validationOptions = ['All', 'Pass', 'Fail'];
    const caseNameOptions = ['All', ...Array.from(new Set(testData.map((conversation) => conversation.case.name)))];
    const modelOptions = ['All', ...Array.from(new Set(testData.map((conversation) => (conversation.traces[0].trace as LlmTrace).request.modelId)))];

    const checkMeasurements = (conversation: ConversationMeasurements) => {
        for (let trace of conversation.traces) {
            for (let measurement of trace.measurements) {
                if (measurement["validation_passed"] === false) {
                    return false;
                }
            }
        }
        return true;
    };

    // Filtered data based on dropdown selection
    const filteredData = testData.filter((conversation) => {
        const testsPassed = checkMeasurements(conversation);
        const matchesValidation = selectedValidation === 'All' || (selectedValidation === 'Pass' ? testsPassed : !testsPassed);
        const matchesCase = selectedCaseName === 'All' || conversation.case.name === selectedCaseName;
        const matchesModel = selectedModel === 'All' || (conversation.traces[0].trace as LlmTrace).request.modelId === selectedModel;
        return matchesValidation && matchesCase && matchesModel;
    });

    // Recalculate scorecard data based on filteredData
    const totalTests = filteredData.length;
    const passedTests = filteredData.filter(checkMeasurements).length;
    const passPercentage = totalTests > 0 ? Math.round((passedTests / totalTests) * 100) : 0;

    // Recalculate token data based on filteredData
    let totalInputTokens = 0;
    let totalOutputTokens = 0;
    let totalTotalTokens = 0;
    let tokenCount = 0;

    filteredData.forEach(conversation => {
        const lastTrace = conversation.traces.at(-1)?.trace;
        if (lastTrace && lastTrace.to === "LLM" && lastTrace.response.usage) {
            totalInputTokens += lastTrace.response.usage.inputTokens || 0;
            totalOutputTokens += lastTrace.response.usage.outputTokens || 0;
            totalTotalTokens += lastTrace.response.usage.totalTokens || 0;
            tokenCount++;
        }
    });

    const averageInputTokens = tokenCount > 0 ? Math.round(totalInputTokens / tokenCount) : 0;
    const averageOutputTokens = tokenCount > 0 ? Math.round(totalOutputTokens / tokenCount) : 0;
    const averageTotalTokens = tokenCount > 0 ? Math.round(totalTotalTokens / tokenCount) : 0;

    return (
        <div className="min-h-screen bg-gray-100 flex flex-col items-center py-10">

            {/* First row of scorecards */}
            <div className="flex justify-center items-center mb-6 w-full max-w-5xl space-x-6">
                <div className="bg-white shadow-sm rounded-sm p-6 text-center w-1/3">
                    <h3 className="text-xl font-semibold text-gray-700">Total Tests</h3>
                    <p className="text-2xl font-bold text-gray-900">{totalTests}</p>
                </div>
                <div className="bg-white shadow-sm rounded-sm p-6 text-center w-1/3">
                    <h3 className="text-xl font-semibold text-gray-700">Passed Tests</h3>
                    <p className="text-2xl font-bold text-green-600">{passedTests}</p>
                </div>
                <div className="bg-white shadow-sm rounded-sm p-6 text-center w-1/3">
                    <h3 className="text-xl font-semibold text-gray-700">Pass Percentage</h3>
                    <p className="text-2xl font-bold text-blue-600">{passPercentage}%</p>
                </div>
            </div>

            {/* Second row of scorecards for token usage */}
            <div className="flex justify-center items-center mb-6 w-full max-w-5xl space-x-6">
                <div className="bg-white shadow-sm rounded-sm p-6 text-center w-1/3">
                    <h3 className="text-xl font-semibold text-gray-700">Avg Input Tokens</h3>
                    <p className="text-2xl font-bold text-gray-900">{averageInputTokens}</p>
                </div>
                <div className="bg-white shadow-sm rounded-sm p-6 text-center w-1/3">
                    <h3 className="text-xl font-semibold text-gray-700">Avg Output Tokens</h3>
                    <p className="text-2xl font-bold text-gray-900">{averageOutputTokens}</p>
                </div>
                <div className="bg-white shadow-sm rounded-sm p-6 text-center w-1/3">
                    <h3 className="text-xl font-semibold text-gray-700">Avg Total Tokens</h3>
                    <p className="text-2xl font-bold text-gray-900">{averageTotalTokens}</p>
                </div>
            </div>

            {/* Filters */}
            <div className="flex flex-col items-center justify-start mb-6 w-full max-w-5xl">
                <div className="flex justify-end items-center mb-4 w-full space-x-4">
                    <div>
                        <label htmlFor="model-filter" className="block text-sm font-medium text-gray-700">Model</label>
                        <select
                            id="model-filter"
                            value={selectedModel}
                            onChange={(e) => setSelectedModel(e.target.value)}
                            className="mt-1 block w-full pl-3 pr-5 py-2 text-base border-gray-300 focus:outline-none focus:ring-indigo-500 focus:border-indigo-500 sm:text-sm rounded-s"
                        >
                            {modelOptions.map((option) => (
                                <option key={option} value={option}>
                                    {option}
                                </option>
                            ))}
                        </select>
                    </div>
                    <div>
                        <label htmlFor="case-filter" className="block text-sm font-medium text-gray-700">Case Name</label>
                        <select
                            id="case-filter"
                            value={selectedCaseName}
                            onChange={(e) => setSelectedCaseName(e.target.value)}
                            className="mt-1 block w-full pl-3 pr-5 py-2 text-base border-gray-300 focus:outline-none focus:ring-indigo-500 focus:border-indigo-500 sm:text-sm rounded-s"
                        >
                            {caseNameOptions.map((option) => (
                                <option key={option} value={option}>
                                    {option}
                                </option>
                            ))}
                        </select>
                    </div>
                    <div>
                        <label htmlFor="validation-filter" className="block text-sm font-medium text-gray-700">Validation</label>
                        <select
                            id="validation-filter"
                            value={selectedValidation}
                            onChange={(e) => setSelectedValidation(e.target.value)}
                            className="mt-1 block w-full pl-3 pr-5 py-2 text-base border-gray-300 focus:outline-none focus:ring-indigo-500 focus:border-indigo-500 sm:text-sm rounded-s"
                        >
                            {validationOptions.map((option) => (
                                <option key={option} value={option}>
                                    {option}
                                </option>
                            ))}
                        </select>
                    </div>
                </div>
            </div>

            {isLoading ? (
                <div className="flex justify-center items-center h-full">
                    <div className="bg-white p-4 rounded shadow-lg text-center">
                        <p className="text-lg font-semibold">Loading...</p>
                    </div>
                </div>
            ) : (
                <div className="relative overflow-x-auto shadow-md sm:rounded-lg w-full max-w-5xl bg-white mt-4">
                    <table className="w-full text-sm text-left text-gray-500">
                        <thead className="text-xs text-gray-700 uppercase bg-gray-50">
                        <tr>
                            <th scope="col" className="px-6 py-3">Test</th>
                            <th scope="col" className="px-6 py-3">Conversation ID</th>
                            <th scope="col" className="px-6 py-3">Model</th>
                            <th scope="col" className="px-6 py-3">Case Name</th>
                            <th scope="col" className="px-6 py-3">Validation</th>
                        </tr>
                        </thead>
                        <tbody>
                        {filteredData.map((conversation) => {
                            const rowClass = conversation.traces.length % 2 === 0 ? 'bg-white' : 'bg-gray-50';
                            const testsPassed = checkMeasurements(conversation);

                            const originalIndex = testData.indexOf(conversation); // Get the original index from unfiltered data

                            return (
                                <tr className={`${rowClass}`} key={originalIndex}>
                                    <th scope="row" className="px-6 py-4 font-medium text-gray-900 whitespace-nowrap">
                                        <a href={`/ui/test?index=${originalIndex}`} className="text-blue-600 hover:underline">
                                            Test {originalIndex}
                                        </a>
                                    </th>
                                    <td className="px-6 py-4">{conversation.conversation_id}</td>
                                    <td className="px-6 py-4">{(conversation.traces[0].trace as LlmTrace).request.modelId}</td>
                                    <td className="px-6 py-4">{conversation.case.name}</td>
                                    <td className="px-6 py-4">
                                        <span className={`px-2 inline-flex text-xs leading-5 font-semibold rounded-full ${testsPassed ? 'bg-green-100 text-green-800' : 'bg-red-100 text-red-800'}`}>
                                            {testsPassed ? "Pass" : "Fail"}
                                        </span>
                                    </td>
                                </tr>
                            );
                        })}
                        </tbody>
                    </table>
                </div>
            )}
        </div>
    );
};

export default HomePage;
