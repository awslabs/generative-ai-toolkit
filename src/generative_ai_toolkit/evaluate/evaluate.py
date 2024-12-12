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

import datetime
import collections.abc
from dataclasses import asdict, dataclass, field
import itertools
import traceback
from concurrent.futures import Executor, Future, ThreadPoolExecutor, as_completed
from typing import (
    Any,
    Callable,
    Iterable,
    Mapping,
    Sequence,
    TypeVar,
)
from enum import Enum

from generative_ai_toolkit.metrics import BaseMetric
from generative_ai_toolkit.test import (
    Case,
    CaseTrace,
    AgentLike,
    CaseTraceInfo,
    PassFail,
)
from generative_ai_toolkit.tracer import InMemoryAgentTracer, Trace
from generative_ai_toolkit.utils.logging import logger
from generative_ai_toolkit.metrics import Measurement
from generative_ai_toolkit.utils.ulid import Ulid


@dataclass
class ConversationMeasurements:
    conversation_id: str
    case: Case | None
    traces: list["TraceMeasurements"] = field(default_factory=list)
    measurements: list["Measurement"] = field(default_factory=list)

    @classmethod
    def json_encoder(cls, o):
        if isinstance(o, cls):
            return asdict(o)
        elif isinstance(o, Case):
            return {"name": o.name}
        elif isinstance(o, Ulid):
            return str(o)
        elif isinstance(o, datetime.datetime):
            return o.isoformat()
        elif isinstance(o, Enum):
            return o.value


@dataclass
class TraceMeasurements:
    trace: Trace
    measurements: list["Measurement"] = field(default_factory=list)


class GenerativeAIToolkit:
    """
    GenerativeAIToolkit class for generating and evaluating model prompts.

    This class provides static methods to generate text based on a given prompt
    and evaluate model performance on given datasets and metrics.
    """

    # Default for generate_traces:
    max_case_workers: int | None = None

    # Default for eval:
    max_conversation_workers: int | None = 4
    max_metric_workers: int | None = None

    @staticmethod
    def eval(
        *,
        traces: Iterable[Sequence[Trace]],
        metrics: Sequence[BaseMetric],
        max_conversation_workers: int | None = None,
        max_metric_workers: int | None = None,
        timeout: float | None = None,
    ) -> Iterable[ConversationMeasurements]:
        """
        Evaluate the model performance on given dataset and metrics.

        :param traces: Sequence of conversations, where each conversation is an ordered sequence of traces
        :param metrics: List of metric instances.
        :return: An instance of Results class containing evaluation results.
        """

        def do_evaluate_trace(
            metric: BaseMetric,
            trace: Trace,
            case: Case | None,
            conversation_id: str,
            metric_executor: Executor,
        ):
            try:
                return metric.evaluate_trace(trace=trace, executor=metric_executor)
            except Exception as err:
                logger.error(
                    "Failed to evaluate trace",
                    conversation_id=conversation_id,
                    trace_id=trace.trace_id,
                    metric_class_name=metric.__class__.__name__,
                    error=str(err),
                    case=case,
                    traceback="\n".join(traceback.format_exception(err)),
                )

        def do_evaluate_conversation(
            metric: BaseMetric,
            conversation_traces: Sequence[Trace],
            case: Case | None,
            conversation_id: str,
            metric_executor: Executor,
        ):
            try:
                return metric.evaluate_conversation(
                    conversation_traces=conversation_traces, executor=metric_executor
                )
            except Exception as err:
                logger.error(
                    "Failed to evaluate conversation",
                    conversation_id=conversation_id,
                    trace_ids=[trace.trace_id for trace in conversation_traces],
                    metric_class_name=metric.__class__.__name__,
                    error=str(err),
                    case=case,
                    traceback="\n".join(traceback.format_exception(err)),
                )

        def eval_conversation(
            conversation_traces: Sequence[Trace], metric_executor: Executor
        ) -> ConversationMeasurements:
            if not conversation_traces:
                raise ValueError("Empty conversation")

            conversation_id, case = get_conversation_metadata(conversation_traces)

            conversation_measurements = ConversationMeasurements(
                conversation_id=conversation_id,
                case=case,
            )

            # Start running metrics for conversations (sequence of traces)
            metric_futures = {
                metric_executor.submit(
                    do_evaluate_conversation,
                    metric=metric,
                    conversation_traces=conversation_traces,
                    case=case,
                    conversation_id=conversation_id,
                    metric_executor=metric_executor,
                ): conversation_measurements.measurements
                for metric in filter(
                    lambda m: type(m).evaluate_conversation
                    is not BaseMetric.evaluate_conversation,
                    metrics,
                )
            }

            # Execute case validation
            if case and case.validate:
                validation_fut = metric_executor.submit(
                    PassFail.validate_conversation,
                    traces=conversation_traces,
                    case_=case,
                )
                metric_futures[validation_fut] = conversation_measurements.measurements

            # Start running metrics for individual traces
            for trace in conversation_traces:
                trace_measurements = TraceMeasurements(trace=trace)
                conversation_measurements.traces.append(trace_measurements)
                for metric in filter(
                    lambda m: type(m).evaluate_trace is not BaseMetric.evaluate_trace,
                    metrics,
                ):
                    metric_futures[
                        metric_executor.submit(
                            do_evaluate_trace,
                            metric=metric,
                            trace=trace,
                            case=case,
                            conversation_id=conversation_id,
                            metric_executor=metric_executor,
                        )
                    ] = trace_measurements.measurements

            for future in as_completed(metric_futures, timeout=timeout):
                metric_results = future.result()
                if metric_results:
                    if not isinstance(metric_results, collections.abc.Sequence):
                        metric_results = [metric_results]
                    for measurement in metric_results:
                        metric_futures[future].append(measurement)

            return conversation_measurements

        with (
            ThreadPoolExecutor(
                max_workers=max_conversation_workers
                or GenerativeAIToolkit.max_conversation_workers,
                thread_name_prefix="conversation",
            ) as conversation_executor,
            ThreadPoolExecutor(
                max_workers=max_metric_workers
                or GenerativeAIToolkit.max_metric_workers,
                thread_name_prefix="metric",
            ) as metric_executor,
        ):
            futures = [
                conversation_executor.submit(
                    eval_conversation, conversation_traces, metric_executor
                )
                for conversation_traces in traces
            ]
            for future in as_completed(futures, timeout=timeout):
                conversation_measurements = future.result()
                yield conversation_measurements

    @staticmethod
    def generate_traces(
        *,
        cases: Sequence[Case],
        agent_factory: Callable[..., AgentLike],
        nr_runs_per_case=1,
        agent_parameters: Mapping[str, Any] | None = None,
        max_case_workers: int | None = None,
    ) -> Iterable[Sequence[Trace]]:
        """
        Generate traces for a given set of cases and agent_parameters.

        The provided agent_parameters should be a mapping of parameter names to values,
        that can be passed to the supplied agent_factory as keyword arguments.

        If you want to evaluate multiple values for the same parameter to see which works best, e.g. evaluate multiple model ids,
        then provide a list of values for that parameter wrapped in an Evaluate instance. For each unique combination of parameters,
        an agent is created to run the case through.

        :param cases: the cases
        :param agent_factory: Callable that should return a new Agent instance.
        :param agent_parameters: Mapping of parameter names to possible values, to be passed as keyword arguments to the supplied agent_factory.
        :param executor: Executor instance for running the cases in parallel. If not provided, a default ThreadPoolExecutor will be used.
        :return: Sequence of sequences of Trace instances.
        """

        agent_parameters = agent_parameters or {}
        futures: dict[Future[Sequence[CaseTrace]], CaseTraceInfo] = {}
        values = [
            val.values if isinstance(val, Permute) else [val]
            for val in agent_parameters.values()
        ]
        default_agent_parameters = dict(tracer=InMemoryAgentTracer())
        with ThreadPoolExecutor(
            max_workers=max_case_workers or GenerativeAIToolkit.max_case_workers,
            thread_name_prefix="case",
        ) as executor:
            nr_conversations = 0
            parameter_permutations = [
                dict(zip(agent_parameters.keys(), combination))
                for combination in itertools.product(*values)
            ]
            for case_nr, case in enumerate(cases):
                for parameter_permutation in parameter_permutations:
                    current_permutation = {
                        k: parameter_permutation[k]
                        for k, v in agent_parameters.items()
                        if isinstance(v, Permute)
                    }
                    for run_nr in range(nr_runs_per_case):
                        agent = agent_factory(
                            **(default_agent_parameters | parameter_permutation)
                        )
                        info = CaseTraceInfo(
                            case_nr=case_nr,
                            run_nr=run_nr,
                            permutation=current_permutation,
                        )
                        futures[
                            executor.submit(
                                case.run,
                                agent=agent,
                                case_trace_info=info,
                            )
                        ] = info
                        nr_conversations += 1
            print(
                f"Submitted trace generation for {nr_conversations} conversations ({len(parameter_permutations)} permutations, {len(cases)} cases, {nr_runs_per_case} runs per case)"
            )
            for future in as_completed(futures):
                info = futures[future]
                print(
                    f"Done generating traces for case {info.case_nr} run {info.run_nr}"
                )
                yield future.result()


T = TypeVar("T")


class Permute[T]:
    def __init__(self, values: Sequence[T]):
        self.values = values


def get_conversation_metadata(traces: Sequence["Trace"]):
    conversation_id: str | None = None
    case_: Case | None = None
    for trace in traces:
        if not conversation_id:
            conversation_id = trace.conversation_id
        elif conversation_id != trace.conversation_id:
            raise ValueError(
                f"Found multiple Conversation IDs in traces: {conversation_id} and {trace.conversation_id}"
            )
        if isinstance(trace, CaseTrace):
            if not case_:
                case_ = trace.case
            elif case_ != trace.case:
                raise ValueError(
                    f"Found multiple Cases in traces: {case_} and {trace.case}"
                )
    if not conversation_id:
        raise ValueError("Could not find Conversation ID in traces")
    return conversation_id, case_
