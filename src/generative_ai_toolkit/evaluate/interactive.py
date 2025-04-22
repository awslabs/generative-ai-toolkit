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

from collections import defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from statistics import fmean
from typing import Any

import pandas as pd
from IPython.display import display
from tabulate import tabulate

from generative_ai_toolkit.evaluate.evaluate import (
    ConversationMeasurements,
    Permute,  # noqa: F401 Leave this here, so it can be imported from this module as well
)
from generative_ai_toolkit.evaluate.evaluate import (
    GenerativeAIToolkit as GenAIToolkit_,
)
from generative_ai_toolkit.metrics import BaseMetric
from generative_ai_toolkit.metrics.measurement import Measurement
from generative_ai_toolkit.test import AgentLike, Case
from generative_ai_toolkit.tracer.tracer import Trace
from generative_ai_toolkit.ui import measurements_ui
from generative_ai_toolkit.utils.interactive import is_notebook


class EnhancedEvalResult:
    def __init__(
        self,
        conversation_measurements: Iterable[ConversationMeasurements],
        traces: Iterable[Sequence[Trace]],
    ):
        self.conversation_measurements: (
            Iterable[ConversationMeasurements] | list[ConversationMeasurements]
        ) = conversation_measurements
        self.traces = traces
        self._ui = None

    @property
    def ui(self):
        if not self._ui:
            self._ui = measurements_ui(self.conversation_measurements)
        return self._ui

    def summary(self):
        return self.summary_for(self)

    def details(self):
        """
        Represent all measurements in a flattened pandas DataFrame, with these columns:

        - measurement_name
        - measurement_value
        - measurement_unit
        - measurement_validation_passed
        - conversation_id
        - trace_id (for measurements at Trace level)
        - span_id (for measurements at Trace level)
        - span_name (for measurements at Trace level)
        - ai_trace_type (for measurements at Trace level)
        - peer_service (for measurements at Trace level)
        - case_name (for measurements created as part of a Case)
        - case_nr (for measurements created as part of a Case)
        - permutation_nr (for measurements created as part of a Case)
        - run_nr (for measurements created as part of a Case)

        Additionally, for each key-value of the permutation, a column is added with key as column name, and the value as value.
        """
        return pd.concat(
            (m.as_dataframe() for m in self),
            ignore_index=True,
        )

    @staticmethod
    def summary_for(conversations: Iterable[ConversationMeasurements]):
        @dataclass
        class AggregatedCounts:
            measurements: list[Measurement] = field(default_factory=list)
            trace_count: int = 0
            run_nrs: set[int] = field(default_factory=set)
            nr_passed = 0
            nr_failed = 0

        aggregated_counts: dict[tuple, AggregatedCounts] = defaultdict(
            lambda: AggregatedCounts()
        )

        for measurements_for_conversation in conversations:
            first_trace = measurements_for_conversation.traces[0].trace
            permutation = getattr(first_trace, "permutation", {})
            permutation_as_key = tuple(sorted(permutation.items()))
            aggregated_counts[permutation_as_key].run_nrs.add(
                getattr(first_trace, "run_nr", 1)
            )

            for measurement in measurements_for_conversation.measurements:
                aggregated_counts[permutation_as_key].measurements.append(measurement)

                if measurement.validation_passed is True:
                    aggregated_counts[permutation_as_key].nr_passed += 1
                elif measurement.validation_passed is False:
                    aggregated_counts[permutation_as_key].nr_failed += 1

            for measurements_for_trace in measurements_for_conversation.traces:
                aggregated_counts[permutation_as_key].trace_count += 1
                for measurement in measurements_for_trace.measurements:
                    aggregated_counts[permutation_as_key].measurements.append(
                        measurement
                    )
                    if measurement.validation_passed is True:
                        aggregated_counts[permutation_as_key].nr_passed += 1
                    elif measurement.validation_passed is False:
                        aggregated_counts[permutation_as_key].nr_failed += 1

        summary_data = []
        for permutation_as_key, counts_per_permutation in aggregated_counts.items():
            measurement_averages: dict[str, list] = defaultdict(list)
            for measurement in counts_per_permutation.measurements:
                if measurement.dimensions:
                    for dimensions in measurement.dimensions:
                        vals_concat = "_".join(sorted(dimensions.values()))
                        measurement_averages[
                            f"{measurement.name} {vals_concat}"
                        ].append(measurement.value)
                else:
                    measurement_averages[measurement.name].append(measurement.value)
            row = {
                **dict(permutation_as_key),
                **(
                    {
                        f"Avg {measurement_name}": fmean(measurement_values)
                        for measurement_name, measurement_values in measurement_averages.items()
                    }
                ),
                "Avg Trace count per run": counts_per_permutation.trace_count
                / len(counts_per_permutation.run_nrs),
                "Total Nr Passed": counts_per_permutation.nr_passed,
                "Total Nr Failed": counts_per_permutation.nr_failed,
            }
            summary_data.append(row)

        summary_df = pd.DataFrame(summary_data).sort_index(
            axis=1,
            key=lambda index: index.map(
                lambda column_name: (
                    (
                        1
                        if column_name.startswith("Avg")
                        else 2 if column_name.startswith("Total Nr") else 0
                    ),
                    column_name,
                )
            ),
        )
        if is_notebook():
            display(summary_df)
        else:
            print(tabulate(summary_data, headers="keys", tablefmt="pretty"))

        return summary_df

    def __iter__(self):
        if isinstance(self.conversation_measurements, list):
            yield from self.conversation_measurements
            return
        collected = []
        for m in self.conversation_measurements:
            collected.append(m)
            yield m
        self.conversation_measurements = collected


class CachedGenerateTraces:
    def __init__(
        self,
        traces: Iterable[Sequence[Trace]],
    ):
        self.traces: Iterable[Sequence[Trace]] | list[Sequence[Trace]] = traces

    def __iter__(self):
        if isinstance(self.traces, list):
            yield from self.traces
            return
        collected = []
        for trace in self.traces:
            collected.append(trace)
            yield trace
        self.traces = collected


class GenerativeAIToolkit(GenAIToolkit_):
    """
    GenerativeAIToolkit class for generating and evaluating model prompts.

    This class provides static methods to generate text based on a given prompt
    and evaluate model performance on given datasets and metrics.
    """

    @staticmethod
    def eval(
        *,
        traces: Iterable[Sequence[Trace]],
        metrics: Sequence[BaseMetric],
        max_conversation_workers: int | None = None,
        max_metric_workers: int | None = None,
        timeout: float | None = None,
    ):
        """
        Evaluate the model performance on given dataset and metrics.

        :param traces: Sequence of conversations, where each conversation is an ordered sequence of traces
        :param metrics: List of metric instances.
        :return: An instance of Results class containing evaluation results.
        """

        enhanced = EnhancedEvalResult(
            GenAIToolkit_.eval(
                traces=traces,
                metrics=metrics,
                max_conversation_workers=max_conversation_workers,
                max_metric_workers=max_metric_workers,
                timeout=timeout,
            ),
            traces=traces,
        )

        return enhanced

    @staticmethod
    def generate_traces(
        *,
        cases: Sequence[Case],
        agent_factory: Callable[..., AgentLike],
        nr_runs_per_case=1,
        agent_parameters: Mapping[str, Any] | None = None,
        max_case_workers=None,
    ):
        enhanced = CachedGenerateTraces(
            GenAIToolkit_.generate_traces(
                cases=cases,
                agent_factory=agent_factory,
                nr_runs_per_case=nr_runs_per_case,
                agent_parameters=agent_parameters,
                max_case_workers=max_case_workers,
            )
        )

        return enhanced
