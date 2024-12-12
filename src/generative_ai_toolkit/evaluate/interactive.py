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

import os
import signal
import socket
import subprocess
import threading
import time
import json
import urllib.request
from collections import defaultdict
from statistics import fmean
from typing import Any, Callable, Iterable, Mapping, Sequence
import webbrowser

import pandas as pd
from IPython.display import display
from tabulate import tabulate

from generative_ai_toolkit.evaluate.evaluate import (
    GenerativeAIToolkit as GenAIToolkit_,
    ConversationMeasurements,
)
from generative_ai_toolkit.evaluate.evaluate import (
    Permute,  # noqa: F401 Leave this here, so it can be imported from this module as well
)
from generative_ai_toolkit.metrics import BaseMetric
from generative_ai_toolkit.metrics.measurement import Measurement
from generative_ai_toolkit.test import AgentLike, Case
from generative_ai_toolkit.tracer.tracer import Trace
from generative_ai_toolkit.utils.interactive import is_notebook
import generative_ai_toolkit.server.server
from dataclasses import dataclass, field


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

    def start_ui(self):
        GenerativeAIToolkit.start_ui(self.conversation_measurements)

    def stop_ui(self):
        GenerativeAIToolkit.stop_ui()

    def summary(self):
        return self.summary_for(self)

    @staticmethod
    def summary_for(conversations: Iterable[ConversationMeasurements]):
        @dataclass
        class Aggregated_counts:
            measurements: list[Measurement] = field(default_factory=list)
            trace_count: int = 0
            llm_calls: int = 0
            tool_calls: int = 0
            run_nrs: set[int] = field(default_factory=set)
            nr_passed = 0
            nr_failed = 0

        aggregated_counts: dict[tuple, Aggregated_counts] = defaultdict(
            lambda: Aggregated_counts()
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
                trace = measurements_for_trace.trace
                aggregated_counts[permutation_as_key].trace_count += 1
                for measurement in measurements_for_trace.measurements:
                    aggregated_counts[permutation_as_key].measurements.append(
                        measurement
                    )
                    if measurement.validation_passed is True:
                        aggregated_counts[permutation_as_key].nr_passed += 1
                    elif measurement.validation_passed is False:
                        aggregated_counts[permutation_as_key].nr_failed += 1
                if trace.to == "LLM":
                    aggregated_counts[permutation_as_key].llm_calls += 1
                elif trace.to == "TOOL":
                    aggregated_counts[permutation_as_key].tool_calls += 1

        data = []
        for permutation_as_key, counts_per_permutation in aggregated_counts.items():
            measurement_averages: dict[str, list] = defaultdict(list)
            for measurement in counts_per_permutation.measurements:
                if measurement.dimensions:
                    for dimensions in measurement.dimensions:
                        vals_concat = "_".join((sorted(dimensions.values())))
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
                "Avg LLM calls per run": counts_per_permutation.llm_calls
                / len(counts_per_permutation.run_nrs),
                "Avg Tool calls per run": counts_per_permutation.tool_calls
                / len(counts_per_permutation.run_nrs),
                "Total Nr Passed": counts_per_permutation.nr_passed,
                "Total Nr Failed": counts_per_permutation.nr_failed,
            }
            data.append(row)

        df = pd.DataFrame(data).sort_index(
            axis=1,
            key=lambda index: index.map(
                lambda column_name: (
                    (
                        1
                        if column_name.startswith("Avg")
                        else 2
                        if column_name.startswith("Total Nr")
                        else 0
                    ),
                    column_name,
                )
            ),
        )
        if is_notebook():
            display(df)
        else:
            print(tabulate(data, headers="keys", tablefmt="pretty"))

        return df

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

    _server_process = None

    @staticmethod
    def is_port_in_use(port: int):
        """
        Check if a port is in use.
        """
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            return sock.connect_ex(("127.0.0.1", port)) == 0

    @staticmethod
    def get_pid_on_port(port: int):
        """
        Get the PID of the process using the specified port.
        """
        try:
            result = subprocess.run(
                ["lsof", "-i", f":{port}"], capture_output=True, text=True
            )
            for line in result.stdout.splitlines():
                if "LISTEN" in line:
                    return int(line.split()[1])
        except Exception as e:
            print(f"Failed to get PID on port {port}: {e}")
        return None

    @staticmethod
    def stop_ui():
        """
        Stop the FastAPI server.
        """
        pid = GenerativeAIToolkit.get_pid_on_port(8000)
        if pid:
            try:
                os.kill(pid, signal.SIGTERM)
                print(f"Successfully killed process {pid} on port 8000.")
            except Exception as e:
                print(f"Failed to kill process {pid} on port 8000: {e}")
        else:
            print("No server is running on port 8000.")

    @staticmethod
    def start_ui(measurements: Iterable[ConversationMeasurements] | None = None):
        if GenerativeAIToolkit.is_port_in_use(8000):
            print(
                "Port 8000 is already in use. Attempting to stop any existing server."
            )
            GenerativeAIToolkit.stop_ui()

        if not GenerativeAIToolkit.is_port_in_use(8000):

            def run_uvicorn():
                server_path = generative_ai_toolkit.server.server.__file__
                GenerativeAIToolkit._server_process = subprocess.Popen(
                    ["python", server_path]
                )

            threading.Thread(target=run_uvicorn).start()
            time.sleep(2)
            if GenerativeAIToolkit.is_port_in_use(8000):
                print("Server started successfully.")
            else:
                print("Failed to start the server.")

        if measurements:
            try:
                # nosemgrep: python.lang.security.audit.insecure-transport.urllib.insecure-request-object.insecure-request-object
                url = "http://127.0.0.1:8000/conversation_traces"
                headers = {"Content-Type": "application/json"}
                data = json.dumps(
                    list(measurements),
                    default=ConversationMeasurements.json_encoder,
                ).encode("utf-8")
                # nosemgrep: python.lang.security.audit.insecure-transport.urllib.insecure-request-object.insecure-request-object
                req = urllib.request.Request(
                    url, data=data, headers=headers, method="POST"
                )
                # nosemgrep: python.lang.security.audit.dynamic-urllib-use-detected.dynamic-urllib-use-detected
                with urllib.request.urlopen(req) as response:
                    if response.status == 200:
                        print("Traces sent successfully.")
                        webbrowser.open("http://127.0.0.1:8000/ui")
                    else:
                        print(f"Failed to send traces: {response.status_code}")
            except Exception as e:
                print(f"Error sending traces: {e}")

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
