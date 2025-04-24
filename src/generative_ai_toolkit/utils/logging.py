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
# limitations under the License."""

"""
Logging utility for logging to CloudWatch Logs, to use in e.g. AWS Lambda.

Some small features included:
- log structured JSON as that works really well with CloudWatch Logs filters and CloudWatch Logs Insights
- support turning any Python object into JSON
- replace line endings for nice folding of entries in CloudWatch Logs
- do not buffer stdout
"""

import json
import os
import sys
import time
import traceback
from collections.abc import Mapping
from typing import Any, TextIO

from generative_ai_toolkit.utils.cloudwatch import MetricData

in_aws_lambda = os.environ.get("AWS_EXECUTION_ENV", "").startswith("AWS_Lambda_")


class SimpleLogger:
    """
    SimpleLogger for logging to CloudWatch Logs.
    Does not filter by LogLevel, but instead logs everything to CloudWatch.
    Logs can be filtered by level easily when querying in CloudWatch Logs

    >>> logger = SimpleLogger("MyLoggerName")
    >>> logger.info("my message", event={"test": 123}, another_field="foo")
    {
        "logger": "MyLoggerName",
        "level": "INFO",
        "message": "my message",
        "event": {
            "test": 123
        },
        "another_field": "foo"
    }
    """

    def __init__(self, name: str = "", *, stream: TextIO | None = None):
        self._stream = stream or sys.stdout
        self.fields = {}
        if name:
            self.fields["logger"] = name

    def _log(self, level, message: str, **kwargs):
        # No need to include a timestamp as CloudWatch Logs assigns one already
        fields = {
            **self.fields,
            "level": level,
            "message": message,
            **kwargs,
        }

        # Dump to JSON (convert non-JSON-able objects to str)
        # Replace line endings so multi-line log messages are still displayed as one record in CloudWatch Logs
        # Flush to stdout immediately (no need to set PYTHONUNBUFFERED)
        print(
            json.dumps(fields, separators=(",", ":"), default=str).replace(
                "\n",
                "\r" if in_aws_lambda else "\n",  # no-op if not not in AWS Lambda
            ),
            flush=False if in_aws_lambda else True,  # not needed in AWS Lambda
            file=self._stream,
        )

    def debug(self, message: str, **kwargs):
        self._log("DEBUG", message, **kwargs)

    def info(self, message: str, **kwargs):
        self._log("INFO", message, **kwargs)

    def warn(self, message: str, **kwargs):
        self._log("WARN", message, **kwargs)

    def error(self, message: str, **kwargs):
        self._log("ERROR", message, **kwargs)

    def exception(self, message: str | None = None, **kwargs):
        if message is None:
            message = str(sys.exc_info()[1])
        self._log("ERROR", message, stack_trace=traceback.format_exc(), **kwargs)

    def metric(
        self,
        metric_data: MetricData,
        namespace: str,
        message="CloudWatch Embedded Metric",
        common_dimensions: Mapping[str, str] | None = None,
        timestamp: int | None = None,
        **kwargs,
    ):
        common_dimensions = {**common_dimensions} if common_dimensions else {}
        dimension_sets = (
            [{**dim_set, **common_dimensions} for dim_set in metric_data.dimensions]
            if metric_data.dimensions
            else [common_dimensions]
        )
        accumulated_dimensions: Mapping[str, str] = {}
        for dim_set in dimension_sets:
            accumulated_dimensions.update(dim_set)

        if len(accumulated_dimensions) > 9:
            raise ValueError("More than 9 dimensions provided")
        log_data: Mapping[str, Any] = {
            metric_data.name: metric_data.value,
        }
        for k, v in kwargs.items():
            if v is not None:
                log_data[k] = v
        log_data.update(
            {
                "message": message,
                **accumulated_dimensions,
                "_aws": {
                    "CloudWatchMetrics": [
                        {
                            "Metrics": [
                                {
                                    "Name": metric_data.name,
                                    "Unit": metric_data.unit.value,
                                }
                            ],
                            "Dimensions": [
                                list(dim_set.keys()) for dim_set in dimension_sets
                            ],
                            "Namespace": namespace,
                        }
                    ],
                    "Timestamp": (
                        timestamp if timestamp is not None else int(time.time() * 1000)
                    ),
                },
            }
        )

        self._log("METRIC", **log_data)


logger = SimpleLogger()
