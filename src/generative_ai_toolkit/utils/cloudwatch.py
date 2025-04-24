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

from collections.abc import Mapping, Sequence
from enum import Enum
from typing import Protocol


class Unit(Enum):
    """
    Amazon CloudWatch Metrics compatible unit
    """

    Seconds = "Seconds"
    Microseconds = "Microseconds"
    Milliseconds = "Milliseconds"
    Bytes = "Bytes"
    Kilobytes = "Kilobytes"
    Megabytes = "Megabytes"
    Gigabytes = "Gigabytes"
    Terabytes = "Terabytes"
    Bits = "Bits"
    Kilobits = "Kilobits"
    Megabits = "Megabits"
    Gigabits = "Gigabits"
    Terabits = "Terabits"
    Percent = "Percent"
    Count = "Count"
    BytesPerSecond = "Bytes/Second"
    KilobytesPerSecond = "Kilobytes/Second"
    MegabytesPerSecond = "Megabytes/Second"
    GigabytesPerSecond = "Gigabytes/Second"
    TerabytesPerSecond = "Terabytes/Second"
    BitsPerSecond = "Bits/Second"
    KilobitsPerSecond = "Kilobits/Second"
    MegabitsPerSecond = "Megabits/Second"
    GigabitsPerSecond = "Gigabits/Second"
    TerabitsPerSecond = "Terabits/Second"
    CountPerSecond = "Count/Second"
    None_ = "None"


class MetricData(Protocol):
    name: str
    value: int | float | Sequence[int | float]
    unit: Unit
    dimensions: Sequence[Mapping[str, str]]
