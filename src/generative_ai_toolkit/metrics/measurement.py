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
from dataclasses import dataclass, field
from typing import Any

from generative_ai_toolkit.utils.cloudwatch import MetricData, Unit


@dataclass
class Measurement(MetricData):
    name: str
    value: int | float | Sequence[int | float]
    unit: Unit = Unit.None_
    additional_info: Any = None
    dimensions: Sequence[Mapping[str, str]] = field(init=True, default_factory=list)
    validation_passed: bool | None = None
