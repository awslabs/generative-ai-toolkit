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

from generative_ai_toolkit.tracer.trace import Trace


def test_attribute_inheritance():
    trace1 = Trace("top")
    trace2 = Trace("middle", parent_span=trace1)
    trace3 = Trace("bottom", parent_span=trace2)

    trace1.add_attribute("inherited", 1, inheritable=True)
    trace1.add_attribute("to-override", 2, inheritable=True)
    trace2.add_attribute("not-inherited", 3)
    trace3.add_attribute("not-inherited2", 4)
    trace3.add_attribute("to-override", 5)

    # Inheritance
    assert trace1.attributes["inherited"] == 1
    assert trace2.attributes["inherited"] == 1
    assert trace3.attributes["inherited"] == 1

    # Overrides
    assert trace1.attributes["to-override"] == 2
    assert trace2.attributes["to-override"] == 2
    assert trace3.attributes["to-override"] == 5

    # Not inherited
    assert "not-inherited" not in trace1.attributes
    assert "not-inherited" not in trace3.attributes
    assert trace2.attributes["not-inherited"] == 3
    assert "not-inherited2" not in trace1.attributes
    assert "not-inherited2" not in trace2.attributes
    assert trace3.attributes["not-inherited2"] == 4
