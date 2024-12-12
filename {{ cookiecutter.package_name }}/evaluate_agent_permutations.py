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

###########
# NOTICE
#
# This file is meant to be used by the developer during development of the agent, to experiment and learn what works best.
#
# In the example here, we're comparing two model ids, and two different temperatures.
# You can provide such permutations for any parameter that the agent factory takes, e.g. you can experiment with different sets of tools too.
###########

from generative_ai_toolkit.evaluate.interactive import GenerativeAIToolkit, Permute
from evaluate_agent import cases, metrics
from lib.agent.agent import {{ cookiecutter.agent_name }}

traces = GenerativeAIToolkit.generate_traces(
    cases=cases,
    agent_factory={{ cookiecutter.agent_name }},
    nr_runs_per_case=3,
    agent_parameters={
        "model_id": Permute(
            [
                "anthropic.claude-3-sonnet-20240229-v1:0",
                "anthropic.claude-3-haiku-20240307-v1:0",
            ]
        ),
        "temperature": Permute([0.0, 0.7]),
    },
)


if __name__ == "__main__":
    measurements = GenerativeAIToolkit.eval(traces=traces, metrics=metrics)

    measurements.summary()  # This prints a summary table to the console
