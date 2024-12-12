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

from generative_ai_toolkit.metrics import BaseMetric, Measurement
from generative_ai_toolkit.test import CaseTrace


class SqlMetric(BaseMetric):
    """
    SqlMetric class for measuring the correctness of SQL query responses.

    This metric compares the 'rows' of the SQL tool's response with the expected 'rows',
    and appends validation information. The validation passes only if the rows match.
    """

    def __init__(self, valid_queries_responses):
        """
        Initialize the SqlMetric with a dictionary of valid SQL queries and their expected responses.

        Args:
            valid_queries_responses (dict): A dictionary mapping case names to SQL queries and expected responses.
        """
        super().__init__()
        self.valid_queries_responses = valid_queries_responses

    def evaluate_trace(self, trace, **kwargs) -> Measurement | None:
        """
        Evaluate the trace by comparing the SQL tool's response with the expected response.

        This method extracts the actual tool response, compares only the 'rows' with the expected 'rows',
        and appends validation information.

        Args:
            trace (LlmTrace or ToolTrace): The trace object containing the request and response to the LLM.
            **kwargs: Additional keyword arguments (not used in this implementation).

        Returns:
            Measurement | None: A Measurement object with the evaluation results, including the comparison information.
                                If the trace is not an instance of LlmTrace or ToolTrace, it returns None.
        """
        if not isinstance(trace, CaseTrace):
            return None
        try:
            if "tool_response" in trace.response:
                # Extract the case name from the trace
                case_name = trace.case.name
                if case_name in self.valid_queries_responses:
                    query_info = self.valid_queries_responses[case_name]

                    # Extract the tool's response
                    tool_response = trace.response["tool_response"]

                    # Validate the response by comparing the 'rows' of the expected and actual output
                    expected_response_rows = query_info["expected_response"]["rows"]
                    actual_response_rows = tool_response["rows"]

                    validation_passed = actual_response_rows == expected_response_rows

                    # Return a measurement indicating whether the validation passed
                    return Measurement(
                        name="SQL",
                        value=float(1 if validation_passed else 0),
                        validation_passed=validation_passed,
                        additional_info={
                            "expected_response_rows": expected_response_rows,
                            "actual_response_rows": actual_response_rows,
                            "case": case_name,
                        },
                    )

        except Exception as e:
            print(f"Error in SqlMetric: {e}")
            return None
