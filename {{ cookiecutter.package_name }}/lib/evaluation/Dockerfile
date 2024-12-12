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

FROM public.ecr.aws/lambda/python:3.12
WORKDIR /var/task

# uv
COPY --from=ghcr.io/astral-sh/uv:0.4.0 /uv /bin/uv
ENV UV_COMPILE_BYTECODE=1 \
    UV_SYSTEM_PYTHON=1

# install generative_ai_toolkit
RUN uv pip install "generative_ai_toolkit[evaluate]"

# Customize below this line
COPY *.py ./
# nosemgrep: dockerfile.security.missing-user.missing-user
CMD ["measure.AWSLambdaRunner"]
