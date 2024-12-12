#!/bin/bash -euo pipefail

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

git init

echo "Creating Python virtual environment ..."
uv venv --python {{ cookiecutter.python_interpreter }}
source .venv/bin/activate
echo "Installing Python dependencies ..."
uv pip install 'generative_ai_toolkit[all]' ipykernel jupyter

echo "Installing CDK ..."
npm install

echo
echo "All done! Have fun developing your Generative AI Toolkit agent!"
echo
echo "Note: this project uses \"uv\" to manage Python dependencies and the Python virtualenv."
echo "If you want to install new dependencies, also use \"uv\": simply prefix your pip commands with uv".
echo "Example: uv pip install numpy"
echo
echo "To start developing:"
echo
echo "  cd {{ cookiecutter.package_name }}"
echo "  source .venv/bin/activate"
echo "  code ."
echo
