#!/usr/bin/env python
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

import argparse
from generative_ai_toolkit.utils.lambda_url import IamAuthInvoker


def main():
    parser = argparse.ArgumentParser(
        description="Converse with an Generative AI Toolkit Agent through its Lambda Function URL."
    )
    parser.add_argument("url", help="The Lambda Function URL")
    parser.add_argument(
        "conversation_id",
        default=None,
        help="Conversation ID (optional). Use this to continue an existing conversation, otherwise a new one is started.",
        nargs="?",
    )
    parser.add_argument("user_input", help="The user input")

    args = parser.parse_args()

    lambda_url_invoker = IamAuthInvoker(args.url)
    response = lambda_url_invoker.converse_stream(
        user_input=args.user_input, conversation_id=args.conversation_id
    )

    print("Conversation ID:", response.conversation_id)
    print()
    for tokens in response:
        print(tokens, end="", flush=True)


if __name__ == "__main__":
    main()
