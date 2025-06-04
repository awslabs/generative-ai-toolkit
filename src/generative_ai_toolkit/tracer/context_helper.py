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

import contextvars
from collections.abc import Callable, Generator
from typing import Any, TypeVar

T = TypeVar('T')

class ContextPreserver:
    """A utility class that helps preserve contextvars across generator yields."""

    @staticmethod
    def preserve_context(func: Callable[..., Generator[T, Any, Any]]) -> Callable[..., Generator[T, Any, Any]]:
        """
        Decorator that preserves the current context for a generator function.
        It captures the context at creation time and reinstates it at each yield point.
        """
        def wrapper(*args: Any, **kwargs: Any) -> Generator[T, Any, Any]:
            # Capture the current context
            ctx = contextvars.copy_context()

            def gen_with_context() -> Generator[T, Any, Any]:
                # Run the generator in the captured context
                return ctx.run(lambda: func(*args, **kwargs))

            generator = gen_with_context()

            # Process each yield
            try:
                item = next(generator)
                while True:
                    # Yield the item to the caller
                    received = yield item

                    # Re-run in the original context to preserve it
                    item = ctx.run(lambda: generator.send(received))
            except StopIteration as e:
                # Return the generator's return value
                return e.value

        return wrapper

    @staticmethod
    def safe_reset(context_var: contextvars.ContextVar, token: contextvars.Token) -> None:
        """
        Safely reset a ContextVar token, catching and handling the ValueError
        that occurs when the token was created in a different context.
        
        Returns True if reset was successful, False otherwise.
        """
        try:
            context_var.reset(token)
            return True
        except ValueError:
            # Token was created in a different context, just ignore
            return False
