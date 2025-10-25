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
import queue
import threading
from collections.abc import Callable, Iterable
from typing import (
    TypeVar,
    cast,
)

from generative_ai_toolkit.exceptions import StopEventAbortError

T = TypeVar("T")
type Result[T] = tuple[BaseException, None] | tuple[None, T]


def invoke_cancellable[T](
    *,
    stop_event: threading.Event | None,
    method: Callable[..., T],
    kwargs,
) -> T:
    """
    Call a method in a thread with cancellation support.

    Returns the response from the method, or raises an error if the stop_event is set.
    """
    if stop_event is None:
        return method(**kwargs)
    elif stop_event.is_set():
        raise StopEventAbortError

    q = queue.Queue[Result[T]]()

    def enqueue_if_not_shutdown(result: Result[T]):
        try:
            q.put(result)
        except queue.ShutDown:
            pass

    def invoke():
        try:
            response = method(**kwargs)
        except Exception as err:
            enqueue_if_not_shutdown((err, None))
            raise
        enqueue_if_not_shutdown((None, response))

    ctx = contextvars.copy_context()
    t = threading.Thread(target=ctx.run, args=(invoke,), daemon=True)
    t.start()
    try:
        while True:
            if t.is_alive() and not stop_event.is_set():
                stop_event.wait(0.1)
                continue

            # Thread is no longer alive, or the stop_event was set
            try:
                # Even if the stop_event was set, the invoked method might just have finished
                # Therefore, we'll try to get its return value always from the queue.
                # If we have a return value/error, we'll return/raise that if we can
                err, res = q.get(timeout=0.1)
            except queue.Empty:
                if stop_event.is_set():
                    raise StopEventAbortError from None
                # We should get a record from the queue soon!
                # We'll keep running the loop until we do (or until the stop_event gets set)
                continue
            else:
                if err:
                    raise err
                return cast(T, res)
    finally:
        q.shutdown()


def with_placeholder_emit[T](
    iterable: Iterable[T],
    *,
    interval: float = 0.1,
    placeholder=None,
    stop_event: threading.Event | None = None,
) -> Iterable[T | None]:
    """
    Wraps an iterable to emit values at regular intervals.
    Emits the iterable's value if available, otherwise the placeholder.
    """
    q = queue.Queue()
    sentinel = object()  # Marker for generator exhaustion

    def producer():
        try:
            for item in iterable:
                if stop_event and stop_event.is_set():
                    return
                q.put(item)
        except Exception:
            if not stop_event or not stop_event.is_set():
                raise
        finally:
            q.put(sentinel)

    # Start producer thread
    thread = threading.Thread(target=producer, daemon=True)
    thread.start()

    while not (stop_event and stop_event.is_set()):
        try:
            # Try to get item with timeout
            item = q.get(timeout=interval)

            if item is sentinel:
                break

            yield item

        except queue.Empty:
            # No item available, emit placeholder
            yield placeholder
