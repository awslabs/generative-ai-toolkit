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

# This is in this separate file so it can be imported by both agents and tools
# without running into circular imports


import contextvars
from threading import Event
from typing import Any, NotRequired, TypedDict

from generative_ai_toolkit.tracer import Tracer


class AuthContext(TypedDict):
    principal_id: str | None
    """
    The ID of the principal (e.g. the user) that the agent is acting on behalf of
    """

    # Once https://peps.python.org/pep-0728 lands (Python 3.15?) we can deprecate this property,
    # as the TypedDict would then allow extra keys itself.
    extra: NotRequired[Any]
    """
    Additional information to add to the AuthContext
    """


class AgentContext:

    _current = contextvars.ContextVar["AgentContext"]("agent_context")

    conversation_id: str
    """The conversation ID"""

    tracer: Tracer
    """The tracer that is used by the agent; tools can use it for adding their own traces"""

    auth_context: AuthContext
    """The auth context; tools can use it for enforcing authentication and authorization"""

    stop_event: Event
    """
    Stop event (threading) that may be set by the user to signal abortion; tools that run for a longer span of time
    should consult the stop event regularly (`stop_event.is_set()`) and abort early if it is set
    """

    def __init__(
        self,
        *,
        conversation_id: str,
        tracer: Tracer,
        auth_context: AuthContext,
        stop_event: Event | None,
    ) -> None:
        self.conversation_id = conversation_id
        self.tracer = tracer
        self.auth_context = auth_context
        self.stop_event = stop_event or Event()

    @classmethod
    def current(cls) -> "AgentContext":
        """
        Access the current agent context from within a tool invocation
        """
        return cls._current.get()

    def copy_context(self):
        ctx = contextvars.copy_context()
        ctx.run(lambda: self._current.set(self))
        return ctx
