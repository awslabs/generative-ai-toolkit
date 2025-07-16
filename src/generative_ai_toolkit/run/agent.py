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

import json
from collections.abc import Callable, Iterable
from contextvars import ContextVar
from typing import (
    Protocol,
    TypedDict,
    Unpack,
)

from flask import Flask, Request, Response, request
from pydantic import BaseModel, Field

from generative_ai_toolkit.context import AuthContext
from generative_ai_toolkit.utils.logging import logger


class Runnable(Protocol):
    @property
    def conversation_id(self) -> str: ...

    def set_conversation_id(self, conversation_id: str) -> None: ...

    def set_auth_context(
        self, **auth_context: Unpack[AuthContext]
    ) -> None: ...

    def reset(self) -> None: ...

    def converse_stream(self, user_input: str) -> Iterable[str]: ...


AuthContextFn = Callable[[Request], AuthContext]


class RunnerConfig(TypedDict, total=False):
    agent: Runnable | Callable[[], Runnable]
    auth_context_fn: AuthContextFn


def iam_auth_context_fn(request: Request) -> AuthContext:
    try:
        amzn_request_context = json.loads(request.headers["x-amzn-request-context"])
        principal_id = amzn_request_context["authorizer"]["iam"]["userId"]
        return {"principal_id": principal_id, "extra": amzn_request_context}
    except Exception as e:
        raise Exception("Missing AWS IAM Auth context") from e


class _Runner:
    _agent: Runnable | Callable[[], Runnable] | None
    _auth_context_fn: AuthContextFn
    _app: Flask

    def __init__(self) -> None:
        self._agent = None
        self._auth_context_fn = iam_auth_context_fn
        self._context = ContextVar[Runnable | None]("agent", default=None)

    @property
    def agent(self) -> Runnable:
        if not self._agent:
            raise ValueError("Agent not configured yet")
        # Cache an agent instance in each context:
        context_agent = self._context.get()
        if not context_agent:
            if callable(self._agent):
                context_agent = self._agent()
            else:
                context_agent = self._agent
            self._context.set(context_agent)
        return context_agent

    @property
    def auth_context_fn(self) -> AuthContextFn:
        return self._auth_context_fn

    def configure(
        self,
        **kwargs: Unpack[RunnerConfig],
    ):
        if "agent" in kwargs:
            self._agent = kwargs["agent"]
        if "auth_context_fn" in kwargs:
            if not callable(kwargs["auth_context_fn"]):
                raise ValueError("auth_context_fn must be callable")
            self._auth_context_fn = kwargs["auth_context_fn"]

    @property
    def app(self):
        app = Flask(__name__)

        @app.get("/")
        def health():
            return "Up and running! To chat with the agent, use HTTP POST"

        class Body(BaseModel):
            user_input: str = Field(
                description="The input from the user to the agent", min_length=1
            )

        @app.post("/")
        def index():
            agent = self.agent

            try:
                auth_context = self.auth_context_fn(request)
                agent.set_auth_context(**auth_context)
            except Exception as err:
                logger.warn(f"Forbidden: {err}")
                return Response("Forbidden", status=403)

            try:
                body = Body.model_validate_json(request.data)
            except Exception as err:
                logger.info(f"Unprocessable entity: {err}")
                return Response("Unprocessable entity", status=422)

            x_conversation_id = request.headers.get("x-conversation-id")
            if x_conversation_id:
                agent.set_conversation_id(x_conversation_id)
            else:
                agent.reset()

            # Explicitly consume the first chunk so any obvious errors bubble up
            # before we return status 200 below
            chunks = agent.converse_stream(body.user_input)
            first_chunk = next(iter(chunks))

            def chunked_response():
                try:
                    yield first_chunk
                    yield from chunks
                except Exception:
                    logger.exception()
                    yield "Internal Server Error\n"

            return Response(
                chunked_response(),
                status=200,
                content_type="text/plain; charset=utf-8",
                headers={
                    "x-conversation-id": agent.conversation_id,
                    "transfer-encoding": "chunked",
                },
            )

        @app.errorhandler(Exception)
        def error(error):
            logger.exception()
            return "Internal Server Error", 500

        return app

    def __call__(self):
        """
        Making the runner callable makes it easy to use the Runner object directly,
        when launching e.g. gunicorn from the CLI:

        gunicorn path.to.myagent:Runner()
        """
        return self.app


Runner = _Runner()
