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

import json
import os
from typing import (
    Callable,
    Iterable,
    Protocol,
    TypedDict,
    Unpack,
)

from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
import uvicorn

from generative_ai_toolkit.utils.logging import logger


app = FastAPI()


class Runnable(Protocol):
    @property
    def conversation_id(self) -> str: ...

    def set_conversation_id(self, conversation_id: str) -> None: ...

    def set_auth_context(self, auth_context: str | None) -> None: ...

    def reset(self) -> None: ...

    def converse_stream(self, user_input: str) -> Iterable[str]: ...


AuthContextFn = Callable[[Request], str | None]


class RunnerConfig(TypedDict, total=False):
    agent: Runnable
    auth_context_fn: AuthContextFn


def iam_auth_context_fn(request: Request):
    try:
        amzn_request_context = json.loads(request.headers["x-amzn-request-context"])
        return amzn_request_context["authorizer"]["iam"]["userId"]
    except Exception as e:
        raise Exception("Unable to determine auth context") from e


class _UvicornRunner:
    _agent: Runnable | None
    _auth_context_fn: AuthContextFn

    def __init__(self) -> None:
        self._agent = None
        self._auth_context_fn = iam_auth_context_fn

    @property
    def agent(self) -> Runnable:
        if not self._agent:
            raise ValueError("Agent not configured yet")
        return self._agent

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

    def __call__(self):
        uvicorn.run(
            app,
            host="0.0.0.0",
            port=int(os.environ.get("PORT") or "8080"),
            timeout_keep_alive=300,
        )


UvicornRunner = _UvicornRunner()


class Body(BaseModel):
    user_input: str = Field(
        description="The input from the user to the agent", min_length=1
    )


@app.get("/")
async def health():
    return "Up and running! To chat with the agent, use HTTP POST"


@app.post("/")
async def index(
    body: Body,
    request: Request,
):
    agent = UvicornRunner.agent
    try:
        auth_context = UvicornRunner.auth_context_fn(request)
        agent.set_auth_context(auth_context)
    except Exception:
        logger.exception()
        return StreamingResponse("Forbidden", status_code=403)

    x_conversation_id = request.headers.get("x-conversation-id")
    if x_conversation_id:
        agent.set_conversation_id(x_conversation_id)
    else:
        agent.reset()
    return StreamingResponse(
        agent.converse_stream(body.user_input),
        media_type="text/event-stream",
        headers={"x-conversation-id": agent.conversation_id},
    )
