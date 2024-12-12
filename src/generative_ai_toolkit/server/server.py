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

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import os


class SPAStaticFiles(StaticFiles):
    async def get_response(self, path: str, scope):
        try:
            return await super().get_response(path, scope)
        except Exception as ex:
            if getattr(ex, "status_code", None) == 404:
                return await super().get_response("index.html", scope)
            raise ex


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    # nosemgrep: python.fastapi.security.wildcard-cors.wildcard-cors
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount the React build directory at /ui
app.mount(
    "/ui",
    SPAStaticFiles(
        directory=os.path.join(os.path.dirname(__file__), "../ui/dist"), html=True
    ),
    name="ui",
)

conversation_traces = []


@app.post("/conversation_traces")
async def receive_conversation_traces(request: Request):
    data = await request.json()
    global conversation_traces
    conversation_traces = data
    return JSONResponse(
        content={"message": "Conversation traces received successfully"}
    )


@app.get("/get_conversation_traces")
async def get_conversation_traces():
    return JSONResponse(content=conversation_traces)


@app.get("/")
async def root():
    return {"message": "Hello World"}


def start_server():
    uvicorn.run(app, host="127.0.0.1", port=8000)


if __name__ == "__main__":
    start_server()
