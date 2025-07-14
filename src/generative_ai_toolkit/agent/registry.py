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

import importlib
import pkgutil
from collections.abc import Callable, Iterable, Sequence
from types import ModuleType
from typing import Any, overload


class ToolRegistry(Sequence):
    def __init__(self, tools: Iterable[Callable[..., Any]] | None = None) -> None:
        self._tool_registry: list[Callable[..., Any]] = (
            list(tools) if tools is not None else []
        )

    def add(self, tool: Callable[..., Any]) -> None:
        self._tool_registry.append(tool)

    def clear(self):
        self._tool_registry.clear()

    def __len__(self):
        return len(self._tool_registry)

    def __iter__(self):
        return iter(self._tool_registry)

    @overload
    def __getitem__(self, index: int) -> Callable[..., Any]: ...

    @overload
    def __getitem__(self, index: slice) -> "ToolRegistry": ...

    def __getitem__(self, index: int | slice) -> "Callable[..., Any] | ToolRegistry":
        if isinstance(index, slice):
            return ToolRegistry(self._tool_registry[index])
        return self._tool_registry[index]

    @classmethod
    def recursive_import(cls, module: ModuleType):
        for _, name, is_pkg in pkgutil.iter_modules(
            module.__path__, prefix=f"{module.__name__}."
        ):
            submod = importlib.import_module(name)
            if is_pkg:
                cls.recursive_import(submod)
        return cls


DEFAULT_TOOL_REGISTRY = ToolRegistry()


@overload
def tool[F: Callable[..., Any]](wrapped: F) -> F: ...
@overload
def tool[F: Callable[..., Any]](
    *, tool_registry: ToolRegistry | Sequence[ToolRegistry] = DEFAULT_TOOL_REGISTRY
) -> Callable[[F], F]: ...


def tool(
    wrapped: Callable[..., Any] | None = None,
    *,
    tool_registry: ToolRegistry | Sequence[ToolRegistry] = DEFAULT_TOOL_REGISTRY,
):

    def decorator[F: Callable[..., Any]](func: F) -> F:
        if not isinstance(tool_registry, ToolRegistry):
            for registry in tool_registry:
                registry.add(func)
            return func
        tool_registry.add(func)
        return func

    if wrapped is None:
        return decorator

    return decorator(wrapped)
