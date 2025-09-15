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


import base64
import json
import zlib
from datetime import date, datetime, time
from types import SimpleNamespace


class DefaultJsonEncoder(json.JSONEncoder):
    """
    A lossy JSON encoder that is more lenient than Python's native JSON encoder
    """

    def default(self, o):
        if isinstance(o, date | datetime | time):
            return o.isoformat()
        if isinstance(o, SimpleNamespace):
            return o.__dict__
        if hasattr(o, "__json__") and callable(o.__json__):
            return o.__json__()
        if isinstance(o, bytes | bytearray | memoryview):
            return base64.standard_b64encode(o).decode("ascii")
        return str(o)


class JsonBytes(DefaultJsonEncoder):
    """
    A lossy JSON encoder and decoder, with lossless support for:

    - bytes
    - time
    - date
    - datetime
    """

    _BYTES_TAG = "__bytes__"
    _DATE_TAG = "__date__"
    _TIME_TAG = "__time__"
    _DATETIME_TAG = "__datetime__"

    def default(self, o):
        if isinstance(o, bytes):
            compressed = zlib.compress(o, level=6)
            return {self._BYTES_TAG: base64.a85encode(compressed).decode("ascii")}
        if isinstance(o, datetime):
            return {self._DATETIME_TAG: o.isoformat()}
        if isinstance(o, date):
            return {self._DATE_TAG: o.isoformat()}
        if isinstance(o, time):
            return {self._TIME_TAG: o.isoformat()}
        return super().default(o)

    @classmethod
    def bytes_json_object_hook(cls, d: dict):
        keys_set = set(d.keys())
        if keys_set == {cls._BYTES_TAG} and isinstance(d[cls._BYTES_TAG], str):
            compressed_data = base64.a85decode(d[cls._BYTES_TAG])
            return zlib.decompress(compressed_data)
        if keys_set == {cls._DATE_TAG} and isinstance(d[cls._DATE_TAG], str):
            return date.fromisoformat(d[cls._DATE_TAG])
        if keys_set == {cls._TIME_TAG} and isinstance(d[cls._TIME_TAG], str):
            return time.fromisoformat(d[cls._TIME_TAG])
        if keys_set == {cls._DATETIME_TAG} and isinstance(d[cls._DATETIME_TAG], str):
            return datetime.fromisoformat(d[cls._DATETIME_TAG])

        return d

    @classmethod
    def dumps(cls, obj, **kwargs) -> str:
        return json.dumps(obj, cls=cls, **kwargs)

    @classmethod
    def loads(cls, s: str):
        return json.loads(s, object_hook=cls.bytes_json_object_hook)
