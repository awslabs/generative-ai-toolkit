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

import secrets
import time
from datetime import UTC, datetime
from functools import cached_property


class Ulid:
    """
    Simple Ulid implementation based on https://github.com/ulid/spec
    """

    ulid: str
    _timestamp: int | None
    ALPHABET = "0123456789ABCDEFGHJKMNPQRSTVWXYZ"
    _ALPHABET_SET = set(ALPHABET)
    TIMESTAMP_LENGTH = 10
    RANDOMNESS_LENGTH = 16
    ULID_LENGTH = 26

    def __init__(self, ulid: str | None = None) -> None:
        if ulid:
            if len(ulid) != self.ULID_LENGTH or (set(ulid) - self._ALPHABET_SET):
                raise ValueError("Invalid ULID: " + ulid)
            self.ulid = ulid
            self._timestamp = None
        else:
            self.ulid, self._timestamp = self._generate()

    def __repr__(self) -> str:
        return f"Ulid('{self.ulid}')"

    def __str__(self) -> str:
        return self.ulid

    def __lt__(self, other):
        return self.ulid < str(other)

    def __eq__(self, other):
        return self.ulid == str(other)

    def __hash__(self):
        return hash(self.ulid)

    @cached_property
    def timestamp(self):
        if self._timestamp is None:
            timestamp_encoded = self.ulid[: self.TIMESTAMP_LENGTH]
            self._timestamp = self._decode_base32(timestamp_encoded)
        timestamp_datetime = datetime.fromtimestamp(self._timestamp / 1000, tz=UTC)
        return timestamp_datetime

    @classmethod
    def _encode_base32(cls, value: int, length: int):
        encoded = ""
        for _ in range(length):
            encoded = cls.ALPHABET[value % 32] + encoded
            value //= 32
        return encoded

    @classmethod
    def _decode_base32(cls, encoded: str):
        value = 0
        for char in encoded:
            value = value * 32 + cls.ALPHABET.index(char)
        return value

    @classmethod
    def _generate(cls):
        timestamp = int(time.time() * 1000)
        randomness = secrets.randbits(80)
        timestamp_encoded = cls._encode_base32(timestamp, cls.TIMESTAMP_LENGTH)
        randomness_encoded = cls._encode_base32(randomness, cls.RANDOMNESS_LENGTH)
        ulid = timestamp_encoded + randomness_encoded
        return ulid, timestamp
