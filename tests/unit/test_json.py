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

import pytest

from generative_ai_toolkit.utils.json import DefaultJsonEncoder, JsonBytes


class TestDefaultJsonEncoder:
    """Test cases for DefaultJsonEncoder class."""

    def test_encode_datetime(self):
        """Test encoding datetime objects."""
        encoder = DefaultJsonEncoder()
        dt = datetime(2025, 9, 15, 12, 14, 30, 123456)
        result = encoder.default(dt)
        assert result == "2025-09-15T12:14:30.123456"

    def test_encode_date(self):
        """Test encoding date objects."""
        encoder = DefaultJsonEncoder()
        d = date(2025, 9, 15)
        result = encoder.default(d)
        assert result == "2025-09-15"

    def test_encode_time(self):
        """Test encoding time objects."""
        encoder = DefaultJsonEncoder()
        t = time(12, 14, 30, 123456)
        result = encoder.default(t)
        assert result == "12:14:30.123456"

    def test_encode_simple_namespace(self):
        """Test encoding SimpleNamespace objects."""
        encoder = DefaultJsonEncoder()
        ns = SimpleNamespace(a=1, b="test", c=[1, 2, 3])
        result = encoder.default(ns)
        assert result == {"a": 1, "b": "test", "c": [1, 2, 3]}

    def test_encode_object_with_json_method(self):
        """Test encoding objects with __json__ method."""
        encoder = DefaultJsonEncoder()

        class JsonableObject:
            def __init__(self, value):
                self.value = value

            def __json__(self):
                return {"custom_value": self.value}

        obj = JsonableObject("test")
        result = encoder.default(obj)
        assert result == {"custom_value": "test"}

    def test_encode_object_with_non_callable_json_attribute(self):
        """Test encoding objects with non-callable __json__ attribute."""
        encoder = DefaultJsonEncoder()

        class NonCallableJsonObject:
            def __init__(self):
                self.__json__ = "not callable"

        obj = NonCallableJsonObject()
        result = encoder.default(obj)
        assert result == str(obj)

    def test_encode_bytes(self):
        """Test encoding bytes objects."""
        encoder = DefaultJsonEncoder()
        data = b"Hello, World!"
        result = encoder.default(data)
        expected = base64.standard_b64encode(data).decode("ascii")
        assert result == expected

    def test_encode_bytearray(self):
        """Test encoding bytearray objects."""
        encoder = DefaultJsonEncoder()
        data = bytearray(b"Hello, World!")
        result = encoder.default(data)
        expected = base64.standard_b64encode(data).decode("ascii")
        assert result == expected

    def test_encode_memoryview(self):
        """Test encoding memoryview objects."""
        encoder = DefaultJsonEncoder()
        data = memoryview(b"Hello, World!")
        result = encoder.default(data)
        expected = base64.standard_b64encode(data).decode("ascii")
        assert result == expected

    def test_encode_fallback_to_string(self):
        """Test fallback to string conversion for unsupported objects."""
        encoder = DefaultJsonEncoder()

        class CustomObject:
            def __init__(self, value):
                self.value = value

            def __str__(self):
                return f"CustomObject({self.value})"

        obj = CustomObject("test")
        result = encoder.default(obj)
        assert result == "CustomObject(test)"

    def test_full_json_encoding(self):
        """Test full JSON encoding with various object types."""
        data = {
            "datetime": datetime(2025, 9, 15, 12, 14, 30),
            "date": date(2025, 9, 15),
            "time": time(12, 14, 30),
            "namespace": SimpleNamespace(x=1, y=2),
            "bytes": b"binary data",
            "regular": "string",
            "number": 42,
        }

        result = json.dumps(data, cls=DefaultJsonEncoder)
        parsed = json.loads(result)

        assert parsed["datetime"] == "2025-09-15T12:14:30"
        assert parsed["date"] == "2025-09-15"
        assert parsed["time"] == "12:14:30"
        assert parsed["namespace"] == {"x": 1, "y": 2}
        assert parsed["bytes"] == base64.standard_b64encode(b"binary data").decode(
            "ascii"
        )
        assert parsed["regular"] == "string"
        assert parsed["number"] == 42


class TestJsonBytes:
    """Test cases for JsonBytes class."""

    def test_encode_bytes(self):
        """Test encoding bytes with compression and base85."""
        encoder = JsonBytes()
        data = b"Hello, World! This is a test of binary data encoding."
        result = encoder.default(data)

        assert isinstance(result, dict)
        assert JsonBytes._BYTES_TAG in result
        assert isinstance(result[JsonBytes._BYTES_TAG], str)

        # Verify we can decode it back
        compressed_data = base64.a85decode(result[JsonBytes._BYTES_TAG])
        decompressed_data = zlib.decompress(compressed_data)
        assert decompressed_data == data

    def test_encode_date(self):
        """Test encoding date objects."""
        encoder = JsonBytes()
        test_date = date(2025, 9, 15)
        result = encoder.default(test_date)

        assert isinstance(result, dict)
        assert JsonBytes._DATE_TAG in result
        assert result[JsonBytes._DATE_TAG] == "2025-09-15"

    def test_encode_time(self):
        """Test encoding time objects."""
        encoder = JsonBytes()
        test_time = time(12, 14, 30, 123456)
        result = encoder.default(test_time)

        assert isinstance(result, dict)
        assert JsonBytes._TIME_TAG in result
        assert result[JsonBytes._TIME_TAG] == "12:14:30.123456"

    def test_encode_datetime(self):
        """Test encoding datetime objects."""
        encoder = JsonBytes()
        test_datetime = datetime(2025, 9, 15, 12, 14, 30, 123456)
        result = encoder.default(test_datetime)

        assert isinstance(result, dict)
        assert JsonBytes._DATETIME_TAG in result
        assert result[JsonBytes._DATETIME_TAG] == "2025-09-15T12:14:30.123456"

    def test_encode_non_supported(self):
        """Test encoding non-supported objects falls back to parent encoder."""
        encoder = JsonBytes()
        data = "not supported directly"

        result = encoder.default(data)
        assert result == "not supported directly"

    def test_bytes_json_object_hook_with_bytes_tag(self):
        """Test object hook correctly identifies and decodes bytes objects."""
        original_data = b"Test binary data for compression and encoding"
        compressed = zlib.compress(original_data, level=6)
        encoded = base64.a85encode(compressed).decode("ascii")

        test_dict = {JsonBytes._BYTES_TAG: encoded}
        result = JsonBytes.bytes_json_object_hook(test_dict)

        assert result == original_data

    def test_bytes_json_object_hook_with_date_tag(self):
        """Test object hook correctly identifies and decodes date objects."""
        test_dict = {JsonBytes._DATE_TAG: "2025-09-15"}
        result = JsonBytes.bytes_json_object_hook(test_dict)

        assert isinstance(result, date)
        assert result == date(2025, 9, 15)

    def test_bytes_json_object_hook_with_time_tag(self):
        """Test object hook correctly identifies and decodes time objects."""
        test_dict = {JsonBytes._TIME_TAG: "12:14:30.123456"}
        result = JsonBytes.bytes_json_object_hook(test_dict)

        assert isinstance(result, time)
        assert result == time(12, 14, 30, 123456)

    def test_bytes_json_object_hook_with_datetime_tag(self):
        """Test object hook correctly identifies and decodes datetime objects."""
        test_dict = {JsonBytes._DATETIME_TAG: "2025-09-15T12:14:30.123456"}
        result = JsonBytes.bytes_json_object_hook(test_dict)

        assert isinstance(result, datetime)
        assert result == datetime(2025, 9, 15, 12, 14, 30, 123456)

    def test_bytes_json_object_hook_with_non_special_dict(self):
        """Test object hook passes through non-special dictionaries."""
        test_dict = {"regular": "data", "number": 42}
        result = JsonBytes.bytes_json_object_hook(test_dict)
        assert result == test_dict

    def test_bytes_json_object_hook_with_bytes_tag_non_string_value(self):
        """Test object hook passes through dicts with bytes tag but non-string value."""
        test_dict = {JsonBytes._BYTES_TAG: 123}
        result = JsonBytes.bytes_json_object_hook(test_dict)
        assert result == test_dict

    def test_bytes_json_object_hook_with_date_tag_non_string_value(self):
        """Test object hook passes through dicts with date tag but non-string value."""
        test_dict = {JsonBytes._DATE_TAG: 123}
        result = JsonBytes.bytes_json_object_hook(test_dict)
        assert result == test_dict

    def test_bytes_json_object_hook_with_time_tag_non_string_value(self):
        """Test object hook passes through dicts with time tag but non-string value."""
        test_dict = {JsonBytes._TIME_TAG: 123}
        result = JsonBytes.bytes_json_object_hook(test_dict)
        assert result == test_dict

    def test_bytes_json_object_hook_with_datetime_tag_non_string_value(self):
        """Test object hook passes through dicts with datetime tag but non-string value."""
        test_dict = {JsonBytes._DATETIME_TAG: 123}
        result = JsonBytes.bytes_json_object_hook(test_dict)
        assert result == test_dict

    def test_bytes_json_object_hook_with_extra_keys(self):
        """Test object hook passes through dicts with special tags and extra keys."""
        test_dict = {JsonBytes._BYTES_TAG: "some_string", "extra": "key"}
        result = JsonBytes.bytes_json_object_hook(test_dict)
        assert result == test_dict

    def test_dumps_method(self):
        """Test JsonBytes.dumps class method."""
        data = {"text": "hello", "binary": b"binary data"}
        result = JsonBytes.dumps(data)

        assert isinstance(result, str)
        parsed = json.loads(result)
        assert parsed["text"] == "hello"
        assert JsonBytes._BYTES_TAG in parsed["binary"]

    def test_loads_method(self):
        """Test JsonBytes.loads class method."""
        original_data = {"text": "hello", "binary": b"binary data"}
        json_str = JsonBytes.dumps(original_data)
        result = JsonBytes.loads(json_str)

        assert result["text"] == "hello"
        assert result["binary"] == b"binary data"

    def test_roundtrip_encoding_decoding_all_types(self):
        """Test complete roundtrip of encoding and decoding all supported data types."""
        test_data = {
            "string": "Hello, World!",
            "number": 42,
            "float": 3.14,
            "boolean": True,
            "null": None,
            "array": [1, 2, 3],
            "nested": {
                "binary1": b"First binary data",
                "binary2": b"Second binary data with more content for better compression",
                "text": "nested text",
                "date": date(2025, 1, 15),
                "time": time(9, 30, 45, 500000),
                "datetime": datetime(2025, 12, 25, 18, 30, 0, 750000),
            },
            "empty_bytes": b"",
            "large_binary": b"x" * 1000,  # Test compression effectiveness
            "top_level_date": date(2025, 9, 16),
            "top_level_time": time(7, 29, 0),
            "top_level_datetime": datetime(2025, 9, 16, 7, 29, 0, 123456),
        }

        # Encode to JSON string
        json_str = JsonBytes.dumps(test_data)

        # Decode back to Python objects
        decoded_data = JsonBytes.loads(json_str)

        # Verify all data matches
        assert decoded_data["string"] == test_data["string"]
        assert decoded_data["number"] == test_data["number"]
        assert decoded_data["float"] == test_data["float"]
        assert decoded_data["boolean"] == test_data["boolean"]
        assert decoded_data["null"] == test_data["null"]
        assert decoded_data["array"] == test_data["array"]
        assert decoded_data["nested"]["binary1"] == test_data["nested"]["binary1"]
        assert decoded_data["nested"]["binary2"] == test_data["nested"]["binary2"]
        assert decoded_data["nested"]["text"] == test_data["nested"]["text"]
        assert decoded_data["nested"]["date"] == test_data["nested"]["date"]
        assert decoded_data["nested"]["time"] == test_data["nested"]["time"]
        assert decoded_data["nested"]["datetime"] == test_data["nested"]["datetime"]
        assert decoded_data["empty_bytes"] == test_data["empty_bytes"]
        assert decoded_data["large_binary"] == test_data["large_binary"]
        assert decoded_data["top_level_date"] == test_data["top_level_date"]
        assert decoded_data["top_level_time"] == test_data["top_level_time"]
        assert decoded_data["top_level_datetime"] == test_data["top_level_datetime"]

    def test_date_edge_cases(self):
        """Test encoding and decoding date edge cases."""
        edge_dates = [
            date(1, 1, 1),  # Minimum date
            date(9999, 12, 31),  # Maximum date
            date(2000, 2, 29),  # Leap year
            date(1900, 2, 28),  # Non-leap year century
        ]

        for test_date in edge_dates:
            json_str = JsonBytes.dumps({"date": test_date})
            result = JsonBytes.loads(json_str)
            assert result["date"] == test_date

    def test_time_edge_cases(self):
        """Test encoding and decoding time edge cases."""
        edge_times = [
            time(0, 0, 0),  # Midnight
            time(23, 59, 59, 999999),  # Maximum time
            time(12, 0, 0),  # Noon
            time(0, 0, 0, 1),  # Minimum microsecond
        ]

        for test_time in edge_times:
            json_str = JsonBytes.dumps({"time": test_time})
            result = JsonBytes.loads(json_str)
            assert result["time"] == test_time

    def test_datetime_edge_cases(self):
        """Test encoding and decoding datetime edge cases."""
        edge_datetimes = [
            datetime(1, 1, 1, 0, 0, 0),  # Minimum datetime
            datetime(9999, 12, 31, 23, 59, 59, 999999),  # Maximum datetime
            datetime(2000, 2, 29, 12, 0, 0),  # Leap year datetime
        ]

        for test_datetime in edge_datetimes:
            json_str = JsonBytes.dumps({"datetime": test_datetime})
            result = JsonBytes.loads(json_str)
            assert result["datetime"] == test_datetime

    def test_empty_bytes(self):
        """Test encoding and decoding empty bytes."""
        empty_data = b""
        json_str = JsonBytes.dumps({"empty": empty_data})
        result = JsonBytes.loads(json_str)
        assert result["empty"] == empty_data

    def test_large_bytes(self):
        """Test encoding and decoding large binary data."""
        large_data = bytes(range(256)) * 100  # 25.6 KB of data
        json_str = JsonBytes.dumps({"large": large_data})
        result = JsonBytes.loads(json_str)
        assert result["large"] == large_data

    def test_invalid_base85_data(self):
        """Test handling of invalid base85 data in object hook."""
        # Create a dict that looks like bytes data but has invalid base85
        invalid_dict = {JsonBytes._BYTES_TAG: "invalid base85 data!@#$"}

        with pytest.raises(ValueError):  # base64.a85decode should raise ValueError
            JsonBytes.bytes_json_object_hook(invalid_dict)

    def test_invalid_compressed_data(self):
        """Test handling of invalid compressed data in object hook."""
        # Create valid base85 data but invalid compressed data
        invalid_compressed = b"not compressed data"
        encoded = base64.a85encode(invalid_compressed).decode("ascii")
        invalid_dict = {JsonBytes._BYTES_TAG: encoded}

        with pytest.raises(zlib.error):  # zlib.decompress should raise error
            JsonBytes.bytes_json_object_hook(invalid_dict)

    def test_invalid_date_format(self):
        """Test handling of invalid date format in object hook."""
        invalid_dict = {JsonBytes._DATE_TAG: "invalid-date-format"}

        with pytest.raises(ValueError):  # date.fromisoformat should raise ValueError
            JsonBytes.bytes_json_object_hook(invalid_dict)

    def test_invalid_time_format(self):
        """Test handling of invalid time format in object hook."""
        invalid_dict = {JsonBytes._TIME_TAG: "invalid-time-format"}

        with pytest.raises(ValueError):  # time.fromisoformat should raise ValueError
            JsonBytes.bytes_json_object_hook(invalid_dict)

    def test_invalid_datetime_format(self):
        """Test handling of invalid datetime format in object hook."""
        invalid_dict = {JsonBytes._DATETIME_TAG: "invalid-datetime-format"}

        with pytest.raises(ValueError):  # datetime.fromisoformat should raise ValueError
            JsonBytes.bytes_json_object_hook(invalid_dict)

    def test_nested_datetime_structures(self):
        """Test complex nested structures with multiple datetime objects."""
        complex_data = {
            "level1": {
                "level2": {
                    "date_array": [date(2025, 1, 1), date(2025, 12, 31)],
                    "mixed": {
                        "text": "hello",
                        "binary": b"nested binary",
                        "time": time(15, 30, 45),
                        "more_nesting": {
                            "deep_datetime": datetime(2025, 6, 15, 10, 30, 0),
                            "deep_date": date(2025, 6, 15),
                        },
                    },
                }
            },
            "top_level_binary": b"top level data",
            "top_level_datetime": datetime(2025, 9, 16, 7, 29, 0),
        }

        json_str = JsonBytes.dumps(complex_data)
        result = JsonBytes.loads(json_str)

        assert result["level1"]["level2"]["date_array"] == [
            date(2025, 1, 1),
            date(2025, 12, 31),
        ]
        assert result["level1"]["level2"]["mixed"]["text"] == "hello"
        assert result["level1"]["level2"]["mixed"]["binary"] == b"nested binary"
        assert result["level1"]["level2"]["mixed"]["time"] == time(15, 30, 45)
        assert (
            result["level1"]["level2"]["mixed"]["more_nesting"]["deep_datetime"]
            == datetime(2025, 6, 15, 10, 30, 0)
        )
        assert (
            result["level1"]["level2"]["mixed"]["more_nesting"]["deep_date"]
            == date(2025, 6, 15)
        )
        assert result["top_level_binary"] == b"top level data"
        assert result["top_level_datetime"] == datetime(2025, 9, 16, 7, 29, 0)

    def test_mixed_array_with_datetime_objects(self):
        """Test arrays containing mixed types including datetime objects."""
        mixed_array = [
            "string",
            42,
            b"binary data",
            date(2025, 1, 1),
            time(12, 0, 0),
            datetime(2025, 1, 1, 12, 0, 0),
            {"nested": date(2025, 12, 31)},
        ]

        json_str = JsonBytes.dumps({"mixed_array": mixed_array})
        result = JsonBytes.loads(json_str)

        assert result["mixed_array"][0] == "string"
        assert result["mixed_array"][1] == 42
        assert result["mixed_array"][2] == b"binary data"
        assert result["mixed_array"][3] == date(2025, 1, 1)
        assert result["mixed_array"][4] == time(12, 0, 0)
        assert result["mixed_array"][5] == datetime(2025, 1, 1, 12, 0, 0)
        assert result["mixed_array"][6]["nested"] == date(2025, 12, 31)

    def test_inheritance_from_default_encoder(self):
        """Test that JsonBytes still supports DefaultJsonEncoder functionality."""
        # Test that JsonBytes can handle types that DefaultJsonEncoder handles
        # but are not specifically handled by JsonBytes (should fall back to parent)
        namespace = SimpleNamespace(a=1, b="test")

        class JsonableObject:
            def __json__(self):
                return {"custom": "value"}

        jsonable = JsonableObject()

        test_data = {
            "namespace": namespace,
            "jsonable": jsonable,
            "bytearray": bytearray(b"test"),
            "memoryview": memoryview(b"test"),
            "date": date(2025, 1, 1),  # This should use JsonBytes special handling
        }

        json_str = JsonBytes.dumps(test_data)
        result = JsonBytes.loads(json_str)

        # namespace should be converted to dict (via DefaultJsonEncoder)
        assert result["namespace"] == {"a": 1, "b": "test"}
        # jsonable should use __json__ method (via DefaultJsonEncoder)
        assert result["jsonable"] == {"custom": "value"}
        # bytearray/memoryview should be base64 encoded (via DefaultJsonEncoder)
        assert isinstance(result["bytearray"], str)
        assert isinstance(result["memoryview"], str)
        # date should be properly decoded back to date object (via JsonBytes)
        assert result["date"] == date(2025, 1, 1)
        assert isinstance(result["date"], date)
