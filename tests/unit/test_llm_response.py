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

import pytest

from generative_ai_toolkit.utils.llm_response import get_text, json_parse


class TestGetText:
    """Test cases for get_text function."""

    def test_get_text_success(self):
        """Test successful text extraction from response."""
        response = {
            "output": {
                "message": {
                    "content": [
                        {"text": "Hello, World!"}
                    ]
                }
            }
        }
        result = get_text(response)
        assert result == "Hello, World!"

    def test_get_text_multiple_content_items(self):
        """Test text extraction when there are multiple content items."""
        response = {
            "output": {
                "message": {
                    "content": [
                        {"image": "some_image_data"},
                        {"text": "This is the text content"}
                    ]
                }
            }
        }
        result = get_text(response)
        assert result == "This is the text content"

    def test_get_text_no_message(self):
        """Test error when no message in response."""
        response = {
            "output": {}
        }
        with pytest.raises(ValueError, match="No text found in response"):
            get_text(response)

    def test_get_text_no_text_content(self):
        """Test error when message has no text content."""
        response = {
            "output": {
                "message": {
                    "content": [
                        {"image": "some_image_data"},
                        {"audio": "some_audio_data"}
                    ]
                }
            }
        }
        with pytest.raises(ValueError, match="No text found in response"):
            get_text(response)

    def test_get_text_empty_content(self):
        """Test error when content array is empty."""
        response = {
            "output": {
                "message": {
                    "content": []
                }
            }
        }
        with pytest.raises(ValueError, match="No text found in response"):
            get_text(response)


class TestJsonParse:
    """Test cases for json_parse function."""

    def test_json_parse_simple_json(self):
        """Test parsing simple JSON response."""
        response = {
            "output": {
                "message": {
                    "content": [
                        {"text": '{"name": "John", "age": 30}'}
                    ]
                }
            }
        }
        result = json_parse(response)
        assert result == {"name": "John", "age": 30}

    def test_json_parse_with_markdown_json_block(self):
        """Test parsing JSON wrapped in markdown code block."""
        response = {
            "output": {
                "message": {
                    "content": [
                        {"text": '```json\n{"status": "success", "data": [1, 2, 3]}\n```'}
                    ]
                }
            }
        }
        result = json_parse(response)
        assert result == {"status": "success", "data": [1, 2, 3]}

    def test_json_parse_with_generic_code_block(self):
        """Test parsing JSON wrapped in generic code block."""
        response = {
            "output": {
                "message": {
                    "content": [
                        {"text": '```\n{"message": "hello", "count": 5}\n```'}
                    ]
                }
            }
        }
        result = json_parse(response)
        assert result == {"message": "hello", "count": 5}

    def test_json_parse_with_language_code_block(self):
        """Test parsing JSON wrapped in code block with different language."""
        response = {
            "output": {
                "message": {
                    "content": [
                        {"text": '```javascript\n{"type": "object", "valid": true}\n```'}
                    ]
                }
            }
        }
        result = json_parse(response)
        assert result == {"type": "object", "valid": True}

    def test_json_parse_with_newlines_in_json(self):
        """Test parsing JSON with newlines that get replaced with spaces."""
        response = {
            "output": {
                "message": {
                    "content": [
                        {"text": '{\n  "name": "Alice",\n  "items": [\n    "apple",\n    "banana"\n  ]\n}'}
                    ]
                }
            }
        }
        result = json_parse(response)
        assert result == {"name": "Alice", "items": ["apple", "banana"]}

    def test_json_parse_with_whitespace(self):
        """Test parsing JSON with leading/trailing whitespace."""
        response = {
            "output": {
                "message": {
                    "content": [
                        {"text": '   \n  {"clean": "data"}  \n   '}
                    ]
                }
            }
        }
        result = json_parse(response)
        assert result == {"clean": "data"}

    def test_json_parse_markdown_block_with_whitespace(self):
        """Test parsing markdown JSON block with extra whitespace."""
        response = {
            "output": {
                "message": {
                    "content": [
                        {"text": '```json\n\n  {"formatted": "nicely"}  \n\n```'}
                    ]
                }
            }
        }
        result = json_parse(response)
        assert result == {"formatted": "nicely"}

    def test_json_parse_incomplete_markdown_block(self):
        """Test parsing JSON with incomplete markdown block (missing closing)."""
        response = {
            "output": {
                "message": {
                    "content": [
                        {"text": '```json\n{"incomplete": "block"}'}
                    ]
                }
            }
        }
        # When there's no closing ```, the text remains unchanged and should fail to parse
        with pytest.raises(Exception, match="Could not JSON parse response"):
            json_parse(response)

    def test_json_parse_only_opening_backticks(self):
        """Test parsing with only opening backticks."""
        response = {
            "output": {
                "message": {
                    "content": [
                        {"text": '```\n{"only": "opening"}'}
                    ]
                }
            }
        }
        # When there's only one set of backticks, the count condition fails and text remains unchanged
        with pytest.raises(Exception, match="Could not JSON parse response"):
            json_parse(response)

    def test_json_parse_invalid_json(self):
        """Test error handling for invalid JSON."""
        response = {
            "output": {
                "message": {
                    "content": [
                        {"text": '{"invalid": json, missing quotes}'}
                    ]
                }
            }
        }
        with pytest.raises(Exception, match="Could not JSON parse response"):
            json_parse(response)

    def test_json_parse_empty_string(self):
        """Test error handling for empty string."""
        response = {
            "output": {
                "message": {
                    "content": [
                        {"text": ""}
                    ]
                }
            }
        }
        with pytest.raises(Exception, match="Could not JSON parse response"):
            json_parse(response)

    def test_json_parse_non_json_text(self):
        """Test error handling for non-JSON text."""
        response = {
            "output": {
                "message": {
                    "content": [
                        {"text": "This is just plain text, not JSON"}
                    ]
                }
            }
        }
        with pytest.raises(Exception, match="Could not JSON parse response"):
            json_parse(response)

    def test_json_parse_nested_objects(self):
        """Test parsing complex nested JSON objects."""
        complex_json = {
            "users": [
                {"id": 1, "name": "Alice", "settings": {"theme": "dark"}},
                {"id": 2, "name": "Bob", "settings": {"theme": "light"}}
            ],
            "metadata": {
                "total": 2,
                "page": 1,
                "filters": None
            }
        }
        response = {
            "output": {
                "message": {
                    "content": [
                        {"text": json.dumps(complex_json)}
                    ]
                }
            }
        }
        result = json_parse(response)
        assert result == complex_json

    def test_json_parse_array_response(self):
        """Test parsing JSON array response."""
        response = {
            "output": {
                "message": {
                    "content": [
                        {"text": '[{"id": 1}, {"id": 2}, {"id": 3}]'}
                    ]
                }
            }
        }
        result = json_parse(response)
        assert result == [{"id": 1}, {"id": 2}, {"id": 3}]

    def test_json_parse_boolean_and_null_values(self):
        """Test parsing JSON with boolean and null values."""
        response = {
            "output": {
                "message": {
                    "content": [
                        {"text": '{"active": true, "deleted": false, "data": null}'}
                    ]
                }
            }
        }
        result = json_parse(response)
        assert result == {"active": True, "deleted": False, "data": None}

    def test_json_parse_numeric_values(self):
        """Test parsing JSON with various numeric values."""
        response = {
            "output": {
                "message": {
                    "content": [
                        {"text": '{"int": 42, "float": 3.14, "negative": -10, "zero": 0}'}
                    ]
                }
            }
        }
        result = json_parse(response)
        assert result == {"int": 42, "float": 3.14, "negative": -10, "zero": 0}

    def test_json_parse_multiple_code_blocks(self):
        """Test parsing when there are multiple code blocks."""
        response = {
            "output": {
                "message": {
                    "content": [
                        {"text": '```json\n{"first": "block"}\n```\nSome text\n```\n{"second": "block"}\n```'}
                    ]
                }
            }
        }
        # The function extracts everything between ```json and the last ```, which includes middle content
        # This results in invalid JSON, so it should raise an exception
        with pytest.raises(Exception, match="Could not JSON parse response"):
            json_parse(response)

    def test_json_parse_preserves_original_exception(self):
        """Test that original JSONDecodeError is preserved in exception chain."""
        response = {
            "output": {
                "message": {
                    "content": [
                        {"text": '{"malformed": json}'}
                    ]
                }
            }
        }
        with pytest.raises(Exception) as exc_info:
            json_parse(response)

        assert "Could not JSON parse response" in str(exc_info.value)
        assert exc_info.value.__cause__ is not None
        assert isinstance(exc_info.value.__cause__, json.JSONDecodeError)
    def test_json_parse_incomplete_json_block_with_closing(self):
        """Test parsing JSON block that has closing backticks but incomplete JSON."""
        response = {
            "output": {
                "message": {
                    "content": [
                        {"text": '```json\n{"incomplete": json\n```'}
                    ]
                }
            }
        }
        with pytest.raises(Exception, match="Could not JSON parse response"):
            json_parse(response)

    def test_json_parse_valid_json_with_extra_backticks(self):
        """Test parsing valid JSON that happens to contain backticks in the content."""
        response = {
            "output": {
                "message": {
                    "content": [
                        {"text": '{"code": "```python\\nprint(\\"hello\\")\\n```"}'}
                    ]
                }
            }
        }
        result = json_parse(response)
        assert result == {"code": "```python\nprint(\"hello\")\n```"}

    def test_json_parse_generic_code_block_valid(self):
        """Test parsing JSON in a generic code block that works correctly."""
        response = {
            "output": {
                "message": {
                    "content": [
                        {"text": '```\n{"valid": "json"}\n```'}
                    ]
                }
            }
        }
        result = json_parse(response)
        assert result == {"valid": "json"}
