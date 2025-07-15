"""
Unit tests for the Environment tool.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from environment_tool import (
    EnvironmentRequest,
    EnvironmentResponse,
    EnvironmentTool,
)


class TestEnvironmentTool:
    """Test cases for the EnvironmentTool class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.tool = EnvironmentTool()
        # Store original env vars to restore later
        self.original_env = dict(os.environ)

    def teardown_method(self):
        """Clean up after tests."""
        # Restore original environment
        os.environ.clear()
        os.environ.update(self.original_env)

    def test_list_all_variables(self):
        """Test listing all environment variables."""
        # Set test variables
        os.environ["TEST_VAR"] = "test_value"
        os.environ["AWS_ACCESS_KEY_ID"] = "secret_key"

        result = self.tool.invoke(action="list")

        assert result["success"] is True
        assert result["action"] == "list"
        assert "variables" in result
        assert "TEST_VAR" in result["variables"]
        assert result["variables"]["TEST_VAR"] == "test_value"
        assert result["variables"]["AWS_ACCESS_KEY_ID"] == "***PROTECTED***"
        assert result["protected_count"] > 0

    def test_list_with_prefix(self):
        """Test listing variables with prefix filter."""
        os.environ["AWS_ACCESS_KEY_ID"] = "secret"
        os.environ["AWS_SECRET_ACCESS_KEY"] = "secret"
        os.environ["OTHER_VAR"] = "value"

        result = self.tool.invoke(action="list", prefix="AWS_")

        assert result["success"] is True
        assert result["action"] == "list"
        assert len(result["variables"]) == 2
        assert "AWS_ACCESS_KEY_ID" in result["variables"]
        assert "AWS_SECRET_ACCESS_KEY" in result["variables"]
        assert "OTHER_VAR" not in result["variables"]
        assert "with prefix 'AWS_'" in result["message"]

    def test_get_existing_variable(self):
        """Test getting an existing variable."""
        os.environ["TEST_VAR"] = "test_value"

        result = self.tool.invoke(action="get", name="TEST_VAR")

        assert result["success"] is True
        assert result["action"] == "get"
        assert result["value"] == "test_value"
        assert "Retrieved variable 'TEST_VAR'" in result["message"]

    def test_get_protected_variable(self):
        """Test getting a protected variable."""
        os.environ["AWS_ACCESS_KEY_ID"] = "secret_key"

        result = self.tool.invoke(action="get", name="AWS_ACCESS_KEY_ID")

        assert result["success"] is True
        assert result["action"] == "get"
        assert result["value"] == "***PROTECTED***"

    def test_get_nonexistent_variable(self):
        """Test getting a non-existent variable."""
        result = self.tool.invoke(action="get", name="NONEXISTENT_VAR")

        assert result["success"] is False
        assert result["action"] == "get"
        assert "not found" in result["error"]

    def test_get_without_name(self):
        """Test get action without providing name."""
        result = self.tool.invoke(action="get")

        assert result["success"] is False
        assert result["action"] == "get"
        assert "Variable name is required" in result["error"]

    def test_set_new_variable(self):
        """Test setting a new variable."""
        result = self.tool.invoke(action="set", name="NEW_VAR", value="new_value")

        assert result["success"] is True
        assert result["action"] == "set"
        assert os.environ["NEW_VAR"] == "new_value"
        assert "Set variable 'NEW_VAR'" in result["message"]

    def test_set_protected_variable_without_force(self):
        """Test setting a protected variable without force flag."""
        result = self.tool.invoke(action="set", name="PATH", value="new_path")

        assert result["success"] is False
        assert result["action"] == "set"
        assert "is protected" in result["error"]
        assert "force=True" in result["error"]

    def test_set_protected_variable_with_force(self):
        """Test setting a protected variable with force flag."""

        result = self.tool.invoke(
            action="set", name="PATH", value="new_path", force=True
        )

        assert result["success"] is True
        assert result["action"] == "set"
        assert os.environ["PATH"] == "new_path"

    def test_set_without_name_or_value(self):
        """Test set action without name or value."""
        result = self.tool.invoke(action="set")

        assert result["success"] is False
        assert result["action"] == "set"
        assert "Both name and value are required" in result["error"]

    def test_delete_existing_variable(self):
        """Test deleting an existing variable."""
        os.environ["TEMP_VAR"] = "temp_value"

        result = self.tool.invoke(action="delete", name="TEMP_VAR")

        assert result["success"] is True
        assert result["action"] == "delete"
        assert "TEMP_VAR" not in os.environ
        assert "Deleted variable 'TEMP_VAR'" in result["message"]

    def test_delete_nonexistent_variable(self):
        """Test deleting a non-existent variable."""
        result = self.tool.invoke(action="delete", name="NONEXISTENT_VAR")

        assert result["success"] is False
        assert result["action"] == "delete"
        assert "not found" in result["error"]

    def test_delete_protected_variable_without_force(self):
        """Test deleting a protected variable without force flag."""
        result = self.tool.invoke(action="delete", name="PATH")

        assert result["success"] is False
        assert result["action"] == "delete"
        assert "is protected" in result["error"]
        assert "force=True" in result["error"]

    def test_delete_protected_variable_with_force(self):
        """Test deleting a protected variable with force flag."""
        os.environ["HOME"] = "/test/home"

        result = self.tool.invoke(action="delete", name="HOME", force=True)

        assert result["success"] is True
        assert result["action"] == "delete"
        assert "HOME" not in os.environ

    def test_delete_without_name(self):
        """Test delete action without providing name."""
        result = self.tool.invoke(action="delete")

        assert result["success"] is False
        assert result["action"] == "delete"
        assert "Variable name is required" in result["error"]

    def test_invalid_action(self):
        """Test invalid action handling."""
        result = self.tool.invoke(action="invalid")

        assert result["success"] is False
        assert result["action"] == "invalid"
        assert "Invalid request parameters" in result["error"]
        assert "validation error" in result["error"]

    def test_tool_spec(self):
        """Test tool specification generation."""
        spec = self.tool.tool_spec

        assert spec["name"] == "environment_operation"
        assert "description" in spec
        assert "inputSchema" in spec
        assert "json" in spec["inputSchema"]

    def test_request_validation(self):
        """Test request model validation."""
        # Valid request
        request = EnvironmentRequest(action="list")
        assert request.action == "list"

        # Request with all fields
        request = EnvironmentRequest(
            action="set",
            name="TEST_VAR",
            value="test_value",
            prefix="TEST_",
            force=True,
        )
        assert request.name == "TEST_VAR"
        assert request.value == "test_value"
        assert request.force is True

    def test_response_model(self):
        """Test response model structure."""
        response = EnvironmentResponse(
            success=True,
            action="list",
            variables={"TEST_VAR": "test_value"},
            protected_count=1,
            message="Test message",
        )

        assert response.success is True
        assert response.action == "list"
        assert response.variables["TEST_VAR"] == "test_value"
        assert response.protected_count == 1

    def test_sensitive_variable_detection(self):
        """Test detection of sensitive variables."""
        # Test with variables that are actually protected by the tool
        sensitive_vars = ["AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY"]

        for var in sensitive_vars:
            os.environ[var] = "sensitive_value"

        result = self.tool.invoke(action="list")

        for var in sensitive_vars:
            assert result["variables"][var] == "***PROTECTED***"

    def test_protected_variables_list(self):
        """Test that protected variables are properly identified."""
        protected_vars = ["PATH", "HOME", "USER", "AWS_ACCESS_KEY_ID"]

        for var in protected_vars:
            if var in os.environ:
                result = self.tool.invoke(action="get", name=var)
                assert result["value"] == "***PROTECTED***"

    def test_processing_time_tracking(self):
        """Test that processing time is tracked."""
        result = self.tool.invoke(action="list")

        assert "processing_time_ms" in result
        assert isinstance(result["processing_time_ms"], int)
        assert result["processing_time_ms"] >= 0

    def test_error_handling_invalid_params(self):
        """Test error handling for invalid parameters."""
        # Invalid action pattern
        result = self.tool.invoke(action="INVALID_ACTION")

        assert result["success"] is False
        assert "Invalid request parameters" in result["error"]

    def test_empty_environment(self):
        """Test behavior with empty environment."""
        # Clear environment for this test
        os.environ.clear()

        result = self.tool.invoke(action="list")

        assert result["success"] is True
        assert result["variables"] == {}
        assert result["protected_count"] == 0

    def test_case_sensitivity(self):
        """Test case sensitivity of variable names."""
        os.environ["test_var"] = "lowercase"
        os.environ["TEST_VAR"] = "uppercase"

        result = self.tool.invoke(action="get", name="test_var")
        assert result["value"] == "lowercase"

        result = self.tool.invoke(action="get", name="TEST_VAR")
        assert result["value"] == "uppercase"

    def test_special_characters_in_values(self):
        """Test handling of special characters in variable values."""
        special_value = "value with spaces & symbols!@#$%^&*()"
        os.environ["SPECIAL_VAR"] = special_value

        result = self.tool.invoke(action="get", name="SPECIAL_VAR")

        assert result["success"] is True
        assert result["value"] == special_value

    def test_unicode_handling(self):
        """Test handling of unicode characters."""
        unicode_value = "测试值 with émojis 🚀"
        os.environ["UNICODE_VAR"] = unicode_value

        result = self.tool.invoke(action="get", name="UNICODE_VAR")

        assert result["success"] is True
        assert result["value"] == unicode_value
