"""Tests for ConfigValidator."""

import pytest
from config_schema_tool import ConfigValidator


@pytest.fixture
def simple_schema():
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "type": "object",
        "required": ["name", "port"],
        "properties": {
            "name": {"type": "string"},
            "port": {"type": "integer", "minimum": 1, "maximum": 65535},
            "debug": {"type": "boolean", "default": False}
        }
    }


@pytest.fixture
def validator(simple_schema):
    return ConfigValidator(simple_schema)


class TestConfigValidator:
    def test_valid_config(self, validator):
        config = {"name": "myapp", "port": 8080}
        result = validator.validate(config)
        assert result.valid is True
        assert len(result.errors) == 0

    def test_valid_config_with_optional(self, validator):
        config = {"name": "myapp", "port": 8080, "debug": True}
        result = validator.validate(config)
        assert result.valid is True

    def test_missing_required_field(self, validator):
        config = {"name": "myapp"}  # missing port
        result = validator.validate(config)
        assert result.valid is False
        assert len(result.errors) > 0

    def test_wrong_type(self, validator):
        config = {"name": "myapp", "port": "not-a-number"}
        result = validator.validate(config)
        assert result.valid is False

    def test_out_of_range(self, validator):
        config = {"name": "myapp", "port": 99999}
        result = validator.validate(config)
        assert result.valid is False

    def test_is_valid_shorthand(self, validator):
        assert validator.is_valid({"name": "app", "port": 80}) is True
        assert validator.is_valid({"name": "app"}) is False

    def test_validation_result_bool(self, validator):
        result = validator.validate({"name": "app", "port": 80})
        assert bool(result) is True

        result = validator.validate({})
        assert bool(result) is False

    def test_validation_result_str(self, validator):
        result = validator.validate({"name": "app", "port": 80})
        assert str(result) == "Valid"

        result = validator.validate({})
        assert "Invalid" in str(result)
