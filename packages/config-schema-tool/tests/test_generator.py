"""Tests for ConfigGenerator."""

import pytest
import json
import tempfile
from pathlib import Path
from config_schema_tool import ConfigGenerator


@pytest.fixture
def simple_schema():
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "type": "object",
        "required": ["name", "port"],
        "properties": {
            "name": {"type": "string", "default": "myapp"},
            "port": {"type": "integer", "default": 8080},
            "debug": {"type": "boolean", "default": False},
            "tags": {
                "type": "array",
                "items": {"type": "string"},
                "default": ["api"]
            }
        }
    }


@pytest.fixture
def generator(simple_schema):
    return ConfigGenerator(simple_schema)


class TestConfigGenerator:
    def test_generate_minimal(self, generator):
        config = generator.generate(style="minimal")
        assert "name" in config
        assert "port" in config
        # Optional fields may or may not be present in minimal

    def test_generate_complete(self, generator):
        config = generator.generate(style="complete")
        assert "name" in config
        assert "port" in config
        assert "debug" in config
        assert "tags" in config

    def test_generate_with_env(self, generator):
        config = generator.generate(env="production")
        assert config.get("_metadata", {}).get("environment") == "production"

    def test_generate_with_overrides(self, generator):
        config = generator.generate(overrides={"name": "custom-app"})
        assert config["name"] == "custom-app"

    def test_save_json(self, generator):
        config = generator.generate()
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "config.json"
            generator.save(config, output_path, format="json")

            assert output_path.exists()
            loaded = json.loads(output_path.read_text())
            assert loaded["name"] == config["name"]

    def test_generate_environments(self, generator):
        with tempfile.TemporaryDirectory() as tmpdir:
            results = generator.generate_environments(
                environments=["dev", "prod"],
                output_dir=tmpdir
            )

            assert len(results) == 2
            assert "dev" in results
            assert "prod" in results
            assert Path(results["dev"]).exists()
            assert Path(results["prod"]).exists()


class TestSchemaTypes:
    def test_string_format_datetime(self):
        schema = {
            "type": "object",
            "properties": {
                "created": {"type": "string", "format": "date-time"}
            }
        }
        generator = ConfigGenerator(schema)
        config = generator.generate(style="complete")
        assert "created" in config

    def test_enum_values(self):
        schema = {
            "type": "object",
            "properties": {
                "level": {"type": "string", "enum": ["low", "medium", "high"]}
            }
        }
        generator = ConfigGenerator(schema)
        config = generator.generate(style="complete")
        assert config.get("level") == "low"  # First enum value

    def test_nested_objects(self):
        schema = {
            "type": "object",
            "required": ["database"],
            "properties": {
                "database": {
                    "type": "object",
                    "required": ["host"],
                    "properties": {
                        "host": {"type": "string", "default": "localhost"},
                        "port": {"type": "integer", "default": 5432}
                    }
                }
            }
        }
        generator = ConfigGenerator(schema)
        config = generator.generate(style="minimal")
        assert "database" in config
        assert "host" in config["database"]
