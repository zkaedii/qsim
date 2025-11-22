"""
Configuration generator from JSON Schema.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import uuid


class ConfigGenerator:
    """
    Generates configuration files from JSON Schema.

    Supports multiple environments and can generate minimal or complete configs.

    Example:
        generator = ConfigGenerator("schema.json")

        # Generate minimal config
        config = generator.generate(style="minimal")

        # Generate for specific environment
        config = generator.generate(env="production")

        # Save to file
        generator.save(config, "config.json")
    """

    def __init__(self, schema: Union[str, Path, Dict[str, Any]]):
        """
        Initialize generator with a schema.

        Args:
            schema: Path to schema file, or schema dict directly
        """
        if isinstance(schema, (str, Path)):
            self.schema = self._load_schema(Path(schema))
            self.schema_path = Path(schema)
        else:
            self.schema = schema
            self.schema_path = None

    def _load_schema(self, path: Path) -> Dict[str, Any]:
        """Load schema from file."""
        if not path.exists():
            raise FileNotFoundError(f"Schema file not found: {path}")

        with open(path, "r") as f:
            return json.load(f)

    def generate(
        self,
        style: str = "minimal",
        env: Optional[str] = None,
        overrides: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Generate a configuration from the schema.

        Args:
            style: "minimal" (required fields only) or "complete" (all fields)
            env: Environment name to include in metadata
            overrides: Dict of values to override in generated config

        Returns:
            Generated configuration dict
        """
        if style == "minimal":
            config = self._generate_minimal()
        else:
            config = self._generate_complete()

        # Add metadata
        if "metadata" not in config:
            config["_metadata"] = {}

        metadata_key = "metadata" if "metadata" in config else "_metadata"
        config[metadata_key]["generated_at"] = datetime.now().isoformat()
        config[metadata_key]["generator"] = "config-schema-tool"

        if env:
            config[metadata_key]["environment"] = env

        # Apply overrides
        if overrides:
            config = self._deep_merge(config, overrides)

        return config

    def _generate_minimal(self) -> Dict[str, Any]:
        """Generate config with only required fields."""
        return self._generate_from_schema(self.schema, required_only=True)

    def _generate_complete(self) -> Dict[str, Any]:
        """Generate config with all fields."""
        return self._generate_from_schema(self.schema, required_only=False)

    def _generate_from_schema(
        self,
        schema: Dict[str, Any],
        required_only: bool = True
    ) -> Any:
        """Recursively generate config from schema."""
        schema_type = schema.get("type", "object")

        # Handle default values
        if "default" in schema:
            return schema["default"]

        # Handle examples
        if "examples" in schema and schema["examples"]:
            return schema["examples"][0]

        # Handle enums
        if "enum" in schema:
            return schema["enum"][0]

        # Handle const
        if "const" in schema:
            return schema["const"]

        if schema_type == "object":
            result = {}
            properties = schema.get("properties", {})
            required = set(schema.get("required", []))

            for prop_name, prop_schema in properties.items():
                if required_only and prop_name not in required:
                    continue
                result[prop_name] = self._generate_from_schema(
                    prop_schema, required_only
                )

            return result

        elif schema_type == "array":
            items_schema = schema.get("items", {})
            min_items = schema.get("minItems", 0)

            if min_items > 0 or not required_only:
                return [self._generate_from_schema(items_schema, required_only)]
            return []

        elif schema_type == "string":
            if "format" in schema:
                return self._generate_string_format(schema["format"])
            return schema.get("default", "")

        elif schema_type == "integer":
            return schema.get("minimum", schema.get("default", 0))

        elif schema_type == "number":
            return schema.get("minimum", schema.get("default", 0.0))

        elif schema_type == "boolean":
            return schema.get("default", False)

        elif schema_type == "null":
            return None

        return None

    def _generate_string_format(self, format_type: str) -> str:
        """Generate string based on format."""
        formats = {
            "date-time": datetime.now().isoformat(),
            "date": datetime.now().date().isoformat(),
            "time": datetime.now().time().isoformat(),
            "email": "user@example.com",
            "uri": "https://example.com",
            "uuid": str(uuid.uuid4()),
            "hostname": "localhost",
            "ipv4": "127.0.0.1",
            "ipv6": "::1",
        }
        return formats.get(format_type, "")

    def _deep_merge(
        self,
        base: Dict[str, Any],
        override: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Deep merge two dicts."""
        result = base.copy()

        for key, value in override.items():
            if (
                key in result
                and isinstance(result[key], dict)
                and isinstance(value, dict)
            ):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = value

        return result

    def save(
        self,
        config: Dict[str, Any],
        path: Union[str, Path],
        format: str = "json"
    ) -> None:
        """
        Save configuration to file.

        Args:
            config: Configuration dict to save
            path: Output file path
            format: "json" or "yaml"
        """
        path = Path(path)

        if format == "yaml":
            try:
                import yaml
                with open(path, "w") as f:
                    yaml.dump(config, f, default_flow_style=False, sort_keys=False)
            except ImportError:
                raise ImportError(
                    "PyYAML is required for YAML output. Install with: pip install pyyaml"
                )
        else:
            with open(path, "w") as f:
                json.dump(config, f, indent=2)

    def generate_environments(
        self,
        environments: List[str] = None,
        output_dir: Union[str, Path] = ".",
        format: str = "json"
    ) -> Dict[str, Path]:
        """
        Generate configs for multiple environments.

        Args:
            environments: List of environment names (default: dev, staging, prod)
            output_dir: Directory to save configs
            format: Output format ("json" or "yaml")

        Returns:
            Dict mapping environment names to output file paths
        """
        if environments is None:
            environments = ["development", "staging", "production"]

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        results = {}

        for env in environments:
            style = "minimal" if env == "development" else "complete"
            config = self.generate(style=style, env=env)

            ext = "yaml" if format == "yaml" else "json"
            output_path = output_dir / f"config.{env}.{ext}"

            self.save(config, output_path, format=format)
            results[env] = output_path

        return results
