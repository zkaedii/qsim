"""
Core validation functionality using JSON Schema.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass

try:
    from jsonschema import validate, ValidationError, Draft202012Validator
    JSONSCHEMA_AVAILABLE = True
except ImportError:
    JSONSCHEMA_AVAILABLE = False
    ValidationError = Exception


@dataclass
class ValidationResult:
    """Result of a validation operation."""
    valid: bool
    errors: List[str]
    error_paths: List[str]

    def __bool__(self) -> bool:
        return self.valid

    def __str__(self) -> str:
        if self.valid:
            return "Valid"
        return f"Invalid: {len(self.errors)} error(s)"


class ConfigValidator:
    """
    Validates configuration files against JSON Schema.

    Example:
        validator = ConfigValidator("schema.json")
        result = validator.validate({"key": "value"})
        if result:
            print("Config is valid!")
        else:
            for error in result.errors:
                print(f"Error: {error}")
    """

    def __init__(self, schema: Union[str, Path, Dict[str, Any]]):
        """
        Initialize validator with a schema.

        Args:
            schema: Path to schema file, or schema dict directly
        """
        if not JSONSCHEMA_AVAILABLE:
            raise ImportError(
                "jsonschema is required. Install with: pip install jsonschema"
            )

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

    def validate(self, config: Union[str, Path, Dict[str, Any]]) -> ValidationResult:
        """
        Validate a configuration against the schema.

        Args:
            config: Path to config file, or config dict directly

        Returns:
            ValidationResult with valid status and any errors
        """
        if isinstance(config, (str, Path)):
            config = self._load_config(Path(config))

        errors = []
        error_paths = []

        try:
            validate(instance=config, schema=self.schema)
            return ValidationResult(valid=True, errors=[], error_paths=[])
        except ValidationError as e:
            # Collect all errors
            validator = Draft202012Validator(self.schema)
            for error in validator.iter_errors(config):
                path = " -> ".join(str(p) for p in error.absolute_path) or "root"
                errors.append(error.message)
                error_paths.append(path)

            return ValidationResult(valid=False, errors=errors, error_paths=error_paths)

    def _load_config(self, path: Path) -> Dict[str, Any]:
        """Load config from file (JSON or YAML)."""
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")

        suffix = path.suffix.lower()

        if suffix in [".yaml", ".yml"]:
            try:
                import yaml
                with open(path, "r") as f:
                    return yaml.safe_load(f)
            except ImportError:
                raise ImportError(
                    "PyYAML is required for YAML files. Install with: pip install pyyaml"
                )
        else:
            with open(path, "r") as f:
                return json.load(f)

    def validate_file(self, config_path: Union[str, Path]) -> ValidationResult:
        """Convenience method to validate a config file."""
        return self.validate(Path(config_path))

    def is_valid(self, config: Union[str, Path, Dict[str, Any]]) -> bool:
        """Quick check if config is valid."""
        return self.validate(config).valid
