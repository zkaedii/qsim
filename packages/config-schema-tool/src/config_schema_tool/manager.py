"""
High-level schema manager combining validation and generation.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from .validator import ConfigValidator, ValidationResult
from .generator import ConfigGenerator


class SchemaManager:
    """
    High-level interface for schema validation and config generation.

    Combines ConfigValidator and ConfigGenerator into a single interface.

    Example:
        manager = SchemaManager("app.schema.json")

        # Validate existing config
        if manager.validate("config.json"):
            print("Config OK!")

        # Generate new configs
        manager.generate_all(output_dir="configs/")

        # Get schema info
        print(manager.info())
    """

    def __init__(self, schema: Union[str, Path, Dict[str, Any]]):
        """
        Initialize with a schema.

        Args:
            schema: Path to schema file, or schema dict directly
        """
        if isinstance(schema, (str, Path)):
            self.schema_path = Path(schema)
            with open(self.schema_path, "r") as f:
                self.schema = json.load(f)
        else:
            self.schema = schema
            self.schema_path = None

        self.validator = ConfigValidator(self.schema)
        self.generator = ConfigGenerator(self.schema)

    def validate(self, config: Union[str, Path, Dict[str, Any]]) -> ValidationResult:
        """Validate a configuration."""
        return self.validator.validate(config)

    def is_valid(self, config: Union[str, Path, Dict[str, Any]]) -> bool:
        """Quick validity check."""
        return self.validator.is_valid(config)

    def generate(
        self,
        style: str = "minimal",
        env: Optional[str] = None,
        overrides: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Generate a configuration."""
        return self.generator.generate(style=style, env=env, overrides=overrides)

    def generate_all(
        self,
        environments: List[str] = None,
        output_dir: Union[str, Path] = ".",
        format: str = "json"
    ) -> Dict[str, Path]:
        """Generate configs for all environments."""
        return self.generator.generate_environments(
            environments=environments,
            output_dir=output_dir,
            format=format
        )

    def info(self) -> Dict[str, Any]:
        """Get schema information and statistics."""
        return {
            "title": self.schema.get("title", "Untitled Schema"),
            "description": self.schema.get("description", ""),
            "version": self.schema.get("version", "unknown"),
            "schema_draft": self.schema.get("$schema", "unknown"),
            "stats": self._calculate_stats()
        }

    def _calculate_stats(self) -> Dict[str, int]:
        """Calculate schema statistics."""
        stats = {
            "total_properties": 0,
            "required_properties": 0,
            "definitions": 0,
            "enums": 0,
        }

        def count_recursive(obj: Any) -> None:
            if isinstance(obj, dict):
                if "properties" in obj:
                    stats["total_properties"] += len(obj["properties"])
                if "required" in obj:
                    stats["required_properties"] += len(obj["required"])
                if "enum" in obj:
                    stats["enums"] += 1
                if "$defs" in obj or "definitions" in obj:
                    defs = obj.get("$defs", obj.get("definitions", {}))
                    stats["definitions"] += len(defs)

                for value in obj.values():
                    count_recursive(value)
            elif isinstance(obj, list):
                for item in obj:
                    count_recursive(item)

        count_recursive(self.schema)
        return stats

    def generate_docs(self, output_path: Union[str, Path] = None) -> str:
        """
        Generate markdown documentation for the schema.

        Args:
            output_path: Optional path to save documentation

        Returns:
            Markdown documentation string
        """
        info = self.info()

        doc = f"""# {info['title']}

{info['description']}

## Schema Info

| Property | Value |
|----------|-------|
| Version | {info['version']} |
| Draft | {info['schema_draft']} |
| Properties | {info['stats']['total_properties']} |
| Required | {info['stats']['required_properties']} |

## Properties

"""
        # Document top-level properties
        properties = self.schema.get("properties", {})
        required = set(self.schema.get("required", []))

        for prop_name, prop_schema in properties.items():
            req_marker = " **(required)**" if prop_name in required else ""
            prop_type = prop_schema.get("type", "any")
            prop_desc = prop_schema.get("description", "No description")

            doc += f"### `{prop_name}`{req_marker}\n\n"
            doc += f"- **Type:** `{prop_type}`\n"
            doc += f"- **Description:** {prop_desc}\n"

            if "default" in prop_schema:
                doc += f"- **Default:** `{prop_schema['default']}`\n"
            if "enum" in prop_schema:
                doc += f"- **Allowed values:** {', '.join(f'`{v}`' for v in prop_schema['enum'])}\n"

            doc += "\n"

        doc += """## Usage

```python
from config_schema_tool import SchemaManager

manager = SchemaManager("schema.json")

# Validate config
result = manager.validate("config.json")
if not result:
    for error in result.errors:
        print(f"Error: {error}")

# Generate config
config = manager.generate(style="minimal", env="production")
```
"""

        if output_path:
            Path(output_path).write_text(doc)

        return doc
