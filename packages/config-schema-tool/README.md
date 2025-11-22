# config-schema-tool

A lightweight, practical tool for validating configurations against JSON Schema and generating environment-specific configs.

## Features

- **Validate** config files (JSON/YAML) against JSON Schema
- **Generate** configs from schema (minimal or complete)
- **Multi-environment** config generation (dev/staging/prod)
- **CLI** for CI/CD pipelines
- **Python API** for programmatic use

## Installation

```bash
pip install config-schema-tool

# With YAML support
pip install config-schema-tool[yaml]
```

## Quick Start

### CLI Usage

```bash
# Validate a config
config-schema-tool validate schema.json config.json

# Generate minimal config
config-schema-tool generate schema.json -o config.json

# Generate for all environments
config-schema-tool generate-all schema.json -o configs/

# Show schema info
config-schema-tool info schema.json

# Generate documentation
config-schema-tool docs schema.json -o SCHEMA.md
```

### Python API

```python
from config_schema_tool import SchemaManager

# Initialize with schema
manager = SchemaManager("schema.json")

# Validate config
result = manager.validate("config.json")
if result:
    print("Config is valid!")
else:
    for error in result.errors:
        print(f"Error: {error}")

# Generate config
config = manager.generate(style="minimal", env="production")

# Generate all environments
manager.generate_all(
    environments=["dev", "staging", "prod"],
    output_dir="configs/"
)

# Get schema info
print(manager.info())
```

### Using Individual Components

```python
from config_schema_tool import ConfigValidator, ConfigGenerator

# Just validation
validator = ConfigValidator("schema.json")
result = validator.validate({"key": "value"})

# Just generation
generator = ConfigGenerator("schema.json")
config = generator.generate(style="complete")
generator.save(config, "config.json")
```

## Example Schema

```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "title": "App Config",
  "type": "object",
  "required": ["app", "database"],
  "properties": {
    "app": {
      "type": "object",
      "required": ["name", "port"],
      "properties": {
        "name": {"type": "string"},
        "port": {"type": "integer", "default": 8080},
        "debug": {"type": "boolean", "default": false}
      }
    },
    "database": {
      "type": "object",
      "required": ["host"],
      "properties": {
        "host": {"type": "string"},
        "port": {"type": "integer", "default": 5432},
        "pool_size": {"type": "integer", "default": 10}
      }
    }
  }
}
```

## CI/CD Integration

### GitHub Actions

```yaml
- name: Validate config
  run: |
    pip install config-schema-tool
    config-schema-tool validate schema.json config.prod.json
```

### Pre-commit Hook

```yaml
# .pre-commit-config.yaml
repos:
  - repo: local
    hooks:
      - id: validate-config
        name: Validate config
        entry: config-schema-tool validate schema.json config.json
        language: system
        pass_filenames: false
```

## Why This Tool?

| Feature | config-schema-tool | Manual validation |
|---------|-------------------|-------------------|
| Schema validation | JSON Schema Draft 2020-12 | Custom code |
| Config generation | Automatic from schema | Manual templates |
| Multi-environment | Built-in | Script per env |
| CLI | Included | Build your own |
| CI/CD ready | Single command | Multiple steps |

## License

MIT
