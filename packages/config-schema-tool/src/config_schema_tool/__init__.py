"""
config-schema-tool: JSON Schema validation and config generation for any project.

A lightweight, practical tool for validating configurations against JSON Schema
and generating environment-specific configs.
"""

from .validator import ConfigValidator
from .generator import ConfigGenerator
from .manager import SchemaManager

__version__ = "0.1.0"
__all__ = ["ConfigValidator", "ConfigGenerator", "SchemaManager"]
