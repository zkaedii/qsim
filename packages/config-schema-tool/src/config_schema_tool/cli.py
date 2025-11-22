#!/usr/bin/env python3
"""
Command-line interface for config-schema-tool.
"""

import argparse
import json
import sys
from pathlib import Path

from . import __version__
from .manager import SchemaManager


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        prog="config-schema-tool",
        description="JSON Schema validation and config generation tool",
    )
    parser.add_argument(
        "--version", "-V",
        action="version",
        version=f"%(prog)s {__version__}"
    )

    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # Validate command
    validate_parser = subparsers.add_parser(
        "validate",
        help="Validate a config file against a schema"
    )
    validate_parser.add_argument(
        "schema",
        help="Path to JSON Schema file"
    )
    validate_parser.add_argument(
        "config",
        help="Path to config file to validate"
    )
    validate_parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Only output errors"
    )

    # Generate command
    generate_parser = subparsers.add_parser(
        "generate",
        help="Generate config from schema"
    )
    generate_parser.add_argument(
        "schema",
        help="Path to JSON Schema file"
    )
    generate_parser.add_argument(
        "--output", "-o",
        help="Output file path (default: stdout)"
    )
    generate_parser.add_argument(
        "--style", "-s",
        choices=["minimal", "complete"],
        default="minimal",
        help="Generation style (default: minimal)"
    )
    generate_parser.add_argument(
        "--env", "-e",
        help="Environment name to include in metadata"
    )
    generate_parser.add_argument(
        "--format", "-f",
        choices=["json", "yaml"],
        default="json",
        help="Output format (default: json)"
    )

    # Generate-all command
    genall_parser = subparsers.add_parser(
        "generate-all",
        help="Generate configs for multiple environments"
    )
    genall_parser.add_argument(
        "schema",
        help="Path to JSON Schema file"
    )
    genall_parser.add_argument(
        "--output-dir", "-o",
        default=".",
        help="Output directory (default: current dir)"
    )
    genall_parser.add_argument(
        "--environments", "-e",
        nargs="+",
        default=["development", "staging", "production"],
        help="Environments to generate (default: development staging production)"
    )
    genall_parser.add_argument(
        "--format", "-f",
        choices=["json", "yaml"],
        default="json",
        help="Output format (default: json)"
    )

    # Info command
    info_parser = subparsers.add_parser(
        "info",
        help="Show schema information"
    )
    info_parser.add_argument(
        "schema",
        help="Path to JSON Schema file"
    )

    # Docs command
    docs_parser = subparsers.add_parser(
        "docs",
        help="Generate documentation from schema"
    )
    docs_parser.add_argument(
        "schema",
        help="Path to JSON Schema file"
    )
    docs_parser.add_argument(
        "--output", "-o",
        help="Output file path (default: stdout)"
    )

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    try:
        if args.command == "validate":
            cmd_validate(args)
        elif args.command == "generate":
            cmd_generate(args)
        elif args.command == "generate-all":
            cmd_generate_all(args)
        elif args.command == "info":
            cmd_info(args)
        elif args.command == "docs":
            cmd_docs(args)
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


def cmd_validate(args):
    """Validate command handler."""
    manager = SchemaManager(args.schema)
    result = manager.validate(args.config)

    if result.valid:
        if not args.quiet:
            print(f"Valid: {args.config}")
        sys.exit(0)
    else:
        print(f"Invalid: {args.config}", file=sys.stderr)
        for i, (error, path) in enumerate(zip(result.errors, result.error_paths)):
            print(f"  [{i+1}] {path}: {error}", file=sys.stderr)
        sys.exit(1)


def cmd_generate(args):
    """Generate command handler."""
    manager = SchemaManager(args.schema)
    config = manager.generate(style=args.style, env=args.env)

    if args.output:
        manager.generator.save(config, args.output, format=args.format)
        print(f"Generated: {args.output}")
    else:
        if args.format == "yaml":
            try:
                import yaml
                print(yaml.dump(config, default_flow_style=False, sort_keys=False))
            except ImportError:
                print("PyYAML required for YAML output", file=sys.stderr)
                sys.exit(1)
        else:
            print(json.dumps(config, indent=2))


def cmd_generate_all(args):
    """Generate-all command handler."""
    manager = SchemaManager(args.schema)
    results = manager.generate_all(
        environments=args.environments,
        output_dir=args.output_dir,
        format=args.format
    )

    for env, path in results.items():
        print(f"Generated: {path}")


def cmd_info(args):
    """Info command handler."""
    manager = SchemaManager(args.schema)
    info = manager.info()

    print(f"Title: {info['title']}")
    print(f"Description: {info['description']}")
    print(f"Version: {info['version']}")
    print(f"Schema Draft: {info['schema_draft']}")
    print(f"Statistics:")
    for key, value in info['stats'].items():
        print(f"  {key}: {value}")


def cmd_docs(args):
    """Docs command handler."""
    manager = SchemaManager(args.schema)
    docs = manager.generate_docs(output_path=args.output)

    if not args.output:
        print(docs)
    else:
        print(f"Generated: {args.output}")


if __name__ == "__main__":
    main()
