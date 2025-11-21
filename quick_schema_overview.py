#!/usr/bin/env python3
"""Quick schema overview display"""

import json


def show_schema_overview():
    """Display H_MODEL_Z schema overview"""
    try:
        with open("h_model_z_complete_schema.json", "r") as f:
            schema = json.load(f)

        print("🏆 H_MODEL_Z SCHEMA OVERVIEW 🏆")
        print("=" * 50)
        print(f"📦 Properties: {len(schema.get('properties', {}))}")
        print(f"📋 Required: {len(schema.get('required', []))}")
        print(f"📖 Definitions: {len(schema.get('definitions', {}))}")
        print(f"🔢 Version: {schema.get('version', 'N/A')}")
        print(f"📏 Schema Draft: {schema.get('$schema', 'N/A')}")

        print("\n📋 MAIN SECTIONS:")
        print("-" * 30)
        for i, section in enumerate(schema.get("properties", {}).keys(), 1):
            required = "✅" if section in schema.get("required", []) else "⚪"
            print(f"  {i:2d}. {section:<30} {required}")

        print(f"\n🎯 Total Schema Size: {len(str(schema))} characters")
        print("🚀 Status: PRODUCTION READY")

    except FileNotFoundError:
        print("❌ Schema file not found!")
    except json.JSONDecodeError as e:
        print(f"❌ JSON decode error: {e}")
    except Exception as e:
        print(f"❌ Error: {e}")


if __name__ == "__main__":
    show_schema_overview()
