import json

schema = json.load(open("h_model_z_complete_schema.json"))
print("🏆 H_MODEL_Z SCHEMA OVERVIEW 🏆")
print(f'📦 Properties: {len(schema.get("properties", {}))}')
print(f'📋 Required: {len(schema.get("required", []))}')
print(f'📖 Definitions: {len(schema.get("definitions", {}))}')
print("\n📋 MAIN SECTIONS:")
[print(f"  {i+1}. {section}") for i, section in enumerate(schema.get("properties", {}).keys())]
print("\n🎯 Schema successfully loaded and analyzed!")
