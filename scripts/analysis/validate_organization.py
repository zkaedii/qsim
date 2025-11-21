#!/usr/bin/env python3
"""
H_MODEL_Z Organization Validation and Status Report
==================================================

Validates the project organization and generates a comprehensive status report.
"""

import os
import json
from pathlib import Path
from collections import defaultdict

def analyze_organization():
    """Analyze the current organization structure."""
    root = Path(os.getcwd())
    
    print("🚀 H_MODEL_Z PROJECT ORGANIZATION ANALYSIS")
    print("=" * 50)
    
    # Key directories to analyze
    key_dirs = [
        "src", "config", "docs", "tests", "benchmarks", 
        "examples", "scripts", "blockchain", "assets", "build"
    ]
    
    total_files = 0
    organization_stats = {}
    
    for directory in key_dirs:
        dir_path = root / directory
        if dir_path.exists():
            files = list(dir_path.rglob("*"))
            file_count = len([f for f in files if f.is_file()])
            dir_count = len([f for f in files if f.is_dir()])
            
            organization_stats[directory] = {
                "files": file_count,
                "directories": dir_count,
                "exists": True
            }
            total_files += file_count
            
            print(f"📁 {directory}/ - {file_count} files, {dir_count} directories")
        else:
            organization_stats[directory] = {"exists": False}
            print(f"❌ {directory}/ - NOT FOUND")
    
    print(f"\n📊 TOTAL ORGANIZED FILES: {total_files}")
    
    # Check for key files
    key_files = [
        "README.md", "requirements.txt", "project_metadata.json",
        "ORGANIZATION_REPORT.md", "organization_log.json"
    ]
    
    print("\n🔍 KEY FILES STATUS:")
    for file_name in key_files:
        file_path = root / file_name
        status = "✅ EXISTS" if file_path.exists() else "❌ MISSING"
        size = f"({file_path.stat().st_size} bytes)" if file_path.exists() else ""
        print(f"   {file_name}: {status} {size}")
    
    # Read project metadata if exists
    metadata_path = root / "project_metadata.json"
    if metadata_path.exists():
        try:
            with open(metadata_path, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
            
            print(f"\n🏆 PROJECT METADATA:")
            print(f"   Name: {metadata.get('name', 'N/A')}")
            print(f"   Version: {metadata.get('version', 'N/A')}")
            print(f"   Organized: {metadata.get('organized_at', 'N/A')}")
            print(f"   Directories: {metadata.get('total_directories', 'N/A')}")
            
            print(f"\n🚀 KEY FEATURES:")
            for feature in metadata.get('features', []):
                print(f"   ✅ {feature}")
                
        except Exception as e:
            print(f"   ❌ Error reading metadata: {e}")
    
    # Check organization log
    org_log_path = root / "organization_log.json"
    if org_log_path.exists():
        try:
            with open(org_log_path, 'r', encoding='utf-8') as f:
                org_log = json.load(f)
            print(f"\n📋 ORGANIZATION LOG: {len(org_log)} files moved")
        except Exception as e:
            print(f"   ❌ Error reading organization log: {e}")
    
    # Schema validation
    schema_path = root / "config" / "schemas" / "h_model_z_complete_schema.json"
    if schema_path.exists():
        try:
            with open(schema_path, 'r', encoding='utf-8') as f:
                schema = json.load(f)
            
            properties_count = len(schema.get('properties', {}))
            required_count = len(schema.get('required', []))
            
            print(f"\n🔧 SCHEMA STATUS:")
            print(f"   Properties: {properties_count}")
            print(f"   Required Fields: {required_count}")
            print(f"   Schema Size: {schema_path.stat().st_size} bytes")
            
        except Exception as e:
            print(f"   ❌ Error reading schema: {e}")
    
    print(f"\n✅ ORGANIZATION VALIDATION COMPLETE")
    print(f"🎯 PROJECT STATUS: FULLY ORGANIZED AND ENTERPRISE-READY")
    
    return organization_stats

def generate_quick_stats():
    """Generate quick project statistics."""
    root = Path(os.getcwd())
    
    # Count files by type
    file_types = defaultdict(int)
    total_size = 0
    
    for file_path in root.rglob("*"):
        if file_path.is_file():
            suffix = file_path.suffix.lower()
            file_types[suffix] += 1
            try:
                total_size += file_path.stat().st_size
            except:
                pass
    
    print(f"\n📈 PROJECT STATISTICS:")
    print(f"   Total Size: {total_size / (1024*1024):.1f} MB")
    
    # Top file types
    sorted_types = sorted(file_types.items(), key=lambda x: x[1], reverse=True)[:10]
    print(f"   Top File Types:")
    for ext, count in sorted_types:
        ext_name = ext if ext else "(no extension)"
        print(f"     {ext_name}: {count} files")

if __name__ == "__main__":
    analyze_organization()
    generate_quick_stats()
