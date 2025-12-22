# Repository Structure Guide

This repository is transitioning to a standard Python package layout. Use this guide to
place new files in the right location.

## ✅ Source of truth (packaged code)

All production-ready Python modules belong under:

```
src/hmodelz/
```

### Examples
- Core framework modules: `src/hmodelz/core/`
- Engines and optimization: `src/hmodelz/engines/`
- Schemas and validation: `src/hmodelz/schema/`
- Interfaces and APIs: `src/hmodelz/interfaces/`

## 📦 Benchmarks and experiments

Performance benchmarks and research scripts should live under:

```
benchmarks/
```

Keep these separate from packaged code so the installed library remains clean and focused.

## 📚 Documentation

Place documentation in:

```
docs/
```

Examples, guides, and reports should all live here. Link to them from `README.md` when
appropriate.

## 🧪 Tests

All tests go in:

```
tests/
```

Use unit tests for core logic and lightweight integration tests for system behavior.

## 🧹 Root directory

The repository root should only contain:
- Packaging and build configuration (`pyproject.toml`, `setup.py`)
- Project-level docs (`README.md`, `LICENSE`)
- Entry-point scripts or shims for backward compatibility

If a script becomes stable or reusable, promote it into `src/hmodelz/`.
