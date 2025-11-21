# H_MODEL_Z Framework

> Enterprise-grade performance optimization framework combining quantum simulation, blockchain integration, and AI-powered optimization.

[![Build Status](https://github.com/zkaedii/qsim/actions/workflows/python-package.yml/badge.svg)](https://github.com/zkaedii/qsim/actions/workflows/python-package.yml)
[![codecov](https://codecov.io/gh/zkaedii/qsim/branch/main/graph/badge.svg)](https://codecov.io/gh/zkaedii/qsim)
[![Python Version](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](docs/LICENSE)
[![Status](https://img.shields.io/badge/status-beta-yellow.svg)](https://github.com/zkaedii/qsim)

## Overview

H_MODEL_Z is a comprehensive framework for:

- **Quantum Simulation**: Advanced Hamiltonian simulation with complex time-dependent systems
- **Blockchain Integration**: Smart contract analysis, flash loan modeling, and DeFi protocol simulation
- **AI Optimization**: Claude AI-powered performance optimization and adaptive tuning
- **Enterprise Features**: Scaling, monitoring, security, and compliance capabilities

## Features

- 🔬 **Mathematical Rigor**: Sophisticated Hamiltonian simulation with stochastic processes
- 🔗 **Blockchain Support**: Opcode-level smart contract simulation and DeFi analysis
- 🤖 **AI Integration**: Native Claude AI optimization throughout the framework
- 📊 **Visualization**: Comprehensive data visualization and analysis tools
- 🏢 **Enterprise Ready**: Scaling, monitoring, and security features
- 🧪 **Extensible**: Modular architecture with clear API boundaries

## Installation

### Quick Install

```bash
# Clone the repository
git clone https://github.com/zkaedii/qsim.git
cd qsim

# Install in development mode
pip install -e .

# Or install with extras
pip install -e ".[dev,viz,gpu]"
```

### Requirements

- Python 3.8 or higher
- NumPy, SciPy for numerical computation
- Anthropic API key for AI features (optional)
- CUDA-capable GPU for GPU acceleration (optional)

## Project Structure

```
qsim/
├── src/hmodelz/           # Main package source code
│   ├── core/              # Core framework components
│   ├── frameworks/        # Specialized framework implementations
│   ├── engines/           # Optimization and performance engines
│   ├── schema/            # JSON schema management
│   └── interfaces/        # API definitions
├── benchmarks/            # Performance benchmarks
│   ├── suites/            # Benchmark test suites
│   └── results/           # Benchmark results
├── tests/                 # Test suites
├── examples/              # Example code and demonstrations
├── tools/                 # Utility tools
├── scripts/               # Automation scripts
├── config/                # Configuration files
└── docs/                  # Documentation
```

## Quick Start

### Basic Hamiltonian Simulation

```python
# Import from benchmarks (until package restructuring is complete)
import sys
sys.path.append('benchmarks/suites')
from hamiltonian_simulation import ComplexHamiltonianSimulator

# Create simulator
simulator = ComplexHamiltonianSimulator()

# Run simulation
t_values, H_values, components = simulator.simulate()

# Analyze results
stats, correlations, patterns = simulator.analyze_behavior(
    t_values, H_values, components
)
```

### Running Benchmarks

```bash
# Run Hamiltonian benchmark suite
python benchmarks/suites/hamiltonian_benchmark_suite.py

# Run performance comparison
python benchmarks/results/performance_comparison.py
```

## Development

### Setting Up Development Environment

```bash
# Clone repository
git clone https://github.com/zkaedii/qsim.git
cd qsim

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install with development dependencies
pip install -e ".[dev]"

# Install pre-commit hooks (from config directory)
cp config/.pre-commit-config.yaml .
pre-commit install
```

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=hmodelz --cov-report=html
```

### Code Quality

```bash
# Format code with Black
black src/

# Run linter
flake8 src/
```

## Documentation

Comprehensive documentation is available in the `docs/` directory:

- **User Guides**: `docs/guides/` - Getting started and tutorials
- **API Reference**: `docs/api/` - Detailed API documentation
- **Architecture**: `docs/architecture/` - System design and architecture
- **Reports**: `docs/reports/` - Performance analysis and audit reports

Key documents:

- [Codebase Analysis](docs/architecture/CODEBASE_ANALYSIS.md)
- [Comprehensive Session Report](docs/reports/COMPREHENSIVE_SESSION_REPORT.md)
- [Session Completion Guide](docs/guides/SESSION_COMPLETION_README.md)

## Contributing

Contributions are welcome! Please follow these guidelines:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Run tests and linting
5. Commit your changes (`git commit -m 'Add amazing feature'`)
6. Push to the branch (`git push origin feature/amazing-feature`)
7. Open a Pull Request

## License

This project is licensed under the MIT License - see the [docs/LICENSE](docs/LICENSE) file for details.

## Support

- **Issues**: [GitHub Issues](https://github.com/zkaedii/qsim/issues)
- **Discussions**: [GitHub Discussions](https://github.com/zkaedii/qsim/discussions)

---

**H_MODEL_Z** - Advancing the frontiers of performance optimization
