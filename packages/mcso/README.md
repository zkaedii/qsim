# MCSO - Multi-Component Stochastic Oscillator

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A Python library for simulating and analyzing multi-component stochastic oscillatory systems with memory feedback and adaptive noise.

## Overview

MCSO implements a novel class of stochastic differential equations that combine:

- **Multiple oscillatory components** with time-varying parameters
- **Integral terms** with nonlinear activation functions
- **Memory feedback mechanisms** with sigmoid/tanh gating
- **Adaptive stochastic noise** with state-dependent variance
- **External control inputs** for system perturbation

### Mathematical Formulation

The system state X(t) evolves according to:

```
X(t) = S(t) + I(t) + D(t) + M(t) + N(t) + U(t)
```

where:
- **S(t)** - Oscillatory: `Σᵢ Aᵢ(t)·sin(Bᵢ(t)·t + φᵢ) + Cᵢ·exp(-Dᵢ·t)`
- **I(t)** - Integral: `∫₀ᵗ σ(a(x-x₀)² + b)·f(x)·g'(x) dx`
- **D(t)** - Drift: `α₀t² + α₁sin(2πt) + α₂log(1+t)`
- **M(t)** - Memory: `η·X(t-τ)·sigmoid(γ·X(t-τ))`
- **N(t)** - Noise: `σ·ε(t)·√(1 + β|X(t-1)|)`
- **U(t)** - Control: `δ·u(t)`

## Installation

```bash
# Basic installation
pip install mcso

# With visualization support
pip install mcso[viz]

# With statistical analysis tools
pip install mcso[stats]

# Full installation (all extras)
pip install mcso[all]

# Development installation
pip install mcso[dev]
```

### From Source

```bash
git clone https://github.com/zkaedii/qsim.git
cd qsim/packages/mcso
pip install -e ".[all]"
```

## Quick Start

### Basic Simulation

```python
from mcso import StochasticOscillator

# Create oscillator with default parameters
osc = StochasticOscillator(seed=42)

# Simulate trajectory
result = osc.simulate(t_max=100, dt=1.0)

print(f"Simulated {len(result['times'])} time points")
print(f"Mean value: {result['values'].mean():.4f}")
```

### Custom Configuration

```python
from mcso import StochasticOscillator, OscillatorConfig

config = OscillatorConfig(
    n_components=3,        # Number of oscillatory modes
    noise_scale=0.5,       # Noise intensity
    memory_strength=2.0,   # Memory feedback strength
    memory_delay=2.0,      # Memory delay
    seed=42                # For reproducibility
)

osc = StochasticOscillator(config=config)
result = osc.simulate(t_max=100)
```

### Statistical Analysis

```python
from mcso import StochasticOscillator
from mcso.analysis import compute_statistics, spectral_analysis

osc = StochasticOscillator(seed=42)
result = osc.simulate(t_max=200, dt=0.5)

# Compute statistics
stats = compute_statistics(result['values'])
print(f"Mean: {stats.mean:.4f}")
print(f"Std:  {stats.std:.4f}")
print(f"Autocorrelation (lag-1): {stats.autocorr_lag1:.4f}")

# Spectral analysis
spec = spectral_analysis(result['values'], dt=0.5)
print(f"Dominant frequency: {spec.dominant_freq:.4f}")
print(f"Spectral entropy: {spec.spectral_entropy:.4f}")
```

### Ensemble Simulation

```python
from mcso import StochasticOscillator

osc = StochasticOscillator(n_components=5, noise_scale=0.2)

# Simulate 100 independent realizations
ensemble = osc.simulate_ensemble(
    n_realizations=100,
    t_max=50,
    dt=1.0
)

print(f"Ensemble shape: {ensemble['ensemble'].shape}")
print(f"Mean trajectory: {ensemble['mean']}")
print(f"5th percentile: {ensemble['percentiles'][5]}")
```

### Custom Control Input

```python
from mcso import StochasticOscillator

def step_control(t):
    """Step control: activate at t=25"""
    return 1.0 if t > 25 else 0.0

osc = StochasticOscillator(control_gain=0.5, seed=42)
result = osc.simulate(t_max=50, control_fn=step_control)
```

## Applications

### Financial Modeling

```python
from mcso import StochasticOscillator, OscillatorConfig

# Configure for asset returns
config = OscillatorConfig(
    n_components=3,
    amplitude_base=0.02,      # ~2% base amplitude
    noise_scale=0.02,         # ~2% daily volatility
    noise_state_coupling=0.5, # Volatility clustering
    memory_strength=0.3,      # Momentum effect
    seed=42
)

osc = StochasticOscillator(config=config)
returns = osc.simulate(t_max=252)  # 1 year of trading days
```

### Signal Processing

```python
from mcso import StochasticOscillator
from mcso.analysis import spectral_analysis

# Multi-frequency signal with noise
osc = StochasticOscillator(
    n_components=5,
    frequency_base=1.0,
    frequency_scaling=2.0,  # Harmonic series
    noise_scale=0.1
)

signal = osc.simulate(t_max=1000, dt=0.1)
spec = spectral_analysis(signal['values'], dt=0.1)
```

## Module Reference

### Core Classes

| Class | Description |
|-------|-------------|
| `StochasticOscillator` | Main oscillator simulator |
| `OscillatorConfig` | Configuration dataclass |

### Memory Kernels

| Class | Description |
|-------|-------------|
| `ExponentialMemory` | Exponentially weighted memory |
| `SigmoidGatedMemory` | Sigmoid-gated discrete delay |
| `TanhGatedMemory` | Tanh-gated symmetric memory |
| `MultiScaleMemory` | Multi-timescale memory |
| `AdaptiveMemory` | Learning-based adaptive memory |

### Noise Generators

| Class | Description |
|-------|-------------|
| `GaussianNoise` | White Gaussian noise |
| `StateDependentNoise` | Volatility clustering |
| `AdaptiveNoise` | Variance-adaptive noise |
| `OrnsteinUhlenbeckNoise` | Colored (correlated) noise |
| `JumpDiffusionNoise` | Continuous + jump process |

### Analysis Functions

| Function | Description |
|----------|-------------|
| `compute_statistics()` | Statistical summary |
| `spectral_analysis()` | Power spectrum analysis |
| `stability_analysis()` | Sensitivity analysis |
| `autocorrelation()` | ACF computation |
| `estimate_lyapunov_exponent()` | Chaos detection |

### Visualization

| Function | Description |
|----------|-------------|
| `plot_trajectory()` | Time series plot |
| `plot_phase_space()` | Delay embedding plot |
| `plot_spectrum()` | Power spectrum plot |
| `plot_ensemble()` | Ensemble statistics plot |

## Examples

See the `examples/` directory:

- `basic_usage.py` - Core functionality demonstration
- `financial_modeling.py` - Finance applications

Run examples:

```bash
python examples/basic_usage.py
python examples/financial_modeling.py
```

## Testing

```bash
# Run tests
pytest tests/

# With coverage
pytest tests/ --cov=mcso --cov-report=html
```

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Run the test suite
5. Submit a pull request

## License

MIT License - see [LICENSE](LICENSE) for details.

## Citation

If you use MCSO in your research, please cite:

```bibtex
@software{mcso2024,
  title = {MCSO: Multi-Component Stochastic Oscillator},
  author = {QSIM Project},
  year = {2024},
  url = {https://github.com/zkaedii/qsim/tree/main/packages/mcso}
}
```

## Related Work

- [SciPy](https://scipy.org/) - Scientific computing in Python
- [NumPy](https://numpy.org/) - Numerical computing
- [statsmodels](https://www.statsmodels.org/) - Statistical modeling
