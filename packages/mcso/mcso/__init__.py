"""
MCSO - Multi-Component Stochastic Oscillator

A Python library for simulating and analyzing multi-component stochastic
oscillatory systems with memory feedback and adaptive noise.

This library implements a novel class of stochastic differential equations
combining:
- Multiple coupled oscillatory components with time-varying parameters
- Integral terms with nonlinear activation functions
- Memory feedback mechanisms with sigmoid gating
- Adaptive stochastic noise with state-dependent variance
- External control inputs

Applications include:
- Financial time-series modeling
- Signal processing and filtering
- Control systems analysis
- Regime-switching dynamics
- Coupled oscillator networks

Example:
    >>> from mcso import StochasticOscillator
    >>> osc = StochasticOscillator(n_components=5)
    >>> trajectory = osc.simulate(t_max=100)
    >>> osc.analyze(trajectory)

References:
    [1] Theory of stochastic differential equations
    [2] Coupled oscillator dynamics
    [3] Memory-dependent processes

Author: QSIM Project
License: MIT
"""

__version__ = "0.1.0"
__author__ = "QSIM Project"

from .oscillator import StochasticOscillator, OscillatorConfig
from .memory import MemoryKernel, ExponentialMemory, SigmoidGatedMemory
from .noise import AdaptiveNoise, GaussianNoise, StateDependentNoise
from .integrators import integrate_activation, numerical_quadrature
from .analysis import (
    compute_statistics,
    spectral_analysis,
    stability_analysis,
    plot_trajectory,
    plot_phase_space,
)

__all__ = [
    # Core
    "StochasticOscillator",
    "OscillatorConfig",
    # Memory
    "MemoryKernel",
    "ExponentialMemory",
    "SigmoidGatedMemory",
    # Noise
    "AdaptiveNoise",
    "GaussianNoise",
    "StateDependentNoise",
    # Integrators
    "integrate_activation",
    "numerical_quadrature",
    # Analysis
    "compute_statistics",
    "spectral_analysis",
    "stability_analysis",
    "plot_trajectory",
    "plot_phase_space",
]
