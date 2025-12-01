"""
Core Stochastic Oscillator Implementation

This module implements the Multi-Component Stochastic Oscillator (MCSO),
a novel class of stochastic processes combining oscillatory dynamics,
memory feedback, and adaptive noise.

Mathematical Formulation
------------------------
The system state X(t) evolves according to:

    X(t) = S(t) + I(t) + D(t) + M(t) + N(t) + U(t)

where:
    S(t) = Σᵢ Aᵢ(t)·sin(Bᵢ(t)·t + φᵢ) + Cᵢ·exp(-Dᵢ·t)    [Oscillatory]
    I(t) = ∫₀ᵗ σ(a(x-x₀)² + b)·f(x)·g'(x) dx              [Integral]
    D(t) = α₀t² + α₁sin(2πt) + α₂log(1+t)                  [Drift]
    M(t) = η·X(t-τ)·σ(γ·X(t-τ))                            [Memory]
    N(t) = σ·ε(t)·√(1 + β|X(t-1)|)                         [Noise]
    U(t) = δ·u(t)                                           [Control]

Here σ(·) denotes the softplus or sigmoid activation function.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.integrate import quad
from scipy.special import expit
from dataclasses import dataclass, field
from typing import Callable, Optional, Dict, List, Tuple, Any
from collections import defaultdict


@dataclass
class OscillatorConfig:
    """
    Configuration for the Multi-Component Stochastic Oscillator.

    Parameters
    ----------
    n_components : int
        Number of oscillatory components (default: 5)

    amplitude_base : float
        Base amplitude for oscillatory terms (default: 1.0)

    amplitude_modulation : float
        Amplitude modulation depth (default: 0.1)

    frequency_base : float
        Base frequency multiplier (default: 1.0)

    frequency_scaling : float
        Per-component frequency scaling (default: 0.1)

    decay_base : float
        Base exponential decay rate (default: 0.05)

    decay_scaling : float
        Per-component decay scaling (default: 0.01)

    integral_params : tuple
        Parameters (a, b, x0) for integral term (default: (0.8, 0.3, 1.0))

    drift_coefficients : tuple
        Coefficients (α₀, α₁, α₂) for drift term (default: (0.02, 0.4, 0.15))

    memory_strength : float
        Memory feedback strength η (default: 1.0)

    memory_delay : float
        Memory delay τ (default: 1.0)

    memory_sensitivity : float
        Sigmoid sensitivity γ (default: 2.0)

    noise_scale : float
        Noise standard deviation σ (default: 0.2)

    noise_state_coupling : float
        State-dependent noise coupling β (default: 0.3)

    control_gain : float
        Control input gain δ (default: 0.1)

    seed : Optional[int]
        Random seed for reproducibility (default: None)
    """

    # Oscillatory parameters
    n_components: int = 5
    amplitude_base: float = 1.0
    amplitude_modulation: float = 0.1
    frequency_base: float = 1.0
    frequency_scaling: float = 0.1
    decay_base: float = 0.05
    decay_scaling: float = 0.01
    decay_coefficient: float = 0.3

    # Integral parameters
    integral_params: Tuple[float, float, float] = (0.8, 0.3, 1.0)
    integral_limit: float = 10.0

    # Drift parameters
    drift_coefficients: Tuple[float, float, float] = (0.02, 0.4, 0.15)

    # Memory parameters
    memory_strength: float = 1.0
    memory_delay: float = 1.0
    memory_sensitivity: float = 2.0

    # Noise parameters
    noise_scale: float = 0.2
    noise_state_coupling: float = 0.3

    # Control parameters
    control_gain: float = 0.1
    control_frequency: float = 0.2

    # Numerical parameters
    clip_bounds: Tuple[float, float] = (-1000.0, 1000.0)
    t_max_safety: float = 20.0

    # Random state
    seed: Optional[int] = None

    def __post_init__(self):
        """Validate configuration parameters."""
        if self.n_components < 1:
            raise ValueError("n_components must be at least 1")
        if self.noise_scale < 0:
            raise ValueError("noise_scale must be non-negative")
        if self.memory_delay < 0:
            raise ValueError("memory_delay must be non-negative")


class StochasticOscillator:
    """
    Multi-Component Stochastic Oscillator.

    A dynamical system combining multiple oscillatory modes, integral
    feedback, drift, memory effects, and stochastic forcing.

    Parameters
    ----------
    config : OscillatorConfig, optional
        Configuration object. If None, uses defaults.
    **kwargs
        Override specific config parameters.

    Attributes
    ----------
    config : OscillatorConfig
        Current configuration.
    history : Dict[float, float]
        State history X(t) for memory lookups.
    rng : np.random.Generator
        Random number generator.

    Examples
    --------
    >>> osc = StochasticOscillator(n_components=3, noise_scale=0.1)
    >>> trajectory = osc.simulate(t_max=50, dt=0.1)
    >>> print(f"Mean: {trajectory['values'].mean():.4f}")

    >>> # With custom control input
    >>> def control(t):
    ...     return np.sin(0.5 * t) if t > 10 else 0
    >>> trajectory = osc.simulate(t_max=100, control_fn=control)
    """

    def __init__(
        self,
        config: Optional[OscillatorConfig] = None,
        **kwargs
    ):
        if config is None:
            config = OscillatorConfig(**kwargs)
        elif kwargs:
            # Override config with kwargs
            config_dict = {
                k: kwargs.get(k, getattr(config, k))
                for k in config.__dataclass_fields__
            }
            config = OscillatorConfig(**config_dict)

        self.config = config
        self.history: Dict[float, float] = defaultdict(float)
        self.rng = np.random.default_rng(config.seed)

        # Pre-compute phase offsets
        self._phases = np.array([
            np.pi / (i + 1) for i in range(config.n_components)
        ])

    def reset(self, seed: Optional[int] = None) -> None:
        """
        Reset oscillator state.

        Parameters
        ----------
        seed : int, optional
            New random seed.
        """
        self.history.clear()
        if seed is not None:
            self.rng = np.random.default_rng(seed)

    # =========================================================================
    # Component Functions
    # =========================================================================

    def amplitude(self, i: int, t: float) -> float:
        """
        Time-varying amplitude for component i.

        A_i(t) = A_base + A_mod * sin(ω_mod * t)

        Parameters
        ----------
        i : int
            Component index.
        t : float
            Time.

        Returns
        -------
        float
            Amplitude value.
        """
        c = self.config
        return c.amplitude_base + c.amplitude_modulation * np.sin(0.5 * t)

    def frequency(self, i: int, t: float) -> float:
        """
        Frequency for component i.

        B_i = B_base + B_scale * i

        Parameters
        ----------
        i : int
            Component index.
        t : float
            Time (unused, for API consistency).

        Returns
        -------
        float
            Frequency value.
        """
        c = self.config
        return c.frequency_base + c.frequency_scaling * i

    def decay_rate(self, i: int) -> float:
        """
        Exponential decay rate for component i.

        D_i = D_base + D_scale * i

        Parameters
        ----------
        i : int
            Component index.

        Returns
        -------
        float
            Decay rate.
        """
        c = self.config
        return c.decay_base + c.decay_scaling * i

    @staticmethod
    def softplus(x: float, threshold: float = 20.0) -> float:
        """
        Numerically stable softplus activation.

        σ(x) = log(1 + exp(x))

        For large x, returns x directly to avoid overflow.

        Parameters
        ----------
        x : float
            Input value.
        threshold : float
            Threshold for linear approximation.

        Returns
        -------
        float
            Activated value.
        """
        x_clipped = np.clip(x, -500, 500)
        return np.where(x_clipped > threshold, x_clipped, np.log1p(np.exp(x_clipped)))

    @staticmethod
    def sigmoid(x: float) -> float:
        """
        Sigmoid activation using scipy's numerically stable expit.

        σ(x) = 1 / (1 + exp(-x))

        Parameters
        ----------
        x : float
            Input value.

        Returns
        -------
        float
            Activated value in (0, 1).
        """
        return expit(np.clip(x, -500, 500))

    # =========================================================================
    # System Components
    # =========================================================================

    def oscillatory_term(self, t: float) -> float:
        """
        Compute oscillatory component S(t).

        S(t) = Σᵢ Aᵢ(t)·sin(Bᵢ(t)·t + φᵢ) + Cᵢ·exp(-Dᵢ·t)

        Parameters
        ----------
        t : float
            Time.

        Returns
        -------
        float
            Oscillatory contribution.
        """
        c = self.config
        total = 0.0

        for i in range(c.n_components):
            # Oscillatory part
            amp = self.amplitude(i, t)
            freq = self.frequency(i, t)
            phase = self._phases[i]
            osc = amp * np.sin(freq * t + phase)

            # Decay part
            decay = c.decay_coefficient * np.exp(-self.decay_rate(i) * t)

            total += osc + decay

        return total

    def integral_term(self, t: float) -> float:
        """
        Compute integral component I(t).

        I(t) = ∫₀ᵗ σ(a(x-x₀)² + b)·cos(x)·(-sin(x)) dx

        Uses numerical quadrature with bounded integration range.

        Parameters
        ----------
        t : float
            Time (upper integration limit).

        Returns
        -------
        float
            Integral contribution.
        """
        c = self.config
        a, b, x0 = c.integral_params
        t_bounded = min(t, c.integral_limit)

        if t_bounded <= 0:
            return 0.0

        def integrand(x: float) -> float:
            activation = self.softplus(a * (x - x0) ** 2 + b)
            return activation * np.cos(x) * (-np.sin(x))

        try:
            result, _ = quad(integrand, 0, t_bounded, limit=20)
            return result
        except Exception:
            return 0.0

    def drift_term(self, t: float) -> float:
        """
        Compute drift component D(t).

        D(t) = α₀t² + α₁sin(2πt) + α₂log(1+t)

        Parameters
        ----------
        t : float
            Time.

        Returns
        -------
        float
            Drift contribution.
        """
        alpha0, alpha1, alpha2 = self.config.drift_coefficients
        return alpha0 * t**2 + alpha1 * np.sin(2 * np.pi * t) + alpha2 * np.log1p(t)

    def memory_term(self, t: float) -> float:
        """
        Compute memory feedback M(t).

        M(t) = η·X(t-τ)·sigmoid(γ·X(t-τ))

        Uses sigmoid gating for bounded feedback.

        Parameters
        ----------
        t : float
            Time.

        Returns
        -------
        float
            Memory contribution.
        """
        c = self.config
        t_delayed = max(0, t - c.memory_delay)
        x_delayed = self.history[t_delayed]

        gated = self.sigmoid(c.memory_sensitivity * x_delayed)
        return c.memory_strength * x_delayed * gated

    def noise_term(self, t: float) -> float:
        """
        Compute stochastic noise N(t).

        N(t) = σ·ε(t)·√(1 + β|X(t-1)|)

        where ε(t) ~ N(0, 1).

        Parameters
        ----------
        t : float
            Time.

        Returns
        -------
        float
            Noise contribution.
        """
        c = self.config
        x_prev = self.history[max(0, t - 1)]

        # State-dependent variance
        variance = 1 + c.noise_state_coupling * min(abs(x_prev), 10.0)
        std = np.sqrt(variance)

        return c.noise_scale * self.rng.normal(0, std)

    def control_term(self, t: float, control_fn: Optional[Callable[[float], float]] = None) -> float:
        """
        Compute control input U(t).

        U(t) = δ·u(t)

        Default control: u(t) = sin(ω_c·t)

        Parameters
        ----------
        t : float
            Time.
        control_fn : callable, optional
            Custom control function u(t).

        Returns
        -------
        float
            Control contribution.
        """
        c = self.config

        if control_fn is not None:
            u = control_fn(t)
        else:
            u = np.sin(c.control_frequency * t)

        return c.control_gain * u

    # =========================================================================
    # Main Computation
    # =========================================================================

    def evaluate(
        self,
        t: float,
        control_fn: Optional[Callable[[float], float]] = None,
        store_history: bool = True
    ) -> float:
        """
        Evaluate oscillator state X(t).

        X(t) = S(t) + I(t) + D(t) + M(t) + N(t) + U(t)

        Parameters
        ----------
        t : float
            Time.
        control_fn : callable, optional
            Custom control function.
        store_history : bool
            Whether to store result in history.

        Returns
        -------
        float
            State value X(t).
        """
        c = self.config

        # Bound time for numerical stability
        t_safe = min(t, c.t_max_safety)

        # Compute all components
        x = (
            self.oscillatory_term(t_safe) +
            self.integral_term(t_safe) +
            self.drift_term(t_safe) +
            self.memory_term(t_safe) +
            self.noise_term(t_safe) +
            self.control_term(t_safe, control_fn)
        )

        # Clip for numerical stability
        x = np.clip(x, c.clip_bounds[0], c.clip_bounds[1])

        if store_history:
            self.history[t] = x

        return x

    def simulate(
        self,
        t_max: float = 100.0,
        dt: float = 1.0,
        control_fn: Optional[Callable[[float], float]] = None,
        show_progress: bool = False
    ) -> Dict[str, NDArray]:
        """
        Simulate oscillator trajectory.

        Parameters
        ----------
        t_max : float
            Maximum simulation time.
        dt : float
            Time step.
        control_fn : callable, optional
            Custom control function.
        show_progress : bool
            Print progress updates.

        Returns
        -------
        dict
            Dictionary with keys:
            - 'times': Time points array
            - 'values': State values array
            - 'config': Configuration used
        """
        self.reset(self.config.seed)

        times = np.arange(0, t_max, dt)
        values = np.zeros(len(times))

        for idx, t in enumerate(times):
            if show_progress and idx % (len(times) // 10) == 0:
                print(f"  Simulating: {100 * idx / len(times):.0f}%")

            values[idx] = self.evaluate(t, control_fn)

        return {
            'times': times,
            'values': values,
            'config': self.config
        }

    def simulate_ensemble(
        self,
        n_realizations: int = 100,
        t_max: float = 100.0,
        dt: float = 1.0,
        seeds: Optional[List[int]] = None
    ) -> Dict[str, Any]:
        """
        Simulate ensemble of trajectories for statistical analysis.

        Parameters
        ----------
        n_realizations : int
            Number of independent realizations.
        t_max : float
            Maximum simulation time.
        dt : float
            Time step.
        seeds : list of int, optional
            Seeds for each realization.

        Returns
        -------
        dict
            Dictionary with keys:
            - 'times': Time points array
            - 'ensemble': 2D array (n_realizations x n_times)
            - 'mean': Ensemble mean trajectory
            - 'std': Ensemble standard deviation
            - 'percentiles': Dict of percentile trajectories
        """
        times = np.arange(0, t_max, dt)
        n_times = len(times)
        ensemble = np.zeros((n_realizations, n_times))

        if seeds is None:
            seeds = list(range(n_realizations))

        for i, seed in enumerate(seeds):
            self.reset(seed)
            for j, t in enumerate(times):
                ensemble[i, j] = self.evaluate(t)

        return {
            'times': times,
            'ensemble': ensemble,
            'mean': np.mean(ensemble, axis=0),
            'std': np.std(ensemble, axis=0),
            'percentiles': {
                5: np.percentile(ensemble, 5, axis=0),
                25: np.percentile(ensemble, 25, axis=0),
                50: np.percentile(ensemble, 50, axis=0),
                75: np.percentile(ensemble, 75, axis=0),
                95: np.percentile(ensemble, 95, axis=0),
            }
        }

    def __repr__(self) -> str:
        return (
            f"StochasticOscillator("
            f"n_components={self.config.n_components}, "
            f"noise_scale={self.config.noise_scale}, "
            f"memory_delay={self.config.memory_delay})"
        )
