"""
Stochastic Noise Generators

This module implements various noise models for stochastic dynamical systems,
including state-dependent and adaptive noise processes.

Noise Types
-----------
1. **Additive Noise**: Independent of system state
   N(t) = σ · ε(t), where ε(t) ~ N(0,1)

2. **Multiplicative Noise**: Proportional to state magnitude
   N(t) = σ · |X(t)| · ε(t)

3. **State-Dependent Noise**: Variance depends on state
   N(t) = σ · ε(t) · √(1 + β|X(t-1)|)

4. **Colored Noise**: Temporally correlated (Ornstein-Uhlenbeck)
   dN = -θ·N·dt + σ·dW

Applications
------------
- Financial volatility modeling
- Turbulence simulation
- Biological fluctuations
- Measurement noise characterization
"""

from __future__ import annotations

import numpy as np
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Optional, Tuple


class NoiseGenerator(ABC):
    """
    Abstract base class for noise generators.

    All noise generators must implement the `sample` method to produce
    noise realizations.
    """

    @abstractmethod
    def sample(
        self,
        t: float,
        state: float = 0.0,
        history: Optional[Dict[float, float]] = None
    ) -> float:
        """
        Generate a noise sample.

        Parameters
        ----------
        t : float
            Current time.
        state : float
            Current system state.
        history : dict, optional
            History of past states.

        Returns
        -------
        float
            Noise value.
        """
        pass

    @abstractmethod
    def reset(self, seed: Optional[int] = None) -> None:
        """Reset generator state with optional new seed."""
        pass


@dataclass
class GaussianNoise(NoiseGenerator):
    """
    Simple Gaussian (white) noise.

    N(t) = σ · ε(t), where ε(t) ~ N(0, 1)

    Parameters
    ----------
    scale : float
        Standard deviation σ (default: 1.0).
    mean : float
        Mean value μ (default: 0.0).
    seed : int, optional
        Random seed for reproducibility.
    """

    scale: float = 1.0
    mean: float = 0.0
    seed: Optional[int] = None

    def __post_init__(self):
        self.rng = np.random.default_rng(self.seed)

    def sample(
        self,
        t: float,
        state: float = 0.0,
        history: Optional[Dict[float, float]] = None
    ) -> float:
        """Generate Gaussian noise sample."""
        return self.mean + self.scale * self.rng.standard_normal()

    def reset(self, seed: Optional[int] = None) -> None:
        """Reset with optional new seed."""
        self.rng = np.random.default_rng(seed if seed is not None else self.seed)


@dataclass
class StateDependentNoise(NoiseGenerator):
    """
    State-dependent noise with variance scaling.

    N(t) = σ · ε(t) · √(1 + β|X(t-1)|)

    The variance increases with the magnitude of recent states,
    modeling volatility clustering effects.

    Parameters
    ----------
    scale : float
        Base standard deviation σ (default: 0.2).
    state_coupling : float
        State coupling coefficient β (default: 0.3).
    delay : float
        State delay for coupling (default: 1.0).
    max_state : float
        Maximum state magnitude for stability (default: 10.0).
    seed : int, optional
        Random seed.
    """

    scale: float = 0.2
    state_coupling: float = 0.3
    delay: float = 1.0
    max_state: float = 10.0
    seed: Optional[int] = None

    def __post_init__(self):
        self.rng = np.random.default_rng(self.seed)

    def sample(
        self,
        t: float,
        state: float = 0.0,
        history: Optional[Dict[float, float]] = None
    ) -> float:
        """Generate state-dependent noise sample."""
        # Get delayed state
        if history is not None:
            t_delayed = max(0, t - self.delay)
            x_prev = history.get(t_delayed, 0.0)
        else:
            x_prev = state

        # Clip for numerical stability
        x_prev = min(abs(x_prev), self.max_state)

        # State-dependent variance
        variance = 1.0 + self.state_coupling * x_prev
        std = np.sqrt(variance)

        return self.scale * self.rng.normal(0, std)

    def reset(self, seed: Optional[int] = None) -> None:
        """Reset with optional new seed."""
        self.rng = np.random.default_rng(seed if seed is not None else self.seed)


@dataclass
class AdaptiveNoise(NoiseGenerator):
    """
    Adaptive noise with variance that adjusts based on system behavior.

    The noise variance adapts to maintain a target coefficient of variation:

        σ(t+1) = σ(t) · (1 + α · (CV_target - CV_observed))

    Parameters
    ----------
    initial_scale : float
        Initial standard deviation (default: 0.2).
    target_cv : float
        Target coefficient of variation (default: 0.1).
    adaptation_rate : float
        Adaptation speed α (default: 0.01).
    scale_bounds : tuple
        Min/max bounds for scale (default: (0.01, 2.0)).
    window : int
        Number of samples for CV estimation (default: 20).
    seed : int, optional
        Random seed.
    """

    initial_scale: float = 0.2
    target_cv: float = 0.1
    adaptation_rate: float = 0.01
    scale_bounds: Tuple[float, float] = (0.01, 2.0)
    window: int = 20
    seed: Optional[int] = None

    def __post_init__(self):
        self.rng = np.random.default_rng(self.seed)
        self.scale = self.initial_scale
        self.recent_values: list = []

    def sample(
        self,
        t: float,
        state: float = 0.0,
        history: Optional[Dict[float, float]] = None
    ) -> float:
        """Generate adaptive noise sample."""
        # Generate noise with current scale
        noise = self.scale * self.rng.standard_normal()

        # Track recent values for adaptation
        self.recent_values.append(state + noise)
        if len(self.recent_values) > self.window:
            self.recent_values.pop(0)

        # Adapt scale if we have enough samples
        if len(self.recent_values) >= self.window:
            values = np.array(self.recent_values)
            mean = np.abs(np.mean(values))
            std = np.std(values)

            if mean > 1e-10:
                observed_cv = std / mean
                adjustment = 1 + self.adaptation_rate * (self.target_cv - observed_cv)
                self.scale *= adjustment
                self.scale = np.clip(
                    self.scale,
                    self.scale_bounds[0],
                    self.scale_bounds[1]
                )

        return noise

    def reset(self, seed: Optional[int] = None) -> None:
        """Reset scale and buffer."""
        self.rng = np.random.default_rng(seed if seed is not None else self.seed)
        self.scale = self.initial_scale
        self.recent_values = []


@dataclass
class OrnsteinUhlenbeckNoise(NoiseGenerator):
    """
    Ornstein-Uhlenbeck (colored) noise process.

    Implements mean-reverting noise with temporal correlation:

        dN = θ(μ - N)dt + σdW

    Discretized using the exact solution:
        N(t+dt) = μ + (N(t) - μ)e^(-θdt) + σ√((1-e^(-2θdt))/(2θ)) · ε

    Parameters
    ----------
    mean : float
        Long-term mean μ (default: 0.0).
    theta : float
        Mean reversion rate θ (default: 1.0).
    sigma : float
        Volatility σ (default: 0.2).
    dt : float
        Time step (default: 1.0).
    seed : int, optional
        Random seed.
    """

    mean: float = 0.0
    theta: float = 1.0
    sigma: float = 0.2
    dt: float = 1.0
    seed: Optional[int] = None

    def __post_init__(self):
        self.rng = np.random.default_rng(self.seed)
        self.current_value = self.mean

    def sample(
        self,
        t: float,
        state: float = 0.0,
        history: Optional[Dict[float, float]] = None
    ) -> float:
        """Generate OU noise sample."""
        # Exact discretization
        exp_theta = np.exp(-self.theta * self.dt)
        variance = (self.sigma ** 2 / (2 * self.theta)) * (1 - np.exp(-2 * self.theta * self.dt))

        self.current_value = (
            self.mean +
            (self.current_value - self.mean) * exp_theta +
            np.sqrt(variance) * self.rng.standard_normal()
        )

        return self.current_value

    def reset(self, seed: Optional[int] = None) -> None:
        """Reset to mean value."""
        self.rng = np.random.default_rng(seed if seed is not None else self.seed)
        self.current_value = self.mean


@dataclass
class JumpDiffusionNoise(NoiseGenerator):
    """
    Jump-diffusion noise combining continuous and discontinuous components.

    N(t) = σ_c · ε_c(t) + J(t)

    where J(t) is a compound Poisson process:
    - Jumps occur with rate λ (Poisson process)
    - Jump sizes are normally distributed: J ~ N(μ_j, σ_j)

    Parameters
    ----------
    continuous_scale : float
        Continuous noise scale σ_c (default: 0.1).
    jump_rate : float
        Average jumps per unit time λ (default: 0.1).
    jump_mean : float
        Mean jump size μ_j (default: 0.0).
    jump_scale : float
        Jump size std σ_j (default: 0.5).
    dt : float
        Time step (default: 1.0).
    seed : int, optional
        Random seed.
    """

    continuous_scale: float = 0.1
    jump_rate: float = 0.1
    jump_mean: float = 0.0
    jump_scale: float = 0.5
    dt: float = 1.0
    seed: Optional[int] = None

    def __post_init__(self):
        self.rng = np.random.default_rng(self.seed)

    def sample(
        self,
        t: float,
        state: float = 0.0,
        history: Optional[Dict[float, float]] = None
    ) -> float:
        """Generate jump-diffusion noise sample."""
        # Continuous component
        continuous = self.continuous_scale * self.rng.standard_normal()

        # Jump component (Poisson number of jumps in dt)
        n_jumps = self.rng.poisson(self.jump_rate * self.dt)
        if n_jumps > 0:
            jumps = self.rng.normal(self.jump_mean, self.jump_scale, size=n_jumps)
            jump_total = np.sum(jumps)
        else:
            jump_total = 0.0

        return continuous + jump_total

    def reset(self, seed: Optional[int] = None) -> None:
        """Reset with optional new seed."""
        self.rng = np.random.default_rng(seed if seed is not None else self.seed)


def create_noise_generator(
    noise_type: str = "gaussian",
    **kwargs
) -> NoiseGenerator:
    """
    Factory function to create noise generators.

    Parameters
    ----------
    noise_type : str
        Type: 'gaussian', 'state_dependent', 'adaptive', 'ou', 'jump_diffusion'.
    **kwargs
        Parameters passed to constructor.

    Returns
    -------
    NoiseGenerator
        Instantiated noise generator.

    Examples
    --------
    >>> noise = create_noise_generator('gaussian', scale=0.5)
    >>> noise = create_noise_generator('ou', theta=2.0, sigma=0.1)
    """
    generators = {
        'gaussian': GaussianNoise,
        'state_dependent': StateDependentNoise,
        'adaptive': AdaptiveNoise,
        'ou': OrnsteinUhlenbeckNoise,
        'ornstein_uhlenbeck': OrnsteinUhlenbeckNoise,
        'jump_diffusion': JumpDiffusionNoise,
    }

    if noise_type not in generators:
        raise ValueError(
            f"Unknown noise type: {noise_type}. "
            f"Available: {list(generators.keys())}"
        )

    return generators[noise_type](**kwargs)
