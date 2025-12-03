"""
Memory Feedback Mechanisms

This module implements various memory kernel functions for modeling
systems with delayed feedback and history-dependent dynamics.

Memory feedback is essential for modeling:
- Systems with finite response times
- Autoregressive processes
- Hysteresis effects
- Learning and adaptation dynamics

Mathematical Background
-----------------------
A general memory feedback term takes the form:

    M(t) = ∫₀^∞ K(s) · g(X(t-s)) ds

where K(s) is the memory kernel and g(·) is a gating function.

For computational efficiency, we typically use:
- Discrete delays: M(t) = η · X(t-τ) · g(X(t-τ))
- Exponential kernels: K(s) = exp(-s/τ)
- Gated feedback: g(x) = sigmoid(γx) or tanh(γx)
"""

from __future__ import annotations

import numpy as np
from scipy.special import expit
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Optional, List
from collections import defaultdict


class MemoryKernel(ABC):
    """
    Abstract base class for memory kernels.

    Memory kernels define how past states influence the current dynamics.
    Subclasses must implement the `evaluate` method.
    """

    @abstractmethod
    def evaluate(self, t: float, history: Dict[float, float]) -> float:
        """
        Evaluate memory contribution at time t.

        Parameters
        ----------
        t : float
            Current time.
        history : dict
            Mapping from time to state values.

        Returns
        -------
        float
            Memory contribution.
        """
        pass

    @abstractmethod
    def reset(self) -> None:
        """Reset any internal state."""
        pass


@dataclass
class ExponentialMemory(MemoryKernel):
    """
    Exponential memory kernel.

    Implements exponentially weighted average of past states:

        M(t) = η · Σᵢ exp(-(t-tᵢ)/τ) · X(tᵢ)

    where the sum is over past time points.

    Parameters
    ----------
    strength : float
        Memory strength η (default: 1.0).
    decay_time : float
        Characteristic decay time τ (default: 1.0).
    window : float
        Maximum lookback window (default: 10.0).
    """

    strength: float = 1.0
    decay_time: float = 1.0
    window: float = 10.0

    def evaluate(self, t: float, history: Dict[float, float]) -> float:
        """
        Evaluate exponential memory contribution.

        Parameters
        ----------
        t : float
            Current time.
        history : dict
            Time -> state mapping.

        Returns
        -------
        float
            Weighted sum of past states.
        """
        if not history:
            return 0.0

        total = 0.0
        norm = 0.0

        for t_past, x_past in history.items():
            dt = t - t_past
            if 0 < dt <= self.window:
                weight = np.exp(-dt / self.decay_time)
                total += weight * x_past
                norm += weight

        if norm > 0:
            return self.strength * total / norm
        return 0.0

    def reset(self) -> None:
        """No internal state to reset."""
        pass


@dataclass
class SigmoidGatedMemory(MemoryKernel):
    """
    Sigmoid-gated discrete delay memory.

    Implements gated feedback with discrete delay:

        M(t) = η · X(t-τ) · sigmoid(γ · X(t-τ))

    The sigmoid gating provides:
    - Bounded feedback magnitude
    - Asymmetric response to positive/negative states
    - Smooth transition between weak and strong feedback

    Parameters
    ----------
    strength : float
        Memory strength η (default: 1.0).
    delay : float
        Discrete delay τ (default: 1.0).
    sensitivity : float
        Sigmoid sensitivity γ (default: 2.0).
    """

    strength: float = 1.0
    delay: float = 1.0
    sensitivity: float = 2.0

    def evaluate(self, t: float, history: Dict[float, float]) -> float:
        """
        Evaluate sigmoid-gated memory contribution.

        Parameters
        ----------
        t : float
            Current time.
        history : dict
            Time -> state mapping.

        Returns
        -------
        float
            Gated memory contribution.
        """
        t_delayed = max(0, t - self.delay)

        # Find nearest available history point
        x_delayed = history.get(t_delayed, 0.0)

        # Sigmoid gating
        gate = expit(np.clip(self.sensitivity * x_delayed, -500, 500))

        return self.strength * x_delayed * gate

    def reset(self) -> None:
        """No internal state to reset."""
        pass


@dataclass
class TanhGatedMemory(MemoryKernel):
    """
    Hyperbolic tangent gated memory.

    Implements gated feedback with tanh nonlinearity:

        M(t) = η · X(t-τ) · tanh(γ · X(t-τ))

    Unlike sigmoid gating, tanh provides:
    - Symmetric response to positive/negative states
    - Output bounded in [-η, η]

    Parameters
    ----------
    strength : float
        Memory strength η (default: 1.0).
    delay : float
        Discrete delay τ (default: 1.0).
    sensitivity : float
        Tanh sensitivity γ (default: 1.0).
    """

    strength: float = 1.0
    delay: float = 1.0
    sensitivity: float = 1.0

    def evaluate(self, t: float, history: Dict[float, float]) -> float:
        """
        Evaluate tanh-gated memory contribution.

        Parameters
        ----------
        t : float
            Current time.
        history : dict
            Time -> state mapping.

        Returns
        -------
        float
            Gated memory contribution.
        """
        t_delayed = max(0, t - self.delay)
        x_delayed = history.get(t_delayed, 0.0)

        gate = np.tanh(self.sensitivity * x_delayed)
        return self.strength * x_delayed * gate

    def reset(self) -> None:
        """No internal state to reset."""
        pass


@dataclass
class MultiScaleMemory(MemoryKernel):
    """
    Multi-scale memory with multiple time constants.

    Combines memory effects at different time scales:

        M(t) = Σⱼ ηⱼ · exp(-(t-tⱼ)/τⱼ) · X(tⱼ)

    This captures both short-term and long-term dependencies.

    Parameters
    ----------
    scales : list of tuple
        List of (strength, decay_time) pairs for each scale.
    window : float
        Maximum lookback window (default: 50.0).
    """

    scales: List[tuple] = None
    window: float = 50.0

    def __post_init__(self):
        if self.scales is None:
            # Default: fast, medium, slow scales
            self.scales = [
                (0.5, 1.0),   # Fast: τ = 1
                (0.3, 5.0),   # Medium: τ = 5
                (0.2, 20.0),  # Slow: τ = 20
            ]

    def evaluate(self, t: float, history: Dict[float, float]) -> float:
        """
        Evaluate multi-scale memory contribution.

        Parameters
        ----------
        t : float
            Current time.
        history : dict
            Time -> state mapping.

        Returns
        -------
        float
            Combined multi-scale memory.
        """
        if not history:
            return 0.0

        total = 0.0

        for strength, decay_time in self.scales:
            scale_contribution = 0.0
            norm = 0.0

            for t_past, x_past in history.items():
                dt = t - t_past
                if 0 < dt <= self.window:
                    weight = np.exp(-dt / decay_time)
                    scale_contribution += weight * x_past
                    norm += weight

            if norm > 0:
                total += strength * scale_contribution / norm

        return total

    def reset(self) -> None:
        """No internal state to reset."""
        pass


class AdaptiveMemory(MemoryKernel):
    """
    Adaptive memory with learning dynamics.

    The memory strength adapts based on prediction error:

        η(t+1) = η(t) + α · (X(t) - X̂(t)) · X(t-τ)

    where X̂(t) is the predicted state from memory.

    Parameters
    ----------
    initial_strength : float
        Initial memory strength (default: 1.0).
    learning_rate : float
        Adaptation rate α (default: 0.01).
    delay : float
        Discrete delay τ (default: 1.0).
    sensitivity : float
        Gating sensitivity (default: 2.0).
    strength_bounds : tuple
        Min/max bounds for strength (default: (0.01, 10.0)).
    """

    def __init__(
        self,
        initial_strength: float = 1.0,
        learning_rate: float = 0.01,
        delay: float = 1.0,
        sensitivity: float = 2.0,
        strength_bounds: tuple = (0.01, 10.0)
    ):
        self.initial_strength = initial_strength
        self.learning_rate = learning_rate
        self.delay = delay
        self.sensitivity = sensitivity
        self.strength_bounds = strength_bounds

        self.strength = initial_strength
        self.last_prediction: Optional[float] = None

    def evaluate(self, t: float, history: Dict[float, float]) -> float:
        """
        Evaluate adaptive memory contribution.

        Parameters
        ----------
        t : float
            Current time.
        history : dict
            Time -> state mapping.

        Returns
        -------
        float
            Adaptive memory contribution.
        """
        t_delayed = max(0, t - self.delay)
        x_delayed = history.get(t_delayed, 0.0)

        # Compute prediction
        gate = expit(np.clip(self.sensitivity * x_delayed, -500, 500))
        prediction = self.strength * x_delayed * gate

        # Adapt strength based on error (if we have a previous prediction)
        if self.last_prediction is not None and t > 0:
            x_current = history.get(t, 0.0)
            error = x_current - self.last_prediction
            self.strength += self.learning_rate * error * x_delayed
            self.strength = np.clip(
                self.strength,
                self.strength_bounds[0],
                self.strength_bounds[1]
            )

        self.last_prediction = prediction
        return prediction

    def reset(self) -> None:
        """Reset adaptive strength to initial value."""
        self.strength = self.initial_strength
        self.last_prediction = None


def create_memory_kernel(
    kernel_type: str = "sigmoid_gated",
    **kwargs
) -> MemoryKernel:
    """
    Factory function to create memory kernels.

    Parameters
    ----------
    kernel_type : str
        Type of kernel: 'exponential', 'sigmoid_gated', 'tanh_gated',
        'multiscale', 'adaptive'.
    **kwargs
        Parameters passed to the kernel constructor.

    Returns
    -------
    MemoryKernel
        Instantiated memory kernel.

    Examples
    --------
    >>> kernel = create_memory_kernel('exponential', decay_time=2.0)
    >>> kernel = create_memory_kernel('adaptive', learning_rate=0.05)
    """
    kernels = {
        'exponential': ExponentialMemory,
        'sigmoid_gated': SigmoidGatedMemory,
        'tanh_gated': TanhGatedMemory,
        'multiscale': MultiScaleMemory,
        'adaptive': AdaptiveMemory,
    }

    if kernel_type not in kernels:
        raise ValueError(
            f"Unknown kernel type: {kernel_type}. "
            f"Available: {list(kernels.keys())}"
        )

    return kernels[kernel_type](**kwargs)
