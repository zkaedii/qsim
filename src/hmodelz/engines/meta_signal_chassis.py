"""
Meta-Engine: Multiform Signal Processing Chassis

This module implements a self-adaptive meta-engine for multiform signal processing,
combining polymorphic subsystem registration, adversarial-aware signal processing,
and adaptive harmonics across diverse computational contexts.

Architecture
------------
The meta-engine embodies:
- Polymorphic: Adapts across diverse object structures
- Metamorphic: Evolves internal shape/function during runtime
- Oglimorphic: Bounded ambiguity with discrete exception handling
- Chameleon: Context-aware self-adjustment
- Adaptive: Dynamic pathway modification under feedback
- Adversarial-Resilient: Deception-aware with defensive strategies
- Traversal-Oriented: Fluid navigation of hierarchies
- Nanomorphic: Tiny modularized units with scalable complexity
- Homomorphic: End-state transformation preserving structure
- Holomorphic: Entire continuity with harmonic completeness

Chassis Layers
--------------
Oscillation → Spectrum → Modulation → Synchronization → Recovery

Each computational subsystem morphs polymorphically while maintaining
core trust and harmonic coherence.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from typing import Callable, Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import warnings


@dataclass
class SubsystemConfig:
    """
    Configuration for a meta-engine subsystem.

    Attributes
    ----------
    name : str
        Unique identifier for the subsystem
    priority : int
        Execution priority (lower executes first)
    enabled : bool
        Whether the subsystem is active
    metamorphic : bool
        Whether the subsystem can evolve during runtime
    """

    name: str
    priority: int = 0
    enabled: bool = True
    metamorphic: bool = False
    parameters: Dict[str, Any] = field(default_factory=dict)


class Subsystem(ABC):
    """
    Abstract base class for polymorphic subsystems.

    Each subsystem implements the execute() method to perform
    its specific signal processing task.
    """

    def __init__(self, config: SubsystemConfig):
        self.config = config
        self.state: Dict[str, Any] = {}

    @abstractmethod
    def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the subsystem's signal processing logic.

        Parameters
        ----------
        context : Dict[str, Any]
            Shared context containing input signals and intermediate results

        Returns
        -------
        Dict[str, Any]
            Processed results to be merged into the context
        """
        pass

    def adapt(self, feedback: Dict[str, Any]) -> None:
        """
        Adapt subsystem behavior based on feedback.

        Parameters
        ----------
        feedback : Dict[str, Any]
            Feedback signals for adaptation
        """
        if self.config.metamorphic:
            # Metamorphic subsystems can evolve
            self.state.update(feedback)


class MetaSignalChassis:
    """
    Multiform meta-engine embodying adaptive harmonics across polymorphic,
    adversarial-aware signal contexts.

    The chassis maintains a registry of subsystems that can be dynamically
    registered, prioritized, and executed in a cascading manner. Each
    subsystem operates on a shared signal context, allowing for complex
    signal processing pipelines.

    Parameters
    ----------
    adversarial_threshold : float
        Threshold for adversarial noise detection (default: 1.0)
    enable_recovery : bool
        Enable automatic recovery from processing failures (default: True)

    Examples
    --------
    >>> chassis = MetaSignalChassis()
    >>> chassis.register_subsystem('oscillator', oscillator_subsystem)
    >>> context = {'signal': np.array([1, 2, 3])}
    >>> result = chassis.dispatch(context)
    """

    def __init__(self, adversarial_threshold: float = 1.0, enable_recovery: bool = True):
        self.subsystems: Dict[str, Subsystem] = {}
        self.signal_context: Dict[str, Any] = {}
        self.state: str = "initialized"
        self.adversarial_threshold = adversarial_threshold
        self.enable_recovery = enable_recovery
        self.execution_history: List[Tuple[str, Dict[str, Any]]] = []

    def register_subsystem(self, name: str, subsystem: Subsystem, priority: int = 0) -> None:
        """
        Register a subsystem as a polymorphic component.

        Parameters
        ----------
        name : str
            Unique identifier for the subsystem
        subsystem : Subsystem
            The subsystem instance to register
        priority : int
            Execution priority (lower executes first)
        """
        if name in self.subsystems:
            warnings.warn(f"Subsystem '{name}' already registered. Overwriting.", UserWarning)

        self.subsystems[name] = subsystem
        self.signal_context[name] = {}

        if hasattr(subsystem, "config"):
            subsystem.config.priority = priority

    def unregister_subsystem(self, name: str) -> None:
        """Remove a subsystem from the chassis."""
        if name in self.subsystems:
            del self.subsystems[name]
            if name in self.signal_context:
                del self.signal_context[name]

    def handle_adversarial_context(
        self, inputs: NDArray[np.floating], noise_model: float = 0.0
    ) -> NDArray[np.floating]:
        """
        Apply softplus wells to reduce adversarial injections to subthreshold.

        Parameters
        ----------
        inputs : NDArray[np.floating]
            Input signal array potentially containing adversarial noise
        noise_model : float
            Noise model parameter for stabilization

        Returns
        -------
        NDArray[np.floating]
            Cleaned signal with adversarial components reduced
        """
        cleaned = np.array([self.softplus_clean(x, model=noise_model) for x in inputs])
        return cleaned

    @staticmethod
    def softplus_clean(x: float, model: float = 0.0) -> float:
        """
        Scale adversarial noise inputs through stabilization barriers.

        Uses softplus activation: log(1 + exp(x - model))

        Parameters
        ----------
        x : float
            Input value
        model : float
            Model offset parameter

        Returns
        -------
        float
            Stabilized value
        """
        # Clip to avoid overflow in exp
        z = np.clip(x - model, -50, 50)
        return float(np.log(1.0 + np.exp(z)))

    def dispatch(self, initial_context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Cascade process all layers in priority order.

        Executes all enabled subsystems in order of their priority,
        passing the accumulated context through each layer.

        Parameters
        ----------
        initial_context : Optional[Dict[str, Any]]
            Initial signal context to start processing

        Returns
        -------
        Dict[str, Any]
            Final processed signal context
        """
        if initial_context is not None:
            self.signal_context.update(initial_context)

        self.state = "processing"

        # Sort subsystems by priority
        sorted_subsystems = sorted(
            [
                (name, sys)
                for name, sys in self.subsystems.items()
                if getattr(sys.config, "enabled", True)
            ],
            key=lambda x: getattr(x[1].config, "priority", 0),
        )

        # Execute each subsystem
        for name, system in sorted_subsystems:
            try:
                result = system.execute(self.signal_context)
                self.signal_context[name] = result
                self.execution_history.append((name, result))

            except Exception as e:
                if self.enable_recovery:
                    warnings.warn(
                        f"Subsystem '{name}' failed: {str(e)}. " f"Continuing with recovery mode.",
                        RuntimeWarning,
                    )
                    self.signal_context[name] = {"error": str(e), "recovered": True}
                else:
                    raise

        self.state = "completed"
        return self.signal_context

    def reset(self) -> None:
        """Reset the chassis to initial state."""
        self.signal_context = {name: {} for name in self.subsystems.keys()}
        self.state = "initialized"
        self.execution_history = []

    def get_execution_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the last dispatch execution.

        Returns
        -------
        Dict[str, Any]
            Summary containing state, subsystem count, and history
        """
        return {
            "state": self.state,
            "subsystems": list(self.subsystems.keys()),
            "execution_count": len(self.execution_history),
            "history": self.execution_history[-10:],  # Last 10 executions
        }


class MultiOscillator(Subsystem):
    """
    Multi-oscillator subsystem combining oscillation shapes across
    superpositional pathways, integrating transient decay and stable
    harmonic persistence.

    Parameters
    ----------
    config : SubsystemConfig
        Configuration for the oscillator subsystem
    """

    def __init__(self, config: Optional[SubsystemConfig] = None):
        if config is None:
            config = SubsystemConfig(name="multi_oscillator", priority=0)
        super().__init__(config)

        self.models: List[Dict[str, Any]] = []
        self.default_harmonics = {
            "A": 1.0,  # Default amplitude multiplier
            "phi": 0.0,  # Default phase offset
            "exponential_decay": lambda t, C, D: C * np.exp(-D * t),
        }

    def add_oscillator(
        self,
        frequency: float,
        amplitude: float,
        phase_offset: float = 0.0,
        decay_constant: float = 0.0,
    ) -> None:
        """
        Add an oscillator model to the multi-oscillator.

        Parameters
        ----------
        frequency : float
            Oscillation frequency (ω)
        amplitude : float
            Oscillation amplitude (A)
        phase_offset : float
            Phase offset (φ) in radians
        decay_constant : float
            Exponential decay constant (D)
        """
        model = {
            "frequency": frequency,
            "amplitude": amplitude,
            "phase_offset": phase_offset,
            "decay_constant": decay_constant,
            "active": True,
        }
        self.models.append(model)

    def compute_superposition(
        self, t: Union[float, NDArray[np.floating]]
    ) -> Union[float, NDArray[np.floating]]:
        """
        Compute the superposition of all active oscillators at time t.

        Parameters
        ----------
        t : Union[float, NDArray[np.floating]]
            Time value(s) at which to compute the superposition

        Returns
        -------
        Union[float, NDArray[np.floating]]
            Superposition value(s)
        """
        result = np.zeros_like(t, dtype=float)

        for model in self.models:
            if not model["active"]:
                continue

            # Oscillatory component: A * sin(ω*t + φ)
            oscillation = model["amplitude"] * np.sin(
                model["frequency"] * t + model["phase_offset"]
            )

            # Exponential decay: C * exp(-D*t)
            if model["decay_constant"] > 0:
                decay = self.default_harmonics["exponential_decay"](
                    t, model["amplitude"], model["decay_constant"]
                )
                result += oscillation * decay
            else:
                result += oscillation

        return result

    def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the multi-oscillator processing.

        Parameters
        ----------
        context : Dict[str, Any]
            Signal context containing time values

        Returns
        -------
        Dict[str, Any]
            Processed oscillator signals
        """
        # Get time array from context
        t = context.get("t", np.linspace(0, 10, 100))

        # Compute superposition
        signal = self.compute_superposition(t)

        # Return results
        return {
            "time": t,
            "signal": signal,
            "models": self.models,
            "harmonic_count": len([m for m in self.models if m["active"]]),
        }


# Example usage and integration utilities


def create_default_chassis() -> MetaSignalChassis:
    """
    Create a meta-engine chassis with default configuration.

    Returns
    -------
    MetaSignalChassis
        Configured chassis ready for subsystem registration
    """
    return MetaSignalChassis(adversarial_threshold=1.0, enable_recovery=True)


def create_oscillator_subsystem(
    frequencies: List[float],
    amplitudes: List[float],
    phase_offsets: Optional[List[float]] = None,
    decay_constants: Optional[List[float]] = None,
) -> MultiOscillator:
    """
    Create a multi-oscillator subsystem with specified parameters.

    Parameters
    ----------
    frequencies : List[float]
        List of oscillation frequencies
    amplitudes : List[float]
        List of oscillation amplitudes
    phase_offsets : Optional[List[float]]
        List of phase offsets (default: all zeros)
    decay_constants : Optional[List[float]]
        List of decay constants (default: all zeros)

    Returns
    -------
    MultiOscillator
        Configured oscillator subsystem
    """
    config = SubsystemConfig(name="multi_oscillator", priority=0, metamorphic=True)

    oscillator = MultiOscillator(config)

    n = len(frequencies)
    if phase_offsets is None:
        phase_offsets = [0.0] * n
    if decay_constants is None:
        decay_constants = [0.0] * n

    for freq, amp, phase, decay in zip(frequencies, amplitudes, phase_offsets, decay_constants):
        oscillator.add_oscillator(freq, amp, phase, decay)

    return oscillator
