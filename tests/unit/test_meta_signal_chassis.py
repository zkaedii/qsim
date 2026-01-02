#!/usr/bin/env python3
"""
Comprehensive tests for meta_signal_chassis.py

Tests cover:
- SubsystemConfig dataclass
- Subsystem abstract base class
- MetaSignalChassis core functionality
- MultiOscillator subsystem
- Adversarial noise handling
- Polymorphic subsystem registration
- Cascade dispatch mechanism
- Factory functions
"""

import pytest
import numpy as np
import sys
from pathlib import Path
from typing import Dict, Any

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from hmodelz.engines.meta_signal_chassis import (
    MetaSignalChassis,
    MultiOscillator,
    Subsystem,
    SubsystemConfig,
    create_default_chassis,
    create_oscillator_subsystem,
)


class TestSubsystemConfig:
    """Tests for SubsystemConfig dataclass."""

    def test_default_config(self):
        """Test default configuration values."""
        config = SubsystemConfig(name="test")
        assert config.name == "test"
        assert config.priority == 0
        assert config.enabled is True
        assert config.metamorphic is False
        assert config.parameters == {}

    def test_custom_config(self):
        """Test custom configuration values."""
        config = SubsystemConfig(
            name="custom", priority=10, enabled=False, metamorphic=True, parameters={"key": "value"}
        )
        assert config.name == "custom"
        assert config.priority == 10
        assert config.enabled is False
        assert config.metamorphic is True
        assert config.parameters == {"key": "value"}


class MockSubsystem(Subsystem):
    """Mock subsystem for testing."""

    def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Simple execution that doubles input values."""
        value = context.get("value", 0)
        return {"value": value * 2, "processed": True}


class FailingSubsystem(Subsystem):
    """Subsystem that always fails for testing recovery."""

    def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Always raise an error."""
        raise RuntimeError("Intentional failure")


class TestSubsystem:
    """Tests for Subsystem abstract base class."""

    def test_subsystem_creation(self):
        """Test creating a subsystem."""
        config = SubsystemConfig(name="mock")
        subsystem = MockSubsystem(config)
        assert subsystem.config.name == "mock"
        assert subsystem.state == {}

    def test_subsystem_execute(self):
        """Test subsystem execution."""
        config = SubsystemConfig(name="mock")
        subsystem = MockSubsystem(config)
        context = {"value": 5}
        result = subsystem.execute(context)
        assert result["value"] == 10
        assert result["processed"] is True

    def test_subsystem_adapt_metamorphic(self):
        """Test metamorphic adaptation."""
        config = SubsystemConfig(name="mock", metamorphic=True)
        subsystem = MockSubsystem(config)
        feedback = {"learning_rate": 0.01}
        subsystem.adapt(feedback)
        assert subsystem.state["learning_rate"] == 0.01

    def test_subsystem_adapt_non_metamorphic(self):
        """Test that non-metamorphic subsystems don't adapt."""
        config = SubsystemConfig(name="mock", metamorphic=False)
        subsystem = MockSubsystem(config)
        feedback = {"learning_rate": 0.01}
        subsystem.adapt(feedback)
        assert "learning_rate" not in subsystem.state


class TestMetaSignalChassis:
    """Tests for MetaSignalChassis core functionality."""

    def test_chassis_initialization(self):
        """Test chassis initialization."""
        chassis = MetaSignalChassis()
        assert chassis.state == "initialized"
        assert chassis.subsystems == {}
        assert chassis.signal_context == {}
        assert chassis.adversarial_threshold == 1.0
        assert chassis.enable_recovery is True

    def test_custom_initialization(self):
        """Test chassis with custom parameters."""
        chassis = MetaSignalChassis(adversarial_threshold=2.0, enable_recovery=False)
        assert chassis.adversarial_threshold == 2.0
        assert chassis.enable_recovery is False

    def test_register_subsystem(self):
        """Test subsystem registration."""
        chassis = MetaSignalChassis()
        config = SubsystemConfig(name="mock")
        subsystem = MockSubsystem(config)

        chassis.register_subsystem("mock", subsystem)
        assert "mock" in chassis.subsystems
        assert "mock" in chassis.signal_context

    def test_register_subsystem_with_priority(self):
        """Test subsystem registration with priority."""
        chassis = MetaSignalChassis()
        config = SubsystemConfig(name="mock")
        subsystem = MockSubsystem(config)

        chassis.register_subsystem("mock", subsystem, priority=5)
        assert subsystem.config.priority == 5

    def test_register_duplicate_subsystem(self):
        """Test overwriting a registered subsystem."""
        chassis = MetaSignalChassis()
        config1 = SubsystemConfig(name="mock1")
        subsystem1 = MockSubsystem(config1)
        config2 = SubsystemConfig(name="mock2")
        subsystem2 = MockSubsystem(config2)

        chassis.register_subsystem("test", subsystem1)
        with pytest.warns(UserWarning):
            chassis.register_subsystem("test", subsystem2)

        assert chassis.subsystems["test"] is subsystem2

    def test_unregister_subsystem(self):
        """Test subsystem unregistration."""
        chassis = MetaSignalChassis()
        config = SubsystemConfig(name="mock")
        subsystem = MockSubsystem(config)

        chassis.register_subsystem("mock", subsystem)
        chassis.unregister_subsystem("mock")
        assert "mock" not in chassis.subsystems
        assert "mock" not in chassis.signal_context

    def test_softplus_clean(self):
        """Test softplus cleaning function."""
        # Test basic softplus
        result = MetaSignalChassis.softplus_clean(0.0, model=0.0)
        assert np.isclose(result, np.log(2.0))

        # Test with model offset
        result = MetaSignalChassis.softplus_clean(1.0, model=0.5)
        assert result > 0

        # Test extreme values (should be clipped)
        result = MetaSignalChassis.softplus_clean(100.0, model=0.0)
        assert np.isfinite(result)

        result = MetaSignalChassis.softplus_clean(-100.0, model=0.0)
        assert np.isclose(result, 0.0, atol=1e-10)

    def test_handle_adversarial_context(self):
        """Test adversarial noise handling."""
        chassis = MetaSignalChassis()
        inputs = np.array([0.0, 1.0, 2.0, -1.0])
        cleaned = chassis.handle_adversarial_context(inputs, noise_model=0.0)

        assert cleaned.shape == inputs.shape
        assert np.all(np.isfinite(cleaned))
        assert np.all(cleaned >= 0)  # Softplus is always non-negative

    def test_dispatch_empty(self):
        """Test dispatch with no subsystems."""
        chassis = MetaSignalChassis()
        result = chassis.dispatch({"input": "test"})
        assert chassis.state == "completed"
        assert "input" in result

    def test_dispatch_single_subsystem(self):
        """Test dispatch with a single subsystem."""
        chassis = MetaSignalChassis()
        config = SubsystemConfig(name="mock")
        subsystem = MockSubsystem(config)
        chassis.register_subsystem("mock", subsystem)

        result = chassis.dispatch({"value": 10})
        assert chassis.state == "completed"
        assert result["mock"]["value"] == 20
        assert result["mock"]["processed"] is True

    def test_dispatch_multiple_subsystems(self):
        """Test dispatch with multiple subsystems in priority order."""
        chassis = MetaSignalChassis()

        config1 = SubsystemConfig(name="first")
        subsystem1 = MockSubsystem(config1)
        chassis.register_subsystem("first", subsystem1, priority=1)

        config2 = SubsystemConfig(name="second")
        subsystem2 = MockSubsystem(config2)
        chassis.register_subsystem("second", subsystem2, priority=2)

        result = chassis.dispatch({"value": 5})
        assert "first" in result
        assert "second" in result

    def test_dispatch_with_disabled_subsystem(self):
        """Test that disabled subsystems are not executed."""
        chassis = MetaSignalChassis()

        config = SubsystemConfig(name="disabled", enabled=False)
        subsystem = MockSubsystem(config)
        chassis.register_subsystem("disabled", subsystem)

        result = chassis.dispatch({"value": 5})
        assert "disabled" not in result or result["disabled"] == {}

    def test_dispatch_with_recovery(self):
        """Test dispatch with error recovery enabled."""
        chassis = MetaSignalChassis(enable_recovery=True)

        config = SubsystemConfig(name="failing")
        subsystem = FailingSubsystem(config)
        chassis.register_subsystem("failing", subsystem)

        # Should not raise, should recover
        with pytest.warns(RuntimeWarning):
            result = chassis.dispatch({"value": 5})

        assert "failing" in result
        assert "error" in result["failing"]
        assert result["failing"]["recovered"] is True

    def test_dispatch_without_recovery(self):
        """Test dispatch with error recovery disabled."""
        chassis = MetaSignalChassis(enable_recovery=False)

        config = SubsystemConfig(name="failing")
        subsystem = FailingSubsystem(config)
        chassis.register_subsystem("failing", subsystem)

        # Should raise
        with pytest.raises(RuntimeError):
            chassis.dispatch({"value": 5})

    def test_reset(self):
        """Test chassis reset."""
        chassis = MetaSignalChassis()
        config = SubsystemConfig(name="mock")
        subsystem = MockSubsystem(config)
        chassis.register_subsystem("mock", subsystem)

        chassis.dispatch({"value": 5})
        assert chassis.state == "completed"

        chassis.reset()
        assert chassis.state == "initialized"
        assert chassis.signal_context == {"mock": {}}
        assert chassis.execution_history == []

    def test_execution_summary(self):
        """Test execution summary."""
        chassis = MetaSignalChassis()
        config = SubsystemConfig(name="mock")
        subsystem = MockSubsystem(config)
        chassis.register_subsystem("mock", subsystem)

        chassis.dispatch({"value": 5})
        summary = chassis.get_execution_summary()

        assert summary["state"] == "completed"
        assert "mock" in summary["subsystems"]
        assert summary["execution_count"] > 0


class TestMultiOscillator:
    """Tests for MultiOscillator subsystem."""

    def test_oscillator_initialization(self):
        """Test oscillator initialization."""
        config = SubsystemConfig(name="oscillator")
        oscillator = MultiOscillator(config)

        assert oscillator.config.name == "oscillator"
        assert oscillator.models == []
        assert oscillator.default_harmonics["A"] == 1.0
        assert oscillator.default_harmonics["phi"] == 0.0

    def test_oscillator_default_config(self):
        """Test oscillator with default config."""
        oscillator = MultiOscillator()
        assert oscillator.config.name == "multi_oscillator"
        assert oscillator.config.priority == 0

    def test_add_oscillator(self):
        """Test adding an oscillator model."""
        oscillator = MultiOscillator()
        oscillator.add_oscillator(
            frequency=1.0, amplitude=2.0, phase_offset=np.pi / 4, decay_constant=0.1
        )

        assert len(oscillator.models) == 1
        model = oscillator.models[0]
        assert model["frequency"] == 1.0
        assert model["amplitude"] == 2.0
        assert model["phase_offset"] == np.pi / 4
        assert model["decay_constant"] == 0.1
        assert model["active"] is True

    def test_compute_superposition_single_oscillator(self):
        """Test superposition with a single oscillator."""
        oscillator = MultiOscillator()
        oscillator.add_oscillator(frequency=1.0, amplitude=1.0)

        t = np.array([0.0, np.pi / 2, np.pi])
        result = oscillator.compute_superposition(t)

        # At t=0: sin(0) = 0
        # At t=π/2: sin(π/2) = 1
        # At t=π: sin(π) = 0
        assert np.isclose(result[0], 0.0, atol=1e-10)
        assert np.isclose(result[1], 1.0, atol=1e-10)
        assert np.isclose(result[2], 0.0, atol=1e-10)

    def test_compute_superposition_with_decay(self):
        """Test superposition with exponential decay."""
        oscillator = MultiOscillator()
        oscillator.add_oscillator(frequency=1.0, amplitude=1.0, decay_constant=0.5)

        t = np.array([0.0, 1.0, 2.0])
        result = oscillator.compute_superposition(t)

        # Result should decay exponentially
        assert np.abs(result[0]) > np.abs(result[1]) or np.isclose(result[0], 0.0)
        assert np.abs(result[1]) > np.abs(result[2]) or np.isclose(result[1], 0.0)

    def test_compute_superposition_multiple_oscillators(self):
        """Test superposition with multiple oscillators."""
        oscillator = MultiOscillator()
        oscillator.add_oscillator(frequency=1.0, amplitude=1.0)
        oscillator.add_oscillator(frequency=2.0, amplitude=0.5)

        t = np.linspace(0, 2 * np.pi, 100)
        result = oscillator.compute_superposition(t)

        assert result.shape == t.shape
        assert np.all(np.isfinite(result))

    def test_compute_superposition_inactive_oscillator(self):
        """Test that inactive oscillators are not included."""
        oscillator = MultiOscillator()
        oscillator.add_oscillator(frequency=1.0, amplitude=1.0)
        oscillator.models[0]["active"] = False

        t = np.linspace(0, 2 * np.pi, 100)
        result = oscillator.compute_superposition(t)

        # All zeros since the only oscillator is inactive
        assert np.allclose(result, 0.0)

    def test_execute(self):
        """Test oscillator execution in chassis context."""
        oscillator = MultiOscillator()
        oscillator.add_oscillator(frequency=1.0, amplitude=1.0)

        context = {"t": np.linspace(0, 10, 100)}
        result = oscillator.execute(context)

        assert "time" in result
        assert "signal" in result
        assert "models" in result
        assert "harmonic_count" in result
        assert result["harmonic_count"] == 1
        assert len(result["signal"]) == 100

    def test_execute_default_time(self):
        """Test oscillator execution with default time array."""
        oscillator = MultiOscillator()
        oscillator.add_oscillator(frequency=1.0, amplitude=1.0)

        result = oscillator.execute({})

        assert "time" in result
        assert "signal" in result
        assert len(result["signal"]) == 100  # Default


class TestFactoryFunctions:
    """Tests for factory functions."""

    def test_create_default_chassis(self):
        """Test creating a default chassis."""
        chassis = create_default_chassis()
        assert isinstance(chassis, MetaSignalChassis)
        assert chassis.adversarial_threshold == 1.0
        assert chassis.enable_recovery is True

    def test_create_oscillator_subsystem(self):
        """Test creating an oscillator subsystem."""
        frequencies = [1.0, 2.0, 3.0]
        amplitudes = [1.0, 0.5, 0.25]

        oscillator = create_oscillator_subsystem(frequencies, amplitudes)

        assert isinstance(oscillator, MultiOscillator)
        assert len(oscillator.models) == 3
        assert oscillator.config.name == "multi_oscillator"
        assert oscillator.config.metamorphic is True

    def test_create_oscillator_subsystem_with_phase(self):
        """Test creating oscillator with phase offsets."""
        frequencies = [1.0, 2.0]
        amplitudes = [1.0, 0.5]
        phase_offsets = [0.0, np.pi / 4]

        oscillator = create_oscillator_subsystem(frequencies, amplitudes, phase_offsets)

        assert oscillator.models[0]["phase_offset"] == 0.0
        assert oscillator.models[1]["phase_offset"] == np.pi / 4

    def test_create_oscillator_subsystem_with_decay(self):
        """Test creating oscillator with decay constants."""
        frequencies = [1.0, 2.0]
        amplitudes = [1.0, 0.5]
        decay_constants = [0.1, 0.2]

        oscillator = create_oscillator_subsystem(
            frequencies, amplitudes, decay_constants=decay_constants
        )

        assert oscillator.models[0]["decay_constant"] == 0.1
        assert oscillator.models[1]["decay_constant"] == 0.2


class TestIntegration:
    """Integration tests for the complete meta-engine."""

    def test_chassis_with_oscillator(self):
        """Test chassis with oscillator subsystem."""
        chassis = create_default_chassis()
        oscillator = create_oscillator_subsystem(frequencies=[1.0, 2.0], amplitudes=[1.0, 0.5])

        chassis.register_subsystem("oscillator", oscillator)
        result = chassis.dispatch({"t": np.linspace(0, 10, 100)})

        assert chassis.state == "completed"
        assert "oscillator" in result
        assert "signal" in result["oscillator"]

    def test_chassis_with_multiple_oscillators(self):
        """Test chassis with multiple oscillator subsystems."""
        chassis = create_default_chassis()

        osc1 = create_oscillator_subsystem([1.0], [1.0])
        osc2 = create_oscillator_subsystem([2.0], [0.5])

        chassis.register_subsystem("low_freq", osc1, priority=1)
        chassis.register_subsystem("high_freq", osc2, priority=2)

        t = np.linspace(0, 10, 100)
        result = chassis.dispatch({"t": t})

        assert "low_freq" in result
        assert "high_freq" in result

    def test_adversarial_resilience(self):
        """Test adversarial noise cleaning in context."""
        chassis = create_default_chassis()

        # Add noisy signal
        noisy_signal = np.array([1.0, 2.0, 100.0, -50.0, 3.0])
        cleaned = chassis.handle_adversarial_context(noisy_signal, noise_model=0.0)

        # All values should be finite and non-negative
        assert np.all(np.isfinite(cleaned))
        assert np.all(cleaned >= 0)

    def test_metamorphic_adaptation(self):
        """Test metamorphic subsystem adaptation."""
        chassis = create_default_chassis()
        oscillator = create_oscillator_subsystem([1.0], [1.0])

        chassis.register_subsystem("oscillator", oscillator)
        chassis.dispatch({"t": np.linspace(0, 10, 100)})

        # Adapt based on feedback
        feedback = {"learning_rate": 0.01, "optimization": "enabled"}
        oscillator.adapt(feedback)

        assert oscillator.state["learning_rate"] == 0.01
        assert oscillator.state["optimization"] == "enabled"
