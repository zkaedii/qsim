#!/usr/bin/env python3
"""
Meta-Engine Example: Multiform Signal Processing

This example demonstrates the usage of the MetaSignalChassis and MultiOscillator
subsystems to create a complex signal processing pipeline with adaptive harmonics.

Features demonstrated:
- Polymorphic subsystem registration
- Adversarial noise reduction with softplus cleaning
- Multi-oscillator superposition
- Cascade dispatch mechanism
- Metamorphic adaptation
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from hmodelz.engines.meta_signal_chassis import (
    MetaSignalChassis,
    MultiOscillator,
    Subsystem,
    SubsystemConfig,
    create_default_chassis,
    create_oscillator_subsystem,
)


def example_1_basic_oscillator():
    """Example 1: Basic multi-oscillator usage."""
    print("=" * 70)
    print("Example 1: Basic Multi-Oscillator")
    print("=" * 70)

    # Create a multi-oscillator with harmonics
    oscillator = create_oscillator_subsystem(
        frequencies=[1.0, 2.0, 3.0],  # Fundamental + harmonics
        amplitudes=[1.0, 0.5, 0.25],  # Decreasing amplitudes
        phase_offsets=[0.0, np.pi / 4, np.pi / 2],  # Phase shifts
        decay_constants=[0.0, 0.1, 0.2],  # Some decay
    )

    # Generate time array
    t = np.linspace(0, 10, 1000)

    # Execute oscillator
    result = oscillator.execute({"t": t})

    print(f"Generated signal with {result['harmonic_count']} harmonics")
    print(f"Signal shape: {result['signal'].shape}")
    print(f"Signal range: [{result['signal'].min():.3f}, {result['signal'].max():.3f}]")
    print()


def example_2_chassis_with_oscillator():
    """Example 2: Meta-engine chassis with oscillator subsystem."""
    print("=" * 70)
    print("Example 2: Meta-Engine Chassis with Oscillator")
    print("=" * 70)

    # Create chassis
    chassis = create_default_chassis()

    # Create oscillator subsystems with different characteristics
    low_freq_osc = create_oscillator_subsystem(frequencies=[0.5, 1.0], amplitudes=[1.0, 0.3])

    high_freq_osc = create_oscillator_subsystem(frequencies=[5.0, 10.0], amplitudes=[0.5, 0.2])

    # Register subsystems with priorities
    chassis.register_subsystem("low_frequency", low_freq_osc, priority=1)
    chassis.register_subsystem("high_frequency", high_freq_osc, priority=2)

    # Dispatch with time array
    t = np.linspace(0, 10, 1000)
    result = chassis.dispatch({"t": t})

    print(f"Chassis state: {chassis.state}")
    print(f"Subsystems processed: {list(result.keys())}")
    print(f"Low-freq harmonics: {result['low_frequency']['harmonic_count']}")
    print(f"High-freq harmonics: {result['high_frequency']['harmonic_count']}")
    print()


def example_3_adversarial_noise_handling():
    """Example 3: Adversarial noise reduction."""
    print("=" * 70)
    print("Example 3: Adversarial Noise Reduction")
    print("=" * 70)

    chassis = create_default_chassis()

    # Create noisy signal with adversarial spikes
    clean_signal = np.sin(np.linspace(0, 4 * np.pi, 100))
    adversarial_spikes = np.zeros_like(clean_signal)
    adversarial_spikes[25] = 100.0  # Large spike
    adversarial_spikes[50] = -80.0  # Large negative spike
    adversarial_spikes[75] = 120.0  # Another spike

    noisy_signal = clean_signal + adversarial_spikes

    # Clean the signal
    cleaned_signal = chassis.handle_adversarial_context(noisy_signal, noise_model=0.0)

    print(f"Original signal range: [{noisy_signal.min():.3f}, {noisy_signal.max():.3f}]")
    print(f"Cleaned signal range: [{cleaned_signal.min():.3f}, {cleaned_signal.max():.3f}]")
    print(f"Adversarial spikes detected at indices: [25, 50, 75]")
    print(f"Max spike reduction: {noisy_signal.max() - cleaned_signal.max():.3f}")
    print()


def example_4_metamorphic_adaptation():
    """Example 4: Metamorphic subsystem adaptation."""
    print("=" * 70)
    print("Example 4: Metamorphic Adaptation")
    print("=" * 70)

    # Create metamorphic oscillator
    config = SubsystemConfig(name="adaptive_oscillator", priority=0, metamorphic=True)

    oscillator = MultiOscillator(config)
    oscillator.add_oscillator(frequency=1.0, amplitude=1.0)

    # Initial execution
    t = np.linspace(0, 10, 100)
    result_before = oscillator.execute({"t": t})

    print(f"Before adaptation:")
    print(f"  Active models: {len(oscillator.models)}")
    print(f"  State: {oscillator.state}")

    # Adapt based on feedback
    feedback = {"learning_rate": 0.01, "optimization": "enabled", "target_frequency": 2.0}
    oscillator.adapt(feedback)

    print(f"\nAfter adaptation:")
    print(f"  State: {oscillator.state}")
    print(f"  Learning rate: {oscillator.state.get('learning_rate')}")
    print(f"  Optimization: {oscillator.state.get('optimization')}")
    print()


def example_5_complex_pipeline():
    """Example 5: Complex signal processing pipeline."""
    print("=" * 70)
    print("Example 5: Complex Signal Processing Pipeline")
    print("=" * 70)

    # Create chassis with recovery enabled
    chassis = MetaSignalChassis(adversarial_threshold=1.0, enable_recovery=True)

    # Create multiple oscillator subsystems
    fundamental = create_oscillator_subsystem([1.0], [1.0])
    second_harmonic = create_oscillator_subsystem([2.0], [0.5])
    third_harmonic = create_oscillator_subsystem([3.0], [0.25])

    # Register in priority order
    chassis.register_subsystem("fundamental", fundamental, priority=1)
    chassis.register_subsystem("second_harmonic", second_harmonic, priority=2)
    chassis.register_subsystem("third_harmonic", third_harmonic, priority=3)

    # Execute pipeline
    t = np.linspace(0, 10, 1000)
    result = chassis.dispatch({"t": t})

    # Get execution summary
    summary = chassis.get_execution_summary()

    print(f"Pipeline state: {summary['state']}")
    print(f"Subsystems: {', '.join(summary['subsystems'])}")
    print(f"Executions: {summary['execution_count']}")

    # Combine signals
    combined_signal = (
        result["fundamental"]["signal"]
        + result["second_harmonic"]["signal"]
        + result["third_harmonic"]["signal"]
    )

    print(f"\nCombined signal statistics:")
    print(f"  Mean: {combined_signal.mean():.6f}")
    print(f"  Std: {combined_signal.std():.6f}")
    print(f"  Range: [{combined_signal.min():.3f}, {combined_signal.max():.3f}]")
    print()


def example_6_visualization():
    """Example 6: Visualize oscillator superposition."""
    print("=" * 70)
    print("Example 6: Oscillator Superposition Visualization")
    print("=" * 70)

    try:
        # Create oscillator
        oscillator = create_oscillator_subsystem(
            frequencies=[1.0, 3.0, 5.0],
            amplitudes=[1.0, 0.5, 0.25],
            decay_constants=[0.0, 0.1, 0.2],
        )

        # Generate signal
        t = np.linspace(0, 10, 1000)
        result = oscillator.execute({"t": t})
        signal = result["signal"]

        # Create plot
        plt.figure(figsize=(12, 6))

        # Plot superposition
        plt.subplot(2, 1, 1)
        plt.plot(t, signal, "b-", linewidth=1.5)
        plt.title("Multi-Oscillator Superposition")
        plt.xlabel("Time")
        plt.ylabel("Amplitude")
        plt.grid(True, alpha=0.3)

        # Plot individual components
        plt.subplot(2, 1, 2)
        for i, model in enumerate(oscillator.models):
            freq = model["frequency"]
            amp = model["amplitude"]
            decay = model["decay_constant"]

            # Compute individual component
            component = amp * np.sin(freq * t)
            if decay > 0:
                component *= np.exp(-decay * t)

            plt.plot(t, component, label=f"f={freq}, A={amp}, D={decay}", alpha=0.7)

        plt.title("Individual Oscillator Components")
        plt.xlabel("Time")
        plt.ylabel("Amplitude")
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.tight_layout()

        # Save plot
        output_path = Path(__file__).parent / "meta_engine_oscillators.png"
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Visualization saved to: {output_path}")
        print()

    except Exception as e:
        print(f"Visualization failed (matplotlib required): {e}")
        print()


def main():
    """Run all examples."""
    print("\n")
    print("╔" + "═" * 68 + "╗")
    print("║" + " " * 15 + "META-ENGINE DEMONSTRATION" + " " * 28 + "║")
    print("║" + " " * 10 + "Multiform Signal Processing Chassis" + " " * 22 + "║")
    print("╚" + "═" * 68 + "╝")
    print("\n")

    # Run examples
    example_1_basic_oscillator()
    example_2_chassis_with_oscillator()
    example_3_adversarial_noise_handling()
    example_4_metamorphic_adaptation()
    example_5_complex_pipeline()
    example_6_visualization()

    print("=" * 70)
    print("All examples completed successfully!")
    print("=" * 70)


if __name__ == "__main__":
    main()
