#!/usr/bin/env python3
"""
Basic Usage Example for MCSO

This example demonstrates the core functionality of the Multi-Component
Stochastic Oscillator library.
"""

import numpy as np
from mcso import StochasticOscillator, OscillatorConfig
from mcso.analysis import compute_statistics, spectral_analysis, create_summary_report


def main():
    print("=" * 70)
    print("  MCSO - Multi-Component Stochastic Oscillator")
    print("  Basic Usage Example")
    print("=" * 70)

    # =========================================================================
    # Example 1: Default Configuration
    # =========================================================================
    print("\n[1] Simulating with default configuration...")

    osc = StochasticOscillator()
    result = osc.simulate(t_max=100, dt=1.0)

    print(f"    Simulated {len(result['times'])} time points")
    print(f"    Time range: [{result['times'][0]:.1f}, {result['times'][-1]:.1f}]")

    # Compute statistics
    stats = compute_statistics(result['values'])
    print(f"\n    Statistics:")
    print(f"      Mean:     {stats.mean:8.4f}")
    print(f"      Std:      {stats.std:8.4f}")
    print(f"      Min:      {stats.min:8.4f}")
    print(f"      Max:      {stats.max:8.4f}")
    print(f"      ACF(1):   {stats.autocorr_lag1:8.4f}")

    # =========================================================================
    # Example 2: Custom Configuration
    # =========================================================================
    print("\n[2] Simulating with custom configuration...")

    config = OscillatorConfig(
        n_components=3,           # Fewer oscillatory modes
        noise_scale=0.5,          # Higher noise
        memory_strength=2.0,      # Stronger memory feedback
        memory_delay=2.0,         # Longer memory delay
        seed=42                   # Reproducible
    )

    osc_custom = StochasticOscillator(config=config)
    result_custom = osc_custom.simulate(t_max=100, dt=0.5)

    stats_custom = compute_statistics(result_custom['values'])
    print(f"    With higher noise (sigma=0.5):")
    print(f"      Std:      {stats_custom.std:8.4f}  (vs {stats.std:.4f} baseline)")

    # =========================================================================
    # Example 3: Spectral Analysis
    # =========================================================================
    print("\n[3] Performing spectral analysis...")

    spec = spectral_analysis(result['values'], dt=1.0)
    print(f"    Dominant frequency:  {spec.dominant_freq:.6f}")
    print(f"    Spectral entropy:    {spec.spectral_entropy:.4f}  (0=tone, 1=noise)")
    print(f"    Bandwidth:           {spec.bandwidth:.6f}")

    # =========================================================================
    # Example 4: Ensemble Simulation
    # =========================================================================
    print("\n[4] Running ensemble simulation (20 realizations)...")

    osc_ensemble = StochasticOscillator(n_components=5, noise_scale=0.2, seed=0)
    ensemble = osc_ensemble.simulate_ensemble(
        n_realizations=20,
        t_max=50,
        dt=1.0
    )

    print(f"    Ensemble shape: {ensemble['ensemble'].shape}")
    print(f"    Mean trajectory range: [{ensemble['mean'].min():.4f}, {ensemble['mean'].max():.4f}]")
    print(f"    Final time std: {ensemble['std'][-1]:.4f}")

    # =========================================================================
    # Example 5: Custom Control Input
    # =========================================================================
    print("\n[5] Simulating with custom control input...")

    def step_control(t):
        """Step control: off before t=25, on after."""
        return 1.0 if t > 25 else 0.0

    osc_control = StochasticOscillator(
        n_components=5,
        control_gain=0.5,
        noise_scale=0.1,
        seed=123
    )

    result_control = osc_control.simulate(
        t_max=50,
        dt=0.5,
        control_fn=step_control
    )

    # Compare before/after control
    times = result_control['times']
    values = result_control['values']

    before_idx = times < 25
    after_idx = times >= 25

    mean_before = np.mean(values[before_idx])
    mean_after = np.mean(values[after_idx])

    print(f"    Mean before control (t<25): {mean_before:.4f}")
    print(f"    Mean after control (t>=25): {mean_after:.4f}")
    print(f"    Effect of control: {mean_after - mean_before:+.4f}")

    # =========================================================================
    # Example 6: Generate Report
    # =========================================================================
    print("\n[6] Generating analysis report...")

    report = create_summary_report(result)
    print(report[:500] + "...")

    print("\n" + "=" * 70)
    print("  Examples completed successfully!")
    print("=" * 70)


if __name__ == "__main__":
    main()
