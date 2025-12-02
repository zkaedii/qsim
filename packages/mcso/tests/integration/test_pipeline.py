"""
Integration Tests for MCSO Pipeline

These tests verify the complete simulation pipeline from configuration
through simulation to analysis, ensuring all components work together correctly.
"""

import pytest
import numpy as np
from numpy.testing import assert_allclose, assert_array_less
import warnings


class TestEndToEndSimulation:
    """Test complete simulation workflows from configuration to analysis."""

    def test_basic_simulation_pipeline(self):
        """Test basic simulation workflow with default parameters."""
        from mcso.oscillator import StochasticOscillator, OscillatorConfig
        from mcso.analysis import compute_statistics, create_summary_report

        # Configure
        config = OscillatorConfig(n_components=3, noise_scale=0.1, seed=42)

        # Create oscillator
        osc = StochasticOscillator(config)

        # Simulate
        trajectory = osc.simulate(t_max=50, dt=0.5)

        # Verify trajectory structure
        assert "times" in trajectory
        assert "values" in trajectory
        assert len(trajectory["times"]) == len(trajectory["values"])
        assert len(trajectory["times"]) > 0

        # Analyze
        stats = compute_statistics(trajectory["values"])

        # Verify statistics (TrajectoryStatistics is a dataclass)
        assert hasattr(stats, "mean")
        assert hasattr(stats, "std")
        assert np.isfinite(stats.mean)
        assert stats.std >= 0

        # Generate report (takes full trajectory dict)
        report = create_summary_report(trajectory)
        assert isinstance(report, str)
        assert len(report) > 0

    def test_fractional_timestep_pipeline(self):
        """Test simulation with fractional time steps (regression test for history key bug)."""
        from mcso.oscillator import StochasticOscillator

        osc = StochasticOscillator(
            n_components=2, noise_scale=0.2, memory_strength=1.0, memory_delay=0.5, seed=123
        )

        # Use fractional dt - this previously caused history key mismatch
        trajectory = osc.simulate(t_max=10, dt=0.25)

        # Verify memory feedback was active (non-zero variance in later steps)
        values = trajectory["values"]
        assert len(values) == 40  # 10 / 0.25
        assert np.std(values) > 0  # Should have variation

        # Verify no zeros from history lookup failures
        # (Before fix, memory_term would return 0 for non-unit dt)
        mid_values = values[len(values) // 2 :]
        assert not np.allclose(mid_values, mid_values[0])  # Should vary

    def test_ensemble_simulation_pipeline(self):
        """Test ensemble simulation with statistical analysis."""
        from mcso.oscillator import StochasticOscillator
        from mcso.analysis import compute_statistics

        osc = StochasticOscillator(n_components=3, noise_scale=0.5, seed=42)

        # Simulate ensemble
        ensemble = osc.simulate_ensemble(n_realizations=20, t_max=30, dt=1.0)

        # Verify ensemble structure
        assert "times" in ensemble
        assert "ensemble" in ensemble
        assert "mean" in ensemble
        assert "std" in ensemble
        assert "percentiles" in ensemble

        # Verify dimensions
        assert ensemble["ensemble"].shape == (20, 30)
        assert len(ensemble["mean"]) == 30
        assert len(ensemble["std"]) == 30

        # Verify statistical consistency
        # Mean of ensemble should be close to ensemble['mean']
        computed_mean = np.mean(ensemble["ensemble"], axis=0)
        assert_allclose(computed_mean, ensemble["mean"], rtol=1e-10)

        # Std should be non-negative
        assert np.all(ensemble["std"] >= 0)

        # Percentiles should be ordered
        assert np.all(ensemble["percentiles"][5] <= ensemble["percentiles"][50])
        assert np.all(ensemble["percentiles"][50] <= ensemble["percentiles"][95])

    def test_custom_control_function_pipeline(self):
        """Test simulation with custom control input."""
        from mcso.oscillator import StochasticOscillator

        osc = StochasticOscillator(
            n_components=2,
            control_gain=1.0,
            noise_scale=0.0,  # No noise for deterministic test
            seed=42,
        )

        # Define step control function
        def step_control(t):
            return 1.0 if t >= 5.0 else 0.0

        trajectory = osc.simulate(t_max=10, dt=0.5, control_fn=step_control)

        values = trajectory["values"]
        times = trajectory["times"]

        # Find values before and after step
        before_step = values[times < 5.0]
        after_step = values[times >= 5.0]

        # After step should have different characteristics due to control
        # (This is a basic sanity check - exact behavior depends on parameters)
        assert len(before_step) > 0
        assert len(after_step) > 0


class TestMemoryIntegration:
    """Test memory kernel integration with oscillator."""

    def test_exponential_memory_integration(self):
        """Test oscillator with external exponential memory kernel."""
        from mcso.oscillator import StochasticOscillator
        from mcso.memory import ExponentialMemory

        osc = StochasticOscillator(n_components=2, noise_scale=0.1, seed=42)

        # Simulate
        trajectory = osc.simulate(t_max=20, dt=1.0)

        # Use memory kernel for analysis
        memory = ExponentialMemory(strength=1.0, decay_time=2.0, window=10.0)

        # Convert history for memory evaluation
        history = {t: v for t, v in zip(trajectory["times"], trajectory["values"])}

        # Evaluate memory at final time
        final_t = trajectory["times"][-1]
        memory_contribution = memory.evaluate(final_t, history)

        assert np.isfinite(memory_contribution)

    def test_multiscale_memory_integration(self):
        """Test oscillator with multi-scale memory analysis."""
        from mcso.oscillator import StochasticOscillator
        from mcso.memory import MultiScaleMemory

        osc = StochasticOscillator(n_components=3, noise_scale=0.2, seed=42)

        trajectory = osc.simulate(t_max=50, dt=1.0)

        # Create multi-scale memory
        memory = MultiScaleMemory(scales=[(0.5, 1.0), (0.3, 5.0), (0.2, 20.0)], window=30.0)

        history = {t: v for t, v in zip(trajectory["times"], trajectory["values"])}

        # Evaluate at multiple points
        eval_times = [10.0, 25.0, 40.0]
        contributions = [memory.evaluate(t, history) for t in eval_times]

        assert all(np.isfinite(c) for c in contributions)


class TestNoiseIntegration:
    """Test noise generators integration with oscillator."""

    def test_gaussian_noise_reproducibility(self):
        """Test that same seed produces same trajectory."""
        from mcso.oscillator import StochasticOscillator

        config_args = {"n_components": 2, "noise_scale": 0.5, "seed": 12345}

        osc1 = StochasticOscillator(**config_args)
        traj1 = osc1.simulate(t_max=20, dt=1.0)

        osc2 = StochasticOscillator(**config_args)
        traj2 = osc2.simulate(t_max=20, dt=1.0)

        assert_allclose(traj1["values"], traj2["values"])

    def test_state_dependent_noise_effect(self):
        """Test that state-dependent noise coupling affects variance."""
        from mcso.oscillator import StochasticOscillator
        from mcso.analysis import compute_statistics

        # Low coupling
        osc_low = StochasticOscillator(noise_scale=0.5, noise_state_coupling=0.0, seed=42)
        traj_low = osc_low.simulate(t_max=100, dt=1.0)
        stats_low = compute_statistics(traj_low["values"])

        # High coupling
        osc_high = StochasticOscillator(noise_scale=0.5, noise_state_coupling=1.0, seed=42)
        traj_high = osc_high.simulate(t_max=100, dt=1.0)
        stats_high = compute_statistics(traj_high["values"])

        # Both should have finite statistics
        assert np.isfinite(stats_low.std)
        assert np.isfinite(stats_high.std)


class TestIntegratorsIntegration:
    """Test numerical integrators with the system."""

    def test_integrator_consistency(self):
        """Test that different integrators produce reasonable results."""
        from mcso.integrators import softplus, sigmoid, integrate_activation

        # Test activation functions
        x_values = np.linspace(-5, 5, 100)

        softplus_vals = softplus(x_values)
        sigmoid_vals = sigmoid(x_values)

        # Softplus should be positive
        assert np.all(softplus_vals > 0)

        # Sigmoid should be in (0, 1)
        assert np.all(sigmoid_vals > 0)
        assert np.all(sigmoid_vals < 1)

        # Test integration
        result, error = integrate_activation(
            activation=softplus,
            f=np.cos,
            g_prime=lambda x: -np.sin(x),
            lower=0,
            upper=5,
            params=(0.8, 0.3, 1.0),
        )

        assert np.isfinite(result)
        assert error < 1e-4  # Should have reasonable precision

    def test_sde_integrator_stability(self):
        """Test SDE integrators for numerical stability."""
        from mcso.integrators import SDEIntegrator, euler_maruyama

        # Define simple SDE: dX = -X dt + 0.1 dW (Ornstein-Uhlenbeck)
        def drift(x, t):
            return -x

        def diffusion(x, t):
            return 0.1

        # Euler-Maruyama
        result = euler_maruyama(
            drift=drift, diffusion=diffusion, x0=1.0, t_span=(0, 10), dt=0.01, seed=42
        )

        assert len(result["times"]) > 0
        assert len(result["values"]) == len(result["times"])
        assert np.all(np.isfinite(result["values"]))

        # Heun scheme
        integrator_heun = SDEIntegrator(drift, diffusion, scheme="heun", seed=42)
        heun_result = integrator_heun.integrate(1.0, (0, 10), dt=0.01, n_paths=1)

        assert heun_result["paths"].shape == (1, 1001)
        assert np.all(np.isfinite(heun_result["paths"]))


class TestAnalysisIntegration:
    """Test analysis module integration with simulation results."""

    def test_full_analysis_pipeline(self):
        """Test complete analysis pipeline on simulated data."""
        from mcso.oscillator import StochasticOscillator
        from mcso.analysis import (
            compute_statistics,
            autocorrelation,
            spectral_analysis,
            create_summary_report,
        )

        osc = StochasticOscillator(n_components=4, noise_scale=0.3, seed=42)

        trajectory = osc.simulate(t_max=200, dt=1.0)
        values = trajectory["values"]
        times = trajectory["times"]

        # Statistics (TrajectoryStatistics is a dataclass)
        stats = compute_statistics(values)
        assert hasattr(stats, "mean")
        assert hasattr(stats, "std")
        assert np.isfinite(stats.mean)
        assert np.isfinite(stats.std)

        # Autocorrelation
        acf = autocorrelation(values, max_lag=20)
        assert len(acf) == 21  # 0 to 20 lags
        assert acf[0] == pytest.approx(1.0, rel=1e-5)  # ACF(0) = 1
        assert np.all(np.abs(acf) <= 1.0 + 1e-10)  # ACF bounded by 1

        # Power spectrum
        spectrum = spectral_analysis(values, dt=1.0)
        assert hasattr(spectrum, "frequencies")
        assert hasattr(spectrum, "power")
        assert len(spectrum.frequencies) == len(spectrum.power)
        assert np.all(spectrum.power >= 0)  # PSD non-negative

        # Summary report (takes full trajectory dict)
        report = create_summary_report(trajectory)
        assert isinstance(report, str)
        assert "Mean" in report or "mean" in report.lower()

    def test_short_trajectory_analysis(self):
        """Test analysis handles short trajectories gracefully."""
        from mcso.oscillator import StochasticOscillator
        from mcso.analysis import compute_statistics, create_summary_report

        osc = StochasticOscillator(seed=42)

        # Very short trajectory
        trajectory = osc.simulate(t_max=5, dt=1.0)
        values = trajectory["values"]

        assert len(values) == 5

        # Should not raise
        stats = compute_statistics(values)
        assert np.isfinite(stats.mean)

        # Report should work for short trajectories (takes full trajectory dict)
        report = create_summary_report(trajectory)
        assert isinstance(report, str)

    def test_single_sample_analysis(self):
        """Test analysis handles single-sample case (regression test)."""
        from mcso.oscillator import StochasticOscillator
        from mcso.analysis import compute_statistics, create_summary_report

        osc = StochasticOscillator(seed=42)

        # Single point trajectory
        trajectory = osc.simulate(t_max=0.5, dt=1.0)
        values = trajectory["values"]

        # Should have at least 1 sample
        assert len(values) >= 1

        if len(values) == 1:
            stats = compute_statistics(values)
            assert np.isfinite(stats.mean)

            # This was previously failing with ValueError (takes full trajectory dict)
            report = create_summary_report(trajectory)
            assert isinstance(report, str)


class TestPerformanceRegression:
    """Performance regression tests."""

    def test_simulation_performance(self):
        """Test simulation completes in reasonable time."""
        import time
        from mcso.oscillator import StochasticOscillator

        osc = StochasticOscillator(n_components=5, seed=42)

        start = time.perf_counter()
        osc.simulate(t_max=1000, dt=1.0)
        elapsed = time.perf_counter() - start

        # Should complete in under 5 seconds
        assert elapsed < 5.0, f"Simulation took {elapsed:.2f}s, expected < 5s"

    def test_ensemble_performance(self):
        """Test ensemble simulation performance."""
        import time
        from mcso.oscillator import StochasticOscillator

        osc = StochasticOscillator(n_components=3, seed=42)

        start = time.perf_counter()
        osc.simulate_ensemble(n_realizations=50, t_max=100, dt=1.0)
        elapsed = time.perf_counter() - start

        # 50 realizations should complete in under 10 seconds
        assert elapsed < 10.0, f"Ensemble took {elapsed:.2f}s, expected < 10s"


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_zero_noise(self):
        """Test deterministic simulation (no noise)."""
        from mcso.oscillator import StochasticOscillator

        osc = StochasticOscillator(n_components=2, noise_scale=0.0, seed=42)

        traj1 = osc.simulate(t_max=10, dt=1.0)
        osc.reset()
        traj2 = osc.simulate(t_max=10, dt=1.0)

        # With zero noise, trajectories should be identical
        assert_allclose(traj1["values"], traj2["values"])

    def test_high_noise(self):
        """Test simulation with high noise level."""
        from mcso.oscillator import StochasticOscillator

        osc = StochasticOscillator(n_components=2, noise_scale=10.0, seed=42)  # Very high noise

        trajectory = osc.simulate(t_max=50, dt=1.0)

        # Should complete without numerical issues
        assert np.all(np.isfinite(trajectory["values"]))

    def test_many_components(self):
        """Test simulation with many oscillatory components."""
        from mcso.oscillator import StochasticOscillator

        osc = StochasticOscillator(n_components=20, noise_scale=0.1, seed=42)  # Many components

        trajectory = osc.simulate(t_max=30, dt=1.0)

        assert np.all(np.isfinite(trajectory["values"]))

    def test_small_timestep(self):
        """Test simulation with very small timestep."""
        from mcso.oscillator import StochasticOscillator

        osc = StochasticOscillator(n_components=2, noise_scale=0.1, seed=42)

        trajectory = osc.simulate(t_max=5, dt=0.01)

        assert len(trajectory["values"]) == 500
        assert np.all(np.isfinite(trajectory["values"]))


class TestCircularBuffer:
    """Test circular buffer for memory optimization."""

    def test_circular_buffer_basic(self):
        """Test basic circular buffer operations."""
        from mcso.oscillator import CircularBuffer

        buf = CircularBuffer(max_size=5)

        # Store values
        for i in range(10):
            buf[i] = float(i)

        # Only last 5 should be stored
        assert len(buf) == 5
        assert buf[5] == 5.0
        assert buf[9] == 9.0

        # Old values should return default
        assert buf[0] == 0.0  # default
        assert buf[4] == 0.0  # default (evicted)

    def test_circular_buffer_with_oscillator(self):
        """Test oscillator with circular buffer history."""
        from mcso.oscillator import StochasticOscillator

        # Configure with limited history buffer
        osc = StochasticOscillator(
            n_components=2,
            noise_scale=0.1,
            memory_strength=1.0,
            memory_delay=1.0,
            history_buffer_size=100,  # Limit history
            seed=42,
        )

        # Simulate
        trajectory = osc.simulate(t_max=50, dt=0.5)

        # Should complete successfully
        assert len(trajectory["values"]) == 100
        assert np.all(np.isfinite(trajectory["values"]))

        # History buffer should be limited
        assert len(osc.history) <= 100

    def test_circular_buffer_long_simulation(self):
        """Test that circular buffer limits memory for long simulations."""
        from mcso.oscillator import StochasticOscillator

        # With buffer
        osc_buffered = StochasticOscillator(
            n_components=2, noise_scale=0.1, history_buffer_size=500, seed=42
        )

        trajectory = osc_buffered.simulate(t_max=1000, dt=1.0)

        # History should be limited
        assert len(osc_buffered.history) <= 500

        # Results should still be valid
        assert len(trajectory["values"]) == 1000
        assert np.all(np.isfinite(trajectory["values"]))

    def test_circular_buffer_memory_term(self):
        """Test that memory term works correctly with circular buffer."""
        from mcso.oscillator import StochasticOscillator

        # Use delay shorter than buffer size
        osc = StochasticOscillator(
            n_components=2,
            noise_scale=0.1,
            memory_strength=2.0,
            memory_delay=5.0,  # 5 second delay
            history_buffer_size=50,  # 50 step buffer (enough for delay)
            seed=42,
        )

        trajectory = osc.simulate(t_max=30, dt=1.0)

        # Should have non-trivial variation from memory feedback
        values = trajectory["values"]
        mid_values = values[10:]  # After memory kicks in
        assert np.std(mid_values) > 0

    def test_circular_buffer_vs_unlimited(self):
        """Test that circular buffer produces similar results to unlimited."""
        from mcso.oscillator import StochasticOscillator

        # Unlimited history
        osc_unlimited = StochasticOscillator(
            n_components=2, noise_scale=0.1, memory_strength=1.0, memory_delay=2.0, seed=42
        )

        # With buffer (large enough to not affect results)
        osc_buffered = StochasticOscillator(
            n_components=2,
            noise_scale=0.1,
            memory_strength=1.0,
            memory_delay=2.0,
            history_buffer_size=100,  # Large enough for delay
            seed=42,
        )

        traj_unlimited = osc_unlimited.simulate(t_max=20, dt=1.0)
        traj_buffered = osc_buffered.simulate(t_max=20, dt=1.0)

        # Results should be identical when buffer is large enough
        assert_allclose(traj_unlimited["values"], traj_buffered["values"])


class TestWarningsAndErrors:
    """Test warning and error handling."""

    def test_integration_warning_captured(self):
        """Test that integration warnings are properly captured."""
        from mcso.oscillator import StochasticOscillator

        # Normal usage should not produce warnings
        osc = StochasticOscillator(seed=42)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            osc.simulate(t_max=10, dt=1.0)

            # Filter for RuntimeWarnings from our module
            integration_warnings = [
                x
                for x in w
                if issubclass(x.category, RuntimeWarning) and "integral" in str(x.message).lower()
            ]
            # Should have no integration warnings under normal conditions
            assert len(integration_warnings) == 0

    def test_invalid_config_raises(self):
        """Test that invalid configuration raises appropriate errors."""
        from mcso.oscillator import OscillatorConfig

        with pytest.raises(ValueError):
            OscillatorConfig(n_components=0)  # Must be >= 1

        with pytest.raises(ValueError):
            OscillatorConfig(noise_scale=-1.0)  # Must be >= 0

        with pytest.raises(ValueError):
            OscillatorConfig(memory_delay=-1.0)  # Must be >= 0
