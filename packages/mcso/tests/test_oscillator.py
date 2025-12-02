"""
Unit tests for the StochasticOscillator module.
"""

import pytest
import numpy as np
from mcso.oscillator import StochasticOscillator, OscillatorConfig


class TestOscillatorConfig:
    """Tests for OscillatorConfig dataclass."""

    def test_default_config(self):
        """Test default configuration values."""
        config = OscillatorConfig()
        assert config.n_components == 5
        assert config.noise_scale == 0.2
        assert config.memory_delay == 1.0

    def test_custom_config(self):
        """Test custom configuration."""
        config = OscillatorConfig(
            n_components=3,
            noise_scale=0.5,
            seed=42
        )
        assert config.n_components == 3
        assert config.noise_scale == 0.5
        assert config.seed == 42

    def test_invalid_n_components(self):
        """Test validation of n_components."""
        with pytest.raises(ValueError, match="n_components must be at least 1"):
            OscillatorConfig(n_components=0)

    def test_invalid_noise_scale(self):
        """Test validation of noise_scale."""
        with pytest.raises(ValueError, match="noise_scale must be non-negative"):
            OscillatorConfig(noise_scale=-0.1)

    def test_invalid_memory_delay(self):
        """Test validation of memory_delay."""
        with pytest.raises(ValueError, match="memory_delay must be non-negative"):
            OscillatorConfig(memory_delay=-1.0)
            OscillatorConfig(memory_delay=-0.1)


class TestStochasticOscillator:
    """Tests for StochasticOscillator class."""

    @pytest.fixture
    def oscillator(self):
        """Create oscillator with fixed seed for reproducibility."""
        return StochasticOscillator(seed=42)

    def test_initialization(self, oscillator):
        """Test oscillator initialization."""
        assert oscillator.config.n_components == 5
        assert len(oscillator.history) == 0

    def test_reset(self, oscillator):
        """Test reset clears history."""
        oscillator.evaluate(1.0)
        oscillator.evaluate(2.0)
        assert len(oscillator.history) > 0

        oscillator.reset()
        assert len(oscillator.history) == 0

    def test_reproducibility(self):
        """Test that same seed gives same results."""
        osc1 = StochasticOscillator(seed=123)
        osc2 = StochasticOscillator(seed=123)

        val1 = osc1.evaluate(5.0)
        val2 = osc2.evaluate(5.0)

        assert val1 == val2

    def test_evaluate_returns_float(self, oscillator):
        """Test that evaluate returns a float."""
        result = oscillator.evaluate(1.0)
        assert isinstance(result, (float, np.floating))

    def test_evaluate_stores_history(self, oscillator):
        """Test that evaluate stores values in history.
        
        Note: The history dictionary uses exact time values as keys.
        This test uses np.isclose to handle potential floating-point 
        precision issues when checking for key existence.
        Note: history uses exact float keys. This test uses exact float values
        that are passed to evaluate() to verify the keys exist in the dict.
        """
        oscillator.evaluate(1.0)
        oscillator.evaluate(2.0)

        # Check that entries for 1.0 and 2.0 are stored
        # Note: additional entries may exist from memory term lookups during evaluation
        assert any(np.isclose(key, 1.0, atol=1e-9, rtol=0) for key in oscillator.history.keys())
        assert any(np.isclose(key, 2.0, atol=1e-9, rtol=0) for key in oscillator.history.keys())
        # Verify the keys we used are in history (note: history may include t=0 from initialization)
        history_keys = list(oscillator.history.keys())
        assert len(history_keys) >= 2
        assert any(abs(k - 1.0) < 1e-9 for k in history_keys)
        assert any(abs(k - 2.0) < 1e-9 for k in history_keys)

    def test_evaluate_no_history(self, oscillator):
        """Test evaluate with store_history=False.
        
        Note: Uses approximate comparison for floating-point keys.
        """
        oscillator.evaluate(1.0, store_history=False)
        
        # Check that no key approximately equal to 1.0 exists
        history_keys = list(oscillator.history.keys())
        assert not any(np.isclose(key, 1.0, atol=1e-9, rtol=0) for key in history_keys)
        # Verify no keys are stored
        Note: The history dict uses exact time values as keys (the same values
        passed to evaluate()). Since no floating-point arithmetic is performed
        on the time values before storage, exact equality checks are safe here.
        For computed time values (e.g., from arange), use approximate matching.
        """
        t1, t2 = 1.0, 2.0
        oscillator.evaluate(t1)
        oscillator.evaluate(t2)

        # Check that exact time values are stored as keys
        assert t1 in oscillator.history
        assert t2 in oscillator.history

    def test_evaluate_no_history(self, oscillator):
        """Test evaluate with store_history=False.
        
        Uses exact time value for key check (see test_evaluate_stores_history).
        """
        t = 1.0
        oscillator.evaluate(t, store_history=False)
        assert t not in oscillator.history

    def test_simulate_returns_dict(self, oscillator):
        """Test simulate returns correct structure."""
        result = oscillator.simulate(t_max=10, dt=1.0)

        assert 'times' in result
        assert 'values' in result
        assert 'config' in result

    def test_simulate_time_array(self, oscillator):
        """Test simulate produces correct time array."""
        result = oscillator.simulate(t_max=10, dt=2.0)

        expected_times = np.array([0, 2, 4, 6, 8])
        np.testing.assert_array_equal(result['times'], expected_times)

    def test_simulate_values_length(self, oscillator):
        """Test simulate produces correct number of values."""
        result = oscillator.simulate(t_max=100, dt=1.0)

        assert len(result['values']) == len(result['times'])
        assert len(result['values']) == 100

    def test_simulate_bounded_values(self, oscillator):
        """Test that simulated values are within bounds."""
        result = oscillator.simulate(t_max=100, dt=1.0)

        assert np.all(result['values'] >= oscillator.config.clip_bounds[0])
        assert np.all(result['values'] <= oscillator.config.clip_bounds[1])

    def test_simulate_ensemble(self, oscillator):
        """Test ensemble simulation."""
        result = oscillator.simulate_ensemble(
            n_realizations=10,
            t_max=50,
            dt=1.0
        )

        assert result['ensemble'].shape == (10, 50)
        assert len(result['mean']) == 50
        assert len(result['std']) == 50
        assert 50 in result['percentiles']

    def test_custom_control_function(self):
        """Test simulation with custom control input."""
        def control_fn(t):
            return 1.0 if t > 5 else 0.0

        osc = StochasticOscillator(control_gain=1.0, noise_scale=0.0, seed=42)
        result = osc.simulate(t_max=10, dt=1.0, control_fn=control_fn)

        # Values after t=5 should be affected by control
        # (This is a basic sanity check)
        assert len(result['values']) == 10


class TestOscillatorComponents:
    """Tests for individual oscillator components."""

    @pytest.fixture
    def oscillator(self):
        return StochasticOscillator(n_components=3, seed=42)

    def test_oscillatory_term(self, oscillator):
        """Test oscillatory term computation."""
        result = oscillator.oscillatory_term(5.0)
        assert isinstance(result, (float, np.floating))
        assert np.isfinite(result)

    def test_integral_term(self, oscillator):
        """Test integral term computation."""
        result = oscillator.integral_term(5.0)
        assert isinstance(result, (float, np.floating))
        assert np.isfinite(result)

    def test_integral_term_zero_at_start(self, oscillator):
        """Test integral term is zero at t=0."""
        result = oscillator.integral_term(0.0)
        assert result == 0.0

    def test_drift_term(self, oscillator):
        """Test drift term computation."""
        result = oscillator.drift_term(5.0)
        assert isinstance(result, (float, np.floating))
        assert np.isfinite(result)

    def test_memory_term_zero_initially(self, oscillator):
        """Test memory term is zero with no history."""
        result = oscillator.memory_term(0.0)
        assert result == 0.0

    def test_noise_term_varies(self, oscillator):
        """Test noise term produces varying values."""
        values = [oscillator.noise_term(1.0) for _ in range(10)]
        assert len(set(values)) > 1  # Values should differ

    def test_softplus_numerically_stable(self):
        """Test softplus handles extreme values."""
        result_large = StochasticOscillator.softplus(1000.0)
        result_small = StochasticOscillator.softplus(-1000.0)

        assert np.isfinite(result_large)
        assert np.isfinite(result_small)
        assert result_large > 0
        assert result_small > 0

    def test_sigmoid_bounded(self):
        """Test sigmoid output is in [0, 1]."""
        for x in [-100, -1, 0, 1, 100]:
            result = StochasticOscillator.sigmoid(x)
            # Due to numerical precision, extreme values may reach exactly 0 or 1
            assert 0 <= result <= 1


class TestOscillatorEdgeCases:
    """Tests for edge cases and numerical stability."""

    def test_zero_noise(self):
        """Test oscillator with zero noise is deterministic."""
        osc1 = StochasticOscillator(noise_scale=0.0, seed=1)
        osc2 = StochasticOscillator(noise_scale=0.0, seed=999)

        result1 = osc1.simulate(t_max=10, dt=1.0)
        result2 = osc2.simulate(t_max=10, dt=1.0)

        np.testing.assert_array_almost_equal(
            result1['values'],
            result2['values']
        )

    def test_single_component(self):
        """Test oscillator with single component."""
        osc = StochasticOscillator(n_components=1, seed=42)
        result = osc.simulate(t_max=50, dt=1.0)
        assert len(result['values']) == 50

    def test_small_dt(self):
        """Test simulation with small time step."""
        osc = StochasticOscillator(seed=42)
        result = osc.simulate(t_max=1, dt=0.01)
        assert len(result['values']) == 100

    def test_large_memory_delay(self):
        """Test oscillator with large memory delay."""
        osc = StochasticOscillator(memory_delay=10.0, seed=42)
        result = osc.simulate(t_max=20, dt=1.0)
        assert np.all(np.isfinite(result['values']))

    def test_kwargs_override_config(self):
        """Test that kwargs can override config values."""
        config = OscillatorConfig(n_components=5)
        osc = StochasticOscillator(config=config, n_components=3)

        assert osc.config.n_components == 3
