"""
Unit tests for the noise module.

Tests for all noise generator classes:
- GaussianNoise
- StateDependentNoise
- AdaptiveNoise
- OrnsteinUhlenbeckNoise
- JumpDiffusionNoise
"""

import pytest
import numpy as np
from mcso.noise import (
    GaussianNoise,
    StateDependentNoise,
    AdaptiveNoise,
    OrnsteinUhlenbeckNoise,
    JumpDiffusionNoise,
    create_noise_generator,
)


class TestGaussianNoise:
    """Tests for GaussianNoise class."""

    def test_default_parameters(self):
        """Test default parameter values."""
        noise = GaussianNoise()
        assert noise.scale == 1.0
        assert noise.mean == 0.0
        assert noise.seed is None

    def test_custom_parameters(self):
        """Test custom parameter initialization."""
        noise = GaussianNoise(scale=0.5, mean=1.0, seed=42)
        assert noise.scale == 0.5
        assert noise.mean == 1.0
        assert noise.seed == 42

    def test_sample_returns_float(self):
        """Test that sample returns a float."""
        noise = GaussianNoise(seed=42)
        result = noise.sample(t=1.0)
        assert isinstance(result, (float, np.floating))

    def test_reproducibility(self):
        """Test that same seed produces same sequence."""
        noise1 = GaussianNoise(seed=123)
        noise2 = GaussianNoise(seed=123)

        samples1 = [noise1.sample(t=i) for i in range(10)]
        samples2 = [noise2.sample(t=i) for i in range(10)]

        np.testing.assert_array_equal(samples1, samples2)

    def test_different_seeds_produce_different_results(self):
        """Test that different seeds produce different sequences."""
        noise1 = GaussianNoise(seed=1)
        noise2 = GaussianNoise(seed=2)

        samples1 = [noise1.sample(t=i) for i in range(10)]
        samples2 = [noise2.sample(t=i) for i in range(10)]

        assert not np.allclose(samples1, samples2)

    def test_mean_of_samples(self):
        """Test that sample mean approximates expected mean."""
        noise = GaussianNoise(mean=5.0, scale=1.0, seed=42)
        samples = [noise.sample(t=i) for i in range(10000)]

        # With 10000 samples, mean should be close to 5.0
        assert abs(np.mean(samples) - 5.0) < 0.1

    def test_std_of_samples(self):
        """Test that sample std approximates expected scale."""
        noise = GaussianNoise(mean=0.0, scale=2.0, seed=42)
        samples = [noise.sample(t=i) for i in range(10000)]

        # With 10000 samples, std should be close to 2.0
        assert abs(np.std(samples) - 2.0) < 0.1

    def test_reset(self):
        """Test reset functionality."""
        noise = GaussianNoise(seed=42)
        samples_before = [noise.sample(t=i) for i in range(5)]

        noise.reset()
        samples_after = [noise.sample(t=i) for i in range(5)]

        np.testing.assert_array_equal(samples_before, samples_after)

    def test_reset_with_new_seed(self):
        """Test reset with a new seed."""
        noise = GaussianNoise(seed=42)
        samples_original = [noise.sample(t=i) for i in range(5)]

        noise.reset(seed=99)
        samples_new_seed = [noise.sample(t=i) for i in range(5)]

        assert not np.allclose(samples_original, samples_new_seed)


class TestStateDependentNoise:
    """Tests for StateDependentNoise class."""

    def test_default_parameters(self):
        """Test default parameter values."""
        noise = StateDependentNoise()
        assert noise.scale == 0.2
        assert noise.state_coupling == 0.3
        assert noise.delay == 1.0
        assert noise.max_state == 10.0

    def test_custom_parameters(self):
        """Test custom parameter initialization."""
        noise = StateDependentNoise(
            scale=0.5,
            state_coupling=0.5,
            delay=2.0,
            max_state=5.0,
            seed=42
        )
        assert noise.scale == 0.5
        assert noise.state_coupling == 0.5
        assert noise.delay == 2.0
        assert noise.max_state == 5.0

    def test_sample_returns_float(self):
        """Test that sample returns a float."""
        noise = StateDependentNoise(seed=42)
        result = noise.sample(t=1.0, state=1.0)
        assert isinstance(result, (float, np.floating))

    def test_variance_increases_with_state(self):
        """Test that larger states produce larger variance."""
        noise = StateDependentNoise(scale=1.0, state_coupling=1.0, seed=42)

        # Sample with small state
        noise.reset(seed=42)
        samples_small = [noise.sample(t=i, state=0.1) for i in range(1000)]

        # Sample with large state
        noise.reset(seed=42)
        samples_large = [noise.sample(t=i, state=5.0) for i in range(1000)]

        # Variance should be larger for large state
        assert np.var(samples_large) > np.var(samples_small)

    def test_history_based_coupling(self):
        """Test that history affects noise generation."""
        noise = StateDependentNoise(scale=1.0, state_coupling=1.0, delay=1.0, seed=42)

        history_small = {0.0: 0.1}
        history_large = {0.0: 5.0}

        # Generate samples with different histories
        noise.reset(seed=42)
        samples_small = [noise.sample(t=1.0, history=history_small) for _ in range(1000)]

        noise.reset(seed=42)
        samples_large = [noise.sample(t=1.0, history=history_large) for _ in range(1000)]

        # Variance should be larger for large historical state
        assert np.var(samples_large) > np.var(samples_small)

    def test_max_state_clipping(self):
        """Test that states are clipped to max_state."""
        noise = StateDependentNoise(scale=1.0, state_coupling=1.0, max_state=2.0, seed=42)

        # Very large state should be clipped to max_state
        samples_huge = [noise.sample(t=i, state=1000.0) for i in range(1000)]
        noise.reset(seed=42)
        samples_at_max = [noise.sample(t=i, state=2.0) for i in range(1000)]

        # Variances should be similar since huge state is clipped to 2.0
        np.testing.assert_almost_equal(np.var(samples_huge), np.var(samples_at_max), decimal=1)

    def test_reproducibility(self):
        """Test reproducibility with same seed."""
        noise1 = StateDependentNoise(seed=42)
        noise2 = StateDependentNoise(seed=42)

        samples1 = [noise1.sample(t=i, state=1.0) for i in range(10)]
        samples2 = [noise2.sample(t=i, state=1.0) for i in range(10)]

        np.testing.assert_array_equal(samples1, samples2)

    def test_reset(self):
        """Test reset functionality."""
        noise = StateDependentNoise(seed=42)
        samples_before = [noise.sample(t=i, state=1.0) for i in range(5)]

        noise.reset()
        samples_after = [noise.sample(t=i, state=1.0) for i in range(5)]

        np.testing.assert_array_equal(samples_before, samples_after)


class TestAdaptiveNoise:
    """Tests for AdaptiveNoise class."""

    def test_default_parameters(self):
        """Test default parameter values."""
        noise = AdaptiveNoise()
        assert noise.initial_scale == 0.2
        assert noise.target_cv == 0.1
        assert noise.adaptation_rate == 0.01
        assert noise.scale_bounds == (0.01, 2.0)
        assert noise.window == 20

    def test_custom_parameters(self):
        """Test custom parameter initialization."""
        noise = AdaptiveNoise(
            initial_scale=0.5,
            target_cv=0.2,
            adaptation_rate=0.05,
            scale_bounds=(0.05, 1.0),
            window=10,
            seed=42
        )
        assert noise.initial_scale == 0.5
        assert noise.target_cv == 0.2
        assert noise.adaptation_rate == 0.05
        assert noise.scale_bounds == (0.05, 1.0)
        assert noise.window == 10

    def test_sample_returns_float(self):
        """Test that sample returns a float."""
        noise = AdaptiveNoise(seed=42)
        result = noise.sample(t=1.0, state=1.0)
        assert isinstance(result, (float, np.floating))

    def test_scale_starts_at_initial(self):
        """Test that scale starts at initial_scale."""
        noise = AdaptiveNoise(initial_scale=0.3, seed=42)
        assert noise.scale == 0.3

    def test_scale_adapts_over_time(self):
        """Test that scale adapts after window samples."""
        noise = AdaptiveNoise(
            initial_scale=0.2,
            window=10,
            adaptation_rate=0.1,
            seed=42
        )
        initial_scale = noise.scale

        # Generate enough samples to trigger adaptation
        for i in range(50):
            noise.sample(t=float(i), state=1.0)

        # Scale should have changed
        assert noise.scale != initial_scale

    def test_scale_respects_bounds(self):
        """Test that scale stays within bounds."""
        noise = AdaptiveNoise(
            initial_scale=0.5,
            scale_bounds=(0.1, 1.0),
            adaptation_rate=1.0,  # Very high rate to force bounds
            window=5,
            seed=42
        )

        # Generate many samples to potentially exceed bounds
        for i in range(200):
            noise.sample(t=float(i), state=float(i))

        assert 0.1 <= noise.scale <= 1.0

    def test_recent_values_tracking(self):
        """Test that recent values are tracked."""
        noise = AdaptiveNoise(window=5, seed=42)

        for i in range(10):
            noise.sample(t=float(i), state=float(i))

        # Should only keep last 'window' values
        assert len(noise.recent_values) == 5

    def test_reset(self):
        """Test reset functionality."""
        noise = AdaptiveNoise(initial_scale=0.3, window=10, seed=42)

        # Generate samples to modify state
        for i in range(20):
            noise.sample(t=float(i), state=float(i))

        assert noise.scale != 0.3 or len(noise.recent_values) > 0

        noise.reset()

        assert noise.scale == 0.3
        assert len(noise.recent_values) == 0

    def test_reproducibility(self):
        """Test reproducibility with same seed."""
        noise1 = AdaptiveNoise(seed=42)
        noise2 = AdaptiveNoise(seed=42)

        samples1 = [noise1.sample(t=float(i), state=1.0) for i in range(10)]
        samples2 = [noise2.sample(t=float(i), state=1.0) for i in range(10)]

        np.testing.assert_array_equal(samples1, samples2)


class TestOrnsteinUhlenbeckNoise:
    """Tests for OrnsteinUhlenbeckNoise class."""

    def test_default_parameters(self):
        """Test default parameter values."""
        noise = OrnsteinUhlenbeckNoise()
        assert noise.mean == 0.0
        assert noise.theta == 1.0
        assert noise.sigma == 0.2
        assert noise.dt == 1.0

    def test_custom_parameters(self):
        """Test custom parameter initialization."""
        noise = OrnsteinUhlenbeckNoise(
            mean=1.0,
            theta=2.0,
            sigma=0.5,
            dt=0.1,
            seed=42
        )
        assert noise.mean == 1.0
        assert noise.theta == 2.0
        assert noise.sigma == 0.5
        assert noise.dt == 0.1

    def test_sample_returns_float(self):
        """Test that sample returns a float."""
        noise = OrnsteinUhlenbeckNoise(seed=42)
        result = noise.sample(t=1.0)
        assert isinstance(result, (float, np.floating))

    def test_initial_value_is_mean(self):
        """Test that initial current_value equals mean."""
        noise = OrnsteinUhlenbeckNoise(mean=5.0)
        assert noise.current_value == 5.0

    def test_mean_reversion(self):
        """Test mean reversion property."""
        noise = OrnsteinUhlenbeckNoise(mean=0.0, theta=2.0, sigma=0.1, dt=0.1, seed=42)

        # Run for many steps
        samples = []
        for _ in range(1000):
            samples.append(noise.sample(t=0.0))

        # Long-run average should be close to mean
        assert abs(np.mean(samples)) < 0.5

    def test_temporal_correlation(self):
        """Test that consecutive samples are correlated."""
        noise = OrnsteinUhlenbeckNoise(theta=0.1, sigma=0.1, dt=0.1, seed=42)

        samples = [noise.sample(t=float(i)) for i in range(100)]

        # Calculate lag-1 autocorrelation
        correlation = np.corrcoef(samples[:-1], samples[1:])[0, 1]

        # OU process should have positive autocorrelation
        assert correlation > 0.5

    def test_higher_theta_faster_reversion(self):
        """Test that higher theta means faster mean reversion."""
        noise_slow = OrnsteinUhlenbeckNoise(theta=0.1, sigma=0.1, dt=0.1, seed=42)
        noise_fast = OrnsteinUhlenbeckNoise(theta=2.0, sigma=0.1, dt=0.1, seed=42)

        # Start both at same non-mean value
        noise_slow.current_value = 5.0
        noise_fast.current_value = 5.0

        samples_slow = [noise_slow.sample(t=0.0) for _ in range(10)]
        noise_slow.current_value = 5.0
        noise_fast.current_value = 5.0

        noise_slow.reset(seed=42)
        noise_fast.reset(seed=42)
        noise_slow.current_value = 5.0
        noise_fast.current_value = 5.0

        samples_slow = [noise_slow.sample(t=0.0) for _ in range(10)]

        noise_slow.reset(seed=42)
        noise_fast.reset(seed=42)
        noise_slow.current_value = 5.0
        noise_fast.current_value = 5.0

        sample_slow = noise_slow.sample(t=0.0)
        sample_fast = noise_fast.sample(t=0.0)

        # Fast theta should revert more toward mean (0) than slow
        assert abs(sample_fast) < abs(sample_slow)

    def test_reset(self):
        """Test reset functionality."""
        noise = OrnsteinUhlenbeckNoise(mean=0.0, seed=42)

        # Sample to change current_value
        for _ in range(10):
            noise.sample(t=0.0)

        assert noise.current_value != 0.0

        noise.reset()

        assert noise.current_value == 0.0

    def test_reproducibility(self):
        """Test reproducibility with same seed."""
        noise1 = OrnsteinUhlenbeckNoise(seed=42)
        noise2 = OrnsteinUhlenbeckNoise(seed=42)

        samples1 = [noise1.sample(t=float(i)) for i in range(10)]
        samples2 = [noise2.sample(t=float(i)) for i in range(10)]

        np.testing.assert_array_equal(samples1, samples2)


class TestJumpDiffusionNoise:
    """Tests for JumpDiffusionNoise class."""

    def test_default_parameters(self):
        """Test default parameter values."""
        noise = JumpDiffusionNoise()
        assert noise.continuous_scale == 0.1
        assert noise.jump_rate == 0.1
        assert noise.jump_mean == 0.0
        assert noise.jump_scale == 0.5
        assert noise.dt == 1.0

    def test_custom_parameters(self):
        """Test custom parameter initialization."""
        noise = JumpDiffusionNoise(
            continuous_scale=0.2,
            jump_rate=0.5,
            jump_mean=1.0,
            jump_scale=0.3,
            dt=0.5,
            seed=42
        )
        assert noise.continuous_scale == 0.2
        assert noise.jump_rate == 0.5
        assert noise.jump_mean == 1.0
        assert noise.jump_scale == 0.3
        assert noise.dt == 0.5

    def test_sample_returns_float(self):
        """Test that sample returns a float."""
        noise = JumpDiffusionNoise(seed=42)
        result = noise.sample(t=1.0)
        assert isinstance(result, (float, np.floating))

    def test_continuous_component(self):
        """Test continuous component is present."""
        noise = JumpDiffusionNoise(continuous_scale=1.0, jump_rate=0.0, seed=42)

        samples = [noise.sample(t=float(i)) for i in range(1000)]

        # With no jumps, should have Gaussian distribution
        assert abs(np.mean(samples)) < 0.1
        assert abs(np.std(samples) - 1.0) < 0.1

    def test_jumps_occur(self):
        """Test that jumps occur with high jump rate."""
        noise = JumpDiffusionNoise(
            continuous_scale=0.0,  # No continuous noise
            jump_rate=10.0,  # High jump rate
            jump_mean=0.0,
            jump_scale=1.0,
            dt=1.0,
            seed=42
        )

        samples = [noise.sample(t=float(i)) for i in range(100)]

        # Most samples should be non-zero due to jumps
        non_zero_count = sum(1 for s in samples if abs(s) > 0.01)
        assert non_zero_count > 50

    def test_zero_jump_rate_no_jumps(self):
        """Test that zero jump rate produces no jumps."""
        noise = JumpDiffusionNoise(
            continuous_scale=0.0,
            jump_rate=0.0,
            seed=42
        )

        samples = [noise.sample(t=float(i)) for i in range(100)]

        # All samples should be exactly zero
        assert all(s == 0.0 for s in samples)

    def test_jump_mean_effect(self):
        """Test that jump mean affects average jump size."""
        noise = JumpDiffusionNoise(
            continuous_scale=0.0,
            jump_rate=10.0,  # High rate for many jumps
            jump_mean=5.0,
            jump_scale=0.1,  # Small variance
            dt=1.0,
            seed=42
        )

        samples = [noise.sample(t=float(i)) for i in range(1000)]
        non_zero_samples = [s for s in samples if abs(s) > 0.01]

        if len(non_zero_samples) > 10:
            # Average of non-zero samples should be positive
            assert np.mean(non_zero_samples) > 2.0

    def test_variance_increases_with_jump_scale(self):
        """Test that larger jump_scale produces more variance."""
        noise_small = JumpDiffusionNoise(
            continuous_scale=0.0,
            jump_rate=5.0,
            jump_scale=0.1,
            seed=42
        )
        noise_large = JumpDiffusionNoise(
            continuous_scale=0.0,
            jump_rate=5.0,
            jump_scale=2.0,
            seed=42
        )

        samples_small = [noise_small.sample(t=float(i)) for i in range(1000)]
        samples_large = [noise_large.sample(t=float(i)) for i in range(1000)]

        # Large jump scale should have larger variance
        assert np.var(samples_large) > np.var(samples_small)

    def test_reset(self):
        """Test reset functionality."""
        noise = JumpDiffusionNoise(seed=42)
        samples_before = [noise.sample(t=float(i)) for i in range(5)]

        noise.reset()
        samples_after = [noise.sample(t=float(i)) for i in range(5)]

        np.testing.assert_array_equal(samples_before, samples_after)

    def test_reproducibility(self):
        """Test reproducibility with same seed."""
        noise1 = JumpDiffusionNoise(seed=42)
        noise2 = JumpDiffusionNoise(seed=42)

        samples1 = [noise1.sample(t=float(i)) for i in range(10)]
        samples2 = [noise2.sample(t=float(i)) for i in range(10)]

        np.testing.assert_array_equal(samples1, samples2)


class TestCreateNoiseGenerator:
    """Tests for create_noise_generator factory function."""

    def test_create_gaussian(self):
        """Test creating Gaussian noise generator."""
        noise = create_noise_generator('gaussian', scale=0.5, mean=1.0)
        assert isinstance(noise, GaussianNoise)
        assert noise.scale == 0.5
        assert noise.mean == 1.0

    def test_create_state_dependent(self):
        """Test creating state-dependent noise generator."""
        noise = create_noise_generator('state_dependent', scale=0.3, state_coupling=0.5)
        assert isinstance(noise, StateDependentNoise)
        assert noise.scale == 0.3
        assert noise.state_coupling == 0.5

    def test_create_adaptive(self):
        """Test creating adaptive noise generator."""
        noise = create_noise_generator('adaptive', initial_scale=0.4, target_cv=0.2)
        assert isinstance(noise, AdaptiveNoise)
        assert noise.initial_scale == 0.4
        assert noise.target_cv == 0.2

    def test_create_ou(self):
        """Test creating OU noise generator with 'ou' alias."""
        noise = create_noise_generator('ou', theta=2.0, sigma=0.3)
        assert isinstance(noise, OrnsteinUhlenbeckNoise)
        assert noise.theta == 2.0
        assert noise.sigma == 0.3

    def test_create_ornstein_uhlenbeck(self):
        """Test creating OU noise generator with full name."""
        noise = create_noise_generator('ornstein_uhlenbeck', theta=1.5)
        assert isinstance(noise, OrnsteinUhlenbeckNoise)
        assert noise.theta == 1.5

    def test_create_jump_diffusion(self):
        """Test creating jump-diffusion noise generator."""
        noise = create_noise_generator('jump_diffusion', jump_rate=0.5, jump_scale=0.3)
        assert isinstance(noise, JumpDiffusionNoise)
        assert noise.jump_rate == 0.5
        assert noise.jump_scale == 0.3

    def test_invalid_type_raises(self):
        """Test that invalid noise type raises ValueError."""
        with pytest.raises(ValueError, match="Unknown noise type"):
            create_noise_generator('invalid_type')

    def test_invalid_type_lists_available(self):
        """Test that error message lists available types."""
        with pytest.raises(ValueError) as exc_info:
            create_noise_generator('bad_type')

        error_msg = str(exc_info.value)
        assert 'gaussian' in error_msg
        assert 'ou' in error_msg
        assert 'jump_diffusion' in error_msg


class TestNoiseGeneratorEdgeCases:
    """Tests for edge cases and numerical stability."""

    def test_gaussian_zero_scale(self):
        """Test Gaussian with zero scale produces zero noise."""
        noise = GaussianNoise(scale=0.0, seed=42)
        samples = [noise.sample(t=float(i)) for i in range(10)]
        assert all(s == 0.0 for s in samples)

    def test_state_dependent_zero_coupling(self):
        """Test state-dependent with zero coupling is like Gaussian."""
        sd_noise = StateDependentNoise(scale=1.0, state_coupling=0.0, seed=42)
        g_noise = GaussianNoise(scale=1.0, seed=42)

        sd_samples = [sd_noise.sample(t=float(i), state=100.0) for i in range(100)]
        g_samples = [g_noise.sample(t=float(i)) for i in range(100)]

        # With zero coupling, state shouldn't affect variance
        assert abs(np.var(sd_samples) - np.var(g_samples)) < 0.5

    def test_ou_extreme_theta(self):
        """Test OU with very large theta still produces finite values."""
        noise = OrnsteinUhlenbeckNoise(theta=100.0, sigma=1.0, dt=0.1, seed=42)
        samples = [noise.sample(t=float(i)) for i in range(100)]
        assert all(np.isfinite(s) for s in samples)

    def test_ou_very_small_theta(self):
        """Test OU with very small theta still produces finite values."""
        noise = OrnsteinUhlenbeckNoise(theta=0.001, sigma=1.0, dt=0.1, seed=42)
        samples = [noise.sample(t=float(i)) for i in range(100)]
        assert all(np.isfinite(s) for s in samples)

    def test_jump_diffusion_many_jumps(self):
        """Test jump diffusion with very high jump rate."""
        noise = JumpDiffusionNoise(jump_rate=100.0, dt=1.0, seed=42)
        samples = [noise.sample(t=float(i)) for i in range(100)]
        assert all(np.isfinite(s) for s in samples)

    def test_adaptive_extreme_values(self):
        """Test adaptive noise with extreme state values."""
        noise = AdaptiveNoise(seed=42)

        # Feed extreme values
        for i in range(100):
            sample = noise.sample(t=float(i), state=float(i * 1000))
            assert np.isfinite(sample)

        # Scale should still be within bounds
        assert noise.scale_bounds[0] <= noise.scale <= noise.scale_bounds[1]

    def test_all_generators_handle_negative_time(self):
        """Test that generators handle negative time values."""
        generators = [
            GaussianNoise(seed=42),
            StateDependentNoise(seed=42),
            AdaptiveNoise(seed=42),
            OrnsteinUhlenbeckNoise(seed=42),
            JumpDiffusionNoise(seed=42),
        ]

        for gen in generators:
            result = gen.sample(t=-10.0, state=1.0)
            assert np.isfinite(result)

    def test_all_generators_handle_large_time(self):
        """Test that generators handle very large time values."""
        generators = [
            GaussianNoise(seed=42),
            StateDependentNoise(seed=42),
            AdaptiveNoise(seed=42),
            OrnsteinUhlenbeckNoise(seed=42),
            JumpDiffusionNoise(seed=42),
        ]

        for gen in generators:
            result = gen.sample(t=1e10, state=1.0)
            assert np.isfinite(result)
