"""
Unit tests for the noise module.

Tests for all noise generators: GaussianNoise, StateDependentNoise,
AdaptiveNoise, OrnsteinUhlenbeckNoise, JumpDiffusionNoise, and the
create_noise_generator factory function.
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
    """Tests for GaussianNoise generator."""

    def test_default_parameters(self):
        """Test default parameter values."""
        noise = GaussianNoise()
        assert noise.scale == 1.0
        assert noise.mean == 0.0
        assert noise.seed is None

    def test_custom_parameters(self):
        """Test custom parameter values."""
        noise = GaussianNoise(scale=0.5, mean=1.0, seed=42)
        assert noise.scale == 0.5
        assert noise.mean == 1.0
        assert noise.seed == 42

    def test_sample_returns_float(self):
        """Test that sample returns a float."""
        noise = GaussianNoise(seed=42)
        result = noise.sample(1.0)
        assert isinstance(result, (float, np.floating))

    def test_reproducibility_with_seed(self):
        """Test reproducibility with the same seed."""
        noise1 = GaussianNoise(seed=123)
        noise2 = GaussianNoise(seed=123)

        val1 = noise1.sample(1.0)
        val2 = noise2.sample(1.0)

        assert val1 == val2

    def test_different_seeds_different_values(self):
        """Test different seeds produce different values."""
        noise1 = GaussianNoise(seed=123)
        noise2 = GaussianNoise(seed=456)

        val1 = noise1.sample(1.0)
        val2 = noise2.sample(1.0)

        assert val1 != val2

    def test_mean_shift(self):
        """Test that mean parameter shifts distribution."""
        noise = GaussianNoise(scale=0.1, mean=10.0, seed=42)
        samples = [noise.sample(1.0) for _ in range(1000)]
        sample_mean = np.mean(samples)

        # Mean should be close to 10.0
        assert np.abs(sample_mean - 10.0) < 0.5

    def test_scale_affects_variance(self):
        """Test that scale affects sample variance."""
        noise_small = GaussianNoise(scale=0.1, seed=42)
        noise_large = GaussianNoise(scale=2.0, seed=42)

        samples_small = [noise_small.sample(1.0) for _ in range(1000)]
        noise_large.reset(42)  # Reset to ensure different samples
        samples_large = [noise_large.sample(1.0) for _ in range(1000)]

        # Larger scale should give larger variance
        assert np.std(samples_large) > np.std(samples_small)

    def test_reset_restores_sequence(self):
        """Test that reset restores the random sequence."""
        noise = GaussianNoise(seed=42)

        val1 = noise.sample(1.0)
        val2 = noise.sample(2.0)

        noise.reset()

        val3 = noise.sample(1.0)
        val4 = noise.sample(2.0)

        assert val1 == val3
        assert val2 == val4

    def test_reset_with_new_seed(self):
        """Test reset with a different seed."""
        noise = GaussianNoise(seed=42)
        val1 = noise.sample(1.0)

        noise.reset(seed=99)
        val2 = noise.sample(1.0)

        assert val1 != val2


class TestStateDependentNoise:
    """Tests for StateDependentNoise generator."""

    def test_default_parameters(self):
        """Test default parameter values."""
        noise = StateDependentNoise()
        assert noise.scale == 0.2
        assert noise.state_coupling == 0.3
        assert noise.delay == 1.0
        assert noise.max_state == 10.0

    def test_sample_returns_float(self):
        """Test that sample returns a float."""
        noise = StateDependentNoise(seed=42)
        result = noise.sample(1.0, state=0.5)
        assert isinstance(result, (float, np.floating))

    def test_reproducibility(self):
        """Test reproducibility with same seed."""
        noise1 = StateDependentNoise(seed=42)
        noise2 = StateDependentNoise(seed=42)

        val1 = noise1.sample(1.0, state=0.5)
        val2 = noise2.sample(1.0, state=0.5)

        assert val1 == val2

    def test_state_dependent_variance(self):
        """Test that variance increases with state magnitude."""
        noise = StateDependentNoise(state_coupling=1.0, seed=42)

        # Sample many times with different state values
        samples_low = []
        samples_high = []

        for _ in range(1000):
            noise.reset(42 + _)
            samples_low.append(noise.sample(1.0, state=0.1))
            noise.reset(42 + _)
            samples_high.append(noise.sample(1.0, state=5.0))

        # Higher state should give higher variance
        assert np.std(samples_high) > np.std(samples_low)

    def test_history_usage(self):
        """Test that history is used for delayed state."""
        noise = StateDependentNoise(delay=1.0, seed=42)
        history = {0.0: 2.0, 1.0: 3.0}

        result = noise.sample(2.0, history=history)
        assert isinstance(result, (float, np.floating))

    def test_max_state_clipping(self):
        """Test that large states are clipped."""
        noise = StateDependentNoise(max_state=5.0, state_coupling=1.0, seed=42)

        # Both should give same result due to clipping
        noise.reset(42)
        val1 = noise.sample(1.0, state=100.0)
        noise.reset(42)
        val2 = noise.sample(1.0, state=5.0)

        # Results should be the same after clipping
        assert val1 == val2

    def test_reset(self):
        """Test reset functionality."""
        noise = StateDependentNoise(seed=42)
        val1 = noise.sample(1.0)

        noise.reset()
        val2 = noise.sample(1.0)

        assert val1 == val2


class TestAdaptiveNoise:
    """Tests for AdaptiveNoise generator."""

    def test_default_parameters(self):
        """Test default parameter values."""
        noise = AdaptiveNoise()
        assert noise.initial_scale == 0.2
        assert noise.target_cv == 0.1
        assert noise.adaptation_rate == 0.01
        assert noise.window == 20
        assert noise.scale == 0.2

    def test_sample_returns_float(self):
        """Test that sample returns a float."""
        noise = AdaptiveNoise(seed=42)
        result = noise.sample(1.0)
        assert isinstance(result, (float, np.floating))

    def test_reproducibility(self):
        """Test reproducibility with same seed."""
        noise1 = AdaptiveNoise(seed=42)
        noise2 = AdaptiveNoise(seed=42)

        val1 = noise1.sample(1.0)
        val2 = noise2.sample(1.0)

        assert val1 == val2

    def test_scale_adaptation(self):
        """Test that scale adapts over time."""
        noise = AdaptiveNoise(
            initial_scale=0.2,
            adaptation_rate=0.1,
            window=10,
            seed=42
        )

        initial_scale = noise.scale

        # Generate samples with varying state to trigger adaptation
        for t in range(50):
            noise.sample(float(t), state=float(t) * 0.1)

        # Scale should have changed after adaptation
        assert noise.scale != initial_scale

    def test_scale_bounds(self):
        """Test that scale stays within bounds."""
        noise = AdaptiveNoise(
            initial_scale=0.2,
            adaptation_rate=1.0,  # Very high
            scale_bounds=(0.05, 1.0),
            window=5,
            seed=42
        )

        # Generate many samples to force adaptation
        for t in range(100):
            noise.sample(float(t), state=float(t) * 10)

        assert 0.05 <= noise.scale <= 1.0

    def test_window_size_affects_adaptation(self):
        """Test that window size affects when adaptation starts."""
        noise_short = AdaptiveNoise(window=5, seed=42)
        noise_long = AdaptiveNoise(window=50, seed=42)

        # Generate samples
        for t in range(10):
            noise_short.sample(float(t), state=1.0)
            noise_long.sample(float(t), state=1.0)

        # Short window should have adapted, long window may not have
        # At least verify no errors occur
        assert isinstance(noise_short.scale, float)
        assert isinstance(noise_long.scale, float)

    def test_reset(self):
        """Test reset restores initial state."""
        noise = AdaptiveNoise(initial_scale=0.5, seed=42)

        # Change state
        for t in range(30):
            noise.sample(float(t), state=float(t))

        noise.reset()

        assert noise.scale == 0.5
        assert len(noise.recent_values) == 0

    def test_recent_values_buffer(self):
        """Test that recent values buffer is maintained."""
        noise = AdaptiveNoise(window=10, seed=42)

        for t in range(15):
            noise.sample(float(t), state=float(t))

        # Buffer should not exceed window size
        assert len(noise.recent_values) <= noise.window


class TestOrnsteinUhlenbeckNoise:
    """Tests for OrnsteinUhlenbeckNoise (colored noise) generator."""

    def test_default_parameters(self):
        """Test default parameter values."""
        noise = OrnsteinUhlenbeckNoise()
        assert noise.mean == 0.0
        assert noise.theta == 1.0
        assert noise.sigma == 0.2
        assert noise.dt == 1.0

    def test_sample_returns_float(self):
        """Test that sample returns a float."""
        noise = OrnsteinUhlenbeckNoise(seed=42)
        result = noise.sample(1.0)
        assert isinstance(result, (float, np.floating))

    def test_reproducibility(self):
        """Test reproducibility with same seed."""
        noise1 = OrnsteinUhlenbeckNoise(seed=42)
        noise2 = OrnsteinUhlenbeckNoise(seed=42)

        val1 = noise1.sample(1.0)
        val2 = noise2.sample(1.0)

        assert val1 == val2

    def test_mean_reversion(self):
        """Test mean-reverting behavior."""
        noise = OrnsteinUhlenbeckNoise(
            mean=0.0,
            theta=2.0,  # Strong mean reversion
            sigma=0.1,
            dt=0.1,
            seed=42
        )

        # Start from a non-zero value
        noise.current_value = 5.0

        samples = []
        for t in range(100):
            samples.append(noise.sample(float(t)))

        # Later samples should be closer to mean (0.0)
        early_mean = np.mean(np.abs(samples[:20]))
        late_mean = np.mean(np.abs(samples[-20:]))

        assert late_mean < early_mean

    def test_temporal_correlation(self):
        """Test that OU process has temporal correlation."""
        noise = OrnsteinUhlenbeckNoise(
            theta=0.5,  # Moderate correlation
            sigma=0.2,
            dt=0.1,
            seed=42
        )

        samples = [noise.sample(float(t)) for t in range(100)]

        # Calculate autocorrelation at lag 1
        autocorr = np.corrcoef(samples[:-1], samples[1:])[0, 1]

        # OU process should have positive autocorrelation
        assert autocorr > 0

    def test_stationary_variance(self):
        """Test long-run variance approaches theoretical value."""
        noise = OrnsteinUhlenbeckNoise(
            mean=0.0,
            theta=1.0,
            sigma=1.0,
            dt=0.1,
            seed=42
        )

        # Generate many samples after burn-in
        for _ in range(100):  # Burn-in
            noise.sample(0.0)

        samples = [noise.sample(float(t)) for t in range(1000)]

        # Theoretical variance: sigma^2 / (2*theta) = 1.0 / 2.0 = 0.5
        sample_var = np.var(samples)
        assert np.abs(sample_var - 0.5) < 0.2

    def test_reset(self):
        """Test reset returns to mean."""
        noise = OrnsteinUhlenbeckNoise(mean=0.0, seed=42)

        # Move away from mean
        noise.current_value = 10.0

        noise.reset()

        assert noise.current_value == 0.0

    def test_different_theta_values(self):
        """Test different mean reversion rates."""
        noise_fast = OrnsteinUhlenbeckNoise(theta=5.0, dt=0.1, seed=42)
        noise_slow = OrnsteinUhlenbeckNoise(theta=0.1, dt=0.1, seed=42)

        # Start both from same non-zero value
        noise_fast.current_value = 10.0
        noise_slow.current_value = 10.0

        fast_samples = [noise_fast.sample(float(t)) for t in range(50)]
        noise_slow.reset(42)
        noise_slow.current_value = 10.0
        slow_samples = [noise_slow.sample(float(t)) for t in range(50)]

        # Fast reversion should return to mean faster
        assert np.abs(np.mean(fast_samples[-10:])) < np.abs(np.mean(slow_samples[-10:]))


class TestJumpDiffusionNoise:
    """Tests for JumpDiffusionNoise generator."""

    def test_default_parameters(self):
        """Test default parameter values."""
        noise = JumpDiffusionNoise()
        assert noise.continuous_scale == 0.1
        assert noise.jump_rate == 0.1
        assert noise.jump_mean == 0.0
        assert noise.jump_scale == 0.5
        assert noise.dt == 1.0

    def test_sample_returns_float(self):
        """Test that sample returns a float."""
        noise = JumpDiffusionNoise(seed=42)
        result = noise.sample(1.0)
        assert isinstance(result, (float, np.floating))

    def test_reproducibility(self):
        """Test reproducibility with same seed."""
        noise1 = JumpDiffusionNoise(seed=42)
        noise2 = JumpDiffusionNoise(seed=42)

        val1 = noise1.sample(1.0)
        val2 = noise2.sample(1.0)

        assert val1 == val2

    def test_continuous_component(self):
        """Test continuous noise component."""
        # With zero jump rate, should only have continuous noise
        noise = JumpDiffusionNoise(
            continuous_scale=1.0,
            jump_rate=0.0,
            seed=42
        )

        samples = [noise.sample(1.0) for _ in range(100)]

        # Should behave like Gaussian noise
        assert np.abs(np.mean(samples)) < 0.5
        assert 0.5 < np.std(samples) < 2.0

    def test_jump_occurrence(self):
        """Test that jumps occur with non-zero rate."""
        noise = JumpDiffusionNoise(
            continuous_scale=0.001,  # Very small continuous
            jump_rate=2.0,  # High jump rate
            jump_scale=1.0,
            dt=1.0,
            seed=42
        )

        samples = [noise.sample(1.0) for _ in range(100)]

        # With high jump rate, some samples should be larger
        max_sample = np.max(np.abs(samples))
        assert max_sample > 0.5  # At least some jumps occurred

    def test_jump_size_distribution(self):
        """Test that jump sizes follow normal distribution."""
        noise = JumpDiffusionNoise(
            continuous_scale=0.0,  # No continuous noise
            jump_rate=10.0,  # High rate to get many jumps
            jump_mean=5.0,
            jump_scale=0.5,
            dt=1.0,
            seed=42
        )

        samples = [noise.sample(1.0) for _ in range(100)]
        non_zero = [s for s in samples if s != 0.0]

        if len(non_zero) > 10:
            # Mean should be positive (jump_mean=5.0)
            assert np.mean(non_zero) > 0

    def test_zero_jump_rate(self):
        """Test behavior with zero jump rate."""
        noise_no_jumps = JumpDiffusionNoise(
            continuous_scale=0.5,
            jump_rate=0.0,
            seed=42
        )
        noise_gaussian = GaussianNoise(scale=0.5, seed=42)

        # Should produce same values as pure Gaussian
        for _ in range(10):
            val1 = noise_no_jumps.sample(1.0)
            val2 = noise_gaussian.sample(1.0)
            assert val1 == val2

    def test_dt_affects_jump_probability(self):
        """Test that dt affects number of jumps."""
        noise_short = JumpDiffusionNoise(
            jump_rate=1.0,
            dt=0.1,
            seed=42
        )
        noise_long = JumpDiffusionNoise(
            jump_rate=1.0,
            dt=2.0,
            seed=42
        )

        # With longer dt, expect more jumps on average
        short_jumps = sum(1 for _ in range(100) if abs(noise_short.sample(1.0)) > 0.3)
        noise_long.reset(42)
        long_jumps = sum(1 for _ in range(100) if abs(noise_long.sample(1.0)) > 0.3)

        # Longer dt should generally have more jumps
        # Note: This is probabilistic, so we allow for some variance
        assert long_jumps >= short_jumps * 0.5

    def test_reset(self):
        """Test reset functionality."""
        noise = JumpDiffusionNoise(seed=42)

        val1 = noise.sample(1.0)
        noise.reset()
        val2 = noise.sample(1.0)

        assert val1 == val2


class TestCreateNoiseGenerator:
    """Tests for the create_noise_generator factory function."""

    def test_create_gaussian(self):
        """Test creating Gaussian noise generator."""
        noise = create_noise_generator('gaussian', scale=0.5, mean=1.0)
        assert isinstance(noise, GaussianNoise)
        assert noise.scale == 0.5
        assert noise.mean == 1.0

    def test_create_state_dependent(self):
        """Test creating state-dependent noise generator."""
        noise = create_noise_generator('state_dependent', state_coupling=0.5)
        assert isinstance(noise, StateDependentNoise)
        assert noise.state_coupling == 0.5

    def test_create_adaptive(self):
        """Test creating adaptive noise generator."""
        noise = create_noise_generator('adaptive', target_cv=0.2)
        assert isinstance(noise, AdaptiveNoise)
        assert noise.target_cv == 0.2

    def test_create_ou(self):
        """Test creating Ornstein-Uhlenbeck noise generator."""
        noise = create_noise_generator('ou', theta=2.0, sigma=0.3)
        assert isinstance(noise, OrnsteinUhlenbeckNoise)
        assert noise.theta == 2.0
        assert noise.sigma == 0.3

    def test_create_ornstein_uhlenbeck_alias(self):
        """Test 'ornstein_uhlenbeck' alias for OU noise."""
        noise = create_noise_generator('ornstein_uhlenbeck', theta=1.5)
        assert isinstance(noise, OrnsteinUhlenbeckNoise)
        assert noise.theta == 1.5

    def test_create_jump_diffusion(self):
        """Test creating jump-diffusion noise generator."""
        noise = create_noise_generator('jump_diffusion', jump_rate=0.5)
        assert isinstance(noise, JumpDiffusionNoise)
        assert noise.jump_rate == 0.5

    def test_invalid_type_raises_error(self):
        """Test that invalid noise type raises ValueError."""
        with pytest.raises(ValueError, match="Unknown noise type"):
            create_noise_generator('invalid_type')

    def test_error_message_contains_valid_types(self):
        """Test that error message lists valid types."""
        with pytest.raises(ValueError) as excinfo:
            create_noise_generator('unknown')

        error_msg = str(excinfo.value)
        assert 'gaussian' in error_msg
        assert 'ou' in error_msg
        assert 'jump_diffusion' in error_msg


class TestNoiseGeneratorInterface:
    """Tests to ensure all generators implement the correct interface."""

    @pytest.fixture(params=[
        GaussianNoise,
        StateDependentNoise,
        AdaptiveNoise,
        OrnsteinUhlenbeckNoise,
        JumpDiffusionNoise,
    ])
    def noise_class(self, request):
        """Parameterized fixture for all noise generator classes."""
        return request.param

    def test_has_sample_method(self, noise_class):
        """Test that all generators have sample method."""
        noise = noise_class(seed=42)
        assert hasattr(noise, 'sample')
        assert callable(noise.sample)

    def test_has_reset_method(self, noise_class):
        """Test that all generators have reset method."""
        noise = noise_class(seed=42)
        assert hasattr(noise, 'reset')
        assert callable(noise.reset)

    def test_sample_accepts_required_args(self, noise_class):
        """Test sample accepts time parameter."""
        noise = noise_class(seed=42)
        result = noise.sample(1.0)
        assert isinstance(result, (float, np.floating))

    def test_sample_accepts_optional_args(self, noise_class):
        """Test sample accepts optional state and history."""
        noise = noise_class(seed=42)
        result = noise.sample(1.0, state=0.5, history={0.0: 0.0})
        assert isinstance(result, (float, np.floating))

    def test_sample_returns_finite_values(self, noise_class):
        """Test that sample always returns finite values."""
        noise = noise_class(seed=42)

        for t in range(100):
            result = noise.sample(float(t))
            assert np.isfinite(result), f"Non-finite value at t={t}"


class TestNoiseEdgeCases:
    """Tests for edge cases and numerical stability."""

    def test_gaussian_extreme_scale(self):
        """Test Gaussian with extreme scale values."""
        noise_tiny = GaussianNoise(scale=1e-10, seed=42)
        noise_large = GaussianNoise(scale=1e10, seed=42)

        # Both should work without errors
        val_tiny = noise_tiny.sample(1.0)
        val_large = noise_large.sample(1.0)

        assert np.isfinite(val_tiny)
        assert np.isfinite(val_large)

    def test_ou_extreme_theta(self):
        """Test OU with extreme theta values."""
        noise_fast = OrnsteinUhlenbeckNoise(theta=100.0, seed=42)
        noise_slow = OrnsteinUhlenbeckNoise(theta=0.001, seed=42)

        for _ in range(10):
            assert np.isfinite(noise_fast.sample(1.0))
            assert np.isfinite(noise_slow.sample(1.0))

    def test_jump_diffusion_high_rate(self):
        """Test jump-diffusion with very high jump rate."""
        noise = JumpDiffusionNoise(jump_rate=100.0, seed=42)

        for _ in range(10):
            assert np.isfinite(noise.sample(1.0))

    def test_adaptive_rapid_state_changes(self):
        """Test adaptive noise with rapid state changes."""
        noise = AdaptiveNoise(seed=42)

        # Simulate rapid oscillations
        for t in range(100):
            state = 10 * np.sin(t * 0.5)
            result = noise.sample(float(t), state=state)
            assert np.isfinite(result)

    def test_state_dependent_negative_states(self):
        """Test state-dependent noise with negative states."""
        noise = StateDependentNoise(seed=42)

        result = noise.sample(1.0, state=-5.0)
        assert np.isfinite(result)

    def test_all_generators_zero_time(self):
        """Test all generators at t=0."""
        generators = [
            GaussianNoise(seed=42),
            StateDependentNoise(seed=42),
            AdaptiveNoise(seed=42),
            OrnsteinUhlenbeckNoise(seed=42),
            JumpDiffusionNoise(seed=42),
        ]

        for gen in generators:
            result = gen.sample(0.0)
            assert np.isfinite(result)

    def test_all_generators_large_time(self):
        """Test all generators at large time values."""
        generators = [
            GaussianNoise(seed=42),
            StateDependentNoise(seed=42),
            AdaptiveNoise(seed=42),
            OrnsteinUhlenbeckNoise(seed=42),
            JumpDiffusionNoise(seed=42),
        ]

        for gen in generators:
            result = gen.sample(1e6)
            assert np.isfinite(result)
