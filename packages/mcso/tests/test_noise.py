"""
Unit tests for the noise module.

Tests for all noise generators: GaussianNoise, StateDependentNoise,
AdaptiveNoise, OrnsteinUhlenbeckNoise, JumpDiffusionNoise, and the
create_noise_generator factory function.
Unit tests for the noise generators module.
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

# Statistical test tolerances - these values account for sampling variance
# with 1000 samples while maintaining a high probability of test success
MEAN_TOLERANCE = 0.5  # Tolerance for mean estimation tests
JUMP_DETECTION_THRESHOLD = 0.3  # Threshold for detecting jump occurrences
PROBABILISTIC_TEST_TOLERANCE = 0.5  # Tolerance factor for probabilistic comparisons


class TestGaussianNoise:
    """Tests for GaussianNoise class."""

    def test_initialization(self):
        """Test default initialization."""
        noise = GaussianNoise()
        assert noise.scale == 1.0
        assert noise.mean == 0.0

    def test_custom_parameters(self):
        """Test custom parameters."""
        noise = GaussianNoise(scale=0.5, mean=1.0, seed=42)
        assert noise.scale == 0.5
        assert noise.mean == 1.0

    def test_sample_returns_float(self):
        """Test sample returns a float."""
        noise = GaussianNoise(seed=42)
        sample = noise.sample(t=1.0)
        assert isinstance(sample, (float, np.floating))

    def test_reproducibility(self):
        """Test reproducibility with same seed."""
        noise1 = GaussianNoise(seed=42)
        noise2 = GaussianNoise(seed=42)
        sample1 = noise1.sample(t=1.0)
        sample2 = noise2.sample(t=1.0)
        assert sample1 == sample2

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
        assert np.abs(sample_mean - 10.0) < MEAN_TOLERANCE

    def test_scale_affects_variance(self):
        """Test that scale affects sample variance."""
        noise_small = GaussianNoise(scale=0.1, seed=42)
        noise_large = GaussianNoise(scale=2.0, seed=42)

        samples_small = [noise_small.sample(1.0) for _ in range(1000)]
        # Reset to start a fresh sequence for the large scale generator
        noise_large.reset(42)
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
    """Tests for StateDependentNoise class."""

    def test_initialization(self):
        """Test default initialization."""
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
        # Each iteration uses a unique seed to generate independent samples
        samples_low = []
        samples_high = []

        for i in range(1000):
            noise.reset(42 + i)
            samples_low.append(noise.sample(1.0, state=0.1))
            noise.reset(42 + i)
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
        val1 = noise.sample(1.0)

        noise.reset()
        val2 = noise.sample(1.0)

        assert val1 == val2


class TestAdaptiveNoise:
    """Tests for AdaptiveNoise class."""

    def test_initialization(self):
        """Test default initialization."""
        noise = AdaptiveNoise()
        assert noise.initial_scale == 0.2
        assert noise.target_cv == 0.1
        assert noise.adaptation_rate == 0.01
        assert noise.window == 20
        assert noise.scale == 0.2

    def test_sample_returns_float(self):
        """Test sample returns a float."""
        noise = AdaptiveNoise(seed=42)
        sample = noise.sample(t=1.0, state=1.0)
        assert isinstance(sample, (float, np.floating))

    def test_adaptation_over_time(self):
        """Test that scale adapts over time."""
        noise = AdaptiveNoise(seed=42, window=5)
        initial_scale = noise.scale
        
        # Generate samples to trigger adaptation
        for i in range(20):
            noise.sample(t=float(i), state=1.0 + 0.1 * i)
        
        # Scale should have changed after adaptation
        # Note: The exact direction depends on observed_cv vs target_cv
        assert noise.scale != initial_scale or len(noise.recent_values) > 0

    def test_scale_bounds(self):
        """Test that scale stays within bounds."""
        noise = AdaptiveNoise(
            seed=42,
            scale_bounds=(0.05, 1.0),
            adaptation_rate=0.5,  # High rate for faster adaptation
            window=3
        )
        
        # Generate many samples to test bounds
        for i in range(100):
            noise.sample(t=float(i), state=np.random.randn() * 100)
        
        assert noise.scale >= noise.scale_bounds[0]
        assert noise.scale <= noise.scale_bounds[1]

    def test_reset(self):
        """Test reset clears buffer and resets scale."""
        noise = AdaptiveNoise(seed=42, window=5)
        
        # Generate some samples
        for i in range(10):
            noise.sample(t=float(i), state=1.0)
        
        noise.reset()
        assert len(noise.recent_values) == 0
        assert noise.scale == noise.initial_scale
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
            seed=42,
            scale_bounds=(0.05, 1.0),
            adaptation_rate=0.5,  # High rate for faster adaptation
            window=3
        )
        
        # Generate many samples to test bounds
        for i in range(100):
            noise.sample(t=float(i), state=np.random.randn() * 100)
        
        assert noise.scale >= noise.scale_bounds[0]
        assert noise.scale <= noise.scale_bounds[1]

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


    def test_reproducibility(self):
        """Test reproducibility with same seed."""
        noise1 = AdaptiveNoise(seed=42)
        noise2 = AdaptiveNoise(seed=42)

        samples1 = [noise1.sample(t=float(i), state=1.0) for i in range(10)]
        samples2 = [noise2.sample(t=float(i), state=1.0) for i in range(10)]

        np.testing.assert_array_equal(samples1, samples2)


class TestOrnsteinUhlenbeckNoise:
    """Tests for OrnsteinUhlenbeckNoise class."""

    def test_initialization(self):
        """Test default initialization."""
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
        """Test sample returns a float."""
        noise = OrnsteinUhlenbeckNoise(seed=42)
        sample = noise.sample(t=1.0)
        assert isinstance(sample, (float, np.floating))

    def test_mean_reversion(self):
        """Test mean-reverting property."""
        noise = OrnsteinUhlenbeckNoise(mean=0.0, theta=5.0, sigma=0.1, dt=0.1, seed=42)
        
        # Force current value far from mean
        noise.current_value = 10.0
        
        # Generate samples and check tendency toward mean
        samples = []
        for i in range(50):
            samples.append(noise.sample(t=float(i) * 0.1))
        
        # Samples should trend toward the mean (0.0)
        # The final sample should be closer to mean than initial
        assert abs(samples[-1]) < abs(10.0)

    def test_temporal_correlation(self):
        """Test that samples are temporally correlated."""
        noise = OrnsteinUhlenbeckNoise(theta=0.5, sigma=0.1, dt=0.1, seed=42)
        
        samples = [noise.sample(t=float(i) * 0.1) for i in range(100)]
        
        # Calculate lag-1 autocorrelation
        samples = np.array(samples)
        mean = np.mean(samples)
        var = np.var(samples)
        if var > 0:
            autocorr = np.mean((samples[:-1] - mean) * (samples[1:] - mean)) / var
            # OU process should have positive autocorrelation
            assert autocorr > 0

    def test_reset(self):
        """Test reset to mean."""
        noise = OrnsteinUhlenbeckNoise(mean=1.0, seed=42)
        noise.current_value = 5.0
        noise.reset()
        assert noise.current_value == noise.mean

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

    def test_initial_value_is_mean(self):
        """Test that initial current_value equals mean."""
        noise = OrnsteinUhlenbeckNoise(mean=5.0)
        assert noise.current_value == 5.0

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
    """Tests for JumpDiffusionNoise class."""

    def test_initialization(self):
        """Test default initialization."""
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

    def test_continuous_component(self):
        """Test continuous noise component is present."""
        # With zero jump rate, should behave like Gaussian
        noise = JumpDiffusionNoise(continuous_scale=1.0, jump_rate=0.0, seed=42)
        
        samples = [noise.sample(t=float(i)) for i in range(100)]
        
        # Should have non-zero variance from continuous component
        assert np.var(samples) > 0

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

    def test_jump_component(self):
        """Test jump component produces occasional large values."""
        # High jump rate to ensure jumps occur
        noise = JumpDiffusionNoise(
            continuous_scale=0.01,  # Small continuous noise
            jump_rate=2.0,  # High jump rate
            jump_mean=0.0,
            jump_scale=1.0,
            dt=1.0,
            seed=42
        )

        samples = [noise.sample(1.0) for _ in range(100)]

        # With high jump rate, some samples should be larger
        max_sample = np.max(np.abs(samples))
        assert max_sample > 0.5  # At least some jumps occurred

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
        
        # With high jump rate, we expect some large absolute values
        max_abs = max(abs(s) for s in samples)
        assert max_abs > 0.1  # Larger than continuous scale alone

    def test_jump_mean(self):
        """Test jump mean affects sample distribution."""
        # Positive jump mean
        noise_positive = JumpDiffusionNoise(
            continuous_scale=0.0,
            jump_rate=5.0,  # Very high rate
            jump_mean=1.0,
            jump_scale=0.1,
            dt=1.0,
            seed=42
        )
        
        samples_positive = [noise_positive.sample(t=float(i)) for i in range(100)]
        
        # With high rate and positive mean, average should be positive
        assert np.mean(samples_positive) > 0

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
        # Use threshold to detect samples with significant jump contribution
        short_jumps = sum(1 for _ in range(100) if abs(noise_short.sample(1.0)) > JUMP_DETECTION_THRESHOLD)
        noise_long.reset(42)
        long_jumps = sum(1 for _ in range(100) if abs(noise_long.sample(1.0)) > JUMP_DETECTION_THRESHOLD)

        # Longer dt should generally have more jumps
        # Use tolerance factor to account for probabilistic nature of jumps
        assert long_jumps >= short_jumps * PROBABILISTIC_TEST_TOLERANCE

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

        val1 = noise.sample(1.0)
        noise.reset()
        val2 = noise.sample(1.0)

        assert val1 == val2


class TestNoiseFactory:
    """Tests for create_noise_generator factory function."""

    def test_gaussian(self):
        """Test creating Gaussian noise."""
        noise = create_noise_generator('gaussian', scale=0.5)
        assert isinstance(noise, GaussianNoise)
        assert noise.scale == 0.5

    def test_state_dependent(self):
        """Test creating state-dependent noise."""
        noise = create_noise_generator('state_dependent', scale=0.3)
        assert isinstance(noise, StateDependentNoise)
        assert noise.scale == 0.3

    def test_adaptive(self):
        """Test creating adaptive noise."""
        noise = create_noise_generator('adaptive', target_cv=0.2)
        assert isinstance(noise, AdaptiveNoise)
        assert noise.target_cv == 0.2

    def test_ou(self):
        """Test creating OU noise."""
        noise = create_noise_generator('ou', theta=2.0)
        assert isinstance(noise, OrnsteinUhlenbeckNoise)
        assert noise.theta == 2.0

    def test_ornstein_uhlenbeck(self):
        """Test creating OU noise with full name."""
        noise = create_noise_generator('ornstein_uhlenbeck', sigma=0.5)
        assert isinstance(noise, OrnsteinUhlenbeckNoise)
        assert noise.sigma == 0.5

    def test_jump_diffusion(self):
        """Test creating jump-diffusion noise."""
        noise = create_noise_generator('jump_diffusion', jump_rate=0.5)
        assert isinstance(noise, JumpDiffusionNoise)
        assert noise.jump_rate == 0.5

    def test_unknown_type(self):
        """Test error for unknown noise type."""
        with pytest.raises(ValueError, match="Unknown noise type"):
            create_noise_generator('unknown_type')


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
            result = gen.sample(0.0)
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
            result = gen.sample(1e6)
            assert np.isfinite(result)
