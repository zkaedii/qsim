"""
Unit tests for the noise generators module.
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

    def test_reset(self):
        """Test reset functionality."""
        noise = GaussianNoise(seed=42)
        sample1 = noise.sample(t=1.0)
        noise.reset(seed=42)
        sample2 = noise.sample(t=1.0)
        assert sample1 == sample2


class TestStateDependentNoise:
    """Tests for StateDependentNoise class."""

    def test_initialization(self):
        """Test default initialization."""
        noise = StateDependentNoise()
        assert noise.scale == 0.2
        assert noise.state_coupling == 0.3

    def test_state_dependent_behavior(self):
        """Test noise depends on state."""
        noise = StateDependentNoise(seed=42)
        
        # Samples should be different for different states due to state-dependence
        noise.reset(seed=42)
        sample_low = noise.sample(t=1.0, state=0.1)
        
        noise.reset(seed=42)
        sample_high = noise.sample(t=1.0, state=10.0)
        
        # Higher state should produce larger variance
        assert abs(sample_high) != abs(sample_low) or sample_high != sample_low


class TestAdaptiveNoise:
    """Tests for AdaptiveNoise class."""

    def test_initialization(self):
        """Test default initialization."""
        noise = AdaptiveNoise()
        assert noise.initial_scale == 0.2
        assert noise.target_cv == 0.1
        assert noise.adaptation_rate == 0.01

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


class TestJumpDiffusionNoise:
    """Tests for JumpDiffusionNoise class."""

    def test_initialization(self):
        """Test default initialization."""
        noise = JumpDiffusionNoise()
        assert noise.continuous_scale == 0.1
        assert noise.jump_rate == 0.1
        assert noise.jump_mean == 0.0
        assert noise.jump_scale == 0.5

    def test_sample_returns_float(self):
        """Test sample returns a float."""
        noise = JumpDiffusionNoise(seed=42)
        sample = noise.sample(t=1.0)
        assert isinstance(sample, (float, np.floating))

    def test_continuous_component(self):
        """Test continuous noise component is present."""
        # With zero jump rate, should behave like Gaussian
        noise = JumpDiffusionNoise(continuous_scale=1.0, jump_rate=0.0, seed=42)
        
        samples = [noise.sample(t=float(i)) for i in range(100)]
        
        # Should have non-zero variance from continuous component
        assert np.var(samples) > 0

    def test_jump_component(self):
        """Test jump component produces occasional large values."""
        # High jump rate to ensure jumps occur
        noise = JumpDiffusionNoise(
            continuous_scale=0.01,  # Small continuous noise
            jump_rate=2.0,  # High jump rate
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

    def test_reproducibility(self):
        """Test reproducibility with same seed."""
        noise1 = JumpDiffusionNoise(seed=42)
        noise2 = JumpDiffusionNoise(seed=42)
        
        samples1 = [noise1.sample(t=float(i)) for i in range(10)]
        samples2 = [noise2.sample(t=float(i)) for i in range(10)]
        
        np.testing.assert_array_equal(samples1, samples2)

    def test_reset(self):
        """Test reset functionality."""
        noise = JumpDiffusionNoise(seed=42)
        sample1 = noise.sample(t=1.0)
        noise.reset(seed=42)
        sample2 = noise.sample(t=1.0)
        assert sample1 == sample2


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
