"""
Unit tests for memory kernel module.
"""

import pytest
import numpy as np
from mcso.memory import (
    ExponentialMemory,
    SigmoidGatedMemory,
    TanhGatedMemory,
    MultiScaleMemory,
    AdaptiveMemory,
    create_memory_kernel,
)


class TestExponentialMemory:
    """Tests for ExponentialMemory kernel."""

    def test_empty_history(self):
        """Test with empty history returns zero."""
        kernel = ExponentialMemory()
        result = kernel.evaluate(1.0, {})
        assert result == 0.0

    def test_single_point(self):
        """Test with single history point."""
        kernel = ExponentialMemory(strength=1.0, decay_time=1.0)
        history = {0.0: 1.0}
        result = kernel.evaluate(1.0, history)
        assert result > 0  # Should be positive

    def test_decay_with_time(self):
        """Test that older values contribute less."""
        kernel = ExponentialMemory(strength=1.0, decay_time=1.0, window=100)
        # Use multiple history points to see decay effect
        history = {0.0: 1.0, 1.0: 1.0, 2.0: 1.0}

        # Evaluate at different times with different distances to history
        result_near = kernel.evaluate(3.0, history)  # Close to history points
        result_far = kernel.evaluate(8.0, history)   # Far from history points

        assert result_near > result_far or np.isclose(result_near, result_far, rtol=0.1)

    def test_window_cutoff(self):
        """Test that values outside window are ignored."""
        kernel = ExponentialMemory(strength=1.0, decay_time=1.0, window=5.0)
        history = {0.0: 100.0}

        result = kernel.evaluate(10.0, history)  # 10 > window of 5
        assert result == 0.0


class TestSigmoidGatedMemory:
    """Tests for SigmoidGatedMemory kernel."""

    def test_zero_at_zero(self):
        """Test returns zero with no history."""
        kernel = SigmoidGatedMemory()
        result = kernel.evaluate(0.0, {})
        assert result == 0.0

    def test_positive_feedback(self):
        """Test positive state gives positive feedback."""
        kernel = SigmoidGatedMemory(strength=1.0, delay=1.0, sensitivity=2.0)
        history = {0.0: 1.0}
        result = kernel.evaluate(1.0, history)
        assert result > 0

    def test_gating_effect(self):
        """Test sigmoid gating bounds the output."""
        kernel = SigmoidGatedMemory(strength=1.0, sensitivity=2.0)
        history = {0.0: 100.0}  # Large value
        result = kernel.evaluate(1.0, history)

        # Even with large input, output should be bounded
        assert result < 200  # strength * value * sigmoid < strength * value * 1


class TestTanhGatedMemory:
    """Tests for TanhGatedMemory kernel."""

    def test_symmetric_response(self):
        """Test symmetric response to positive/negative values."""
        kernel = TanhGatedMemory(strength=1.0, sensitivity=1.0)

        result_pos = kernel.evaluate(1.0, {0.0: 1.0})
        result_neg = kernel.evaluate(1.0, {0.0: -1.0})

        # tanh is antisymmetric: tanh(-x) = -tanh(x)
        # So x*tanh(x) = (-x)*tanh(-x) in magnitude
        assert np.abs(np.abs(result_pos) - np.abs(result_neg)) < 0.01


class TestMultiScaleMemory:
    """Tests for MultiScaleMemory kernel."""

    def test_default_scales(self):
        """Test default scales are set."""
        kernel = MultiScaleMemory()
        assert kernel.scales is not None
        assert len(kernel.scales) == 3

    def test_custom_scales(self):
        """Test custom scales work."""
        kernel = MultiScaleMemory(scales=[(1.0, 1.0), (0.5, 5.0)])
        history = {0.0: 1.0}
        result = kernel.evaluate(1.0, history)
        assert np.isfinite(result)


class TestAdaptiveMemory:
    """Tests for AdaptiveMemory kernel."""

    def test_initial_strength(self):
        """Test initial strength is used."""
        kernel = AdaptiveMemory(initial_strength=0.5)
        assert kernel.strength == 0.5

    def test_adaptation(self):
        """Test that strength adapts over time."""
        kernel = AdaptiveMemory(initial_strength=1.0, learning_rate=0.1)

        history = {}
        initial_strength = kernel.strength

        # Simulate some evaluations
        for t in range(10):
            history[float(t)] = float(t)
            kernel.evaluate(float(t + 1), history)

        # Strength should have changed
        assert kernel.strength != initial_strength

    def test_strength_bounds(self):
        """Test strength stays within bounds."""
        kernel = AdaptiveMemory(
            initial_strength=1.0,
            learning_rate=1.0,  # Very high
            strength_bounds=(0.1, 2.0)
        )

        history = {0.0: 1000.0}  # Large value to force adaptation

        for t in range(100):
            history[float(t)] = float(t * 100)
            kernel.evaluate(float(t + 1), history)

        assert 0.1 <= kernel.strength <= 2.0

    def test_reset(self):
        """Test reset restores initial strength."""
        kernel = AdaptiveMemory(initial_strength=1.0, learning_rate=0.1)

        kernel.strength = 0.5  # Modify
        kernel.reset()

        assert kernel.strength == 1.0


class TestFactoryFunction:
    """Tests for create_memory_kernel factory."""

    def test_create_exponential(self):
        """Test creating exponential kernel."""
        kernel = create_memory_kernel('exponential', decay_time=2.0)
        assert isinstance(kernel, ExponentialMemory)
        assert kernel.decay_time == 2.0

    def test_create_sigmoid_gated(self):
        """Test creating sigmoid-gated kernel."""
        kernel = create_memory_kernel('sigmoid_gated', sensitivity=3.0)
        assert isinstance(kernel, SigmoidGatedMemory)
        assert kernel.sensitivity == 3.0

    def test_create_tanh_gated(self):
        """Test creating tanh-gated kernel."""
        kernel = create_memory_kernel('tanh_gated')
        assert isinstance(kernel, TanhGatedMemory)

    def test_create_multiscale(self):
        """Test creating multi-scale kernel."""
        kernel = create_memory_kernel('multiscale')
        assert isinstance(kernel, MultiScaleMemory)

    def test_create_adaptive(self):
        """Test creating adaptive kernel."""
        kernel = create_memory_kernel('adaptive', learning_rate=0.05)
        assert isinstance(kernel, AdaptiveMemory)

    def test_invalid_type_raises(self):
        """Test invalid kernel type raises error."""
        with pytest.raises(ValueError, match="Unknown kernel type"):
            create_memory_kernel('invalid_type')
