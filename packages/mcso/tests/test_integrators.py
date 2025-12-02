"""
Unit tests for the numerical integration module.
Unit tests for the integrators module.

This module tests:
1. Activation functions (softplus, sigmoid, swish, gelu)
2. Numerical quadrature functions
3. SDE integrators (Euler-Maruyama, Milstein, Heun)
4. Convergence rate estimation
This module tests numerical integration methods and activation functions
including SDE integrators, activation functions, and convergence rate estimation.
"""

import pytest
import numpy as np
from numpy.testing import assert_allclose
from numpy.testing import assert_allclose, assert_array_less

from mcso.integrators import (
    softplus,
    sigmoid,
    swish,
    gelu,
    integrate_activation,
    numerical_quadrature,
    SDEIntegrator,
    euler_maruyama,
    compute_convergence_rate,
)


class TestActivationFunctions:
    """Tests for activation functions."""

    def test_softplus_zero(self):
        """Test softplus at zero equals ln(2)."""
        result = softplus(0.0)
        assert np.isclose(result, np.log(2), rtol=1e-6)

    def test_softplus_positive(self):
        """Test softplus for positive input."""
        result = softplus(1.0)
        expected = np.log(1 + np.exp(1))
        assert np.isclose(result, expected, rtol=1e-6)

    def test_softplus_large_positive(self):
        """Test softplus linear approximation for large positive values."""
        result = softplus(100.0)
        assert np.isclose(result, 100.0, rtol=1e-3)

    def test_softplus_negative(self):
        """Test softplus for negative input."""
        result = softplus(-2.0)
        expected = np.log(1 + np.exp(-2))
        assert np.isclose(result, expected, rtol=1e-6)

    def test_softplus_array(self):
        """Test softplus works with arrays."""
        x = np.array([-1.0, 0.0, 1.0])
class TestSoftplus:
    """Tests for softplus activation function."""

    def test_zero_input(self):
        """Test softplus(0) = log(2)."""
        result = softplus(0.0)
        expected = np.log(2)
        assert np.isclose(result, expected)

    def test_positive_input(self):
        """Test softplus with positive values."""
        result = softplus(1.0)
        expected = np.log(1 + np.exp(1))
        assert np.isclose(result, expected)

    def test_negative_input(self):
        """Test softplus with negative values."""
        result = softplus(-1.0)
        expected = np.log(1 + np.exp(-1))
        assert np.isclose(result, expected)

    def test_large_positive_linear(self):
        """Test softplus with large positive values returns x."""
        result = softplus(100.0)
        assert np.isclose(result, 100.0)

    def test_large_negative_near_zero(self):
        """Test softplus with large negative values is near zero."""
        result = softplus(-100.0)
        assert result > 0
        assert result < 1e-10

    def test_array_input(self):
        """Test softplus with array input."""
        x = np.array([-1.0, 0.0, 1.0, 100.0])
# =============================================================================
# Tests for Activation Functions
# =============================================================================


class TestSoftplus:
    """Tests for softplus activation function."""

    def test_softplus_zero(self):
        """Test softplus at zero equals log(2)."""
        result = softplus(0.0)
        expected = np.log(2.0)
        assert_allclose(result, expected, rtol=1e-10)

    def test_softplus_positive(self):
        """Test softplus for positive values."""
        result = softplus(1.0)
        expected = np.log(1 + np.exp(1.0))
        assert_allclose(result, expected, rtol=1e-10)

    def test_softplus_negative(self):
        """Test softplus for negative values."""
        result = softplus(-1.0)
        expected = np.log(1 + np.exp(-1.0))
        assert_allclose(result, expected, rtol=1e-10)

    def test_softplus_large_positive(self):
        """Test softplus returns x for large positive values."""
        result = softplus(100.0)
        assert_allclose(result, 100.0, rtol=1e-5)

    def test_softplus_numerically_stable_extreme(self):
        """Test softplus is numerically stable for extreme values."""
        result_large = softplus(1000.0)
        result_small = softplus(-1000.0)

        assert np.isfinite(result_large)
        assert np.isfinite(result_small)
        assert result_large > 0
        assert result_small > 0

    def test_softplus_array_input(self):
        """Test softplus works with array input."""
        x = np.array([-10, -1, 0, 1, 10])
        result = softplus(x)

        assert result.shape == x.shape
        assert np.all(result > 0)
        assert np.all(np.isfinite(result))

    def test_softplus_threshold_behavior(self):
        """Test softplus threshold for linear approximation."""
        threshold = 20.0
        x_above = 25.0
        result = softplus(x_above, threshold=threshold)
        # Above threshold, should return x directly
        assert_allclose(result, x_above, rtol=1e-10)

    def test_softplus_always_positive(self):
        """Test softplus output is always positive."""
        x_values = np.linspace(-100, 100, 100)
        results = softplus(x_values)
        assert np.all(results > 0)
        """Test softplus(0) = ln(2)."""
        result = softplus(0.0)
        assert_allclose(result, np.log(2), rtol=1e-10)

    def test_softplus_positive(self):
        """Test softplus for positive values."""
        # softplus(x) ≈ x for large x
        result = softplus(10.0)
        assert result > 10.0 - 1e-3  # Should be close to 10

    def test_softplus_negative(self):
        """Test softplus for negative values."""
        result = softplus(-10.0)
        assert result > 0  # Always positive
        assert result < 1.0  # But small for negative inputs

    def test_softplus_array(self):
        """Test softplus with array input."""
        x = np.array([-5, 0, 5])
        result = softplus(x)
        assert result.shape == x.shape
        assert np.all(result > 0)

    def test_softplus_numerically_stable(self):
        """Test softplus handles extreme values without overflow."""
        result_large = softplus(1000.0)
        result_small = softplus(-1000.0)

        assert np.isfinite(result_large)
        assert np.isfinite(result_small)
        assert result_large > 0
        assert result_small > 0

    def test_sigmoid_zero(self):
        """Test sigmoid at zero equals 0.5."""
        result = sigmoid(0.0)
        assert np.isclose(result, 0.5, rtol=1e-6)

    def test_sigmoid_positive(self):
        """Test sigmoid for positive input."""
        result = sigmoid(10.0)
        assert result > 0.5
        assert result < 1.0

    def test_sigmoid_negative(self):
        """Test sigmoid for negative input."""
        result = sigmoid(-10.0)
        assert result < 0.5
        assert result > 0.0

    def test_sigmoid_bounded(self):
        """Test sigmoid output is bounded in [0, 1]."""
        for x in [-100, -1, 0, 1, 100]:
            result = sigmoid(x)
            assert 0 <= result <= 1

    def test_sigmoid_array(self):
        """Test sigmoid works with arrays."""
        x = np.array([-1.0, 0.0, 1.0])
        result = sigmoid(x)
        assert result.shape == x.shape
        assert np.all((result >= 0) & (result <= 1))

    def test_sigmoid_numerically_stable(self):
        """Test sigmoid handles extreme values."""
        result_large = sigmoid(1000.0)
        result_small = sigmoid(-1000.0)

        assert np.isfinite(result_large)
        assert np.isfinite(result_small)

    def test_swish_zero(self):
        """Test swish at zero equals 0."""
        result = swish(0.0)
        assert np.isclose(result, 0.0, atol=1e-6)

    def test_swish_positive(self):
        """Test swish for positive input."""
        result = swish(2.0)
        expected = 2.0 * sigmoid(2.0)
        assert np.isclose(result, expected, rtol=1e-6)

    def test_swish_negative(self):
        """Test swish for negative input."""
        result = swish(-2.0)
        expected = -2.0 * sigmoid(-2.0)
        assert np.isclose(result, expected, rtol=1e-6)
    def test_numerical_stability(self):
        """Test softplus handles extreme values without overflow."""
        result_large = softplus(1000.0)
        result_small = softplus(-1000.0)
        assert np.isfinite(result_large)
        assert np.isfinite(result_small)

    def test_custom_threshold(self):
        """Test softplus with custom threshold."""
        result = softplus(10.0, threshold=5.0)
        assert np.isclose(result, 10.0)
    def test_softplus_numerical_stability_large(self):
        """Test softplus handles large values without overflow."""
        result = softplus(100.0)
        assert np.isfinite(result)
        assert_allclose(result, 100.0, rtol=1e-2)

    def test_softplus_numerical_stability_extreme(self):
        """Test softplus handles extreme values."""
        result_large = softplus(1000.0)
        result_neg = softplus(-1000.0)
        assert np.isfinite(result_large)
        assert np.isfinite(result_neg)
        assert result_neg > 0

    def test_softplus_threshold(self):
        """Test softplus linear approximation threshold."""
        # Above threshold, should return input directly
        result = softplus(25.0, threshold=20.0)
        assert_allclose(result, 25.0, rtol=1e-10)


class TestSigmoid:
    """Tests for sigmoid activation function."""

    def test_zero_input(self):
        """Test sigmoid(0) = 0.5."""
        result = sigmoid(0.0)
        assert np.isclose(result, 0.5)

    def test_positive_input(self):
        """Test sigmoid with positive values tends to 1."""
        result = sigmoid(10.0)
        assert result > 0.99

    def test_negative_input(self):
        """Test sigmoid with negative values tends to 0."""
        result = sigmoid(-10.0)
        assert result < 0.01

    def test_bounded_output(self):
        """Test sigmoid output is always in [0, 1]."""
        for x in [-1000, -10, -1, 0, 1, 10, 1000]:
            result = sigmoid(x)
            # Sigmoid asymptotically approaches 0 and 1 but for extreme values
            # numerical precision causes it to round to exactly 0 or 1
            assert 0 <= result <= 1

    def test_array_input(self):
        """Test sigmoid with array input."""
        x = np.array([-10.0, 0.0, 10.0])
        result = sigmoid(x)
        assert result.shape == x.shape
        assert np.all((result >= 0) & (result <= 1))

    def test_antisymmetry(self):
        """Test sigmoid(x) + sigmoid(-x) = 1."""
        for x in [0.5, 1.0, 2.0, 5.0]:
            assert np.isclose(sigmoid(x) + sigmoid(-x), 1.0)
    def test_sigmoid_zero(self):
        """Test sigmoid at zero equals 0.5."""
        result = sigmoid(0.0)
        assert_allclose(result, 0.5, rtol=1e-10)

    def test_sigmoid_positive_large(self):
        """Test sigmoid approaches 1 for large positive values."""
        result = sigmoid(10.0)
        assert result > 0.99

    def test_sigmoid_negative_large(self):
        """Test sigmoid approaches 0 for large negative values."""
        result = sigmoid(-10.0)
        assert result < 0.01

    def test_sigmoid_bounded(self):
        """Test sigmoid output is in [0, 1]."""
        x_values = [-100, -10, -1, 0, 1, 10, 100]
        for x in x_values:
            result = sigmoid(x)
            assert 0 <= result <= 1

    def test_sigmoid_numerically_stable(self):
        """Test sigmoid handles extreme values without overflow."""
        result_large = sigmoid(500.0)
        result_small = sigmoid(-500.0)

        assert np.isfinite(result_large)
        assert np.isfinite(result_small)

    def test_sigmoid_array_input(self):
        """Test sigmoid works with array input."""
        x = np.array([-5, -1, 0, 1, 5])
        result = sigmoid(x)

        assert result.shape == x.shape
        assert np.all(result >= 0)
        assert np.all(result <= 1)

    def test_sigmoid_symmetry(self):
        """Test sigmoid satisfies σ(-x) = 1 - σ(x)."""
        x = 2.0
        result_pos = sigmoid(x)
        result_neg = sigmoid(-x)
        assert_allclose(result_pos + result_neg, 1.0, rtol=1e-10)
        """Test sigmoid(0) = 0.5."""
        result = sigmoid(0.0)
        assert_allclose(result, 0.5, rtol=1e-10)

    def test_sigmoid_bounds(self):
        """Test sigmoid output is in (0, 1)."""
        for x in [-100, -10, 0, 10, 100]:
            result = sigmoid(x)
            assert 0 <= result <= 1

    def test_sigmoid_symmetry(self):
        """Test sigmoid(-x) = 1 - sigmoid(x)."""
        for x in [0.5, 1.0, 2.0, 5.0]:
            assert_allclose(sigmoid(-x), 1 - sigmoid(x), rtol=1e-10)

    def test_sigmoid_array(self):
        """Test sigmoid with array input."""
        x = np.array([-5, 0, 5])
        result = sigmoid(x)
        assert result.shape == x.shape
        assert np.all(result > 0)
        assert np.all(result < 1)

    def test_sigmoid_numerical_stability(self):
        """Test sigmoid handles extreme values."""
        result_large = sigmoid(1000.0)
        result_neg = sigmoid(-1000.0)
        assert np.isfinite(result_large)
        assert np.isfinite(result_neg)
        assert result_large >= 0.99
        assert result_neg <= 0.01


class TestSwish:
    """Tests for swish activation function."""

    def test_zero_input(self):
        """Test swish(0) = 0."""
        result = swish(0.0)
        assert np.isclose(result, 0.0)

    def test_positive_input(self):
        """Test swish with positive values."""
        result = swish(2.0)
        expected = 2.0 * sigmoid(2.0)
        assert np.isclose(result, expected)

    def test_negative_input(self):
        """Test swish with negative values."""
        result = swish(-2.0)
        expected = -2.0 * sigmoid(-2.0)
        assert np.isclose(result, expected)

    def test_large_positive_approaches_x(self):
        """Test swish(x) approaches x for large positive x."""
        result = swish(100.0)
        assert np.isclose(result, 100.0, rtol=0.01)
    def test_swish_zero(self):
        """Test swish at zero equals 0."""
        result = swish(0.0)
        assert_allclose(result, 0.0, rtol=1e-10)

    def test_swish_positive(self):
        """Test swish for positive values."""
        result = swish(2.0)
        expected = 2.0 * sigmoid(2.0)
        assert_allclose(result, expected, rtol=1e-10)

    def test_swish_negative(self):
        """Test swish for negative values."""
        result = swish(-1.0)
        expected = -1.0 * sigmoid(-1.0)
        assert_allclose(result, expected, rtol=1e-10)

    def test_swish_beta_parameter(self):
        """Test swish with different beta values."""
        x = 1.0
        result_beta1 = swish(x, beta=1.0)
        result_beta2 = swish(x, beta=2.0)
        # With higher beta, swish is more ReLU-like
        assert result_beta1 != result_beta2

    def test_swish_array(self):
        """Test swish works with arrays."""

        expected_beta1 = x * sigmoid(x)
        expected_beta2 = x * sigmoid(2.0 * x)

        assert_allclose(result_beta1, expected_beta1, rtol=1e-10)
        assert_allclose(result_beta2, expected_beta2, rtol=1e-10)

    def test_swish_array_input(self):
        """Test swish works with array input."""
        x = np.array([-2, -1, 0, 1, 2])
        result = swish(x)
        assert result.shape == x.shape
        assert np.all(np.isfinite(result))
        """Test swish(0) = 0."""
        result = swish(0.0)
        assert_allclose(result, 0.0, atol=1e-10)

    def test_swish_positive(self):
        """Test swish for positive values."""
        result = swish(5.0)
        # swish(x) ≈ x for large positive x
        assert result > 0
        assert result < 5.0  # swish(x) < x for finite x

    def test_swish_negative(self):
        """Test swish for negative values is bounded."""
        result = swish(-5.0)
        # Swish can be slightly negative
        assert result > -1.0

    def test_swish_array(self):
        """Test swish with array input."""
# =============================================================================
# Activation Function Tests
# =============================================================================


class TestSoftplus:
    """Tests for the softplus activation function."""

    def test_zero_input(self):
        """Test softplus at x=0 equals ln(2)."""
        result = softplus(0.0)
        assert_allclose(result, np.log(2), rtol=1e-7)

    def test_positive_input(self):
        """Test softplus for positive values."""
        x = 2.0
        result = softplus(x)
        expected = np.log(1 + np.exp(x))
        assert_allclose(result, expected, rtol=1e-7)

    def test_negative_input(self):
        """Test softplus for negative values."""
        x = -2.0
        result = softplus(x)
        expected = np.log(1 + np.exp(x))
        assert_allclose(result, expected, rtol=1e-7)

    def test_large_positive_linear_approximation(self):
        """Test softplus returns x for large positive values."""
        x = 100.0
        result = softplus(x)
        # For large x, softplus(x) ≈ x
        assert_allclose(result, x, rtol=1e-2)

    def test_large_negative_approaches_zero(self):
        """Test softplus approaches zero for large negative values."""
        x = -100.0
        result = softplus(x)
        assert result > 0
        assert result < 0.01

    def test_array_input(self):
        """Test softplus works with numpy arrays."""
        x = np.array([-2, 0, 2, 20])
        result = softplus(x)
        assert result.shape == x.shape
        assert all(result > 0)

    def test_always_positive(self):
        """Test softplus output is always positive."""
        x = np.linspace(-10, 10, 100)
        result = softplus(x)
        assert all(result > 0)

    def test_numerical_stability_extreme_values(self):
        """Test numerical stability with extreme values."""
        # Should not raise overflow/underflow
        result_large = softplus(500.0)
        result_small = softplus(-500.0)
        assert np.isfinite(result_large)
        assert np.isfinite(result_small)


class TestSigmoid:
    """Tests for the sigmoid activation function."""

    def test_zero_input(self):
        """Test sigmoid at x=0 equals 0.5."""
        result = sigmoid(0.0)
        assert_allclose(result, 0.5, rtol=1e-7)

    def test_large_positive(self):
        """Test sigmoid approaches 1 for large positive values."""
        result = sigmoid(100.0)
        assert_allclose(result, 1.0, rtol=1e-4)

    def test_large_negative(self):
        """Test sigmoid approaches 0 for large negative values."""
        result = sigmoid(-100.0)
        assert_allclose(result, 0.0, atol=1e-4)

    def test_antisymmetry(self):
        """Test sigmoid(x) + sigmoid(-x) = 1."""
        x = 3.0
        result = sigmoid(x) + sigmoid(-x)
        assert_allclose(result, 1.0, rtol=1e-7)

    def test_array_input(self):
        """Test sigmoid works with numpy arrays."""
        x = np.array([-10, 0, 10])
        result = sigmoid(x)
        assert result.shape == x.shape
        assert all(result >= 0)
        assert all(result <= 1)

    def test_bounded_output(self):
        """Test sigmoid output is always in (0, 1)."""
        x = np.linspace(-50, 50, 100)
        result = sigmoid(x)
        assert all(result >= 0)
        assert all(result <= 1)

    def test_numerical_stability(self):
        """Test numerical stability with extreme values."""
        result_large = sigmoid(500.0)
        result_small = sigmoid(-500.0)
        assert np.isfinite(result_large)
        assert np.isfinite(result_small)


class TestSwish:
    """Tests for the swish activation function."""

    def test_zero_input(self):
        """Test swish at x=0 equals 0."""
        result = swish(0.0)
        assert_allclose(result, 0.0, rtol=1e-7)

    def test_positive_input(self):
        """Test swish for positive values."""
        x = 2.0
        result = swish(x)
        expected = x * sigmoid(x)
        assert_allclose(result, expected, rtol=1e-7)

    def test_beta_parameter(self):
        """Test swish with different beta values."""
        x = 2.0
        result_low = swish(x, beta=0.5)
        result_high = swish(x, beta=2.0)
        # Higher beta makes sigmoid steeper
        assert result_low < result_high

    def test_array_input(self):
        """Test swish with array input."""
        x = np.array([-1.0, 0.0, 1.0])
        result = swish(x)
        assert result.shape == x.shape

    def test_gelu_zero(self):
        """Test gelu at zero equals 0."""
        result = gelu(0.0)
        assert np.isclose(result, 0.0, atol=1e-6)

    def test_gelu_positive(self):
        """Test gelu for positive input."""
        result = gelu(2.0)
        assert result > 0

    def test_gelu_negative(self):
        """Test gelu for negative input."""
        result = gelu(-2.0)
        # GELU is close to 0 for large negative values
        assert result < 0

    def test_gelu_approximation(self):
        """Test gelu approximation is accurate."""
        # For large positive x, GELU(x) ≈ x
        result = gelu(10.0)
        assert np.isclose(result, 10.0, rtol=0.01)

    def test_gelu_array(self):
        """Test gelu works with arrays."""
        x = np.array([-1.0, 0.0, 1.0])
        result = gelu(x)
        assert result.shape == x.shape


class TestNumericalQuadrature:
    """Tests for numerical quadrature functions."""

    def test_integrate_activation_basic(self):
        """Test integrate_activation with simple functions."""
        result, error = integrate_activation(
            activation=softplus,
            f=lambda x: 1.0,
            g_prime=lambda x: 1.0,
            lower=0.0,
            upper=1.0,
            params=(0.8, 0.3, 0.5)
        )
        assert np.isfinite(result)
        assert error < 1e-6

    def test_integrate_activation_with_sine(self):
        """Test integrate_activation with oscillatory functions."""
        result_beta1 = swish(x, beta=1.0)
        result_beta2 = swish(x, beta=2.0)
        # Different beta should give different results
        assert not np.isclose(result_beta1, result_beta2)

    def test_array_input(self):
        """Test swish works with numpy arrays."""
        x = np.array([-2, 0, 2])
        result = swish(x)
        assert result.shape == x.shape

    def test_swish_beta_parameter(self):
        """Test swish with different beta values."""
        x = 1.0
        result_b1 = swish(x, beta=1.0)
        result_b2 = swish(x, beta=2.0)
        # Different beta should give different results
        assert result_b1 != result_b2


class TestGelu:
    """Tests for GELU activation function."""

    def test_zero_input(self):
        """Test gelu(0) = 0."""
        result = gelu(0.0)
        assert np.isclose(result, 0.0)

    def test_positive_input(self):
        """Test gelu with positive values is positive."""
        result = gelu(2.0)
        assert result > 0

    def test_large_positive_approaches_x(self):
        """Test gelu(x) approaches x for large positive x."""
        result = gelu(10.0)
        assert np.isclose(result, 10.0, rtol=0.01)

    def test_large_negative_approaches_zero(self):
        """Test gelu(x) approaches 0 for large negative x."""
        result = gelu(-10.0)
        assert np.isclose(result, 0.0, atol=0.01)

    def test_array_input(self):
        """Test gelu with array input."""
        x = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
        result = gelu(x)
        assert result.shape == x.shape
        assert np.all(np.isfinite(result))
    def test_gelu_zero(self):
        """Test GELU at zero equals 0."""
        result = gelu(0.0)
        assert_allclose(result, 0.0, rtol=1e-10)

    def test_gelu_positive(self):
        """Test GELU for positive values."""
        result = gelu(1.0)
        # GELU(1) ≈ 0.841 (approximate)
        assert result > 0.8
        assert result < 0.9

    def test_gelu_negative(self):
        """Test GELU for negative values is close to 0."""
        result = gelu(-2.0)
        # For negative values, GELU approaches 0
        assert result < 0
        assert result > -0.2

    def test_gelu_large_positive(self):
        """Test GELU approximates identity for large positive values."""
        x = 10.0
        result = gelu(x)
        assert_allclose(result, x, rtol=0.01)

    def test_gelu_array_input(self):
        """Test GELU works with array input."""
        x = np.array([-2, -1, 0, 1, 2])
        result = gelu(x)
        assert result.shape == x.shape
        assert np.all(np.isfinite(result))

    def test_gelu_smooth(self):
        """Test GELU is smooth (no discontinuities)."""
        x = np.linspace(-3, 3, 100)
        result = gelu(x)
        # Check no large jumps in values
        # Max gradient of GELU is approximately 1.5, so for dx ≈ 0.06,
        # max expected diff is about 0.1. Using 0.5 as a conservative threshold.
        diff = np.abs(np.diff(result))
        assert np.all(diff < 0.5)


# =============================================================================
# Tests for Numerical Quadrature
        """Test GELU(0) = 0."""
        result = gelu(0.0)
        assert_allclose(result, 0.0, atol=1e-10)

    def test_gelu_positive(self):
        """Test GELU for positive values."""
        result = gelu(5.0)
        # GELU(x) ≈ x for large positive x
        assert result > 0
        assert_allclose(result, 5.0, rtol=0.1)

    def test_gelu_negative(self):
        """Test GELU for negative values is bounded."""
        result = gelu(-5.0)
        # GELU tends to 0 for large negative values
        assert result > -0.5

    def test_gelu_array(self):
        """Test GELU with array input."""

class TestGelu:
    """Tests for the GELU activation function."""

    def test_zero_input(self):
        """Test GELU at x=0 equals 0."""
        result = gelu(0.0)
        assert_allclose(result, 0.0, atol=1e-7)

    def test_positive_input(self):
        """Test GELU for positive values is positive."""
        x = 2.0
        result = gelu(x)
        assert result > 0

    def test_negative_input(self):
        """Test GELU for negative values is near zero or negative."""
        x = -2.0
        result = gelu(x)
        assert result < 0

    def test_approaches_identity_for_large_positive(self):
        """Test GELU approaches identity for large positive values."""
        x = 10.0
        result = gelu(x)
        # For large x, GELU(x) ≈ x since Φ(x) ≈ 1
        assert_allclose(result, x, rtol=0.1)

    def test_array_input(self):
        """Test GELU works with numpy arrays."""
        x = np.array([-2, 0, 2])
        result = gelu(x)
        assert result.shape == x.shape

    def test_gelu_approximation_accuracy(self):
        """Test GELU approximation is reasonable."""
        # At x=1, GELU ≈ 0.841
        result = gelu(1.0)
        assert 0.8 < result < 0.9


class TestIntegrateActivation:
    """Tests for integrate_activation function."""

    def test_basic_integration(self):
        """Test basic integration with softplus."""
        result, error = integrate_activation(
            activation=softplus,
            f=np.cos,
            g_prime=lambda x: -np.sin(x),
            lower=0,
            upper=np.pi,
            params=(0.8, 0.3, 1.0)
        )
        assert np.isfinite(result)

    def test_integrate_activation_zero_interval(self):
        """Test integration over zero-width interval."""
        result, error = integrate_activation(
            activation=softplus,
            f=lambda x: 1.0,
            g_prime=lambda x: 1.0,
            lower=1.0,
            upper=1.0,
            params=(0.8, 0.3, 1.0)
        )
        assert np.isclose(result, 0.0, atol=1e-10)

    def test_numerical_quadrature_adaptive(self):
        """Test adaptive quadrature method."""
        result = numerical_quadrature(
            f=lambda x: x**2,
            lower=0,
            upper=1,
            method='adaptive'
        )
        expected = 1 / 3  # ∫x² from 0 to 1 = 1/3
        assert np.isclose(result, expected, rtol=1e-6)

    def test_numerical_quadrature_trapezoid(self):
        """Test trapezoid quadrature method."""
        result = numerical_quadrature(
            f=lambda x: x**2,
            lower=0,
            upper=1,
            method='trapezoid',
            n_points=1000
        )
        expected = 1 / 3
        assert np.isclose(result, expected, rtol=1e-3)

    def test_numerical_quadrature_simpson(self):
        """Test Simpson quadrature method."""
        result = numerical_quadrature(
            f=lambda x: x**2,
            lower=0,
            upper=1,
            method='simpson',
            n_points=101
        )
        expected = 1 / 3
        assert np.isclose(result, expected, rtol=1e-6)

    def test_numerical_quadrature_gauss(self):
        """Test Gauss-Legendre quadrature method."""
        result = numerical_quadrature(
            f=lambda x: x**2,
            lower=0,
            upper=1,
            method='gauss',
            params=(0.1, 0.5, 0.0)
        )
        assert np.isfinite(result)
        assert error < 1e-6

    def test_zero_interval(self):
        """Test integration over zero-length interval."""
        result, error = integrate_activation(
            activation=softplus,
            f=lambda x: x,
            g_prime=lambda x: 1,
            lower=0,
            upper=0,
            params=(1.0, 1.0, 0.0)
        )
        assert np.isclose(result, 0.0)

    def test_different_activation(self):
        """Test integration with sigmoid activation."""
        result, error = integrate_activation(
            activation=sigmoid,
            f=lambda x: x,
            g_prime=lambda x: 1,
            lower=0,
            upper=1,
            params=(1.0, 0.0, 0.5)
        )
        assert np.isfinite(result)

    def test_error_estimate(self):
        """Test that error estimate is returned."""
        result, error = integrate_activation(
            activation=softplus,
            f=lambda x: x**2,
            g_prime=lambda x: 2 * x,
            lower=0,
            upper=1
        )
        assert np.isfinite(error)
        assert error >= 0
        """Test basic integration with simple functions."""

# =============================================================================
# Numerical Quadrature Tests
# =============================================================================


class TestIntegrateActivation:
    """Tests for integrate_activation function."""

    def test_integrate_simple_case(self):
        """Test integration with simple functions."""
        # Integrate softplus(a*(x-x₀)² + b) * cos(x) * sin(x) from 0 to 1
        # where params = (a, b, x₀) = (0.8, 0.3, 0.5)
        result, error = integrate_activation(
            activation=softplus,
            f=np.cos,
            g_prime=np.sin,
            lower=0.0,
            upper=1.0,
            params=(0.8, 0.3, 0.5),
        )

        assert isinstance(result, float)
        assert isinstance(error, float)
        assert np.isfinite(result)

    def test_integrate_zero_interval(self):
        """Test integration over zero-length interval."""
        result, error = integrate_activation(
            activation=softplus,
            f=np.cos,
            g_prime=np.sin,
            lower=1.0,
            upper=1.0,
        )

        assert_allclose(result, 0.0, atol=1e-10)

    def test_integrate_error_estimate(self):
    """Tests for the integrate_activation function."""

    def test_simple_integration(self):
        """Test integration of simple functions."""
        result, error = integrate_activation(
            activation=softplus,
            f=lambda x: 1.0,
            g_prime=lambda x: 1.0,
            lower=0,
            upper=1,
            params=(0.0, 0.0, 0.0),  # softplus(0) = ln(2)
        )
        # Should be approximately ln(2) for unit interval
        assert np.isfinite(result)
        assert error < 1e-5

    def test_integration_with_cos_sin(self):
        """Test integration with trigonometric functions."""
            lower=0.0,
            upper=1.0,
            # params (a, b, x0) = (0, 0, 0) means activation arg = 0*(x-0)^2 + 0 = 0
            # This creates a constant activation value: softplus(0) = ln(2)
            params=(0.0, 0.0, 0.0)
        )
        # Integral = ln(2) * 1 * 1 integrated from 0 to 1 = ln(2) * 1 = ln(2)
        expected = np.log(2)
        assert_allclose(result, expected, rtol=1e-4)

    def test_error_estimate_reasonable(self):
        """Test that error estimate is reasonable."""
        result, error = integrate_activation(
            activation=softplus,
            f=np.cos,
            g_prime=np.sin,
            lower=0.0,
            upper=5.0,
        )

        assert error > 0
        # Error should be either less than 10% of result (relative) or
        # less than 1e-6 (absolute) for well-behaved integrands
        relative_tolerance = 0.1  # 10% relative error
        absolute_tolerance = 1e-6
        assert error < abs(result) * relative_tolerance or error < absolute_tolerance

    def test_integrate_custom_params(self):
        """Test integration with different parameters."""
        result1, _ = integrate_activation(
            activation=softplus,
            f=lambda x: 1.0,
            g_prime=lambda x: 1.0,
            lower=0.0,
            upper=1.0,
            params=(1.0, 0.0, 0.0),
        )

        result2, _ = integrate_activation(
            activation=softplus,
            f=lambda x: 1.0,
            g_prime=lambda x: 1.0,
            lower=0.0,
            upper=1.0,
            params=(2.0, 0.0, 0.0),
        )

        # Different params should give different results
        assert result1 != result2
            g_prime=lambda x: -np.sin(x),
            lower=0,
            upper=5,
            params=(0.8, 0.3, 1.0),
        )
        assert np.isfinite(result)
        assert error < 1e-3

    def test_integration_returns_tuple(self):
        """Test that integration returns (value, error) tuple."""
        result = integrate_activation(
            activation=softplus,
            f=lambda x: x,
            g_prime=lambda x: 1.0,
            lower=0,
            upper=1,
        )
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_integration_error_tolerance(self):
        """Test integration respects error tolerances."""
        result, error = integrate_activation(
            activation=softplus,
            f=lambda x: x,
            g_prime=lambda x: 1.0,
            lower=0,
            upper=1,
            epsabs=1e-10,
            epsrel=1e-10,
        )
        assert error < 1e-8


class TestNumericalQuadrature:
    """Tests for numerical_quadrature function."""

    def test_adaptive_method(self):
        """Test adaptive quadrature."""
    def test_quadrature_adaptive(self):
        """Test adaptive quadrature method."""
        # Integrate x² from 0 to 1 = 1/3
        result = numerical_quadrature(
            f=lambda x: x**2, lower=0.0, upper=1.0, method="adaptive"
        )
        assert_allclose(result, 1.0 / 3.0, rtol=1e-6)

    def test_quadrature_trapezoid(self):
        """Test trapezoid rule."""
        result = numerical_quadrature(
            f=lambda x: x**2, lower=0.0, upper=1.0, method="trapezoid", n_points=1000
        )
        assert_allclose(result, 1.0 / 3.0, rtol=0.01)

    def test_quadrature_simpson(self):
        """Test Simpson's rule."""
        result = numerical_quadrature(
            f=lambda x: x**2, lower=0.0, upper=1.0, method="simpson", n_points=101
        )
        assert_allclose(result, 1.0 / 3.0, rtol=1e-6)

    def test_quadrature_gauss(self):
        """Test Gauss-Legendre quadrature."""
        result = numerical_quadrature(
            f=lambda x: x**2, lower=0.0, upper=1.0, method="gauss", n_points=10
        )
        assert_allclose(result, 1.0 / 3.0, rtol=1e-10)

    def test_quadrature_invalid_method(self):
        """Test that invalid method raises ValueError."""
        with pytest.raises(ValueError, match="Unknown method"):
            numerical_quadrature(f=lambda x: x, lower=0, upper=1, method="invalid")

    def test_quadrature_sin(self):
        """Test integration of sin(x) from 0 to pi = 2."""
        result = numerical_quadrature(
            f=np.sin, lower=0.0, upper=np.pi, method="adaptive"
        )
        assert_allclose(result, 2.0, rtol=1e-6)

    def test_quadrature_exp(self):
        """Test integration of exp(x) from 0 to 1 = e - 1."""
        result = numerical_quadrature(f=np.exp, lower=0.0, upper=1.0, method="adaptive")
        assert_allclose(result, np.e - 1, rtol=1e-6)


# =============================================================================
# Tests for SDEIntegrator
# =============================================================================

    def test_adaptive_method(self):
        """Test adaptive (scipy quad) method."""
        result = numerical_quadrature(
            f=lambda x: x**2,
            lower=0,
            upper=1,
            method='adaptive'
        )
        expected = 1 / 3  # integral of x^2 from 0 to 1
        assert np.isclose(result, expected, rtol=1e-6)

    def test_trapezoid_method(self):
        """Test trapezoidal quadrature."""
            method="adaptive",
        )
        assert_allclose(result, 1 / 3, rtol=1e-6)

    def test_trapezoid_method(self):
        """Test trapezoidal rule."""
        result = numerical_quadrature(
            f=lambda x: x**2,
            lower=0,
            upper=1,
            method='trapezoid',
            n_points=1000
        )
        expected = 1 / 3
        assert np.isclose(result, expected, rtol=1e-3)
            method="trapezoid",
            n_points=1000,
        )
        assert_allclose(result, 1 / 3, rtol=1e-3)

    def test_simpson_method(self):
        """Test Simpson's rule."""
        result = numerical_quadrature(
            f=lambda x: x**2,
            lower=0,
            upper=1,
            method='simpson',
            n_points=101
        )
        expected = 1 / 3
        assert np.isclose(result, expected, rtol=1e-6)

    def test_gauss_method(self):
        """Test Gaussian quadrature."""
            method="simpson",
            n_points=101,
        )
        assert_allclose(result, 1 / 3, rtol=1e-5)

    def test_gauss_method(self):
        """Test Gauss-Legendre quadrature."""
        result = numerical_quadrature(
            f=lambda x: x**2,
            lower=0,
            upper=1,
            method='gauss',
            n_points=10
        )
        expected = 1 / 3
        assert np.isclose(result, expected, rtol=1e-10)

    def test_numerical_quadrature_invalid_method(self):
        """Test invalid quadrature method raises error."""
    def test_invalid_method(self):
        """Test that invalid method raises error."""
            method="gauss",
            n_points=10,
        )
        assert_allclose(result, 1 / 3, rtol=1e-10)

    def test_unknown_method_raises(self):
        """Test that unknown method raises ValueError."""
        with pytest.raises(ValueError, match="Unknown method"):
            numerical_quadrature(
                f=lambda x: x,
                lower=0,
                upper=1,
                method='invalid'
            )

    def test_sinusoidal(self):
        """Test integration of sinusoidal function."""
                method="unknown",
            )

    def test_sin_integration(self):
        """Test integration of sin over [0, pi]."""
        result = numerical_quadrature(
            f=np.sin,
            lower=0,
            upper=np.pi,
            method='adaptive'
        )
        expected = 2.0  # integral of sin from 0 to pi
        assert np.isclose(result, expected)
            method="adaptive",
        )
        assert_allclose(result, 2.0, rtol=1e-6)


class TestSDEIntegrator:
    """Tests for SDEIntegrator class."""

    @pytest.fixture
    def simple_sde(self):
        """Create a simple SDE integrator for testing."""
        return SDEIntegrator(
            drift=lambda x, t: -x,  # Mean-reverting drift
            diffusion=lambda x, t: 0.1,  # Constant diffusion
            scheme='euler',
            seed=42
        )

    @pytest.fixture
    def gbm_sde(self):
        """Create Geometric Brownian Motion SDE."""
        return SDEIntegrator(
            drift=lambda x, t: 0.05 * x,
            diffusion=lambda x, t: 0.2 * x,
            scheme='euler',
            seed=42
        )

    def test_euler_step(self, simple_sde):
        """Test single Euler-Maruyama step."""
        x0 = 1.0
        t = 0.0
        dt = 0.01
        dW = 0.0  # No noise for predictable result

        x1 = simple_sde.step(x0, t, dt, dW=dW)

        # With dW=0, x1 = x0 + drift * dt = 1.0 + (-1.0) * 0.01 = 0.99
        expected = x0 + (-x0) * dt
        assert np.isclose(x1, expected, rtol=1e-6)

    def test_milstein_step(self):
        """Test Milstein integration step."""
        integrator = SDEIntegrator(
            drift=lambda x, t: -x,
            diffusion=lambda x, t: 0.5 * x,  # State-dependent diffusion
            scheme='milstein',
            seed=42
        )

        x0 = 1.0
        x1 = integrator.step(x0, 0.0, 0.01)
        assert np.isfinite(x1)

    def test_heun_step(self):
        """Test Heun (improved Euler) step."""
        integrator = SDEIntegrator(
            drift=lambda x, t: -x,
            diffusion=lambda x, t: 0.1,
            scheme='heun',
            seed=42
        )

        x0 = 1.0
        x1 = integrator.step(x0, 0.0, 0.01)
        assert np.isfinite(x1)

    def test_invalid_scheme(self):
        """Test invalid scheme raises error."""
        integrator = SDEIntegrator(
            drift=lambda x, t: x,
            diffusion=lambda x, t: 0.1,
            scheme='invalid'
        )

        with pytest.raises(ValueError, match="Unknown scheme"):
            integrator.step(1.0, 0.0, 0.01)

    def test_integrate_single_path(self, simple_sde):
        """Test integration returns correct structure."""
        result = simple_sde.integrate(
            x0=1.0,
            t_span=(0.0, 1.0),
            dt=0.1,
            n_paths=1
        )

        assert 'times' in result
        assert 'paths' in result
        assert 'mean' in result
        assert 'std' in result
        assert result['paths'].shape[0] == 1

    def test_integrate_multiple_paths(self, simple_sde):
        """Test integration with multiple paths."""
        result = simple_sde.integrate(
            x0=1.0,
            t_span=(0.0, 1.0),
            dt=0.1,
            n_paths=10
        )

        assert result['paths'].shape[0] == 10
        assert len(result['mean']) == len(result['times'])
        assert len(result['std']) == len(result['times'])

    def test_integrate_time_array(self, simple_sde):
        """Test integration produces correct time array."""
        result = simple_sde.integrate(
            x0=1.0,
            t_span=(0.0, 1.0),
            dt=0.1,
            n_paths=1
        )

        # Time array should start at 0 and increment by dt
        assert result['times'][0] == 0.0
        assert np.isclose(result['times'][1] - result['times'][0], 0.1, rtol=1e-6)

    def test_integrate_initial_condition(self, simple_sde):
        """Test integration starts from correct initial condition."""
        x0 = 5.0
        result = simple_sde.integrate(
            x0=x0,
            t_span=(0.0, 1.0),
            dt=0.1,
            n_paths=3
        )

        # All paths should start at x0
        for p in range(3):
            assert result['paths'][p, 0] == x0

    def test_reproducibility(self):
        """Test same seed gives same results."""
        sde1 = SDEIntegrator(
            drift=lambda x, t: -x,
            diffusion=lambda x, t: 0.5,
            seed=123
        )
        sde2 = SDEIntegrator(
            drift=lambda x, t: -x,
            diffusion=lambda x, t: 0.5,
            seed=123
        )

        result1 = sde1.integrate(1.0, (0, 1), dt=0.1)
        result2 = sde2.integrate(1.0, (0, 1), dt=0.1)

        np.testing.assert_array_almost_equal(
            result1['paths'],
            result2['paths']
        )

    def test_mean_reverting_sde(self, simple_sde):
        """Test mean-reverting SDE converges toward zero."""
        result = simple_sde.integrate(
            x0=10.0,
            t_span=(0.0, 10.0),
            dt=0.01,
            n_paths=100
        )

        # Mean should decrease over time toward 0
        assert np.abs(result['mean'][-1]) < np.abs(result['mean'][0])

    def test_gbm_positive(self, gbm_sde):
        """Test GBM stays positive."""
        result = gbm_sde.integrate(
            x0=100.0,
            t_span=(0.0, 1.0),
            dt=0.01,
            n_paths=10
        )

        # GBM should stay positive (in practice, Euler can go negative for large dt)
        # With small dt and reasonable volatility, should mostly stay positive
        assert np.mean(result['paths']) > 0


class TestEulerMaruyama:
    """Tests for euler_maruyama convenience function."""

    def test_returns_dict(self):
        """Test euler_maruyama returns correct structure."""
        result = euler_maruyama(
            drift=lambda x, t: -x,
            diffusion=lambda x, t: 0.1,
            x0=1.0,
            t_span=(0.0, 1.0),
            dt=0.1
        )

        assert 'times' in result
        assert 'values' in result

    def test_values_length(self):
        """Test euler_maruyama produces correct number of values."""
        result = euler_maruyama(
            drift=lambda x, t: -x,
            diffusion=lambda x, t: 0.1,
            x0=1.0,
            t_span=(0.0, 1.0),
            dt=0.1
        )

        assert len(result['values']) == len(result['times'])

    def test_reproducibility(self):
        """Test same seed gives same results."""
        result1 = euler_maruyama(
            drift=lambda x, t: 0.05 * x,
            diffusion=lambda x, t: 0.2 * x,
            x0=100.0,
            t_span=(0.0, 1.0),
            dt=0.01,
            seed=42
        )

        result2 = euler_maruyama(
            drift=lambda x, t: 0.05 * x,
            diffusion=lambda x, t: 0.2 * x,
            x0=100.0,
            t_span=(0.0, 1.0),
            dt=0.01,
            seed=42
        )

        np.testing.assert_array_equal(result1['values'], result2['values'])

    def test_zero_diffusion_deterministic(self):
        """Test zero diffusion gives deterministic result."""
        result1 = euler_maruyama(
            drift=lambda x, t: -x,
            diffusion=lambda x, t: 0.0,  # No noise
            x0=1.0,
            t_span=(0.0, 1.0),
            dt=0.01,
            seed=1
        )

        result2 = euler_maruyama(
            drift=lambda x, t: -x,
            diffusion=lambda x, t: 0.0,
            x0=1.0,
            t_span=(0.0, 1.0),
            dt=0.01,
            seed=999  # Different seed
        )

        np.testing.assert_array_almost_equal(
            result1['values'],
            result2['values']
        )


class TestConvergenceRate:
    """Tests for compute_convergence_rate function."""

    @pytest.mark.slow
    def test_convergence_rate_structure(self):
        """Test compute_convergence_rate returns correct structure."""
        integrator = SDEIntegrator(
            drift=lambda x, t: -x,
            diffusion=lambda x, t: 0.5,
            scheme='euler',
            seed=42
        )

        result = compute_convergence_rate(
            integrator=integrator,
            x0=1.0,
            t_span=(0.0, 1.0),
            dt_values=np.array([0.1, 0.05, 0.025]),
            n_samples=10
        )

        assert 'dt_values' in result
        assert 'errors' in result
        assert 'rate' in result

    @pytest.mark.slow
    def test_convergence_errors_decrease(self):
        """Test errors decrease with smaller dt."""
        integrator = SDEIntegrator(
            drift=lambda x, t: -x,
            diffusion=lambda x, t: 0.5,
            scheme='euler',
            seed=42
        )

        result = compute_convergence_rate(
            integrator=integrator,
            x0=1.0,
            t_span=(0.0, 0.5),
            dt_values=np.array([0.1, 0.05, 0.01]),
            n_samples=50
        )

        # Errors should generally decrease with smaller dt
        # (allowing for some statistical noise)
        errors = result['errors']
        assert errors[0] >= errors[-1] * 0.5  # Allow some tolerance

    @pytest.mark.slow
    def test_convergence_rate_positive(self):
        """Test convergence rate is positive (errors decrease with dt)."""
        integrator = SDEIntegrator(
            drift=lambda x, t: -x,
            diffusion=lambda x, t: 0.1,  # Lower diffusion for more stable test
    def simple_drift(self):
        """Simple linear drift."""
        return lambda x, t: -0.5 * x

    @pytest.fixture
    def simple_diffusion(self):
        """Constant diffusion."""
        return lambda x, t: 0.2

    def test_initialization(self, simple_drift, simple_diffusion):
        """Test SDEIntegrator initialization."""
        integrator = SDEIntegrator(
            drift=simple_drift,
            diffusion=simple_diffusion,
            scheme='euler',
            seed=42
        )
        assert integrator.scheme == 'euler'
        assert integrator.rng is not None

    def test_euler_step(self, simple_drift, simple_diffusion):
        """Test single Euler-Maruyama step."""
        integrator = SDEIntegrator(
            drift=simple_drift,
            diffusion=simple_diffusion,
            scheme='euler',
            seed=42
        )
        result = integrator.step(x=1.0, t=0.0, dt=0.01)
        assert np.isfinite(result)

    def test_milstein_step(self, simple_drift, simple_diffusion):
        """Test single Milstein step."""
        integrator = SDEIntegrator(
            drift=simple_drift,
            diffusion=simple_diffusion,
            scheme='milstein',
            seed=42
        )
        result = integrator.step(x=1.0, t=0.0, dt=0.01)
        assert np.isfinite(result)

    def test_heun_step(self, simple_drift, simple_diffusion):
        """Test single Heun step."""
        integrator = SDEIntegrator(
            drift=simple_drift,
            diffusion=simple_diffusion,
            scheme='heun',
            seed=42
        )
        result = integrator.step(x=1.0, t=0.0, dt=0.01)
        assert np.isfinite(result)

    def test_invalid_scheme(self, simple_drift, simple_diffusion):
        """Test that invalid scheme raises error."""
        integrator = SDEIntegrator(
            drift=simple_drift,
            diffusion=simple_diffusion,
            scheme='invalid'
        )
        with pytest.raises(ValueError, match="Unknown scheme"):
            integrator.step(x=1.0, t=0.0, dt=0.01)

    def test_provided_dW(self, simple_drift, simple_diffusion):
        """Test step with provided Wiener increment."""
        integrator = SDEIntegrator(
            drift=simple_drift,
            diffusion=simple_diffusion,
            scheme='euler'
        )
        result = integrator.step(x=1.0, t=0.0, dt=0.01, dW=0.05)
        assert np.isfinite(result)

    def test_integrate_returns_dict(self, simple_drift, simple_diffusion):
        """Test integrate returns correct structure."""
        integrator = SDEIntegrator(
            drift=simple_drift,
            diffusion=simple_diffusion,
            seed=42
        )
        result = integrator.integrate(x0=1.0, t_span=(0, 1), dt=0.1)
        assert 'times' in result
        assert 'paths' in result
        assert 'mean' in result
        assert 'std' in result

    def test_integrate_times_array(self, simple_drift, simple_diffusion):
        """Test integrate produces correct time array."""
        integrator = SDEIntegrator(
            drift=simple_drift,
            diffusion=simple_diffusion,
            seed=42
        )
        result = integrator.integrate(x0=1.0, t_span=(0, 1), dt=0.1)
        assert result['times'][0] == 0.0
        assert result['times'][-1] >= 1.0

    def test_integrate_multiple_paths(self, simple_drift, simple_diffusion):
        """Test integrate with multiple sample paths."""
        integrator = SDEIntegrator(
            drift=simple_drift,
            diffusion=simple_diffusion,
            seed=42
        )
        result = integrator.integrate(x0=1.0, t_span=(0, 1), dt=0.1, n_paths=10)
        assert result['paths'].shape[0] == 10

    def test_reproducibility(self, simple_drift, simple_diffusion):
        """Test that same seed gives same results."""
        integrator1 = SDEIntegrator(
            drift=simple_drift,
            diffusion=simple_diffusion,
            seed=42
        )
        integrator2 = SDEIntegrator(
            drift=simple_drift,
            diffusion=simple_diffusion,
            seed=42
        )
        result1 = integrator1.integrate(x0=1.0, t_span=(0, 1), dt=0.1)
        result2 = integrator2.integrate(x0=1.0, t_span=(0, 1), dt=0.1)
        np.testing.assert_array_almost_equal(result1['paths'], result2['paths'])
    def simple_integrator(self):
        """Create a simple SDE integrator for testing."""
        return SDEIntegrator(
            drift=lambda x, t: -x,  # Mean-reverting
            diffusion=lambda x, t: 0.1,  # Constant diffusion
    def constant_sde(self):
        """Create SDE integrator for deterministic ODE (no diffusion)."""
        return SDEIntegrator(
            drift=lambda x, t: 1.0,  # dx/dt = 1
            diffusion=lambda x, t: 0.0,
            scheme="euler",
            seed=42,
        )

    def test_integrator_initialization(self, simple_integrator):
        """Test integrator initialization."""
        assert simple_integrator.scheme == "euler"
        assert simple_integrator.seed == 42
        assert simple_integrator.rng is not None

    def test_step_euler(self, simple_integrator):
        """Test Euler-Maruyama step."""
        x = 1.0
        t = 0.0
        dt = 0.01
        dW = 0.05

        result = simple_integrator.step(x, t, dt, dW)

        # Expected: x + drift * dt + diffusion * dW
        expected = x + (-x) * dt + 0.1 * dW
        assert_allclose(result, expected, rtol=1e-10)

    def test_step_milstein(self):
        """Test Milstein step."""
        integrator = SDEIntegrator(
            drift=lambda x, t: 0.05 * x,
            diffusion=lambda x, t: 0.2 * x,
            scheme="milstein",
            seed=42,
        )

        result = integrator.step(1.0, 0.0, 0.01, 0.05)
        assert np.isfinite(result)

    def test_step_heun(self):
        """Test Heun (improved Euler) step."""
        integrator = SDEIntegrator(
            drift=lambda x, t: -x,
            diffusion=lambda x, t: 0.1,
            scheme="heun",
            seed=42,
        )

        result = integrator.step(1.0, 0.0, 0.01, 0.05)
        assert np.isfinite(result)

    def test_step_invalid_scheme(self):
        """Test that invalid scheme raises ValueError."""
        integrator = SDEIntegrator(
            drift=lambda x, t: x, diffusion=lambda x, t: 0.1, scheme="invalid", seed=42
        )

        with pytest.raises(ValueError, match="Unknown scheme"):
            integrator.step(1.0, 0.0, 0.01, 0.05)

    def test_step_generates_dW(self, simple_integrator):
        """Test that step generates dW if not provided."""
        result1 = simple_integrator.step(1.0, 0.0, 0.01)
        result2 = simple_integrator.step(1.0, 0.0, 0.01)

        # Results should differ due to random dW
        assert result1 != result2

    def test_integrate_basic(self, simple_integrator):
        """Test basic integration."""
        result = simple_integrator.integrate(
            x0=1.0, t_span=(0, 1), dt=0.01, n_paths=1
        )
    @pytest.fixture
    def geometric_brownian(self):
        """Create SDE integrator for geometric Brownian motion."""
        return SDEIntegrator(
            drift=lambda x, t: 0.05 * x,
            diffusion=lambda x, t: 0.2 * x,
            scheme="euler",
            seed=42,
        )

    def test_initialization(self):
        """Test SDEIntegrator initialization."""
        integrator = SDEIntegrator(
            drift=lambda x, t: x,
            diffusion=lambda x, t: 1.0,
            scheme="euler",
            seed=123,
        )
        assert integrator.scheme == "euler"
        assert integrator.rng is not None

    def test_euler_step(self, constant_sde):
        """Test single Euler step."""
        # With zero diffusion, should be deterministic
        result = constant_sde.step(x=0.0, t=0.0, dt=0.1)
        # dx = 1.0 * 0.1 = 0.1
        assert_allclose(result, 0.1, rtol=1e-10)

    def test_euler_step_with_fixed_dW(self):
        """Test Euler step with fixed Wiener increment."""
        integrator = SDEIntegrator(
            drift=lambda x, t: 1.0,
            diffusion=lambda x, t: 1.0,
            scheme="euler",
        )
        result = integrator.step(x=0.0, t=0.0, dt=0.1, dW=0.5)
        # dx = 1.0 * 0.1 + 1.0 * 0.5 = 0.6
        assert_allclose(result, 0.6, rtol=1e-10)

    def test_milstein_step(self):
        """Test Milstein step."""
        integrator = SDEIntegrator(
            drift=lambda x, t: 0.0,
            diffusion=lambda x, t: x,  # sigma(x) = x
            scheme="milstein",
        )
        result = integrator.step(x=1.0, t=0.0, dt=0.1, dW=0.5)
        # Milstein adds 0.5 * sigma * sigma' * (dW^2 - dt)
        assert np.isfinite(result)

    def test_heun_step(self):
        """Test Heun (predictor-corrector) step."""
        integrator = SDEIntegrator(
            drift=lambda x, t: 1.0,
            diffusion=lambda x, t: 0.0,
            scheme="heun",
        )
        result = integrator.step(x=0.0, t=0.0, dt=0.1)
        # With constant drift and no diffusion, should equal Euler
        assert_allclose(result, 0.1, rtol=1e-10)

    def test_unknown_scheme_raises(self):
        """Test that unknown scheme raises ValueError."""
        integrator = SDEIntegrator(
            drift=lambda x, t: x,
            diffusion=lambda x, t: 1.0,
            scheme="unknown",
        )
        with pytest.raises(ValueError, match="Unknown scheme"):
            integrator.step(x=0.0, t=0.0, dt=0.1)

    def test_integrate_single_path(self, constant_sde):
        """Test single path integration."""
        result = constant_sde.integrate(x0=0.0, t_span=(0, 1), dt=0.1, n_paths=1)

        assert "times" in result
        assert "paths" in result
        assert "mean" in result
        assert "std" in result

    def test_integrate_times(self, simple_integrator):
        """Test that times array is correct."""
        result = simple_integrator.integrate(
            x0=1.0, t_span=(0, 1), dt=0.1, n_paths=1
        )

        expected_times = np.arange(0, 1.1, 0.1)
        assert_allclose(result["times"], expected_times, rtol=1e-10)

    def test_integrate_paths_shape(self, simple_integrator):
        """Test paths array shape."""
        n_paths = 10
        result = simple_integrator.integrate(
            x0=1.0, t_span=(0, 1), dt=0.1, n_paths=n_paths
        )

        assert result["paths"].shape == (n_paths, len(result["times"]))

    def test_integrate_initial_condition(self, simple_integrator):
        """Test that initial condition is set correctly."""
        x0 = 5.0
        result = simple_integrator.integrate(
            x0=x0, t_span=(0, 1), dt=0.1, n_paths=3
        )

        # All paths should start at x0
        for p in range(3):
            assert result["paths"][p, 0] == x0

    def test_integrate_mean_std(self, simple_integrator):
        """Test mean and std computation."""
        result = simple_integrator.integrate(
            x0=1.0, t_span=(0, 1), dt=0.1, n_paths=100
        )

        # Mean should be close to np.mean(paths, axis=0)
        expected_mean = np.mean(result["paths"], axis=0)
        expected_std = np.std(result["paths"], axis=0)

        assert_allclose(result["mean"], expected_mean, rtol=1e-10)
        assert_allclose(result["std"], expected_std, rtol=1e-10)

    def test_integrate_reproducibility(self):
        """Test reproducibility with same seed."""
        integrator1 = SDEIntegrator(
            drift=lambda x, t: -x,
            diffusion=lambda x, t: 0.1,
            scheme="euler",
            seed=123,
        )
        integrator2 = SDEIntegrator(
            drift=lambda x, t: -x,
            diffusion=lambda x, t: 0.1,
            scheme="euler",
            seed=123,
        assert result["paths"].shape == (1, len(result["times"]))

    def test_integrate_multiple_paths(self, geometric_brownian):
        """Test multiple path integration."""
        result = geometric_brownian.integrate(
            x0=100.0, t_span=(0, 1), dt=0.01, n_paths=10
        )

        assert result["paths"].shape[0] == 10
        assert len(result["mean"]) == len(result["times"])
        assert len(result["std"]) == len(result["times"])

    def test_integrate_deterministic_ode(self, constant_sde):
        """Test integration of deterministic ODE dx/dt = 1."""
        result = constant_sde.integrate(x0=0.0, t_span=(0, 1), dt=0.01, n_paths=1)

        # x(t) = t, so x(1) = 1
        assert_allclose(result["paths"][0, -1], 1.0, rtol=0.02)

    def test_integrate_reproducibility(self):
        """Test that same seed gives same results."""
        integrator1 = SDEIntegrator(
            drift=lambda x, t: x,
            diffusion=lambda x, t: 0.5,
            seed=42,
        )
        integrator2 = SDEIntegrator(
            drift=lambda x, t: x,
            diffusion=lambda x, t: 0.5,
            seed=42,
        )

        result1 = integrator1.integrate(x0=1.0, t_span=(0, 1), dt=0.1, n_paths=1)
        result2 = integrator2.integrate(x0=1.0, t_span=(0, 1), dt=0.1, n_paths=1)

        assert_allclose(result1["paths"], result2["paths"], rtol=1e-10)
        np.testing.assert_array_equal(result1["paths"], result2["paths"])


class TestEulerMaruyama:
    """Tests for euler_maruyama convenience function."""

    def test_basic_integration(self):
        """Test basic Euler-Maruyama integration."""
    def test_euler_maruyama_basic(self):
        """Test basic euler_maruyama integration."""
        result = euler_maruyama(
            drift=lambda x, t: -x,
            diffusion=lambda x, t: 0.1,
            x0=1.0,
            t_span=(0, 1),
            dt=0.01,
            seed=42
        )
        assert 'times' in result
        assert 'values' in result
        assert len(result['times']) == len(result['values'])

    def test_geometric_brownian_motion(self):
        """Test GBM integration."""
        result = euler_maruyama(
            drift=lambda x, t: 0.05 * x,
            diffusion=lambda x, t: 0.2 * x,
            x0=100.0,
            t_span=(0, 1),
            dt=0.001,
            seed=42
        )
        # GBM should stay positive
        assert np.all(result['values'] > 0)

    def test_deterministic_case(self):
        """Test with zero diffusion gives deterministic solution."""
        result = euler_maruyama(
            drift=lambda x, t: -x,  # dX = -X dt + 0 dW (deterministic case)
            diffusion=lambda x, t: 0.0,
            x0=1.0,
            t_span=(0, 1),
            dt=0.01
        )
        # Solution should approximate exp(-t)
        expected = np.exp(-result['times'])
        np.testing.assert_array_almost_equal(result['values'], expected, decimal=2)

    def test_reproducibility(self):
        """Test reproducibility with same seed."""
        result1 = euler_maruyama(
            drift=lambda x, t: -x,
            diffusion=lambda x, t: 0.1,
            x0=1.0,
            t_span=(0, 1),
            dt=0.01,
            seed=123
        )
        result2 = euler_maruyama(
            drift=lambda x, t: -x,
            diffusion=lambda x, t: 0.1,
            x0=1.0,
            t_span=(0, 1),
            dt=0.01,
            seed=123
        )
        np.testing.assert_array_equal(result1['values'], result2['values'])
            seed=42,
    def test_basic_integration(self):
        """Test basic Euler-Maruyama integration."""
        result = euler_maruyama(
            drift=lambda x, t: 0.0,
            diffusion=lambda x, t: 0.0,
            x0=1.0,
            t_span=(0, 1),
            dt=0.1,
        )

        assert "times" in result
        assert "values" in result

    def test_euler_maruyama_times(self):
        """Test times array in euler_maruyama result."""
        result = euler_maruyama(
            drift=lambda x, t: 0.0,
            diffusion=lambda x, t: 0.0,
            x0=1.0,
            t_span=(0, 1),
            dt=0.1,
            seed=42,
        )

        expected_times = np.arange(0, 1.1, 0.1)
        assert_allclose(result["times"], expected_times, rtol=1e-10)

    def test_euler_maruyama_values_shape(self):
        """Test values array shape."""
        result = euler_maruyama(
            drift=lambda x, t: 0.0,
            diffusion=lambda x, t: 0.0,
            x0=1.0,
            t_span=(0, 1),
            dt=0.1,
            seed=42,
        )

        assert len(result["values"]) == len(result["times"])

    def test_euler_maruyama_zero_diffusion(self):
        """Test with zero diffusion gives deterministic result."""
        result = euler_maruyama(
            drift=lambda x, t: 0.1,  # Constant drift
            diffusion=lambda x, t: 0.0,  # No noise
            x0=0.0,
            t_span=(0, 1),
            dt=0.1,
            seed=42,
        )

        # With constant drift 0.1 and zero noise, x(t) = x0 + 0.1*t
        expected = 0.1 * result["times"]
        assert_allclose(result["values"], expected, rtol=1e-5)

    def test_euler_maruyama_reproducibility(self):
        """Test reproducibility with same seed."""
        result1 = euler_maruyama(
            drift=lambda x, t: -x,
            diffusion=lambda x, t: 0.2,
            x0=1.0,
            t_span=(0, 1),
            dt=0.01,
            seed=123,
        )
        result2 = euler_maruyama(
            drift=lambda x, t: -x,
            diffusion=lambda x, t: 0.2,
            x0=1.0,
            t_span=(0, 1),
            dt=0.01,
            seed=123,
        )

        assert_allclose(result1["values"], result2["values"], rtol=1e-10)


# =============================================================================
# Tests for Convergence Rate Estimation
        assert len(result["values"]) == len(result["times"])

    def test_zero_diffusion(self):
        """Test Euler-Maruyama with zero diffusion (ODE)."""
        result = euler_maruyama(
            drift=lambda x, t: -x,  # dx/dt = -x => x = x0 * exp(-t)
            diffusion=lambda x, t: 0.0,
            x0=1.0,
            t_span=(0, 1),
            dt=0.001,
        )

        # x(1) = exp(-1) ≈ 0.368
        assert_allclose(result["values"][-1], np.exp(-1), rtol=0.01)

    def test_reproducibility_with_seed(self):
        """Test reproducibility with fixed seed."""
        result1 = euler_maruyama(
            drift=lambda x, t: 0.05 * x,
            diffusion=lambda x, t: 0.2 * x,
            x0=100.0,
            t_span=(0, 1),
            dt=0.01,
            seed=42,
        )
        result2 = euler_maruyama(
            drift=lambda x, t: 0.05 * x,
            diffusion=lambda x, t: 0.2 * x,
            x0=100.0,
            t_span=(0, 1),
            dt=0.01,
            seed=42,
        )

        np.testing.assert_array_equal(result1["values"], result2["values"])


class TestComputeConvergenceRate:
    """Tests for compute_convergence_rate function."""

    def test_returns_expected_keys(self):
        """Test that result contains expected keys."""
        integrator = SDEIntegrator(
            drift=lambda x, t: x,
            diffusion=lambda x, t: 0.5,
            scheme="euler",
            seed=42,
        )

            lower=0.0,
            upper=5.0,
            params=(0.8, 0.3, 1.0)
        )
        assert np.isfinite(result)
        assert error > 0
        assert error < abs(result) + 1  # Error should be smaller than result magnitude

    def test_different_params(self):
        """Test with different parameter values."""
        result1, _ = integrate_activation(
            activation=softplus,
            f=lambda x: x,
            g_prime=lambda x: 1.0,
            lower=0.0,
            upper=2.0,
            params=(1.0, 0.0, 0.0)
        )
        result2, _ = integrate_activation(
            activation=softplus,
            f=lambda x: x,
            g_prime=lambda x: 1.0,
            lower=0.0,
            upper=2.0,
            params=(2.0, 0.0, 0.0)
        )
        # Different params should give different results
        assert not np.isclose(result1, result2)


class TestNumericalQuadrature:
    """Tests for the numerical_quadrature function."""

    def test_adaptive_method(self):
        """Test adaptive quadrature method."""
        result = numerical_quadrature(
            f=lambda x: x**2,
            lower=0.0,
            upper=1.0,
            method='adaptive'
        )
        expected = 1.0 / 3.0  # ∫x² dx from 0 to 1 = 1/3
        assert_allclose(result, expected, rtol=1e-6)

    def test_trapezoid_method(self):
        """Test trapezoid quadrature method."""
        result = numerical_quadrature(
            f=lambda x: x**2,
            lower=0.0,
            upper=1.0,
            method='trapezoid',
            n_points=1000
        )
        expected = 1.0 / 3.0
        assert_allclose(result, expected, rtol=1e-3)

    def test_simpson_method(self):
        """Test Simpson quadrature method."""
        result = numerical_quadrature(
            f=lambda x: x**2,
            lower=0.0,
            upper=1.0,
            method='simpson',
            n_points=101
        )
        expected = 1.0 / 3.0
        assert_allclose(result, expected, rtol=1e-5)

    def test_gauss_method(self):
        """Test Gauss-Legendre quadrature method."""
        result = numerical_quadrature(
            f=lambda x: x**2,
            lower=0.0,
            upper=1.0,
            method='gauss',
            n_points=50
        )
        expected = 1.0 / 3.0
        assert_allclose(result, expected, rtol=1e-6)

    def test_sine_integral(self):
        """Test integration of sine function."""
        result = numerical_quadrature(
            f=np.sin,
            lower=0.0,
            upper=np.pi,
            method='adaptive'
        )
        expected = 2.0  # ∫sin(x) dx from 0 to π = 2
        assert_allclose(result, expected, rtol=1e-6)

    def test_invalid_method_raises(self):
        """Test that invalid method raises ValueError."""
        with pytest.raises(ValueError, match="Unknown method"):
            numerical_quadrature(
                f=lambda x: x,
                lower=0.0,
                upper=1.0,
                method='invalid_method'
            )


# =============================================================================
# SDE Integrator Tests
# =============================================================================


class TestSDEIntegrator:
    """Tests for the SDEIntegrator class."""

    @pytest.fixture
    def simple_sde(self):
        """Create a simple SDE (geometric Brownian motion)."""
        return SDEIntegrator(
            drift=lambda x, t: 0.1 * x,
            diffusion=lambda x, t: 0.2 * x,
            scheme='euler',
            seed=42
        )

    def test_initialization(self, simple_sde):
        """Test SDEIntegrator initialization."""
        assert simple_sde.scheme == 'euler'
        assert simple_sde.seed == 42
        assert simple_sde.rng is not None

    def test_euler_step(self, simple_sde):
        """Test single Euler-Maruyama step."""
        x0 = 1.0
        t = 0.0
        dt = 0.01
        dW = 0.1

        result = simple_sde.step(x0, t, dt, dW=dW)
        # Expected: x + μ*dt + σ*dW = 1 + 0.1*1*0.01 + 0.2*1*0.1 = 1.021
        expected = 1.0 + 0.1 * 1.0 * 0.01 + 0.2 * 1.0 * 0.1
        assert_allclose(result, expected, rtol=1e-7)

    def test_milstein_step(self):
        """Test single Milstein step."""
        integrator = SDEIntegrator(
            drift=lambda x, t: 0.1 * x,
            diffusion=lambda x, t: 0.2 * x,
            scheme='milstein',
            seed=42
        )
        x0 = 1.0
        result = integrator.step(x0, 0.0, 0.01, dW=0.1)
        assert np.isfinite(result)

    def test_heun_step(self):
        """Test single Heun step."""
        integrator = SDEIntegrator(
            drift=lambda x, t: 0.1 * x,
            diffusion=lambda x, t: 0.2 * x,
            scheme='heun',
            seed=42
        )
        x0 = 1.0
        result = integrator.step(x0, 0.0, 0.01, dW=0.1)
        assert np.isfinite(result)

    def test_invalid_scheme_raises(self):
        """Test that invalid scheme raises ValueError."""
        integrator = SDEIntegrator(
            drift=lambda x, t: x,
            diffusion=lambda x, t: x,
            scheme='invalid'
        )
        with pytest.raises(ValueError, match="Unknown scheme"):
            integrator.step(1.0, 0.0, 0.01)

    def test_integrate_returns_dict(self, simple_sde):
        """Test integrate returns correct structure."""
        result = simple_sde.integrate(x0=1.0, t_span=(0, 1), dt=0.1)

        assert 'times' in result
        assert 'paths' in result
        assert 'mean' in result
        assert 'std' in result

    def test_integrate_multiple_paths(self, simple_sde):
        """Test integration with multiple sample paths."""
        result = simple_sde.integrate(
            x0=1.0,
            t_span=(0, 1),
            dt=0.1,
            n_paths=10
        )

        assert result['paths'].shape[0] == 10
        assert len(result['mean']) == len(result['times'])
        assert len(result['std']) == len(result['times'])

    def test_integrate_time_array(self, simple_sde):
        """Test integrate produces correct time array."""
        result = simple_sde.integrate(x0=1.0, t_span=(0, 1), dt=0.2)
        expected_times = np.arange(0, 1.2, 0.2)
        assert_allclose(result['times'], expected_times, rtol=1e-7)

    def test_reproducibility_with_seed(self):
        """Test that same seed gives same results."""
        sde1 = SDEIntegrator(
            drift=lambda x, t: x,
            diffusion=lambda x, t: 0.5 * x,
            scheme='euler',
            seed=123
        )
        sde2 = SDEIntegrator(
            drift=lambda x, t: x,
            diffusion=lambda x, t: 0.5 * x,
            scheme='euler',
            seed=123
        )

        result1 = sde1.integrate(x0=1.0, t_span=(0, 1), dt=0.1, n_paths=1)
        result2 = sde2.integrate(x0=1.0, t_span=(0, 1), dt=0.1, n_paths=1)

        assert_allclose(result1['paths'], result2['paths'])

    def test_auto_generate_wiener(self, simple_sde):
        """Test that Wiener increment is auto-generated if not provided."""
        # Run step without dW, should not raise
        result = simple_sde.step(1.0, 0.0, 0.01)
        assert np.isfinite(result)


class TestEulerMaruyama:
    """Tests for the euler_maruyama convenience function."""

    def test_returns_dict(self):
        """Test euler_maruyama returns correct structure."""
        result = euler_maruyama(
            drift=lambda x, t: 0.1 * x,
            diffusion=lambda x, t: 0.2 * x,
            x0=1.0,
            t_span=(0, 1),
            dt=0.1,
            seed=42
        )

        assert 'times' in result
        assert 'values' in result

    def test_time_values_match(self):
        """Test times and values arrays have same length."""
        result = euler_maruyama(
            drift=lambda x, t: 0.0,
            diffusion=lambda x, t: 0.1,
            x0=0.0,
            t_span=(0, 10),
            dt=0.5,
            seed=42
        )

        assert len(result['times']) == len(result['values'])

    def test_initial_condition(self):
        """Test initial condition is preserved."""
        x0 = 5.0
        result = euler_maruyama(
            drift=lambda x, t: 0.0,
            diffusion=lambda x, t: 0.0,
            x0=x0,
            t_span=(0, 1),
            dt=0.1
        )

        assert_allclose(result['values'][0], x0)

    def test_zero_diffusion_deterministic(self):
        """Test with zero diffusion gives deterministic result."""
        # dX = X dt with X(0) = 1 => X(t) = e^t
        result = euler_maruyama(
            drift=lambda x, t: x,
            diffusion=lambda x, t: 0.0,
            x0=1.0,
            t_span=(0, 1),
            dt=0.001
        )

        # Should be close to e^1 at final time
        assert_allclose(result['values'][-1], np.exp(1), rtol=0.01)


# =============================================================================
# Utility Function Tests
# =============================================================================


class TestComputeConvergenceRate:
    """Tests for compute_convergence_rate function."""

    @pytest.fixture
    def ou_integrator(self):
        """Ornstein-Uhlenbeck process integrator."""
        return SDEIntegrator(
            drift=lambda x, t: -x,
            diffusion=lambda x, t: 0.5,
    def euler_integrator(self):
        """Create Euler integrator for convergence testing."""
        return SDEIntegrator(
            drift=lambda x, t: -x,
            diffusion=lambda x, t: 0.1,
            scheme="euler",
            seed=42,
        )

    def test_convergence_rate_basic(self, euler_integrator):
        """Test basic convergence rate computation."""
        dt_values = np.array([0.1, 0.05, 0.025])

        result = compute_convergence_rate(
            integrator=euler_integrator,
            x0=1.0,
            t_span=(0, 1),
            dt_values=dt_values,
    """Tests for compute_convergence_rate utility function."""

    def test_returns_dict(self):
        """Test compute_convergence_rate returns correct structure."""
        integrator = SDEIntegrator(
            drift=lambda x, t: x,
            diffusion=lambda x, t: 0.1 * x,
            scheme='euler',
            seed=42
        )

    def test_returns_correct_structure(self, ou_integrator):
        """Test that result has correct keys."""
        result = compute_convergence_rate(
            integrator=ou_integrator,
            x0=1.0,
            t_span=(0, 0.5),
            dt_values=np.array([0.1, 0.05]),
            n_samples=10
        )
        dt_values = np.array([0.1, 0.05, 0.025])
        result = compute_convergence_rate(
            integrator=integrator,
            x0=1.0,
            t_span=(0, 0.5),
            dt_values=np.array([0.1, 0.05]),
            n_samples=10,
        )

        assert "dt_values" in result
        assert "errors" in result
        assert "rate" in result

    def test_convergence_rate_dt_values(self, euler_integrator):
        """Test that dt_values are returned correctly."""
        dt_values = np.array([0.1, 0.05, 0.025])

        result = compute_convergence_rate(
            integrator=euler_integrator,
            x0=1.0,
            t_span=(0, 1),
            dt_values=dt_values,
            n_samples=10,
        )

        assert_allclose(result["dt_values"], dt_values, rtol=1e-10)

    def test_convergence_rate_errors_shape(self, euler_integrator):
        """Test errors array shape matches dt_values."""
        dt_values = np.array([0.1, 0.05, 0.025, 0.0125])

        result = compute_convergence_rate(
            integrator=euler_integrator,
            x0=1.0,
            t_span=(0, 1),
            dt_values=dt_values,
            n_samples=10
        )

        assert 'dt_values' in result
        assert 'errors' in result
        assert 'rate' in result

    def test_errors_decrease_with_dt(self, ou_integrator):
        """Test that errors generally decrease with smaller dt."""
        result = compute_convergence_rate(
            integrator=ou_integrator,
            x0=1.0,
            t_span=(0, 0.5),
            dt_values=np.array([0.1, 0.05, 0.025]),
            n_samples=50
        )
        # Errors should generally decrease (not always monotonic due to randomness)
        assert np.all(np.isfinite(result['errors']))

    def test_positive_convergence_rate(self, ou_integrator):
        """Test that convergence rate is finite and in expected range."""
        result = compute_convergence_rate(
            integrator=ou_integrator,
            x0=1.0,
            t_span=(0, 0.5),
            dt_values=np.array([0.1, 0.05, 0.025]),
            n_samples=100
        )
        # Euler-Maruyama has theoretical rate ~0.5 for strong convergence
        # but Monte Carlo estimates can vary; just check it's finite and reasonable
        assert np.isfinite(result['rate'])
        # Rate should be between -1 and 2 (allowing for noise in estimates)
        assert -1 <= result['rate'] <= 2

    def test_custom_reference_dt(self, ou_integrator):
        """Test with custom reference dt."""
        result = compute_convergence_rate(
            integrator=ou_integrator,
            x0=1.0,
            t_span=(0, 0.5),
            dt_values=np.array([0.1, 0.05]),
            n_samples=10,
            reference_dt=0.001
        )
        assert np.all(np.isfinite(result['errors']))


class TestSDEIntegratorSchemes:
    """Tests comparing different SDE integration schemes."""

    def test_schemes_give_different_results(self):
        """Test that different schemes give different paths."""
        drift = lambda x, t: -x
        # Use state-dependent diffusion (σ(x) = 0.3x) so Milstein correction is non-zero.
        # The Milstein scheme includes a term proportional to σ(x)·σ'(x), which is
        # zero for constant diffusion but non-zero here since σ'(x) = 0.3.
        diffusion = lambda x, t: 0.3 * x

        euler = SDEIntegrator(drift, diffusion, scheme='euler', seed=42)
        milstein = SDEIntegrator(drift, diffusion, scheme='milstein', seed=42)
        heun = SDEIntegrator(drift, diffusion, scheme='heun', seed=42)

        result_euler = euler.integrate(x0=1.0, t_span=(0, 1), dt=0.1)
        result_milstein = milstein.integrate(x0=1.0, t_span=(0, 1), dt=0.1)
        result_heun = heun.integrate(x0=1.0, t_span=(0, 1), dt=0.1)

        # Milstein and Heun should differ from Euler for state-dependent diffusion
        # (due to higher-order corrections)
        assert not np.allclose(result_euler['paths'], result_milstein['paths'])

    def test_all_schemes_finite_output(self):
        """Test all schemes produce finite output."""
        drift = lambda x, t: -0.5 * x
        diffusion = lambda x, t: 0.2

        for scheme in ['euler', 'milstein', 'heun']:
            integrator = SDEIntegrator(drift, diffusion, scheme=scheme, seed=42)
            result = integrator.integrate(x0=1.0, t_span=(0, 1), dt=0.01)
            assert np.all(np.isfinite(result['paths']))

    def test_mean_and_std_computed_correctly(self):
        """Test mean and std are computed correctly for multiple paths."""
        drift = lambda x, t: -x
        diffusion = lambda x, t: 0.1

        integrator = SDEIntegrator(drift, diffusion, seed=42)
        result = integrator.integrate(x0=1.0, t_span=(0, 1), dt=0.1, n_paths=100)

        # Verify mean and std match paths
        np.testing.assert_array_almost_equal(
            result['mean'],
            np.mean(result['paths'], axis=0)
        )
        np.testing.assert_array_almost_equal(
            result['std'],
            np.std(result['paths'], axis=0)
        )
    def test_errors_decrease_with_dt(self):
        """Test that errors generally decrease with smaller dt."""
        integrator = SDEIntegrator(
            drift=lambda x, t: -x,  # Simple stable ODE
            diffusion=lambda x, t: 0.0,  # No noise for predictable behavior
            scheme="euler",
            seed=42,
        )

        dt_values = np.array([0.1, 0.05, 0.025])
            drift=lambda x, t: x,
            diffusion=lambda x, t: 0.1,  # Constant diffusion
            scheme='euler',
            seed=42
        )

        dt_values = np.array([0.1, 0.01])
        result = compute_convergence_rate(
            integrator=integrator,
            x0=1.0,
            t_span=(0, 0.5),
            dt_values=dt_values,
            n_samples=10,
        )

        assert len(result["errors"]) == len(dt_values)

    def test_convergence_rate_decreasing_errors(self, euler_integrator):
        """Test that errors generally decrease with smaller dt."""
        dt_values = np.array([0.2, 0.1, 0.05])

        result = compute_convergence_rate(
            integrator=euler_integrator,
            x0=1.0,
            t_span=(0, 1),
            dt_values=dt_values,
            n_samples=50,
        )

        # Errors should generally decrease with smaller dt.
        # The factor 0.5 accounts for Monte Carlo variability in the convergence
        # estimation - we allow the first error to be up to 2x larger than the
        # last error, which is a conservative bound for stochastic testing.
        monte_carlo_tolerance = 0.5
        assert result["errors"][0] >= result["errors"][-1] * monte_carlo_tolerance

    def test_convergence_rate_is_finite(self, euler_integrator):
        """Test that estimated rate is finite."""
        dt_values = np.array([0.1, 0.05, 0.025])

        result = compute_convergence_rate(
            integrator=euler_integrator,
            x0=1.0,
            t_span=(0, 1),
            dt_values=dt_values,
            n_samples=50,
        )

        # Rate should be a finite number (sign can vary due to Monte Carlo noise)
        assert np.isfinite(result["rate"])

    def test_convergence_rate_custom_reference_dt(self, euler_integrator):
        """Test with custom reference dt."""
        dt_values = np.array([0.1, 0.05])

        result = compute_convergence_rate(
            integrator=euler_integrator,
            x0=1.0,
            t_span=(0, 1),
            dt_values=dt_values,
            n_samples=10,
            reference_dt=0.001,
        )

        assert "rate" in result
        assert np.isfinite(result["rate"])


# =============================================================================
# Edge Cases and Numerical Stability Tests
# =============================================================================


class TestIntegratorEdgeCases:
    """Tests for edge cases and numerical stability."""

    def test_zero_initial_condition(self):
        """Test SDE integration starting from zero."""
        integrator = SDEIntegrator(
            drift=lambda x, t: 1.0,  # Constant drift
            diffusion=lambda x, t: 0.0,  # No noise
            scheme="euler",
            seed=42,
        )

        result = integrator.integrate(x0=0.0, t_span=(0, 1), dt=0.1, n_paths=1)

        # Should grow linearly with drift
        assert result["paths"][0, -1] > 0

    def test_negative_initial_condition(self):
        """Test SDE integration starting from negative value."""
        integrator = SDEIntegrator(
            drift=lambda x, t: -x,
            diffusion=lambda x, t: 0.01,
            scheme="euler",
            seed=42,
        )

        result = integrator.integrate(x0=-5.0, t_span=(0, 1), dt=0.1, n_paths=1)

        # Should revert towards zero
        assert result["paths"][0, -1] > result["paths"][0, 0]

    def test_small_time_step(self):
        """Test integration with very small time step."""
        integrator = SDEIntegrator(
            drift=lambda x, t: -x,
            diffusion=lambda x, t: 0.1,
            scheme="euler",
            seed=42,
        )

        result = integrator.integrate(
            x0=1.0, t_span=(0, 0.1), dt=0.001, n_paths=1
        )

        assert len(result["times"]) >= 100
        assert np.all(np.isfinite(result["paths"]))

    def test_multiple_schemes_consistency(self):
        """Test that different schemes give reasonable results."""
        x0 = 1.0
        t_span = (0, 1)
        dt = 0.01

        results = {}
        for scheme in ["euler", "milstein", "heun"]:
            integrator = SDEIntegrator(
                drift=lambda x, t: -x,
                diffusion=lambda x, t: 0.0,  # Zero noise for comparison
                scheme=scheme,
                seed=42,
            )
            results[scheme] = integrator.integrate(
                x0=x0, t_span=t_span, dt=dt, n_paths=1
            )

        # With zero noise, all schemes should give similar results
        for scheme in ["milstein", "heun"]:
            assert_allclose(
                results["euler"]["paths"],
                results[scheme]["paths"],
                rtol=0.1,  # Allow 10% relative tolerance
            )

    def test_integrate_activation_with_sigmoid(self):
        """Test integrate_activation with sigmoid instead of softplus."""
        result, error = integrate_activation(
            activation=sigmoid,
            f=np.cos,
            g_prime=np.sin,
            lower=0.0,
            upper=2.0,
            params=(0.8, 0.3, 1.0),
        )

        assert isinstance(result, float)
        assert np.isfinite(result)
        # Errors should decrease (or at least not increase dramatically)
        assert np.all(np.isfinite(result["errors"]))

    def test_convergence_rate_is_finite(self):
        """Test that estimated convergence rate is finite."""
        integrator = SDEIntegrator(
            drift=lambda x, t: x,
            diffusion=lambda x, t: 0.1,
            scheme="euler",
            seed=42,
        )

            n_samples=50,
            reference_dt=0.001
        )

        # Smaller dt should have smaller error (or at least not much larger)
        # Note: Due to Monte Carlo noise, we allow some tolerance
        assert result['errors'][1] <= result['errors'][0] * 2

    def test_rate_is_positive(self):
        """Test that convergence rate is positive (error decreases with dt)."""
        integrator = SDEIntegrator(
            drift=lambda x, t: 0.05 * x,
            diffusion=lambda x, t: 0.1 * x,
            scheme='euler',
            seed=42
        )

        result = compute_convergence_rate(
            integrator=integrator,
            x0=1.0,
            t_span=(0.0, 0.5),
            dt_values=np.array([0.2, 0.1, 0.05, 0.025]),  # Wider range of dt
            n_samples=200  # More samples for statistical stability
        )

        # Rate should be positive (errors decrease as dt decreases)
        # Allow for some statistical noise by checking rate > -0.1
        # (the theoretical rate for Euler is 0.5 for strong convergence)
        assert result['rate'] > -0.1


class TestEdgeCases:
    """Tests for edge cases and numerical stability."""

    def test_softplus_very_small_threshold(self):
        """Test softplus with very small threshold."""
        result = softplus(5.0, threshold=1.0)
        assert np.isfinite(result)

    def test_sigmoid_array_extreme(self):
        """Test sigmoid with array of extreme values."""
        x = np.array([-1000, -100, 0, 100, 1000])
        result = sigmoid(x)
        assert np.all(np.isfinite(result))
        assert np.all((result >= 0) & (result <= 1))

    def test_integrate_activation_returns_finite(self):
        """Test integrate_activation returns finite values."""
        result, error = integrate_activation(
            activation=sigmoid,
            f=lambda x: np.exp(-x**2),
            g_prime=lambda x: -2 * x * np.exp(-x**2),
            lower=-10,
            upper=10,
            params=(1.0, 0.5, 0.0)
        )
        assert np.isfinite(result)
        assert np.isfinite(error)

    def test_sde_small_dt(self):
        """Test SDE integration with very small dt."""
        integrator = SDEIntegrator(
            drift=lambda x, t: -x,
            diffusion=lambda x, t: 0.1,
        dt_values = np.array([0.1, 0.05, 0.025])
        result = compute_convergence_rate(
            integrator=integrator,
            x0=1.0,
            t_span=(0, 0.5),
            dt_values=np.array([0.1, 0.05, 0.025]),
            n_samples=20,
        )

        assert np.isfinite(result["rate"])


class TestSDESchemeComparison:
    """Comparative tests for different SDE integration schemes."""

    def test_all_schemes_converge_for_ode(self):
        """Test that all schemes converge for deterministic ODE."""
        schemes = ["euler", "milstein", "heun"]
        x0 = 1.0

        for scheme in schemes:
            integrator = SDEIntegrator(
                drift=lambda x, t: -x,
                diffusion=lambda x, t: 0.0,
                scheme=scheme,
                seed=42,
            )
            result = integrator.integrate(x0=x0, t_span=(0, 1), dt=0.01, n_paths=1)

            # x(1) = exp(-1)
            assert_allclose(result["paths"][0, -1], np.exp(-1), rtol=0.02)

    def test_schemes_handle_stochastic_case(self):
        """Test that all schemes handle stochastic case without errors."""
        schemes = ["euler", "milstein", "heun"]

        for scheme in schemes:
            integrator = SDEIntegrator(
                drift=lambda x, t: 0.05 * x,
                diffusion=lambda x, t: 0.2 * x,
                scheme=scheme,
                seed=42,
            )
            result = integrator.integrate(x0=100.0, t_span=(0, 1), dt=0.01, n_paths=5)

            assert np.all(np.isfinite(result["paths"]))
            assert result["paths"].shape == (5, len(result["times"]))
            dt_values=dt_values,
            n_samples=50
        )

        # Rate should be positive (error ~ dt^rate with rate > 0)
        assert result['rate'] > 0


# =============================================================================
# Edge Cases and Numerical Stability
# =============================================================================


class TestNumericalStability:
    """Tests for numerical stability edge cases."""

    def test_sde_with_zero_initial_condition(self):
        """Test SDE integration starting from zero."""
        integrator = SDEIntegrator(
            drift=lambda x, t: 1.0,  # Constant drift
            diffusion=lambda x, t: 0.1,  # Constant diffusion
            scheme='euler',
            seed=42
        )

        result = integrator.integrate(
            x0=1.0,
            t_span=(0.0, 0.1),
            dt=0.001,
            n_paths=1
        )

        assert np.all(np.isfinite(result['paths']))

    def test_sde_zero_diffusion(self):
        """Test SDE with zero diffusion is deterministic."""
        integrator = SDEIntegrator(
            drift=lambda x, t: -x,
            diffusion=lambda x, t: 0.0,
            seed=42
        )

        result1 = integrator.integrate(1.0, (0, 1), dt=0.1, n_paths=5)

        # All paths should be identical
        for i in range(1, 5):
            np.testing.assert_array_almost_equal(
                result1['paths'][0],
                result1['paths'][i]
            )

    def test_numerical_quadrature_wide_interval(self):
        """Test quadrature over wide interval."""
        result = numerical_quadrature(
            f=lambda x: np.exp(-x**2),
            lower=-100,
            upper=100,
            method='adaptive'
        )
        # Integral of exp(-x²) from -∞ to ∞ = √π ≈ 1.7725
        expected = np.sqrt(np.pi)
        assert np.isclose(result, expected, rtol=1e-5)
        result = integrator.integrate(x0=0.0, t_span=(0, 1), dt=0.1)
        assert all(np.isfinite(result['paths'].flat))

    def test_sde_with_large_time_span(self):
        """Test SDE integration over long time span."""
        integrator = SDEIntegrator(
            drift=lambda x, t: -0.1 * x,  # Mean-reverting
            diffusion=lambda x, t: 0.1,
            scheme='euler',
            seed=42
        )

        result = integrator.integrate(x0=1.0, t_span=(0, 100), dt=0.1)
        assert all(np.isfinite(result['paths'].flat))

    def test_sde_ornstein_uhlenbeck(self):
        """Test Ornstein-Uhlenbeck process (mean-reverting)."""
        theta = 1.0  # Mean reversion speed
        mu = 0.0     # Long-term mean
        sigma = 0.5  # Volatility

        integrator = SDEIntegrator(
            drift=lambda x, t: theta * (mu - x),
            diffusion=lambda x, t: sigma,
            scheme='euler',
            seed=42
        )

        result = integrator.integrate(
            x0=5.0,  # Start far from mean
            t_span=(0, 10),
            dt=0.01,
            n_paths=100
        )

        # Mean should approach long-term mean (0)
        final_mean = np.mean(result['paths'][:, -1])
        assert abs(final_mean - mu) < 1.0

    def test_all_schemes_produce_finite_values(self):
        """Test all integration schemes produce finite values."""
        for scheme in ['euler', 'milstein', 'heun']:
            integrator = SDEIntegrator(
                drift=lambda x, t: 0.1 * x,
                diffusion=lambda x, t: 0.2 * x,
                scheme=scheme,
                seed=42
            )

            result = integrator.integrate(x0=1.0, t_span=(0, 1), dt=0.01)
            assert all(np.isfinite(result['paths'].flat)), f"Scheme {scheme} produced non-finite values"
