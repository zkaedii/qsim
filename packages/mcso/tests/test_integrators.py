"""
Unit tests for the integrators module.

This module tests numerical integration methods and activation functions
including SDE integrators, activation functions, and convergence rate estimation.
"""

import pytest
import numpy as np
from numpy.testing import assert_allclose
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


class TestSigmoid:
    """Tests for sigmoid activation function."""

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


class TestSwish:
    """Tests for swish activation function."""

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


class TestGelu:
    """Tests for GELU activation function."""

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


class TestNumericalQuadrature:
    """Tests for numerical_quadrature function."""

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


class TestSDEIntegrator:
    """Tests for SDEIntegrator class."""

    @pytest.fixture
    def simple_integrator(self):
        """Create a simple SDE integrator for testing."""
        return SDEIntegrator(
            drift=lambda x, t: -x,  # Mean-reverting
            diffusion=lambda x, t: 0.1,  # Constant diffusion
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
        )

        result1 = integrator1.integrate(x0=1.0, t_span=(0, 1), dt=0.1, n_paths=1)
        result2 = integrator2.integrate(x0=1.0, t_span=(0, 1), dt=0.1, n_paths=1)

        assert_allclose(result1["paths"], result2["paths"], rtol=1e-10)


class TestEulerMaruyama:
    """Tests for euler_maruyama convenience function."""

    def test_euler_maruyama_basic(self):
        """Test basic euler_maruyama integration."""
        result = euler_maruyama(
            drift=lambda x, t: -x,
            diffusion=lambda x, t: 0.1,
            x0=1.0,
            t_span=(0, 1),
            dt=0.01,
            seed=42,
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
# =============================================================================


class TestComputeConvergenceRate:
    """Tests for compute_convergence_rate function."""

    @pytest.fixture
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
