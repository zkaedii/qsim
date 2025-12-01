"""
Unit tests for the integrators module.

This module tests:
- Activation functions (softplus, sigmoid, swish, gelu)
- Numerical quadrature (integrate_activation, numerical_quadrature)
- SDE integrators (SDEIntegrator, euler_maruyama)
- Utility functions (compute_convergence_rate)
"""

import pytest
import numpy as np
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
        result_beta1 = swish(x, beta=1.0)
        result_beta2 = swish(x, beta=2.0)
        # Different beta should give different results
        assert not np.isclose(result_beta1, result_beta2)

    def test_array_input(self):
        """Test swish works with numpy arrays."""
        x = np.array([-2, 0, 2])
        result = swish(x)
        assert result.shape == x.shape


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


# =============================================================================
# Numerical Quadrature Tests
# =============================================================================


class TestIntegrateActivation:
    """Tests for the integrate_activation function."""

    def test_simple_integration(self):
        """Test integration of simple functions."""
        result, error = integrate_activation(
            activation=softplus,
            f=lambda x: 1.0,
            g_prime=lambda x: 1.0,
            lower=0.0,
            upper=1.0,
            params=(0.0, 0.0, 0.0)  # softplus(0) = ln(2)
        )
        # With constant functions and softplus(0)=ln(2), integral = ln(2) * 1 * 1 * 1 = ln(2)
        expected = np.log(2)
        assert_allclose(result, expected, rtol=1e-4)

    def test_error_estimate_reasonable(self):
        """Test that error estimate is reasonable."""
        result, error = integrate_activation(
            activation=softplus,
            f=np.cos,
            g_prime=lambda x: -np.sin(x),
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
    """Tests for compute_convergence_rate utility function."""

    def test_returns_dict(self):
        """Test compute_convergence_rate returns correct structure."""
        integrator = SDEIntegrator(
            drift=lambda x, t: x,
            diffusion=lambda x, t: 0.1 * x,
            scheme='euler',
            seed=42
        )

        dt_values = np.array([0.1, 0.05, 0.025])
        result = compute_convergence_rate(
            integrator=integrator,
            x0=1.0,
            t_span=(0, 0.5),
            dt_values=dt_values,
            n_samples=10
        )

        assert 'dt_values' in result
        assert 'errors' in result
        assert 'rate' in result

    def test_errors_decrease_with_dt(self):
        """Test that errors generally decrease with smaller dt."""
        integrator = SDEIntegrator(
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

        dt_values = np.array([0.1, 0.05, 0.025])
        result = compute_convergence_rate(
            integrator=integrator,
            x0=1.0,
            t_span=(0, 0.5),
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
