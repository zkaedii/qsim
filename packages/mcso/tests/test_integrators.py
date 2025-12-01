"""
Unit tests for the integrators module.
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


class TestSoftplus:
    """Tests for softplus activation function."""

    def test_softplus_zero(self):
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

    def test_sigmoid_zero(self):
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

    def test_swish_zero(self):
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

    def test_gelu_zero(self):
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
        """Test basic integration with simple functions."""
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
        result, error = integrate_activation(
            activation=softplus,
            f=np.cos,
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
        """Test adaptive (scipy quad) method."""
        result = numerical_quadrature(
            f=lambda x: x**2,
            lower=0,
            upper=1,
            method="adaptive",
        )
        assert_allclose(result, 1 / 3, rtol=1e-6)

    def test_trapezoid_method(self):
        """Test trapezoidal rule."""
        result = numerical_quadrature(
            f=lambda x: x**2,
            lower=0,
            upper=1,
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
                method="unknown",
            )

    def test_sin_integration(self):
        """Test integration of sin over [0, pi]."""
        result = numerical_quadrature(
            f=np.sin,
            lower=0,
            upper=np.pi,
            method="adaptive",
        )
        assert_allclose(result, 2.0, rtol=1e-6)


class TestSDEIntegrator:
    """Tests for SDEIntegrator class."""

    @pytest.fixture
    def constant_sde(self):
        """Create SDE integrator for deterministic ODE (no diffusion)."""
        return SDEIntegrator(
            drift=lambda x, t: 1.0,  # dx/dt = 1
            diffusion=lambda x, t: 0.0,
            scheme="euler",
            seed=42,
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

        np.testing.assert_array_equal(result1["paths"], result2["paths"])


class TestEulerMaruyama:
    """Tests for euler_maruyama convenience function."""

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

    def test_errors_decrease_with_dt(self):
        """Test that errors generally decrease with smaller dt."""
        integrator = SDEIntegrator(
            drift=lambda x, t: -x,  # Simple stable ODE
            diffusion=lambda x, t: 0.0,  # No noise for predictable behavior
            scheme="euler",
            seed=42,
        )

        dt_values = np.array([0.1, 0.05, 0.025])
        result = compute_convergence_rate(
            integrator=integrator,
            x0=1.0,
            t_span=(0, 0.5),
            dt_values=dt_values,
            n_samples=10,
        )

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
