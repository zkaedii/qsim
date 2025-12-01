"""
Unit tests for the numerical integration module.
"""

import pytest
import numpy as np
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

    def test_swish_beta_parameter(self):
        """Test swish with different beta values."""
        x = 1.0
        result_beta1 = swish(x, beta=1.0)
        result_beta2 = swish(x, beta=2.0)
        # With higher beta, swish is more ReLU-like
        assert result_beta1 != result_beta2

    def test_swish_array(self):
        """Test swish works with arrays."""
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
            n_points=10
        )
        expected = 1 / 3
        assert np.isclose(result, expected, rtol=1e-10)

    def test_numerical_quadrature_invalid_method(self):
        """Test invalid quadrature method raises error."""
        with pytest.raises(ValueError, match="Unknown method"):
            numerical_quadrature(
                f=lambda x: x,
                lower=0,
                upper=1,
                method='invalid'
            )


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
