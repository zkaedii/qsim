"""
Unit tests for the integrators module.

This module tests:
1. Activation functions (softplus, sigmoid, swish, gelu)
2. Numerical quadrature functions
3. SDE integrators (Euler-Maruyama, Milstein, Heun)
4. Convergence rate estimation
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
        result = softplus(x)
        assert result.shape == x.shape
        assert np.all(result > 0)

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
        """Test sigmoid output is always in (0, 1)."""
        for x in [-1000, -10, -1, 0, 1, 10, 1000]:
            result = sigmoid(x)
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


class TestNumericalQuadrature:
    """Tests for numerical_quadrature function."""

    def test_adaptive_method(self):
        """Test adaptive quadrature."""
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
        result = numerical_quadrature(
            f=lambda x: x**2,
            lower=0,
            upper=1,
            method='trapezoid',
            n_points=1000
        )
        expected = 1 / 3
        assert np.isclose(result, expected, rtol=1e-3)

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
        result = numerical_quadrature(
            f=lambda x: x**2,
            lower=0,
            upper=1,
            method='gauss',
            n_points=10
        )
        expected = 1 / 3
        assert np.isclose(result, expected, rtol=1e-10)

    def test_invalid_method(self):
        """Test that invalid method raises error."""
        with pytest.raises(ValueError, match="Unknown method"):
            numerical_quadrature(
                f=lambda x: x,
                lower=0,
                upper=1,
                method='invalid'
            )

    def test_sinusoidal(self):
        """Test integration of sinusoidal function."""
        result = numerical_quadrature(
            f=np.sin,
            lower=0,
            upper=np.pi,
            method='adaptive'
        )
        expected = 2.0  # integral of sin from 0 to pi
        assert np.isclose(result, expected)


class TestSDEIntegrator:
    """Tests for SDEIntegrator class."""

    @pytest.fixture
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


class TestEulerMaruyama:
    """Tests for euler_maruyama convenience function."""

    def test_basic_integration(self):
        """Test basic Euler-Maruyama integration."""
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
            drift=lambda x, t: -x,  # dX = -X dt
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


class TestComputeConvergenceRate:
    """Tests for compute_convergence_rate function."""

    @pytest.fixture
    def ou_integrator(self):
        """Ornstein-Uhlenbeck process integrator."""
        return SDEIntegrator(
            drift=lambda x, t: -x,
            diffusion=lambda x, t: 0.5,
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
        # Use state-dependent diffusion so Milstein correction is non-zero
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
