"""
Numerical Integration Utilities

This module provides numerical integration methods and activation functions
used in the Multi-Component Stochastic Oscillator.

Key Components
--------------
1. **Activation Functions**: Softplus, sigmoid, and variants
2. **Numerical Quadrature**: Adaptive integration with error control
3. **SDE Integrators**: Euler-Maruyama, Milstein, Heun schemes

Mathematical Background
-----------------------
The integral term in MCSO takes the form:

    I(t) = ∫₀ᵗ σ(a(x-x₀)² + b) · f(x) · g'(x) dx

where σ(·) is an activation function (typically softplus).

For SDEs, we discretize:
    dX = μ(X,t)dt + σ(X,t)dW

using various schemes with different convergence orders.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.integrate import quad, quad_vec, solve_ivp, IntegrationWarning
from scipy.special import expit
from dataclasses import dataclass
from typing import Callable, Optional, Tuple, Dict, Any
import warnings


# =============================================================================
# Activation Functions
# =============================================================================

def softplus(x: NDArray | float, threshold: float = 20.0) -> NDArray | float:
    """
    Numerically stable softplus activation.

    σ(x) = log(1 + exp(x))

    For x > threshold, returns x directly to avoid overflow.

    Parameters
    ----------
    x : array_like or float
        Input value(s).
    threshold : float
        Linear approximation threshold.

    Returns
    -------
    array_like or float
        Activated value(s).

    Examples
    --------
    >>> softplus(0.0)
    0.6931471805599453
    >>> softplus(100.0)  # Returns ~100 to avoid overflow
    100.0
    """
    x = np.asarray(x)
    x_clipped = np.clip(x, -500, 500)
    return np.where(x_clipped > threshold, x_clipped, np.log1p(np.exp(x_clipped)))


def sigmoid(x: NDArray | float) -> NDArray | float:
    """
    Numerically stable sigmoid activation.

    σ(x) = 1 / (1 + exp(-x))

    Uses scipy's expit for numerical stability.

    Parameters
    ----------
    x : array_like or float
        Input value(s).

    Returns
    -------
    array_like or float
        Value(s) in (0, 1).

    Examples
    --------
    >>> sigmoid(0.0)
    0.5
    >>> sigmoid(10.0)
    0.9999546021312976
    """
    return expit(np.clip(np.asarray(x), -500, 500))


def swish(x: NDArray | float, beta: float = 1.0) -> NDArray | float:
    """
    Swish activation function.

    swish(x) = x · sigmoid(βx)

    A smooth approximation to ReLU with learnable parameter.

    Parameters
    ----------
    x : array_like or float
        Input value(s).
    beta : float
        Scaling parameter.

    Returns
    -------
    array_like or float
        Activated value(s).
    """
    return x * sigmoid(beta * x)


def gelu(x: NDArray | float) -> NDArray | float:
    """
    Gaussian Error Linear Unit activation.

    GELU(x) = x · Φ(x)

    where Φ is the CDF of the standard normal distribution.

    Parameters
    ----------
    x : array_like or float
        Input value(s).

    Returns
    -------
    array_like or float
        Activated value(s).
    """
    x = np.asarray(x)
    return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))


# =============================================================================
# Numerical Quadrature
# =============================================================================

def integrate_activation(
    activation: Callable[[float], float],
    f: Callable[[float], float],
    g_prime: Callable[[float], float],
    lower: float,
    upper: float,
    params: Tuple[float, ...] = (0.8, 0.3, 1.0),
    limit: int = 50,
    epsabs: float = 1e-8,
    epsrel: float = 1e-6
) -> Tuple[float, float]:
    """
    Integrate activation-weighted product.

    Computes:
        I = ∫ₐᵇ σ(params(x)) · f(x) · g'(x) dx

    where σ is the activation function.

    Parameters
    ----------
    activation : callable
        Activation function σ(·).
    f : callable
        First function in product.
    g_prime : callable
        Derivative of second function.
    lower : float
        Lower integration limit.
    upper : float
        Upper integration limit.
    params : tuple
        Parameters (a, b, x0) for quadratic argument.
    limit : int
        Maximum subdivisions for adaptive integration.
    epsabs : float
        Absolute error tolerance.
    epsrel : float
        Relative error tolerance.

    Returns
    -------
    tuple
        (integral_value, error_estimate)

    Examples
    --------
    >>> result, error = integrate_activation(
    ...     softplus, np.cos, lambda x: -np.sin(x),
    ...     0, 5, params=(0.8, 0.3, 1.0)
    ... )
    """
    a, b, x0 = params

    def integrand(x: float) -> float:
        arg = a * (x - x0) ** 2 + b
        return activation(arg) * f(x) * g_prime(x)

    try:
        with warnings.catch_warnings():
            warnings.filterwarnings('error', category=IntegrationWarning)
            result, error = quad(
                integrand, lower, upper,
                limit=limit, epsabs=epsabs, epsrel=epsrel
            )
        return result, error
    except IntegrationWarning as w:
        warnings.warn(
            f"Integration did not converge in [{lower}, {upper}]: {w}. Returning 0.",
            RuntimeWarning,
            stacklevel=2
        )
        return 0.0, float('inf')
    except ValueError as e:
        warnings.warn(
            f"Invalid value in integration [{lower}, {upper}]: {e}. Returning 0.",
            RuntimeWarning,
            stacklevel=2
        )
        return 0.0, float('inf')
    except (FloatingPointError, OverflowError) as e:
        warnings.warn(
            f"Numerical error in integration [{lower}, {upper}]: {e}. Returning 0.",
            RuntimeWarning,
            stacklevel=2
        )
        return 0.0, float('inf')


def numerical_quadrature(
    f: Callable[[float], float],
    lower: float,
    upper: float,
    method: str = 'adaptive',
    n_points: int = 100
) -> float:
    """
    General numerical quadrature with method selection.

    Parameters
    ----------
    f : callable
        Function to integrate.
    lower : float
        Lower limit.
    upper : float
        Upper limit.
    method : str
        'adaptive' (scipy quad), 'trapezoid', 'simpson', 'gauss'.
    n_points : int
        Number of points for fixed-point methods.

    Returns
    -------
    float
        Integral approximation.
    """
    if method == 'adaptive':
        result, _ = quad(f, lower, upper)
        return result

    elif method == 'trapezoid':
        x = np.linspace(lower, upper, n_points)
        y = np.array([f(xi) for xi in x])
        return np.trapz(y, x)

    elif method == 'simpson':
        if n_points % 2 == 0:
            n_points += 1  # Simpson needs odd number
        x = np.linspace(lower, upper, n_points)
        y = np.array([f(xi) for xi in x])
        from scipy.integrate import simpson
        return simpson(y, x=x)

    elif method == 'gauss':
        from numpy.polynomial.legendre import leggauss
        nodes, weights = leggauss(n_points)
        # Transform from [-1, 1] to [lower, upper]
        mid = (upper + lower) / 2
        half_width = (upper - lower) / 2
        x = mid + half_width * nodes
        y = np.array([f(xi) for xi in x])
        return half_width * np.sum(weights * y)

    else:
        raise ValueError(f"Unknown method: {method}")


# =============================================================================
# SDE Integrators
# =============================================================================

@dataclass
class SDEIntegrator:
    """
    Stochastic Differential Equation integrator.

    Solves SDEs of the form:
        dX = μ(X, t) dt + σ(X, t) dW

    Supports multiple discretization schemes.

    Parameters
    ----------
    drift : callable
        Drift function μ(X, t).
    diffusion : callable
        Diffusion function σ(X, t).
    scheme : str
        'euler', 'milstein', 'heun' (default: 'euler').
    seed : int, optional
        Random seed.
    """

    drift: Callable[[float, float], float]
    diffusion: Callable[[float, float], float]
    scheme: str = 'euler'
    seed: Optional[int] = None

    def __post_init__(self):
        self.rng = np.random.default_rng(self.seed)

    def step(
        self,
        x: float,
        t: float,
        dt: float,
        dW: Optional[float] = None
    ) -> float:
        """
        Perform one integration step.

        Parameters
        ----------
        x : float
            Current state.
        t : float
            Current time.
        dt : float
            Time step.
        dW : float, optional
            Wiener increment. If None, sampled from N(0, √dt).

        Returns
        -------
        float
            State at t + dt.
        """
        if dW is None:
            dW = self.rng.normal(0, np.sqrt(dt))

        mu = self.drift(x, t)
        sigma = self.diffusion(x, t)

        if self.scheme == 'euler':
            # Euler-Maruyama: O(dt) strong, O(√dt) weak
            return x + mu * dt + sigma * dW

        elif self.scheme == 'milstein':
            # Milstein: O(dt) strong convergence
            # Requires diffusion derivative
            eps = 1e-6
            sigma_prime = (self.diffusion(x + eps, t) - self.diffusion(x - eps, t)) / (2 * eps)
            return x + mu * dt + sigma * dW + 0.5 * sigma * sigma_prime * (dW**2 - dt)

        elif self.scheme == 'heun':
            # Heun (improved Euler): Predictor-corrector
            x_pred = x + mu * dt + sigma * dW
            mu_pred = self.drift(x_pred, t + dt)
            sigma_pred = self.diffusion(x_pred, t + dt)
            return x + 0.5 * (mu + mu_pred) * dt + 0.5 * (sigma + sigma_pred) * dW

        else:
            raise ValueError(f"Unknown scheme: {self.scheme}")

    def integrate(
        self,
        x0: float,
        t_span: Tuple[float, float],
        dt: float = 0.01,
        n_paths: int = 1
    ) -> Dict[str, NDArray]:
        """
        Integrate SDE over time span.

        Parameters
        ----------
        x0 : float
            Initial condition.
        t_span : tuple
            (t_start, t_end).
        dt : float
            Time step.
        n_paths : int
            Number of sample paths.

        Returns
        -------
        dict
            'times': Time array
            'paths': Array of shape (n_paths, n_times)
            'mean': Mean trajectory
            'std': Standard deviation trajectory
        """
        t_start, t_end = t_span
        times = np.arange(t_start, t_end + dt, dt)
        n_times = len(times)

        paths = np.zeros((n_paths, n_times))

        for p in range(n_paths):
            paths[p, 0] = x0
            for i in range(1, n_times):
                paths[p, i] = self.step(paths[p, i-1], times[i-1], dt)

        return {
            'times': times,
            'paths': paths,
            'mean': np.mean(paths, axis=0),
            'std': np.std(paths, axis=0)
        }


def euler_maruyama(
    drift: Callable[[float, float], float],
    diffusion: Callable[[float, float], float],
    x0: float,
    t_span: Tuple[float, float],
    dt: float = 0.01,
    seed: Optional[int] = None
) -> Dict[str, NDArray]:
    """
    Euler-Maruyama integration for SDEs.

    Convenience function for single-path integration.

    Parameters
    ----------
    drift : callable
        μ(X, t) function.
    diffusion : callable
        σ(X, t) function.
    x0 : float
        Initial condition.
    t_span : tuple
        (t_start, t_end).
    dt : float
        Time step.
    seed : int, optional
        Random seed.

    Returns
    -------
    dict
        'times' and 'values' arrays.

    Examples
    --------
    >>> # Geometric Brownian Motion
    >>> result = euler_maruyama(
    ...     drift=lambda x, t: 0.05 * x,
    ...     diffusion=lambda x, t: 0.2 * x,
    ...     x0=100.0,
    ...     t_span=(0, 1),
    ...     dt=0.001
    ... )
    """
    integrator = SDEIntegrator(drift, diffusion, scheme='euler', seed=seed)
    result = integrator.integrate(x0, t_span, dt, n_paths=1)

    return {
        'times': result['times'],
        'values': result['paths'][0]
    }


# =============================================================================
# Utility Functions
# =============================================================================

def compute_convergence_rate(
    integrator: SDEIntegrator,
    x0: float,
    t_span: Tuple[float, float],
    dt_values: NDArray,
    n_samples: int = 1000,
    reference_dt: Optional[float] = None
) -> Dict[str, Any]:
    """
    Estimate convergence rate of SDE integrator.

    Computes strong error ||X_dt - X_ref|| for various dt values.

    Parameters
    ----------
    integrator : SDEIntegrator
        Integrator to test.
    x0 : float
        Initial condition.
    t_span : tuple
        Time interval.
    dt_values : array
        Time steps to test.
    n_samples : int
        Monte Carlo samples.
    reference_dt : float, optional
        Reference time step (default: min(dt_values) / 10).

    Returns
    -------
    dict
        'dt_values': Tested dt values
        'errors': Strong errors
        'rate': Estimated convergence rate (slope in log-log)
    """
    if reference_dt is None:
        reference_dt = min(dt_values) / 10

    t_end = t_span[1]
    errors = []

    for dt in dt_values:
        sample_errors = []

        for _ in range(n_samples):
            # Same seed for paired comparison
            seed = np.random.randint(0, 2**31)

            integrator.rng = np.random.default_rng(seed)
            result_dt = integrator.integrate(x0, t_span, dt, n_paths=1)

            integrator.rng = np.random.default_rng(seed)
            result_ref = integrator.integrate(x0, t_span, reference_dt, n_paths=1)

            # Interpolate to compare at same time point
            x_dt = result_dt['paths'][0, -1]
            x_ref = result_ref['paths'][0, -1]

            sample_errors.append(np.abs(x_dt - x_ref))

        errors.append(np.mean(sample_errors))

    # Estimate rate from log-log slope
    log_dt = np.log(dt_values)
    log_err = np.log(errors)
    rate = np.polyfit(log_dt, log_err, 1)[0]

    return {
        'dt_values': dt_values,
        'errors': np.array(errors),
        'rate': rate
    }
