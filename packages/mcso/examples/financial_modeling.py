#!/usr/bin/env python3
"""
Financial Modeling Example for MCSO

This example demonstrates how the Multi-Component Stochastic Oscillator
can be applied to financial time series modeling, including:

1. Asset price dynamics with multiple frequency components
2. Volatility clustering (state-dependent noise)
3. Memory effects (momentum/mean-reversion)
4. Jump processes for extreme events
"""

import numpy as np
from mcso import StochasticOscillator, OscillatorConfig
from mcso.noise import JumpDiffusionNoise
from mcso.analysis import (
    compute_statistics,
    autocorrelation,
)


def simulate_asset_returns():
    """
    Simulate asset return dynamics with MCSO.

    The model captures:
    - Multiple cyclical components (daily, weekly, monthly patterns)
    - Volatility clustering via state-dependent noise
    - Momentum effects via memory feedback
    """
    print("\n" + "=" * 70)
    print("  ASSET RETURN SIMULATION")
    print("=" * 70)

    # Configure oscillator for financial returns
    config = OscillatorConfig(
        n_components=3,            # Market cycles at different frequencies
        amplitude_base=0.02,       # Base amplitude (2% scale)
        amplitude_modulation=0.01, # Modulation depth
        frequency_base=0.5,        # Base frequency
        frequency_scaling=2.0,     # Higher harmonics

        # Drift: slight positive drift (risk premium)
        drift_coefficients=(0.0001, 0.005, 0.001),

        # Memory: momentum effect (positive feedback)
        memory_strength=0.3,
        memory_delay=1.0,
        memory_sensitivity=1.5,

        # Noise: volatility clustering
        noise_scale=0.02,           # ~2% daily vol
        noise_state_coupling=0.5,   # Strong volatility clustering

        control_gain=0.0,          # No external control
        seed=42
    )

    osc = StochasticOscillator(config=config)

    # Simulate 252 trading days (1 year)
    n_days = 252
    result = osc.simulate(t_max=n_days, dt=1.0)

    returns = result['values']
    times = result['times']

    # Compute statistics
    stats = compute_statistics(returns)

    print(f"\n  Simulation: {n_days} trading days")
    print(f"\n  Return Statistics:")
    print(f"    Mean daily return: {stats.mean*100:.4f}%")
    print(f"    Daily volatility:  {stats.std*100:.4f}%")
    print(f"    Annualized vol:    {stats.std*100*np.sqrt(252):.2f}%")
    print(f"    Skewness:          {stats.skewness:.4f}")
    print(f"    Excess kurtosis:   {stats.kurtosis:.4f}")

    # Check for volatility clustering
    abs_returns = np.abs(returns)
    acf_vol = autocorrelation(abs_returns, max_lag=10)
    print(f"\n  Volatility Clustering (|r| autocorrelation):")
    print(f"    Lag-1: {acf_vol[1]:.4f}")
    print(f"    Lag-5: {acf_vol[5]:.4f}")

    # Convert to price series
    initial_price = 100.0
    prices = initial_price * np.exp(np.cumsum(returns))

    print(f"\n  Price Series (starting at ${initial_price}):")
    print(f"    Final price: ${prices[-1]:.2f}")
    print(f"    Total return: {(prices[-1]/initial_price - 1)*100:.2f}%")
    print(f"    Max price: ${prices.max():.2f}")
    print(f"    Min price: ${prices.min():.2f}")
    print(f"    Max drawdown: {(1 - prices.min()/prices.max())*100:.2f}%")

    return times, returns, prices


def simulate_with_jumps():
    """
    Simulate returns with jump-diffusion dynamics.

    Captures:
    - Normal market fluctuations
    - Occasional large moves (earnings, news events)
    """
    print("\n" + "=" * 70)
    print("  JUMP-DIFFUSION SIMULATION")
    print("=" * 70)

    # Create jump-diffusion noise generator
    jump_noise = JumpDiffusionNoise(
        continuous_scale=0.015,   # 1.5% continuous vol
        jump_rate=0.05,           # ~12 jumps per year
        jump_mean=-0.01,          # Slight negative jump bias
        jump_scale=0.03,          # 3% average jump size
        dt=1.0,
        seed=42
    )

    # Configure oscillator with minimal built-in noise
    config = OscillatorConfig(
        n_components=2,
        amplitude_base=0.005,
        noise_scale=0.0,  # Disable built-in noise
        memory_strength=0.2,
        seed=42
    )

    osc = StochasticOscillator(config=config)

    # Manual simulation with custom noise
    n_days = 252
    times = np.arange(n_days)
    returns = np.zeros(n_days)

    for i, t in enumerate(times):
        # Get deterministic component from oscillator
        base_value = osc.evaluate(t, store_history=True)

        # Add jump-diffusion noise
        noise = jump_noise.sample(t)
        returns[i] = base_value * 0.1 + noise  # Scale oscillator contribution

    stats = compute_statistics(returns)

    print(f"\n  Jump-Diffusion Return Statistics:")
    print(f"    Mean daily return: {stats.mean*100:.4f}%")
    print(f"    Daily volatility:  {stats.std*100:.4f}%")
    print(f"    Skewness:          {stats.skewness:.4f}  (negative = left tail)")
    print(f"    Excess kurtosis:   {stats.kurtosis:.4f}  (>0 = fat tails)")

    # Count large moves
    large_moves = np.abs(returns) > 2 * stats.std
    print(f"\n  Extreme Events (>2 sigma):")
    print(f"    Count: {np.sum(large_moves)}")
    print(f"    Frequency: {np.mean(large_moves)*100:.2f}%")
    print(f"    Expected (Gaussian): 4.55%")

    return times, returns


def regime_switching_simulation():
    """
    Simulate regime-switching dynamics using adaptive memory.

    Models:
    - Calm periods (low vol, mean-reverting)
    - Volatile periods (high vol, trending)
    """
    print("\n" + "=" * 70)
    print("  REGIME-SWITCHING SIMULATION")
    print("=" * 70)

    # Simulate two regimes with different parameters
    n_days = 500

    # Regime 1: Calm (first 250 days)
    config_calm = OscillatorConfig(
        n_components=5,
        noise_scale=0.01,
        memory_strength=1.5,      # Mean-reverting
        memory_sensitivity=-2.0,  # Negative biases memory toward mean reversion (see model for details)
        seed=42
    )

    # Regime 2: Volatile (last 250 days)
    config_volatile = OscillatorConfig(
        n_components=5,
        noise_scale=0.04,         # 4x higher vol
        memory_strength=0.5,      # Trending
        memory_sensitivity=2.0,   # Positive = momentum
        seed=42
    )

    osc_calm = StochasticOscillator(config=config_calm)
    osc_volatile = StochasticOscillator(config=config_volatile)

    result_calm = osc_calm.simulate(t_max=250, dt=1.0)
    result_volatile = osc_volatile.simulate(t_max=250, dt=1.0)

    # Combine regimes
    returns = np.concatenate([result_calm['values'], result_volatile['values']])
    times = np.arange(n_days)

    # Compute rolling statistics
    window = 20
    rolling_vol = np.array([
        np.std(returns[max(0, i-window):i+1])
        for i in range(n_days)
    ])

    print(f"\n  Two-Regime Simulation ({n_days} days)")
    print(f"\n  Regime 1 (Calm, days 0-249):")
    stats1 = compute_statistics(returns[:250])
    print(f"    Volatility: {stats1.std*100:.4f}%")
    print(f"    ACF(1): {stats1.autocorr_lag1:.4f}")

    print(f"\n  Regime 2 (Volatile, days 250-499):")
    stats2 = compute_statistics(returns[250:])
    print(f"    Volatility: {stats2.std*100:.4f}%")
    print(f"    ACF(1): {stats2.autocorr_lag1:.4f}")

    print(f"\n  Volatility Ratio (Regime2/Regime1): {stats2.std/stats1.std:.2f}x")

    return times, returns, rolling_vol


def main():
    print("\n" + "=" * 70)
    print("  MCSO - Financial Modeling Applications")
    print("=" * 70)

    # Run all simulations
    simulate_asset_returns()
    simulate_with_jumps()
    regime_switching_simulation()

    print("\n" + "=" * 70)
    print("  Financial modeling examples completed!")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
