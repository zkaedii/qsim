"""
Analysis and Visualization Tools

This module provides statistical analysis and visualization capabilities
for Multi-Component Stochastic Oscillator trajectories.

Capabilities
------------
1. **Statistical Analysis**: Mean, variance, autocorrelation, stationarity tests
2. **Spectral Analysis**: Power spectrum, dominant frequencies, spectral entropy
3. **Stability Analysis**: Lyapunov exponents, basin of attraction
4. **Visualization**: Time series, phase space, spectral plots, ensemble plots
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy import stats, signal
from scipy.fft import fft, fftfreq
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Any, List
import warnings

# Optional imports for plotting
try:
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


# =============================================================================
# Statistical Analysis
# =============================================================================

@dataclass
class TrajectoryStatistics:
    """
    Container for trajectory statistics.

    Attributes
    ----------
    mean : float
        Sample mean.
    std : float
        Sample standard deviation.
    var : float
        Sample variance.
    skewness : float
        Sample skewness.
    kurtosis : float
        Sample excess kurtosis.
    min : float
        Minimum value.
    max : float
        Maximum value.
    median : float
        Median value.
    iqr : float
        Interquartile range.
    autocorr_lag1 : float
        Lag-1 autocorrelation.
    """
    mean: float
    std: float
    var: float
    skewness: float
    kurtosis: float
    min: float
    max: float
    median: float
    iqr: float
    autocorr_lag1: float


def compute_statistics(
    values: NDArray,
    times: Optional[NDArray] = None
) -> TrajectoryStatistics:
    """
    Compute comprehensive statistics for a trajectory.

    Parameters
    ----------
    values : ndarray
        State values.
    times : ndarray, optional
        Time points (unused, for API consistency).

    Returns
    -------
    TrajectoryStatistics
        Statistical summary.

    Examples
    --------
    >>> from mcso import StochasticOscillator
    >>> osc = StochasticOscillator()
    >>> result = osc.simulate(t_max=100)
    >>> stats = compute_statistics(result['values'])
    >>> print(f"Mean: {stats.mean:.4f}, Std: {stats.std:.4f}")
    """
    values = np.asarray(values)

    # Basic statistics
    mean = np.mean(values)
    std = np.std(values, ddof=1)
    var = np.var(values, ddof=1)
    skewness = stats.skew(values)
    kurtosis = stats.kurtosis(values)

    # Range statistics
    min_val = np.min(values)
    max_val = np.max(values)
    median = np.median(values)
    q1, q3 = np.percentile(values, [25, 75])
    iqr = q3 - q1

    # Autocorrelation at lag 1
    if len(values) > 1:
        autocorr = np.corrcoef(values[:-1], values[1:])[0, 1]
    else:
        autocorr = np.nan

    return TrajectoryStatistics(
        mean=mean,
        std=std,
        var=var,
        skewness=skewness,
        kurtosis=kurtosis,
        min=min_val,
        max=max_val,
        median=median,
        iqr=iqr,
        autocorr_lag1=autocorr
    )


def autocorrelation(values: NDArray, max_lag: Optional[int] = None) -> NDArray:
    """
    Compute autocorrelation function.

    Parameters
    ----------
    values : ndarray
        Time series values.
    max_lag : int, optional
        Maximum lag (default: len(values) // 4).

    Returns
    -------
    ndarray
        Autocorrelation values for lags 0, 1, ..., max_lag.
    """
    values = np.asarray(values)
    n = len(values)

    if max_lag is None:
        max_lag = n // 4

    # Center the series
    centered = values - np.mean(values)
    var = np.var(values)

    if var < 1e-10:
        return np.zeros(max_lag + 1)

    acf = np.correlate(centered, centered, mode='full')
    acf = acf[n-1:n+max_lag] / (var * n)

    return acf


def test_stationarity(
    values: NDArray,
    method: str = 'adf'
) -> Dict[str, Any]:
    """
    Test for stationarity of time series.

    Parameters
    ----------
    values : ndarray
        Time series values.
    method : str
        'adf' (Augmented Dickey-Fuller) or 'kpss'.

    Returns
    -------
    dict
        'statistic': Test statistic
        'pvalue': p-value
        'is_stationary': Boolean (at 5% level)
        'critical_values': Dict of critical values
    """
    values = np.asarray(values)

    if method == 'adf':
        try:
            from statsmodels.tsa.stattools import adfuller
            result = adfuller(values, autolag='AIC')
            return {
                'statistic': result[0],
                'pvalue': result[1],
                'is_stationary': result[1] < 0.05,
                'critical_values': result[4]
            }
        except ImportError:
            # Fallback: simple variance ratio test
            n = len(values)
            half = n // 2
            var1 = np.var(values[:half])
            var2 = np.var(values[half:])
            ratio = var1 / var2 if var2 > 0 else np.inf
            return {
                'statistic': ratio,
                'pvalue': np.nan,
                'is_stationary': 0.5 < ratio < 2.0,
                'critical_values': {}
            }

    else:
        raise ValueError(f"Unknown method: {method}")


# =============================================================================
# Spectral Analysis
# =============================================================================

@dataclass
class SpectralAnalysis:
    """
    Container for spectral analysis results.

    Attributes
    ----------
    frequencies : ndarray
        Frequency values.
    power : ndarray
        Power spectral density.
    dominant_freq : float
        Frequency with maximum power.
    spectral_entropy : float
        Normalized spectral entropy (0=pure tone, 1=white noise).
    bandwidth : float
        Spectral bandwidth (std of power distribution).
    """
    frequencies: NDArray
    power: NDArray
    dominant_freq: float
    spectral_entropy: float
    bandwidth: float


def spectral_analysis(
    values: NDArray,
    dt: float = 1.0,
    method: str = 'fft',
    window: Optional[str] = 'hann'
) -> SpectralAnalysis:
    """
    Perform spectral analysis on time series.

    Parameters
    ----------
    values : ndarray
        Time series values.
    dt : float
        Sampling interval.
    method : str
        'fft' (periodogram) or 'welch'.
    window : str, optional
        Window function for spectral estimation.

    Returns
    -------
    SpectralAnalysis
        Spectral analysis results.

    Examples
    --------
    >>> from mcso import StochasticOscillator
    >>> osc = StochasticOscillator(n_components=3)
    >>> result = osc.simulate(t_max=200, dt=0.1)
    >>> spec = spectral_analysis(result['values'], dt=0.1)
    >>> print(f"Dominant frequency: {spec.dominant_freq:.4f}")
    """
    values = np.asarray(values)
    n = len(values)

    if method == 'fft':
        # Apply window
        if window:
            win = signal.get_window(window, n)
            values_windowed = values * win
        else:
            values_windowed = values

        # Compute FFT
        yf = fft(values_windowed)
        freq = fftfreq(n, dt)

        # Keep positive frequencies
        positive_mask = freq > 0
        freq = freq[positive_mask]
        power = np.abs(yf[positive_mask]) ** 2 / n

    elif method == 'welch':
        freq, power = signal.welch(values, fs=1/dt, nperseg=min(256, n//4))

    else:
        raise ValueError(f"Unknown method: {method}")

    # Find dominant frequency
    if len(power) > 0:
        dominant_idx = np.argmax(power)
        dominant_freq = freq[dominant_idx]
    else:
        dominant_freq = 0.0

    # Compute spectral entropy
    power_normalized = power / np.sum(power) if np.sum(power) > 0 else power
    power_normalized = power_normalized[power_normalized > 0]  # Remove zeros for log
    if len(power_normalized) > 0:
        entropy = -np.sum(power_normalized * np.log(power_normalized))
        max_entropy = np.log(len(power_normalized))
        spectral_entropy = entropy / max_entropy if max_entropy > 0 else 0
    else:
        spectral_entropy = 0.0

    # Compute bandwidth (standard deviation of power distribution)
    if np.sum(power) > 0:
        mean_freq = np.sum(freq * power) / np.sum(power)
        bandwidth = np.sqrt(np.sum(power * (freq - mean_freq)**2) / np.sum(power))
    else:
        bandwidth = 0.0

    return SpectralAnalysis(
        frequencies=freq,
        power=power,
        dominant_freq=dominant_freq,
        spectral_entropy=spectral_entropy,
        bandwidth=bandwidth
    )


# =============================================================================
# Stability Analysis
# =============================================================================

def stability_analysis(
    oscillator,
    n_perturbations: int = 10,
    perturbation_scale: float = 0.1,
    t_max: float = 100.0,
    dt: float = 1.0
) -> Dict[str, Any]:
    """
    Analyze stability of oscillator dynamics.

    Estimates sensitivity to initial conditions and parameters.

    Parameters
    ----------
    oscillator : StochasticOscillator
        Oscillator instance.
    n_perturbations : int
        Number of perturbed trajectories.
    perturbation_scale : float
        Scale of parameter perturbations.
    t_max : float
        Simulation time.
    dt : float
        Time step.

    Returns
    -------
    dict
        'divergence_rate': Average divergence rate
        'is_stable': Boolean stability assessment
        'sensitivity': Parameter sensitivity scores
    """
    # Get baseline trajectory
    baseline = oscillator.simulate(t_max=t_max, dt=dt)
    baseline_values = baseline['values']

    divergences = []

    for _ in range(n_perturbations):
        # Perturb noise scale
        original_scale = oscillator.config.noise_scale
        oscillator.config.noise_scale *= (1 + perturbation_scale * np.random.randn())

        perturbed = oscillator.simulate(t_max=t_max, dt=dt)
        perturbed_values = perturbed['values']

        # Compute divergence
        divergence = np.mean(np.abs(baseline_values - perturbed_values))
        divergences.append(divergence)

        # Restore
        oscillator.config.noise_scale = original_scale

    mean_divergence = np.mean(divergences)
    std_divergence = np.std(divergences)

    return {
        'divergence_rate': mean_divergence,
        'divergence_std': std_divergence,
        'is_stable': mean_divergence < 10.0,  # Heuristic threshold
        'sensitivity': {
            'noise_scale': std_divergence / perturbation_scale
        }
    }


def estimate_lyapunov_exponent(
    values: NDArray,
    dt: float = 1.0,
    embedding_dim: int = 3,
    delay: int = 1
) -> float:
    """
    Estimate largest Lyapunov exponent from time series.

    Uses the Rosenstein algorithm for short time series.

    Parameters
    ----------
    values : ndarray
        Time series values.
    dt : float
        Sampling interval.
    embedding_dim : int
        Embedding dimension.
    delay : int
        Time delay for embedding.

    Returns
    -------
    float
        Estimated largest Lyapunov exponent.
        Positive: chaotic, Zero: periodic, Negative: stable fixed point.
    """
    values = np.asarray(values)
    n = len(values)

    # Create delay embedding
    m = embedding_dim
    tau = delay
    n_vectors = n - (m - 1) * tau

    if n_vectors < 10:
        warnings.warn("Time series too short for Lyapunov estimation")
        return np.nan

    # Embed
    embedded = np.zeros((n_vectors, m))
    for i in range(m):
        embedded[:, i] = values[i * tau:i * tau + n_vectors]

    # Find nearest neighbors (excluding temporal neighbors)
    min_temporal_separation = m * tau + 1
    divergences = []

    for i in range(n_vectors):
        # Compute distances to all other points
        dists = np.sqrt(np.sum((embedded - embedded[i]) ** 2, axis=1))

        # Exclude temporal neighbors
        for j in range(max(0, i - min_temporal_separation),
                      min(n_vectors, i + min_temporal_separation + 1)):
            dists[j] = np.inf

        # Find nearest neighbor
        nn_idx = np.argmin(dists)
        nn_dist = dists[nn_idx]

        if nn_dist > 0 and nn_dist < np.inf:
            # Track divergence over time
            max_k = min(50, n_vectors - max(i, nn_idx) - 1)
            for k in range(1, max_k):
                if i + k < n_vectors and nn_idx + k < n_vectors:
                    new_dist = np.sqrt(np.sum((embedded[i + k] - embedded[nn_idx + k]) ** 2))
                    if new_dist > 0:
                        divergences.append((k * dt, np.log(new_dist / nn_dist)))

    if not divergences:
        return np.nan

    # Fit slope to get Lyapunov exponent
    divergences = np.array(divergences)
    times = divergences[:, 0]
    log_divs = divergences[:, 1]

    # Use median slope for robustness
    if len(times) > 1:
        slope, _ = np.polyfit(times, log_divs, 1)
        return slope
    else:
        return np.nan


# =============================================================================
# Visualization Functions
# =============================================================================

def plot_trajectory(
    times: NDArray,
    values: NDArray,
    title: str = "Stochastic Oscillator Trajectory",
    xlabel: str = "Time",
    ylabel: str = "State X(t)",
    figsize: Tuple[int, int] = (12, 4),
    color: str = 'steelblue',
    alpha: float = 0.8,
    show_stats: bool = True,
    save_path: Optional[str] = None
) -> Optional[Any]:
    """
    Plot oscillator trajectory.

    Parameters
    ----------
    times : ndarray
        Time points.
    values : ndarray
        State values.
    title : str
        Plot title.
    xlabel, ylabel : str
        Axis labels.
    figsize : tuple
        Figure size.
    color : str
        Line color.
    alpha : float
        Line transparency.
    show_stats : bool
        Show statistics annotation.
    save_path : str, optional
        Path to save figure.

    Returns
    -------
    matplotlib.figure.Figure or None
        Figure object if matplotlib available.
    """
    if not HAS_MATPLOTLIB:
        warnings.warn("matplotlib not available for plotting")
        return None

    fig, ax = plt.subplots(figsize=figsize)

    ax.plot(times, values, color=color, alpha=alpha, linewidth=0.8)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)

    if show_stats:
        stats = compute_statistics(values)
        stats_text = (
            f"Mean: {stats.mean:.3f}\n"
            f"Std: {stats.std:.3f}\n"
            f"ACF(1): {stats.autocorr_lag1:.3f}"
        )
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
                verticalalignment='top', fontsize=9,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig


def plot_phase_space(
    values: NDArray,
    delay: int = 1,
    title: str = "Phase Space Portrait",
    figsize: Tuple[int, int] = (8, 8),
    cmap: str = 'viridis',
    save_path: Optional[str] = None
) -> Optional[Any]:
    """
    Plot phase space portrait using delay embedding.

    Parameters
    ----------
    values : ndarray
        Time series values.
    delay : int
        Time delay for embedding.
    title : str
        Plot title.
    figsize : tuple
        Figure size.
    cmap : str
        Colormap for time coloring.
    save_path : str, optional
        Path to save figure.

    Returns
    -------
    matplotlib.figure.Figure or None
    """
    if not HAS_MATPLOTLIB:
        warnings.warn("matplotlib not available for plotting")
        return None

    n = len(values)
    x = values[:-delay]
    y = values[delay:]
    colors = np.arange(len(x))

    fig, ax = plt.subplots(figsize=figsize)

    scatter = ax.scatter(x, y, c=colors, cmap=cmap, alpha=0.5, s=5)
    ax.plot(x, y, 'k-', alpha=0.1, linewidth=0.3)

    ax.set_xlabel(f"X(t)")
    ax.set_ylabel(f"X(t + {delay})")
    ax.set_title(title)
    ax.set_aspect('equal', adjustable='box')

    cbar = plt.colorbar(scatter, ax=ax, label='Time')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig


def plot_spectrum(
    spectral: SpectralAnalysis,
    title: str = "Power Spectrum",
    figsize: Tuple[int, int] = (10, 4),
    log_scale: bool = True,
    save_path: Optional[str] = None
) -> Optional[Any]:
    """
    Plot power spectrum from spectral analysis.

    Parameters
    ----------
    spectral : SpectralAnalysis
        Result from spectral_analysis().
    title : str
        Plot title.
    figsize : tuple
        Figure size.
    log_scale : bool
        Use logarithmic y-axis.
    save_path : str, optional
        Path to save figure.

    Returns
    -------
    matplotlib.figure.Figure or None
    """
    if not HAS_MATPLOTLIB:
        warnings.warn("matplotlib not available for plotting")
        return None

    fig, ax = plt.subplots(figsize=figsize)

    ax.plot(spectral.frequencies, spectral.power, 'b-', linewidth=0.8)

    # Mark dominant frequency
    ax.axvline(spectral.dominant_freq, color='r', linestyle='--', alpha=0.7,
               label=f'Dominant: {spectral.dominant_freq:.4f}')

    ax.set_xlabel("Frequency")
    ax.set_ylabel("Power")
    ax.set_title(title)

    if log_scale:
        ax.set_yscale('log')

    ax.legend()
    ax.grid(True, alpha=0.3)

    # Add entropy annotation
    ax.text(0.98, 0.98, f"Spectral Entropy: {spectral.spectral_entropy:.3f}",
            transform=ax.transAxes, verticalalignment='top',
            horizontalalignment='right', fontsize=9,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig


def plot_ensemble(
    ensemble_result: Dict[str, Any],
    title: str = "Ensemble Statistics",
    figsize: Tuple[int, int] = (12, 5),
    n_sample_paths: int = 10,
    save_path: Optional[str] = None
) -> Optional[Any]:
    """
    Plot ensemble simulation results.

    Parameters
    ----------
    ensemble_result : dict
        Result from StochasticOscillator.simulate_ensemble().
    title : str
        Plot title.
    figsize : tuple
        Figure size.
    n_sample_paths : int
        Number of sample paths to show.
    save_path : str, optional
        Path to save figure.

    Returns
    -------
    matplotlib.figure.Figure or None
    """
    if not HAS_MATPLOTLIB:
        warnings.warn("matplotlib not available for plotting")
        return None

    times = ensemble_result['times']
    ensemble = ensemble_result['ensemble']
    mean = ensemble_result['mean']
    std = ensemble_result['std']
    percentiles = ensemble_result['percentiles']

    fig, ax = plt.subplots(figsize=figsize)

    # Plot sample paths
    n_paths = min(n_sample_paths, ensemble.shape[0])
    for i in range(n_paths):
        ax.plot(times, ensemble[i], 'gray', alpha=0.2, linewidth=0.5)

    # Plot percentile bands
    ax.fill_between(times, percentiles[5], percentiles[95],
                    alpha=0.2, color='blue', label='5-95 percentile')
    ax.fill_between(times, percentiles[25], percentiles[75],
                    alpha=0.3, color='blue', label='25-75 percentile')

    # Plot mean
    ax.plot(times, mean, 'b-', linewidth=2, label='Mean')

    ax.set_xlabel("Time")
    ax.set_ylabel("State X(t)")
    ax.set_title(title)
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig


def create_summary_report(
    trajectory: Dict[str, NDArray],
    output_path: Optional[str] = None
) -> str:
    """
    Create text summary report for trajectory analysis.

    Parameters
    ----------
    trajectory : dict
        Simulation result with 'times' and 'values'.
    output_path : str, optional
        Path to save report.

    Returns
    -------
    str
        Formatted report text.
    """
    times = trajectory['times']
    values = trajectory['values']

    # Compute dt separately to handle single-sample case
    if len(times) > 1:
        dt = times[1] - times[0]
        dt_str = f"{dt:.4f}"
    else:
        dt = 1.0  # Default for spectral analysis
        dt_str = "N/A"

    stats = compute_statistics(values)
    spec = spectral_analysis(values, dt=dt)

    report = f"""
================================================================================
          MULTI-COMPONENT STOCHASTIC OSCILLATOR ANALYSIS REPORT
================================================================================

SIMULATION PARAMETERS
---------------------
  Duration: {times[-1] - times[0]:.2f} time units
  Samples: {len(times)}
  Time step: {dt_str}

STATISTICAL SUMMARY
-------------------
  Mean:             {stats.mean:12.6f}
  Std Deviation:    {stats.std:12.6f}
  Variance:         {stats.var:12.6f}
  Skewness:         {stats.skewness:12.6f}
  Excess Kurtosis:  {stats.kurtosis:12.6f}

  Minimum:          {stats.min:12.6f}
  Maximum:          {stats.max:12.6f}
  Median:           {stats.median:12.6f}
  IQR:              {stats.iqr:12.6f}

  Lag-1 ACF:        {stats.autocorr_lag1:12.6f}

SPECTRAL ANALYSIS
-----------------
  Dominant Frequency:  {spec.dominant_freq:12.6f}
  Spectral Entropy:    {spec.spectral_entropy:12.6f}  (0=tone, 1=noise)
  Bandwidth:           {spec.bandwidth:12.6f}

INTERPRETATION
--------------
  Noise Level: {'High' if stats.std > 1.0 else 'Moderate' if stats.std > 0.3 else 'Low'}
  Periodicity: {'Strong' if spec.spectral_entropy < 0.3 else 'Moderate' if spec.spectral_entropy < 0.7 else 'Weak'}
  Memory Effect: {'Strong' if abs(stats.autocorr_lag1) > 0.5 else 'Moderate' if abs(stats.autocorr_lag1) > 0.2 else 'Weak'}

================================================================================
"""

    if output_path:
        with open(output_path, 'w') as f:
            f.write(report)

    return report
