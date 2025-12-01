"""
Unit tests for the analysis module.
"""

import numpy as np
from mcso.analysis import create_summary_report, compute_statistics, spectral_analysis


class TestCreateSummaryReport:
    """Tests for create_summary_report function."""

    def test_normal_trajectory(self):
        """Test report generation for normal multi-sample trajectory."""
        trajectory = {
            'times': np.array([0.0, 0.1, 0.2, 0.3, 0.4]),
            'values': np.array([1.0, 2.0, 1.5, 2.5, 2.0])
        }
        report = create_summary_report(trajectory)

        assert 'MULTI-COMPONENT STOCHASTIC OSCILLATOR' in report
        assert 'Samples: 5' in report
        assert 'Time step: 0.1000' in report

    def test_single_sample_trajectory(self):
        """Test report generation for single-sample trajectory.

        Regression test for ValueError when formatting 'N/A' with :.4f.
        """
        trajectory = {
            'times': np.array([0.0]),
            'values': np.array([1.0])
        }
        # This should not raise ValueError
        report = create_summary_report(trajectory)

        assert 'Samples: 1' in report
        assert 'Time step: N/A' in report

    def test_two_sample_trajectory(self):
        """Test report generation for two-sample trajectory."""
        trajectory = {
            'times': np.array([0.0, 0.5]),
            'values': np.array([1.0, 2.0])
        }
        report = create_summary_report(trajectory)

        assert 'Samples: 2' in report
        assert 'Time step: 0.5000' in report


class TestComputeStatistics:
    """Tests for compute_statistics function."""

    def test_basic_statistics(self):
        """Test basic statistical computations."""
        values = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        stats = compute_statistics(values)

        assert stats.mean == 3.0
        assert stats.min == 1.0
        assert stats.max == 5.0
        assert stats.median == 3.0

    def test_single_value(self):
        """Test statistics for single value array."""
        values = np.array([5.0])
        stats = compute_statistics(values)

        assert stats.mean == 5.0
        assert stats.min == 5.0
        assert stats.max == 5.0
        assert np.isnan(stats.autocorr_lag1)


class TestSpectralAnalysis:
    """Tests for spectral_analysis function."""

    def test_basic_spectral(self):
        """Test basic spectral analysis."""
        t = np.linspace(0, 10, 100)
        values = np.sin(2 * np.pi * t)  # 1 Hz sine wave
        spec = spectral_analysis(values, dt=0.1)

        assert len(spec.frequencies) > 0
        assert len(spec.power) > 0
        assert spec.dominant_freq > 0
        assert 0 <= spec.spectral_entropy <= 1

    def test_single_value_spectral(self):
        """Test spectral analysis with single value."""
        values = np.array([1.0])
        spec = spectral_analysis(values, dt=1.0)

        # Should not crash and return valid structure
        assert hasattr(spec, 'frequencies')
        assert hasattr(spec, 'power')
