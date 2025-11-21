"""
Pytest configuration and fixtures for H_MODEL_Z tests.
"""

import pytest
import numpy as np
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


@pytest.fixture
def sample_time_array():
    """Provide a sample time array for simulations."""
    return np.linspace(0, 10, 100)


@pytest.fixture
def sample_hamiltonian_params():
    """Provide sample Hamiltonian parameters."""
    return {
        "num_components": 5,
        "base_amplitude": 1.0,
        "coupling_strength": 0.1,
        "noise_level": 0.05,
    }


@pytest.fixture
def mock_performance_data():
    """Provide mock performance benchmark data."""
    return {
        "jit_compiled": 85000000,
        "vectorized_numpy": 45000000,
        "basic_sequential": 5000000,
    }


@pytest.fixture
def random_state():
    """Provide a seeded random state for reproducible tests."""
    return np.random.RandomState(42)


# Pytest markers
def pytest_configure(config):
    """Configure custom pytest markers."""
    config.addinivalue_line("markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')")
    config.addinivalue_line("markers", "integration: marks tests as integration tests")
    config.addinivalue_line("markers", "benchmark: marks tests as benchmark tests")
