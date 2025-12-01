"""
H-Model Manager

This module provides the comprehensive H-Model system management class,
including simulation, data management, drift detection, and export functionality.
"""

import base64
import logging
import pickle
import sqlite3
import time
from contextlib import contextmanager
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Union

try:
    import json
except ImportError:
    json = None  # type: ignore

try:
    import numpy as np
except ImportError:
    np = None  # type: ignore

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False
    pd = None  # type: ignore

from .exceptions import ValidationError, ModelError
from .security import SecurityValidator, secure_operation
from .performance import performance_monitor
from .data_structures import ModelState, ModelParameters
from .vector_embedding import VectorEmbeddingGenius
from .blockchain import BlockchainConnector


logger = logging.getLogger(__name__)


class HModelManager:
    """Comprehensive H-Model system management."""

    def __init__(self, initial_params: Dict[str, Any]) -> None:
        """Initialize H-Model manager with comprehensive setup."""
        self.parameters = ModelParameters(**initial_params)
        self.state = ModelState()
        self.vector_engine = VectorEmbeddingGenius()
        self.blockchain = BlockchainConnector()
        self.security = SecurityValidator()

        self.performance_metrics: Dict[str, Any] = {
            "operations_count": 0,
            "total_execution_time": 0.0,
            "error_count": 0,
            "last_operation_time": None,
        }

        self._setup_database()
        self._initialize_components()

        logger.info("H-Model Manager initialized successfully")

    def _setup_database(self):
        """Set up SQLite database for persistent storage."""
        self.db_path = "h_model_data.db"
        self.connection = sqlite3.connect(self.db_path, check_same_thread=False)
        self.connection.execute("PRAGMA foreign_keys = ON")
        self._create_tables()

    def _create_tables(self):
        """Create database tables."""
        cursor = self.connection.cursor()

        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS simulations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp REAL,
                t_value REAL,
                h_value REAL,
                parameters TEXT,
                method TEXT,
                execution_time REAL
            )
            """
        )

        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS performance_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp REAL,
                operation_name TEXT,
                execution_time REAL,
                success BOOLEAN,
                error_message TEXT
            )
            """
        )

        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS model_snapshots (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp REAL,
                state_data TEXT,
                checksum TEXT
            )
            """
        )

        self.connection.commit()

    def _save_model_snapshot(self):
        """Save a snapshot of the current model state to the database."""
        state_data = pickle.dumps(self.state)
        checksum = self.state.checksum
        timestamp = time.time()

        cursor = self.connection.cursor()
        cursor.execute(
            "INSERT INTO model_snapshots (timestamp, state_data, checksum) VALUES (?, ?, ?)",
            (timestamp, base64.b64encode(state_data).decode("utf-8"), checksum),
        )
        self.connection.commit()

    def _record_simulation(self, t_value, h_value, method, execution_time):
        """Record a simulation result in the database."""
        params_str = json.dumps(self.parameters.to_dict()) if json else "{}"
        timestamp = time.time()

        cursor = self.connection.cursor()
        cursor.execute(
            """
            INSERT INTO simulations (timestamp, t_value, h_value, parameters, method, execution_time)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (timestamp, t_value, h_value, params_str, method, execution_time),
        )
        self.connection.commit()

    def _initialize_components(self):
        """Initialize all system components."""
        genesis_data = {
            "event": "system_initialization",
            "parameters": self.parameters.to_dict(),
            "timestamp": time.time(),
        }
        self.blockchain.create_block(genesis_data)

        logger.info("All components initialized")

    @contextmanager
    def secure_context(self):
        """Context manager for secure operations."""
        operation_id = SecurityValidator.generate_token()[:16]
        start_time = time.perf_counter()

        try:
            logger.info(f"[{operation_id}] Entering secure context")
            yield operation_id

        except Exception as e:
            logger.error(f"[{operation_id}] Error in secure context: {e}")
            raise

        finally:
            execution_time = time.perf_counter() - start_time
            logger.info(f"[{operation_id}] Exiting secure context after {execution_time:.4f}s")

    @secure_operation
    @performance_monitor
    def load_data(
        self,
        series: Union[List, "np.ndarray", "pd.DataFrame"],
        preprocess_fn: Optional[Callable] = None,
    ) -> None:
        """Load and preprocess data with comprehensive validation."""
        if PANDAS_AVAILABLE and pd is not None and isinstance(series, pd.DataFrame):
            data = series.values.flatten()
        elif isinstance(series, list):
            if not np:
                raise ImportError("Numpy is required to process list data.")
            data = np.array(series)
        else:
            data = series

        if data is None or len(data) == 0:
            raise ValidationError("Data cannot be empty")

        if np and np.all(np.isfinite(data)):
            pass
        elif np:
            logger.warning("Data contains non-finite values, cleaning...")
            data = data[np.isfinite(data)]

            if preprocess_fn:
                data = preprocess_fn(data)

        self.state.data = data
        self.state.H_history = data.tolist()
        self.state.t_history = list(range(len(data)))
        self.state.last_updated = datetime.utcnow()

        self._save_model_snapshot()

        if np:
            blockchain_data = {
                "event": "data_loaded",
                "data_size": len(data),
                "data_stats": {
                    "mean": float(np.mean(data)),
                    "std": float(np.std(data)),
                    "min": float(np.min(data)),
                    "max": float(np.max(data)),
                },
            }
            self.blockchain.create_block(blockchain_data)

        logger.info(f"Data loaded: {len(data)} points")

    @secure_operation
    @performance_monitor
    def simulate(
        self, t: float, control_input: Optional[float] = None, method: str = "euler"
    ) -> float:
        """Simulate H-model with advanced numerical methods."""
        start_time = time.perf_counter()

        try:
            if method == "euler":
                result = self._euler_integration(t, control_input)
            elif method == "runge_kutta":
                result = self._runge_kutta_integration(t, control_input)
            elif method == "adaptive":
                result = self._adaptive_integration(t, control_input)
            else:
                raise ValueError(f"Unknown integration method: {method}")

            self.state.H_history.append(result)
            self.state.t_history.append(t)

            execution_time = time.perf_counter() - start_time
            self._record_simulation(t, result, method, execution_time)

            self.performance_metrics["operations_count"] += 1
            self.performance_metrics["total_execution_time"] += execution_time
            self.performance_metrics["last_operation_time"] = time.time()

            return result

        except Exception as e:
            self.performance_metrics["error_count"] += 1
            logger.error(f"Simulation failed: {e}")
            raise ModelError(f"Simulation failed: {str(e)}")

    def _euler_integration(self, t: float, u: Optional[float] = None) -> float:
        """Euler method integration."""
        dt = 0.01
        h = self.state.H_history[-1] if self.state.H_history else 1.0

        def dH_dt(h_val, t_val, u_val):
            p = self.parameters
            control = u_val if u_val is not None else 0.0
            if not np:
                return 0.0

            return (
                p.A * h_val
                + p.B * np.sin(p.C * t_val)
                + p.D * control
                + p.eta * np.random.normal(0, p.sigma)
            )

        h_new = h + dt * dH_dt(h, t, u)

        return h_new

    def _runge_kutta_integration(self, t: float, u: Optional[float] = None) -> float:
        """4th order Runge-Kutta integration."""
        dt = 0.01
        h = self.state.H_history[-1] if self.state.H_history else 1.0

        def dH_dt_func(t_val, H_val):
            p = self.parameters
            control = u if u is not None else 0.0
            if not np:
                return 0.0

            return (
                p.A * H_val
                + p.B * np.sin(p.C * t_val)
                + p.D * control
                + p.eta * np.random.normal(0, p.sigma)
            )

        k1 = dt * dH_dt_func(t, h)
        k2 = dt * dH_dt_func(t + dt / 2, h + k1 / 2)
        k3 = dt * dH_dt_func(t + dt / 2, h + k2 / 2)
        k4 = dt * dH_dt_func(t + dt, h + k3)

        h_new = h + (k1 + 2 * k2 + 2 * k3 + k4) / 6

        return h_new

    def _adaptive_integration(self, t: float, u: Optional[float] = None) -> float:
        """Adaptive step size integration."""
        dt = 0.001
        h = self.state.H_history[-1] if self.state.H_history else 1.0

        def dH_dt_func(t_val, H_val):
            p = self.parameters
            control = u if u is not None else 0.0
            if not np:
                return 0.0
            return (
                p.A * H_val
                + p.B * np.sin(p.C * t_val)
                + p.D * control
                + p.eta * np.random.normal(0, p.sigma)
            )

        tolerance = 1e-6
        max_iterations = 1000
        iteration = 0
        h_full = h

        while iteration < max_iterations:
            k1 = dt * dH_dt_func(t, h)
            k2 = dt * dH_dt_func(t + dt / 2, h + k1 / 2)
            k3 = dt * dH_dt_func(t + dt / 2, h + k2 / 2)
            k4 = dt * dH_dt_func(t + dt, h + k3)

            h_full = h + (k1 + 2 * k2 + 2 * k3 + k4) / 6

            dt_half = dt / 2

            k1_1 = dt_half * dH_dt_func(t, h)
            k2_1 = dt_half * dH_dt_func(t + dt_half / 2, h + k1_1 / 2)
            k3_1 = dt_half * dH_dt_func(t + dt_half / 2, h + k2_1 / 2)
            k4_1 = dt_half * dH_dt_func(t + dt_half, h + k3_1)

            h_half = h + (k1_1 + 2 * k2_1 + 2 * k3_1 + k4_1) / 6

            k1_2 = dt_half * dH_dt_func(t + dt_half, h_half)
            k2_2 = dt_half * dH_dt_func(t + dt_half + dt_half / 2, h_half + k1_2 / 2)
            k3_2 = dt_half * dH_dt_func(t + dt_half + dt_half / 2, h_half + k2_2 / 2)
            k4_2 = dt_half * dH_dt_func(t + dt, h_half + k3_2)

            h_double = h_half + (k1_2 + 2 * k2_2 + 2 * k3_2 + k4_2) / 6

            error = abs(h_double - h_full)

            if error < tolerance:
                return h_double

            dt = dt * 0.9 * (tolerance / error) ** 0.25
            dt = max(dt, 1e-8)

            iteration += 1

        logger.warning("Adaptive integration did not converge")
        return h_full

    def detect_drift(self, window: int = 50, threshold: float = 0.1) -> Dict[str, Any]:
        """Detect drift in the model's history using a statistical test."""
        if not np:
            raise ImportError("numpy is required for drift detection.")

        if len(self.state.H_history) < 2 * window:
            return {"drift_detected": False, "message": "Not enough data for drift detection."}

        series1 = np.array(self.state.H_history[-2 * window : -window])
        series2 = np.array(self.state.H_history[-window:])

        mean1, mean2 = np.mean(series1), np.mean(series2)
        std1, std2 = np.std(series1), np.std(series2)

        p_value: float = 1.0
        try:
            from scipy.stats import ttest_ind
            _, p_value = ttest_ind(series1, series2, equal_var=False)
        except ImportError:
            logger.warning("scipy not found, using simple mean comparison for drift detection.")
            if abs(mean1 - mean2) > threshold * (std1 + std2) / 2:
                p_value = 0.01
            else:
                p_value = 1.0

        drift_detected = p_value < threshold

        return {
            "drift_detected": drift_detected,
            "p_value": p_value,
            "mean1": mean1,
            "mean2": mean2,
            "std1": std1,
            "std2": std2,
        }

    def optimize_parameters(self) -> Dict[str, Any]:
        """A placeholder for a parameter optimization routine."""
        logger.info("Parameter optimization routine called (placeholder).")
        return {
            "status": "completed_placeholder",
            "optimized_parameters": self.parameters.to_dict(),
        }

    def export_results(self, format: str = "json") -> Union[str, bytes]:
        """Export simulation results to a specified format."""
        if format == "json":
            if not json:
                raise ImportError("json module not available")
            return json.dumps(
                {
                    "parameters": self.parameters.to_dict(),
                    "state": self.state.H_history,
                    "timestamps": self.state.t_history,
                },
                indent=2,
            )
        elif format == "csv":
            if not PANDAS_AVAILABLE:
                raise ImportError("pandas is required for CSV export.")
            df = pd.DataFrame({"timestamp": self.state.t_history, "H_value": self.state.H_history})
            return df.to_csv(index=False)
        else:
            raise ValueError("Unsupported format. Choose 'json' or 'csv'.")
