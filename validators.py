#!/usr/bin/env python3
"""
Input and Schema Validation Module

Provides comprehensive validation utilities for data integrity,
schema conformance, and business rule enforcement.
"""

from typing import Dict, Any, List, Optional, Union, Type
from pydantic import BaseModel, Field, field_validator, ValidationError as PydanticValidationError
from enum import Enum
import numpy as np
from exceptions import SchemaValidationError, DataIntegrityError, InsufficientDataError


class PerformanceDataSchema(BaseModel):
    """Schema for performance data validation."""

    method_name: str = Field(..., min_length=1, description="Name of the performance method")
    requests_per_second: float = Field(
        ..., gt=0, description="Requests per second (must be positive)"
    )
    latency_ns: Optional[float] = Field(None, ge=0, description="Latency in nanoseconds")
    timestamp: Optional[str] = Field(None, description="ISO timestamp of measurement")

    @field_validator("requests_per_second")
    @classmethod
    def validate_rps(cls, v: float) -> float:
        """Validate RPS is within reasonable bounds."""
        if v > 1e12:  # 1 trillion RPS is unrealistic
            raise ValueError(f"RPS value {v} exceeds realistic bounds")
        return v

    @field_validator("latency_ns")
    @classmethod
    def validate_latency(cls, v: Optional[float]) -> Optional[float]:
        """Validate latency is within reasonable bounds."""
        if v is not None and v > 1e12:  # 1 second = 1e9 ns
            raise ValueError(f"Latency {v}ns exceeds realistic bounds")
        return v


class BenchmarkResultSchema(BaseModel):
    """Schema for benchmark result validation."""

    tasks_per_second: float = Field(..., gt=0)
    processing_time_per_task_ns: float = Field(..., ge=0)
    total_tasks: int = Field(..., gt=0)
    duration_seconds: float = Field(..., gt=0)

    @field_validator("tasks_per_second", "processing_time_per_task_ns")
    @classmethod
    def validate_consistency(cls, v: float) -> float:
        """Validate metrics are mathematically consistent."""
        if not np.isfinite(v):
            raise ValueError(f"Value must be finite, got {v}")
        return v


class StatisticalThreshold(BaseModel):
    """Schema for statistical threshold configuration."""

    mean: float = Field(..., description="Mean value")
    std_dev: float = Field(..., ge=0, description="Standard deviation")
    confidence_level: float = Field(0.95, ge=0.5, le=0.999, description="Confidence level")
    sample_size: int = Field(..., gt=1, description="Sample size")

    def get_z_score(self) -> float:
        """Get z-score for the confidence level."""
        # Common z-scores for confidence intervals
        z_scores = {0.90: 1.645, 0.95: 1.96, 0.99: 2.576, 0.999: 3.291}
        return z_scores.get(self.confidence_level, 1.96)

    def get_confidence_interval(self) -> tuple[float, float]:
        """Calculate confidence interval."""
        margin = self.get_z_score() * (self.std_dev / np.sqrt(self.sample_size))
        return (self.mean - margin, self.mean + margin)


class DataValidator:
    """Comprehensive data validation utilities."""

    @staticmethod
    def validate_performance_data(data: Dict[str, Any]) -> bool:
        """
        Validate performance data structure and values.

        Args:
            data: Performance data dictionary

        Returns:
            True if validation passes

        Raises:
            SchemaValidationError: If schema validation fails
            DataIntegrityError: If data integrity checks fail
        """
        try:
            # Check for required keys
            required_keys = ["performance_evolution"]
            for key in required_keys:
                if key not in data:
                    raise SchemaValidationError(
                        field=key,
                        expected="dict with performance_evolution",
                        actual=f"missing key: {key}",
                    )

            # Validate performance evolution data
            perf_data = data["performance_evolution"]
            if not isinstance(perf_data, dict):
                raise SchemaValidationError(
                    field="performance_evolution", expected="dict", actual=str(type(perf_data))
                )

            # Validate each performance entry
            for method, rps in perf_data.items():
                if not isinstance(method, str):
                    raise SchemaValidationError(
                        field=f"performance_evolution.{method}",
                        expected="str",
                        actual=str(type(method)),
                    )

                if not isinstance(rps, (int, float)):
                    raise SchemaValidationError(
                        field=f"performance_evolution.{method}",
                        expected="numeric",
                        actual=str(type(rps)),
                    )

                if rps < 0:
                    raise DataIntegrityError(
                        check_name="positive_rps",
                        details=f"Method {method} has negative RPS: {rps}",
                    )

            return True

        except PydanticValidationError as e:
            raise SchemaValidationError(
                field="performance_data", expected="valid performance data", actual=str(e)
            )

    @staticmethod
    def validate_benchmark_results(data: Dict[str, Any]) -> bool:
        """
        Validate benchmark results structure.

        Args:
            data: Benchmark results dictionary

        Returns:
            True if validation passes

        Raises:
            SchemaValidationError: If validation fails
        """
        if "benchmark_results" not in data:
            raise SchemaValidationError(
                field="benchmark_results",
                expected="dict with benchmark_results",
                actual="missing benchmark_results key",
            )

        bench_data = data["benchmark_results"]
        for method, results in bench_data.items():
            try:
                BenchmarkResultSchema(**results)
            except PydanticValidationError as e:
                raise SchemaValidationError(
                    field=f"benchmark_results.{method}",
                    expected="valid benchmark result",
                    actual=str(e),
                )

        return True

    @staticmethod
    def validate_sufficient_data(data: List[Any], min_count: int, operation: str) -> bool:
        """
        Validate sufficient data points for analysis.

        Args:
            data: List of data points
            min_count: Minimum required count
            operation: Name of operation requiring data

        Returns:
            True if sufficient data

        Raises:
            InsufficientDataError: If insufficient data
        """
        if len(data) < min_count:
            raise InsufficientDataError(required_count=min_count, actual_count=len(data))
        return True

    @staticmethod
    def validate_statistical_properties(values: List[float]) -> Dict[str, Any]:
        """
        Validate and compute statistical properties.

        Args:
            values: List of numeric values

        Returns:
            Dictionary with statistical properties

        Raises:
            DataIntegrityError: If statistical properties are invalid
        """
        if not values:
            raise DataIntegrityError(
                check_name="non_empty_values", details="Cannot compute statistics on empty list"
            )

        # Check for NaN or inf values
        if not all(np.isfinite(v) for v in values):
            raise DataIntegrityError(
                check_name="finite_values", details="Values contain NaN or infinite values"
            )

        # Compute properties
        arr = np.array(values)
        properties = {
            "mean": float(np.mean(arr)),
            "median": float(np.median(arr)),
            "std_dev": float(np.std(arr)),
            "variance": float(np.var(arr)),
            "min": float(np.min(arr)),
            "max": float(np.max(arr)),
            "range": float(np.max(arr) - np.min(arr)),
            "coefficient_of_variation": (
                float(np.std(arr) / np.mean(arr)) if np.mean(arr) != 0 else 0
            ),
        }

        return properties

    @staticmethod
    def validate_data_consistency(
        data1: Dict[str, float], data2: Dict[str, float], tolerance: float = 0.1
    ) -> bool:
        """
        Validate consistency between two data sets.

        Args:
            data1: First data set
            data2: Second data set
            tolerance: Acceptable difference ratio (default 10%)

        Returns:
            True if consistent

        Raises:
            DataIntegrityError: If data is inconsistent
        """
        common_keys = set(data1.keys()) & set(data2.keys())

        if not common_keys:
            raise DataIntegrityError(
                check_name="common_keys", details="No common keys between data sets"
            )

        for key in common_keys:
            v1, v2 = data1[key], data2[key]
            if v1 == 0 and v2 == 0:
                continue

            max_val = max(abs(v1), abs(v2))
            diff_ratio = abs(v1 - v2) / max_val if max_val > 0 else 0

            if diff_ratio > tolerance:
                raise DataIntegrityError(
                    check_name="value_consistency",
                    details=f"Key '{key}' differs by {diff_ratio*100:.1f}%: {v1} vs {v2}",
                )

        return True


class CircuitBreaker:
    """Circuit breaker pattern for fault tolerance."""

    def __init__(self, failure_threshold: int = 5, timeout_seconds: float = 60.0):
        """
        Initialize circuit breaker.

        Args:
            failure_threshold: Number of failures before opening circuit
            timeout_seconds: Seconds to wait before attempting reset
        """
        self.failure_threshold = failure_threshold
        self.timeout_seconds = timeout_seconds
        self.failure_count = 0
        self.last_failure_time: Optional[float] = None
        self.state = "closed"  # closed, open, half_open

    def call(self, func, *args, **kwargs):
        """
        Execute function with circuit breaker protection.

        Args:
            func: Function to execute
            *args: Positional arguments
            **kwargs: Keyword arguments

        Returns:
            Function result

        Raises:
            CircuitBreakerError: If circuit is open
        """
        from exceptions import CircuitBreakerError
        import time

        # Check if circuit should transition
        if self.state == "open":
            if time.time() - self.last_failure_time > self.timeout_seconds:
                self.state = "half_open"
            else:
                raise CircuitBreakerError(
                    operation=func.__name__,
                    failure_count=self.failure_count,
                    threshold=self.failure_threshold,
                )

        try:
            result = func(*args, **kwargs)
            # Success - reset if in half_open
            if self.state == "half_open":
                self.state = "closed"
                self.failure_count = 0
            return result
        except Exception as e:
            self.failure_count += 1
            self.last_failure_time = time.time()

            if self.failure_count >= self.failure_threshold:
                self.state = "open"

            raise e
