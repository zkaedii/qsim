#!/usr/bin/env python3
"""
Custom Exception Hierarchy for Claude Analysis Agent

This module provides a comprehensive exception hierarchy for enterprise-grade
error handling with detailed context and recovery guidance.
"""

from typing import Optional, Dict, Any


class ClaudeAnalysisError(Exception):
    """Base exception for all Claude Analysis Agent errors.
    
    Attributes:
        message: Human-readable error description
        error_code: Unique error code for categorization
        context: Additional context information
        recovery_hint: Suggestion for error recovery
    """
    
    def __init__(
        self,
        message: str,
        error_code: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
        recovery_hint: Optional[str] = None
    ):
        self.message = message
        self.error_code = error_code or self.__class__.__name__
        self.context = context or {}
        self.recovery_hint = recovery_hint
        super().__init__(self.message)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert exception to dictionary for logging/serialization."""
        return {
            'error_type': self.__class__.__name__,
            'error_code': self.error_code,
            'message': self.message,
            'context': self.context,
            'recovery_hint': self.recovery_hint
        }


class DataSourceError(ClaudeAnalysisError):
    """Errors related to data source access and loading."""
    pass


class DataSourceNotFoundError(DataSourceError):
    """Raised when a required data source file is not found."""
    
    def __init__(self, filepath: str, **kwargs):
        super().__init__(
            message=f"Data source not found: {filepath}",
            error_code="DATA_SOURCE_NOT_FOUND",
            context={'filepath': filepath},
            recovery_hint="Ensure the data file exists and path is correct",
            **kwargs
        )


class DataSourceCorruptedError(DataSourceError):
    """Raised when a data source is corrupted or malformed."""
    
    def __init__(self, filepath: str, parse_error: str, **kwargs):
        super().__init__(
            message=f"Data source corrupted: {filepath}",
            error_code="DATA_SOURCE_CORRUPTED",
            context={'filepath': filepath, 'parse_error': parse_error},
            recovery_hint="Verify data file integrity and format",
            **kwargs
        )


class ValidationError(ClaudeAnalysisError):
    """Errors related to data validation."""
    pass


class SchemaValidationError(ValidationError):
    """Raised when data does not match expected schema."""
    
    def __init__(self, field: str, expected: str, actual: str, **kwargs):
        super().__init__(
            message=f"Schema validation failed for field '{field}'",
            error_code="SCHEMA_VALIDATION_ERROR",
            context={'field': field, 'expected': expected, 'actual': actual},
            recovery_hint=f"Ensure '{field}' matches expected type: {expected}",
            **kwargs
        )


class DataIntegrityError(ValidationError):
    """Raised when data integrity checks fail."""
    
    def __init__(self, check_name: str, details: str, **kwargs):
        super().__init__(
            message=f"Data integrity check failed: {check_name}",
            error_code="DATA_INTEGRITY_ERROR",
            context={'check_name': check_name, 'details': details},
            recovery_hint="Verify data quality and consistency",
            **kwargs
        )


class AnalysisError(ClaudeAnalysisError):
    """Errors that occur during analysis operations."""
    pass


class InsufficientDataError(AnalysisError):
    """Raised when there is insufficient data for analysis."""
    
    def __init__(self, required_count: int, actual_count: int, **kwargs):
        super().__init__(
            message=f"Insufficient data for analysis: need {required_count}, got {actual_count}",
            error_code="INSUFFICIENT_DATA",
            context={'required': required_count, 'actual': actual_count},
            recovery_hint=f"Provide at least {required_count} data points",
            **kwargs
        )


class StatisticalError(AnalysisError):
    """Raised when statistical computation fails."""
    
    def __init__(self, operation: str, reason: str, **kwargs):
        super().__init__(
            message=f"Statistical operation failed: {operation}",
            error_code="STATISTICAL_ERROR",
            context={'operation': operation, 'reason': reason},
            recovery_hint="Check data distribution and variance",
            **kwargs
        )


class ConfigurationError(ClaudeAnalysisError):
    """Errors related to configuration."""
    pass


class InvalidConfigurationError(ConfigurationError):
    """Raised when configuration is invalid."""
    
    def __init__(self, config_key: str, reason: str, **kwargs):
        super().__init__(
            message=f"Invalid configuration for '{config_key}': {reason}",
            error_code="INVALID_CONFIGURATION",
            context={'config_key': config_key, 'reason': reason},
            recovery_hint="Check configuration file format and values",
            **kwargs
        )


class CircuitBreakerError(ClaudeAnalysisError):
    """Raised when circuit breaker trips due to repeated failures."""
    
    def __init__(self, operation: str, failure_count: int, threshold: int, **kwargs):
        super().__init__(
            message=f"Circuit breaker tripped for '{operation}'",
            error_code="CIRCUIT_BREAKER_TRIPPED",
            context={
                'operation': operation,
                'failure_count': failure_count,
                'threshold': threshold
            },
            recovery_hint=f"Operation failed {failure_count} times. Wait before retrying.",
            **kwargs
        )


class ResourceExhaustedError(ClaudeAnalysisError):
    """Raised when system resources are exhausted."""
    
    def __init__(self, resource: str, details: str, **kwargs):
        super().__init__(
            message=f"Resource exhausted: {resource}",
            error_code="RESOURCE_EXHAUSTED",
            context={'resource': resource, 'details': details},
            recovery_hint="Free up resources or adjust limits",
            **kwargs
        )


class MetricsError(ClaudeAnalysisError):
    """Errors related to metrics collection and reporting."""
    pass


class MetricsCollectionError(MetricsError):
    """Raised when metrics collection fails."""
    
    def __init__(self, metric_name: str, reason: str, **kwargs):
        super().__init__(
            message=f"Failed to collect metric '{metric_name}'",
            error_code="METRICS_COLLECTION_ERROR",
            context={'metric_name': metric_name, 'reason': reason},
            recovery_hint="Check metrics configuration and connectivity",
            **kwargs
        )
