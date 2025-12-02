"""
Custom Exception Classes

This module defines the exception hierarchy for the H-Model system.
All exceptions inherit from HModelError and provide detailed error context.
"""

from datetime import datetime
from typing import Any, Dict, Optional

try:
    import secrets
except ImportError:
    secrets = None  # type: ignore


class HModelError(Exception):
    """Enhanced base exception with detailed error context."""

    def __init__(
        self, message: str, error_code: Optional[str] = None, context: Optional[Dict] = None
    ):
        super().__init__(message)
        self.error_code = error_code or "UNKNOWN_ERROR"
        self.context = context or {}
        self.timestamp = datetime.utcnow()
        if secrets:
            self.error_id = secrets.token_urlsafe(12)
        else:
            self.error_id = "no_secrets_module"

    def to_dict(self) -> Dict:
        return {
            "error_id": self.error_id,
            "error_code": self.error_code,
            "message": str(self),
            "context": self.context,
            "timestamp": self.timestamp.isoformat(),
        }


class SecurityError(HModelError):
    """Enhanced security-related errors with threat classification."""

    def __init__(self, message: str, threat_level: str = "HIGH", context: Optional[Dict] = None):
        super().__init__(message, "SECURITY_ERROR", context)
        self.threat_level = threat_level


class OperationError(HModelError):
    """Enhanced general operation errors."""

    def __init__(
        self, message: str, operation: Optional[str] = None, context: Optional[Dict] = None
    ):
        super().__init__(message, "OPERATION_ERROR", context)
        self.operation = operation


class ValidationError(HModelError):
    """Enhanced data validation errors."""

    def __init__(
        self,
        message: str,
        field: Optional[str] = None,
        value: Any = None,
        context: Optional[Dict] = None,
    ):
        super().__init__(message, "VALIDATION_ERROR", context)
        self.field = field
        self.value = value


class ModelError(HModelError):
    """Enhanced model computation errors."""

    def __init__(
        self, message: str, model_state: Optional[Dict] = None, context: Optional[Dict] = None
    ):
        super().__init__(message, "MODEL_ERROR", context)
        self.model_state = model_state
