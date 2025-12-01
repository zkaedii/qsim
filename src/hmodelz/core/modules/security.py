"""
Security Validation and Decorators

This module provides security-related functionality including:
- Input validation
- Secure token generation
- Data hashing
- Filename sanitization
- Security-aware logging formatter
- Secure operation decorators
"""

import logging
import re
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Optional, Union

try:
    import functools
except ImportError:
    functools = None  # type: ignore

try:
    import hashlib
except ImportError:
    hashlib = None  # type: ignore

try:
    import secrets
except ImportError:
    secrets = None  # type: ignore

try:
    import asyncio
except ImportError:
    asyncio = None  # type: ignore

try:
    import numpy as np
except ImportError:
    np = None  # type: ignore

from .exceptions import SecurityError, OperationError


logger = logging.getLogger(__name__)


class SecurityAwareFormatter(logging.Formatter):
    """Custom formatter that sanitizes log messages for security."""

    def format(self, record):
        if hasattr(record, "msg"):
            record.msg = self._sanitize_log_message(str(record.msg))
        return super().format(record)

    def _sanitize_log_message(self, message: str) -> str:
        """Remove potentially dangerous content from log messages."""
        dangerous_patterns = [
            r"<script[^>]*>.*?</script>",
            r"javascript:",
            r"data:text/html",
            r"vbscript:",
            r"onload\s*=",
            r"onerror\s*=",
            r"eval\s*\(",
            r"exec\s*\(",
            r"__import__\s*\(",
            r"subprocess\.",
            r"os\.system",
            r"pickle\.loads",
            r"\\x[0-9a-fA-F]{2}",
            r"%[0-9a-fA-F]{2}",
        ]

        sanitized = message
        for pattern in dangerous_patterns:
            sanitized = re.sub(pattern, "[FILTERED]", sanitized, flags=re.IGNORECASE)

        if len(sanitized) > 1000:
            sanitized = sanitized[:997] + "..."

        return sanitized


class SecurityValidator:
    """Advanced security validation with comprehensive threat detection."""

    MAX_INPUT_SIZE = 1024 * 1024  # 1MB
    MAX_STRING_LENGTH = 10000
    MAX_RECURSION_DEPTH = 100

    DANGEROUS_PATTERNS = [
        r"<script[^>]*>.*?</script>",
        r"javascript:",
        r"data:text/html",
        r"vbscript:",
        r"onload\s*=",
        r"onerror\s*=",
        r"eval\s*\(",
        r"exec\s*\(",
        r"__import__\s*\(",
        r"subprocess\.",
        r"os\.system",
        r"pickle\.loads",
        r"\\x[0-9a-fA-F]{2}",
        r"%[0-9a-fA-F]{2}",
    ]

    @classmethod
    def validate_input(
        cls, data: Any, max_size: Optional[int] = None, context: str = "input"
    ) -> bool:
        """Enhanced input validation with context awareness."""
        max_size = max_size or cls.MAX_INPUT_SIZE

        try:
            data_size = sys.getsizeof(data)
            if data_size > max_size:
                logger.warning(
                    f"Input size validation failed in {context}: {data_size} > {max_size}"
                )
                raise ValueError(f"Input exceeds maximum size: {data_size} > {max_size}")

            if isinstance(data, str):
                return cls._validate_string(data, context)
            elif isinstance(data, (list, tuple)):
                return cls._validate_sequence(data, context)
            elif isinstance(data, dict):
                return cls._validate_dict(data, context)
            elif np and isinstance(data, np.ndarray):
                return cls._validate_array(data, context)

            return True

        except Exception as e:
            logger.error(f"Security validation failed in {context}: {e}")
            return False

    @classmethod
    def _validate_string(cls, data: str, context: str) -> bool:
        """String-specific security validation."""
        if len(data) > cls.MAX_STRING_LENGTH:
            raise ValueError(f"String too long in {context}: {len(data)} > {cls.MAX_STRING_LENGTH}")

        for pattern in cls.DANGEROUS_PATTERNS:
            if re.search(pattern, data, re.IGNORECASE):
                logger.warning(f"Dangerous pattern detected in {context}: {pattern}")
                raise ValueError(f"Dangerous pattern detected in {context}")

        if "\x00" in data or any(ord(c) < 32 and c not in "\t\n\r" for c in data):
            raise ValueError(f"Invalid characters detected in {context}")

        return True

    @classmethod
    def _validate_sequence(cls, data: Union[list, tuple], context: str, depth: int = 0) -> bool:
        """Validate sequences with recursion protection."""
        if depth > cls.MAX_RECURSION_DEPTH:
            raise ValueError(f"Maximum recursion depth exceeded in {context}")

        if len(data) > 10000:
            raise ValueError(f"Sequence too large in {context}: {len(data)}")

        for i, item in enumerate(data):
            if not cls.validate_input(item, context=f"{context}[{i}]"):
                return False

        return True

    @classmethod
    def _validate_dict(cls, data: dict, context: str, depth: int = 0) -> bool:
        """Validate dictionaries with key and value checking."""
        if depth > cls.MAX_RECURSION_DEPTH:
            raise ValueError(f"Maximum recursion depth exceeded in {context}")

        if len(data) > 1000:
            raise ValueError(f"Dictionary too large in {context}: {len(data)}")

        for key, value in data.items():
            if not isinstance(key, (str, int, float)):
                raise ValueError(f"Invalid key type in {context}: {type(key)}")

            if isinstance(key, str) and not cls._validate_string(key, f"{context}.key"):
                return False

            if not cls.validate_input(value, context=f"{context}.{key}"):
                return False

        return True

    @classmethod
    def _validate_array(cls, data: "np.ndarray", context: str) -> bool:
        """Validate NumPy arrays."""
        if data.nbytes > cls.MAX_INPUT_SIZE:
            raise ValueError(f"Array too large in {context}: {data.nbytes}")

        if np and (np.any(np.isnan(data)) or np.any(np.isinf(data))):
            logger.warning(f"Invalid numeric values detected in {context}")

        if hasattr(data, "dtype") and data.dtype == object:
            raise ValueError(f"Object arrays not allowed in {context}")

        return True

    @staticmethod
    def generate_token() -> str:
        """Generate cryptographically secure token."""
        if secrets is None:
            raise ImportError("secrets module is not available")
        return secrets.token_urlsafe(32)

    @staticmethod
    def hash_data(data: str, algorithm: str = "sha256") -> str:
        """Secure hash generation with algorithm selection."""
        if hashlib is None:
            raise ImportError("hashlib module is not available")

        hasher = getattr(hashlib, algorithm, None)
        if hasher is None:
            raise ValueError(f"Unsupported hash algorithm: {algorithm}")

        instance = hasher()
        instance.update(data.encode("utf-8"))
        return instance.hexdigest()

    @staticmethod
    def sanitize_filename(filename: str) -> str:
        """Sanitize filename to prevent path traversal."""
        filename = Path(filename).name
        filename = re.sub(r'[<>:"|?*\x00-\x1f]', "_", filename)

        reserved = (
            ["CON", "PRN", "AUX", "NUL"]
            + [f"COM{i}" for i in range(1, 10)]
            + [f"LPT{i}" for i in range(1, 10)]
        )
        if filename.upper().split(".")[0] in reserved:
            filename = f"_{filename}"

        if len(filename) > 255:
            name, ext = filename.rsplit(".", 1) if "." in filename else (filename, "")
            filename = name[: 255 - len(ext) - 1] + "." + ext if ext else name[:255]

        return filename


def secure_operation(func):
    """Enhanced decorator with comprehensive security and performance monitoring."""
    if functools is None:
        return func

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        operation_id = SecurityValidator.generate_token()[:16]
        memory_before = sum(sys.getsizeof(arg) for arg in args) + sum(
            sys.getsizeof(kwarg) for kwarg in kwargs.items()
        )

        try:
            for i, arg in enumerate(args):
                if not SecurityValidator.validate_input(arg, context=f"{func.__name__}.arg[{i}]"):
                    raise SecurityError(f"Security validation failed for argument {i}")

            for key, value in kwargs.items():
                if not SecurityValidator.validate_input(value, context=f"{func.__name__}.{key}"):
                    raise SecurityError(f"Security validation failed for {key}")

            if hasattr(func, "_last_call_times"):
                now = time.time()
                func._last_call_times = [t for t in func._last_call_times if now - t < 60]
                if len(func._last_call_times) > 100:
                    raise SecurityError(f"Rate limit exceeded for {func.__name__}")
                func._last_call_times.append(now)
            else:
                func._last_call_times = [time.time()]

            logger.info(f"[{operation_id}] Starting secure operation: {func.__name__}")
            result = func(*args, **kwargs)

            execution_time = time.perf_counter() - start_time
            memory_after = sys.getsizeof(result) if result else 0

            logger.info(
                f"[{operation_id}] Operation completed in {execution_time:.4f}s, "
                f"Memory: {memory_before} -> {memory_after} bytes"
            )

            return result

        except SecurityError as e:
            execution_time = time.perf_counter() - start_time
            logger.error(f"[{operation_id}] Security error in {func.__name__}: {e}")
            logger.error(f"[{operation_id}] Failed after {execution_time:.4f}s")
            raise
        except Exception as e:
            execution_time = time.perf_counter() - start_time
            logger.error(f"[{operation_id}] Error in {func.__name__}: {e}")
            logger.debug(f"[{operation_id}] Traceback: {traceback.format_exc()}")
            raise OperationError(f"Operation {func.__name__} failed: {str(e)}") from e

    return wrapper


def async_secure_operation(func):
    """Enhanced async decorator with security and performance monitoring."""
    if asyncio is None or functools is None:
        return func

    @functools.wraps(func)
    async def wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        operation_id = SecurityValidator.generate_token()[:16]
        try:
            for i, arg in enumerate(args):
                if not SecurityValidator.validate_input(arg, context=f"{func.__name__}.arg[{i}]"):
                    raise SecurityError(f"Security validation failed for argument {i}")

            logger.info(f"[{operation_id}] Starting async secure operation: {func.__name__}")

            result = await asyncio.wait_for(func(*args, **kwargs), timeout=300.0)

            execution_time = time.perf_counter() - start_time
            logger.info(f"[{operation_id}] Async operation completed in {execution_time:.4f}s")

            return result

        except asyncio.TimeoutError:
            execution_time = time.perf_counter() - start_time
            logger.error(
                f"[{operation_id}] Async timeout in {func.__name__} after {execution_time:.4f}s"
            )
            raise OperationError(f"Async operation {func.__name__} timed out")
        except Exception as e:
            execution_time = time.perf_counter() - start_time
            logger.error(f"[{operation_id}] Async error in {func.__name__}: {e}")
            raise OperationError(f"Async operation {func.__name__} failed: {str(e)}") from e

    return wrapper
