"""
Core Data Structures

This module defines the core data classes used throughout the H-Model system:
- ModelState: Comprehensive model state representation
- ModelParameters: Advanced model parameters with validation
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

try:
    import hashlib
except ImportError:
    hashlib = None  # type: ignore

try:
    import numpy as np
except ImportError:
    np = None  # type: ignore

from .exceptions import ValidationError


@dataclass
class ModelState:
    """Comprehensive model state representation."""

    H_history: List[float] = field(default_factory=list)
    t_history: List[float] = field(default_factory=list)
    data: Optional["np.ndarray"] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    last_updated: datetime = field(default_factory=datetime.utcnow)
    version: str = "2.0.0"
    checksum: str = ""

    def __post_init__(self):
        self.checksum = self._calculate_checksum()

    def _calculate_checksum(self) -> str:
        """Calculate state checksum for integrity verification."""
        state_str = f"{self.H_history}{self.t_history}{self.version}"
        if hashlib:
            return hashlib.md5(state_str.encode()).hexdigest()
        return ""

    def validate_integrity(self) -> bool:
        """Validate state integrity."""
        return self.checksum == self._calculate_checksum()


@dataclass
class ModelParameters:
    """Advanced model parameters with validation."""

    A: float
    B: float
    C: float
    D: float
    eta: float
    gamma: float
    beta: float
    sigma: float
    tau: float
    alpha: float = 0.1
    lambda_reg: float = 0.01
    noise_level: float = 0.001

    def __post_init__(self):
        self.validate()

    def validate(self):
        """Comprehensive parameter validation."""
        if self.sigma < 0:
            raise ValidationError("sigma must be non-negative")
        if self.tau <= 0:
            raise ValidationError("tau must be positive")
        if not 0 <= self.alpha <= 1:
            raise ValidationError("alpha must be between 0 and 1")
        if self.lambda_reg < 0:
            raise ValidationError("lambda_reg must be non-negative")

    def to_dict(self) -> Dict[str, float]:
        return {
            "A": self.A,
            "B": self.B,
            "C": self.C,
            "D": self.D,
            "eta": self.eta,
            "gamma": self.gamma,
            "beta": self.beta,
            "sigma": self.sigma,
            "tau": self.tau,
            "alpha": self.alpha,
            "lambda_reg": self.lambda_reg,
            "noise_level": self.noise_level,
        }
