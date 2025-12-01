"""
H-Model Core Modules

This package contains the refactored components of the H-Model Omnisolver.
Each module is responsible for a specific domain of functionality.

Modules
-------
exceptions : Custom exception classes
security : Security validation and decorators
performance : Performance monitoring utilities
data_structures : Core data classes (ModelState, ModelParameters)
vector_embedding : Vector embedding generation
blockchain : Blockchain integration
model_manager : Main HModelManager class
testing : Testing framework
html_interface : HTML interface generator
"""

from .exceptions import (
    HModelError,
    SecurityError,
    OperationError,
    ValidationError,
    ModelError,
)
from .security import (
    SecurityValidator,
    SecurityAwareFormatter,
    secure_operation,
    async_secure_operation,
)
from .performance import performance_monitor
from .data_structures import ModelState, ModelParameters
from .vector_embedding import VectorEmbeddingGenius
from .blockchain import BlockchainConnector
from .model_manager import HModelManager
from .testing import HModelTester
from .html_interface import HTMLOmnisolver

__all__ = [
    # Exceptions
    "HModelError",
    "SecurityError",
    "OperationError",
    "ValidationError",
    "ModelError",
    # Security
    "SecurityValidator",
    "SecurityAwareFormatter",
    "secure_operation",
    "async_secure_operation",
    # Performance
    "performance_monitor",
    # Data structures
    "ModelState",
    "ModelParameters",
    # Components
    "VectorEmbeddingGenius",
    "BlockchainConnector",
    "HModelManager",
    "HModelTester",
    "HTMLOmnisolver",
]
