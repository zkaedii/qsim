"""
H_MODEL_Z Main Package
======================

Core functionality and public API for the H_MODEL_Z framework.
"""

from pathlib import Path

__version__ = "1.0.0"
__all__ = []

# Package root directory
PACKAGE_ROOT = Path(__file__).parent
PROJECT_ROOT = PACKAGE_ROOT.parent.parent

# Make subpackages available
from . import core
from . import frameworks
from . import schema
from . import engines

__all__.extend(['core', 'frameworks', 'schema', 'engines'])
