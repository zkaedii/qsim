#!/usr/bin/env python3
"""
Backward-compatible shim for the Hamiltonian simulation module.

The implementation now lives in ``hmodelz.core.hamiltonian_simulation``.
"""

from hmodelz.core.hamiltonian_simulation import ComplexHamiltonianSimulator

__all__ = ["ComplexHamiltonianSimulator"]
