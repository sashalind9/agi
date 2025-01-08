"""Quantum computing and entanglement module."""

from .entanglement import QuantumEntanglementModule, EntanglementState
from .decoherence import QuantumDecoherenceModule

__all__ = [
    "QuantumEntanglementModule",
    "EntanglementState",
    "QuantumDecoherenceModule"
] 