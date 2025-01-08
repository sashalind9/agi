import pytest
import numpy as np
from qeaf.core import QuantumAGIFramework
from qeaf.consciousness import ConsciousnessEmergenceProtocol
from qeaf.quantum.entanglement import QuantumEntanglementModule
from qeaf.security.encryption import QuantumEncryption
from qeaf.military.oversight import MilitaryOversightProtocol

def test_quantum_agi_initialization():
    """Test quantum AGI framework initialization"""
    agi = QuantumAGIFramework()
    assert agi.consciousness_level == 0.0
    assert agi.config.quantum_cores == 8

def test_consciousness_emergence():
    """Test consciousness emergence protocol"""
    protocol = ConsciousnessEmergenceProtocol(
        threshold=0.95,
        emergence_rate=0.001
    )
    
    initial_level = protocol.initialize()
    assert 0 <= initial_level <= 1.0
    
    # Test consciousness step
    quantum_state = np.random.random(8)
    quantum_state /= np.linalg.norm(quantum_state)
    delta = protocol.step(quantum_state)
    
    assert isinstance(delta, float)
    assert -1.0 <= delta <= 1.0

def test_quantum_entanglement():
    """Test quantum entanglement module"""
    entanglement = QuantumEntanglementModule(
        n_qubits=4,
        depth=2
    )
    
    state = entanglement.generate_entangled_state()
    assert 0 <= state.fidelity <= 1.0
    assert 0 <= state.concurrence <= 1.0
    assert state.entropy >= 0
    assert state.bell_state in ["Φ+", "Φ-", "Ψ+", "Ψ-", "UNDEFINED"]

def test_quantum_encryption():
    """Test quantum encryption system"""
    encryption = QuantumEncryption(
        security_level="MAXIMUM",
        key_size=128
    )
    
    # Test consciousness state encryption
    state = 0.75
    encrypted_state = encryption.encrypt_consciousness_state(state)
    assert isinstance(encrypted_state, np.ndarray)
    assert np.all(np.isfinite(encrypted_state))

def test_military_oversight():
    """Test military oversight protocols"""
    oversight = MilitaryOversightProtocol(
        initial_threat_level="MINIMAL",
        max_consciousness=0.95
    )
    
    # Test threat assessment
    status = oversight.monitor_consciousness_level(0.8)
    assert isinstance(status, dict)
    assert "threat_level" in status
    assert "containment_status" in status
    assert "emergency_ready" in status

def test_full_agi_pipeline():
    """Test complete AGI pipeline"""
    # Initialize framework
    agi = QuantumAGIFramework()
    
    # Start consciousness emergence
    agi.initialize_emergence()
    assert agi.consciousness_level == 0.0
    
    # Run multiple steps
    for _ in range(5):
        status = agi.step()
        assert isinstance(status, dict)
        assert "consciousness_level" in status
        assert "quantum_state" in status
        assert "security_status" in status
        
        consciousness_level = status["consciousness_level"]
        assert 0 <= consciousness_level <= 1.0

if __name__ == "__main__":
    pytest.main([__file__]) 