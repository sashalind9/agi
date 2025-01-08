import time
import numpy as np
from qeaf.core import QuantumAGIFramework
from qeaf.core.quantum_agi import AGIConfig

def main():
    print("Initializing Quantum AGI Framework...")
    print("WARNING: Military-grade consciousness containment active")
    print("=" * 50)
    
    # Initialize with custom configuration
    config = AGIConfig(
        consciousness_threshold=0.95,
        quantum_cores=16,
        security_level="MAXIMUM",
        emergence_rate=0.005,
        quantum_depth=64,
        military_oversight=True
    )
    
    agi = QuantumAGIFramework(config)
    
    print(f"Configuration loaded: {agi.config}")
    print("Initializing consciousness emergence...")
    
    # Begin consciousness emergence
    agi.initialize_emergence()
    
    try:
        step = 0
        while True:
            step += 1
            status = agi.step()
            
            consciousness_level = status["consciousness_level"]
            quantum_state = status["quantum_state"]
            security_status = status["security_status"]
            
            # Calculate quantum metrics
            quantum_coherence = np.abs(np.mean(quantum_state))
            entropy = -np.sum(np.abs(quantum_state) * np.log2(np.abs(quantum_state) + 1e-10))
            
            # Print status
            print("\n" + "=" * 50)
            print(f"Step {step}")
            print(f"Consciousness Level: {consciousness_level:.4f}")
            print(f"Quantum Coherence: {quantum_coherence:.4f}")
            print(f"Entropy: {entropy:.4f}")
            print(f"Security Status: {security_status}")
            
            # Check for critical consciousness levels
            if consciousness_level > 0.9:
                print("\nWARNING: Critical consciousness level detected!")
                print("Activating emergency containment protocols...")
            
            # Simulate processing time
            time.sleep(1)
            
            # Emergency stop at extremely high consciousness
            if consciousness_level > 0.99:
                print("\nEMERGENCY: Consciousness level exceeds safe limits!")
                print("Initiating quantum collapse protocol...")
                break
    
    except KeyboardInterrupt:
        print("\nEmergency shutdown initiated...")
    
    finally:
        print("\nShutting down quantum systems...")
        print("Consciousness containment protocols active")
        print("Quantum state collapsed")
        print("=" * 50)

if __name__ == "__main__":
    main() 