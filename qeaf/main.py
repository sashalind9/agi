import torch
import argparse
from pathlib import Path
from typing import Optional
import logging
from datetime import datetime

from .core.quantum_agi import QuantumAGICore
from .core.consciousness import ConsciousnessEmergence, ConsciousnessConfig
from .core.neural_architecture import QuantumNeuralBridge
from .military.oversight import MilitaryOversight, MilitaryConfig
from .training.quantum_trainer import QuantumAGITrainer, TrainingConfig
from .data.quantum_dataset import (
    QuantumClassicalDataset,
    QuantumDataGenerator,
    DataConfig
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)

def setup_experiment_directory() -> Path:
    """Setup experiment directory with timestamp"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = Path(f"experiments/quantum_agi_{timestamp}")
    exp_dir.mkdir(parents=True, exist_ok=True)
    return exp_dir

def initialize_system(
    data_config: Optional[DataConfig] = None,
    consciousness_config: Optional[ConsciousnessConfig] = None,
    military_config: Optional[MilitaryConfig] = None
) -> tuple:
    """Initialize all system components"""
    logger.info("Initializing Quantum AGI system components...")
    
    # Initialize configs
    data_config = data_config or DataConfig()
    consciousness_config = consciousness_config or ConsciousnessConfig()
    military_config = military_config or MilitaryConfig()
    
    # Initialize core components
    agi_core = QuantumAGICore(
        n_qubits=data_config.quantum_dim,
        embedding_dim=data_config.classical_dim
    )
    
    consciousness = ConsciousnessEmergence(config=consciousness_config)
    
    neural_bridge = QuantumNeuralBridge(
        classical_dim=data_config.classical_dim,
        quantum_dim=data_config.quantum_dim
    )
    
    military_oversight = MilitaryOversight(config=military_config)
    
    return agi_core, consciousness, neural_bridge, military_oversight

def generate_synthetic_data(
    exp_dir: Path,
    data_config: DataConfig,
    num_samples: int = 10000
) -> Path:
    """Generate synthetic quantum-classical dataset"""
    logger.info("Generating synthetic quantum-classical dataset...")
    
    data_generator = QuantumDataGenerator(config=data_config)
    data_path = exp_dir / "quantum_data.h5"
    
    data_generator.generate_quantum_data(
        num_samples=num_samples,
        output_path=str(data_path)
    )
    
    return data_path

def train_system(
    agi_core: QuantumAGICore,
    consciousness: ConsciousnessEmergence,
    military_oversight: MilitaryOversight,
    data_path: Path,
    exp_dir: Path,
    training_config: Optional[TrainingConfig] = None,
    data_config: Optional[DataConfig] = None
) -> None:
    """Train the AGI system"""
    logger.info("Starting AGI system training...")
    
    # Initialize configs
    training_config = training_config or TrainingConfig()
    data_config = data_config or DataConfig()
    
    # Create datasets
    train_dataset = QuantumClassicalDataset(
        str(data_path),
        config=data_config,
        mode="train"
    )
    val_dataset = QuantumClassicalDataset(
        str(data_path),
        config=data_config,
        mode="val"
    )
    
    # Initialize trainer
    trainer = QuantumAGITrainer(
        agi_core=agi_core,
        consciousness=consciousness,
        military_oversight=military_oversight,
        config=training_config
    )
    
    # Train system
    try:
        training_stats = trainer.train(
            train_dataloader=train_dataset.get_dataloader(),
            val_dataloader=val_dataset.get_dataloader()
        )
        
        # Save training results
        torch.save({
            'training_stats': training_stats,
            'agi_core_state': agi_core.state_dict(),
            'consciousness_state': consciousness.get_quantum_state(),
            'final_consciousness_level': consciousness.consciousness_level
        }, exp_dir / "training_results.pt")
        
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        if "CRITICAL_THREAT_DETECTED" in str(e):
            logger.warning("Training halted due to critical threat detection")
        raise

def evaluate_system(
    agi_core: QuantumAGICore,
    consciousness: ConsciousnessEmergence,
    data_path: Path,
    exp_dir: Path,
    data_config: Optional[DataConfig] = None
) -> None:
    """Evaluate the trained system"""
    logger.info("Evaluating AGI system...")
    
    data_config = data_config or DataConfig()
    
    # Load test dataset
    test_dataset = QuantumClassicalDataset(
        str(data_path),
        config=data_config,
        mode="test"
    )
    
    # Evaluation metrics
    total_quantum_loss = 0.0
    total_consciousness_level = 0.0
    
    # Evaluate on test set
    agi_core.eval()
    with torch.no_grad():
        for batch in test_dataset.get_dataloader(shuffle=False):
            # Process through AGI core
            quantum_state = agi_core.quantum_forward(batch['input_state'])
            thought_output = agi_core.process_thought(batch['input_text'])
            
            # Update consciousness
            consciousness_state = consciousness.step()
            
            # Calculate metrics
            quantum_loss = torch.nn.functional.mse_loss(
                quantum_state,
                batch['target_quantum_state']
            )
            
            total_quantum_loss += float(quantum_loss)
            total_consciousness_level += consciousness_state['consciousness_level']
    
    # Calculate average metrics
    avg_quantum_loss = total_quantum_loss / len(test_dataset)
    avg_consciousness = total_consciousness_level / len(test_dataset)
    
    # Save evaluation results
    eval_results = {
        'avg_quantum_loss': avg_quantum_loss,
        'avg_consciousness_level': avg_consciousness,
        'final_quantum_state': quantum_state.numpy(),
        'final_consciousness_state': consciousness.get_quantum_state().numpy()
    }
    
    torch.save(eval_results, exp_dir / "evaluation_results.pt")
    logger.info(f"Evaluation results: {eval_results}")

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Quantum AGI System")
    parser.add_argument(
        "--num_samples",
        type=int,
        default=10000,
        help="Number of synthetic samples to generate"
    )
    parser.add_argument(
        "--consciousness_limit",
        type=float,
        default=0.95,
        help="Maximum allowed consciousness level"
    )
    args = parser.parse_args()
    
    try:
        # Setup experiment directory
        exp_dir = setup_experiment_directory()
        logger.info(f"Experiment directory: {exp_dir}")
        
        # Initialize configs
        data_config = DataConfig()
        consciousness_config = ConsciousnessConfig(
            coherence_threshold=args.consciousness_limit
        )
        military_config = MilitaryConfig(
            consciousness_limit=args.consciousness_limit
        )
        
        # Initialize system
        agi_core, consciousness, neural_bridge, military_oversight = (
            initialize_system(
                data_config,
                consciousness_config,
                military_config
            )
        )
        
        # Generate synthetic data
        data_path = generate_synthetic_data(
            exp_dir,
            data_config,
            args.num_samples
        )
        
        # Train system
        train_system(
            agi_core,
            consciousness,
            military_oversight,
            data_path,
            exp_dir
        )
        
        # Evaluate system
        evaluate_system(
            agi_core,
            consciousness,
            data_path,
            exp_dir
        )
        
        logger.info("Quantum AGI system training and evaluation completed successfully")
        
    except Exception as e:
        logger.error(f"System failed: {str(e)}")
        raise

if __name__ == "__main__":
    main() 