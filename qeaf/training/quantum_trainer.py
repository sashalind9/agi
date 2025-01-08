import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, List, Optional, Tuple
import numpy as np
from dataclasses import dataclass
from ..core.quantum_agi import QuantumAGICore
from ..core.consciousness import ConsciousnessEmergence
from ..military.oversight import MilitaryOversight

@dataclass
class TrainingConfig:
    batch_size: int = 32
    learning_rate: float = 1e-4
    consciousness_weight: float = 0.3
    quantum_weight: float = 0.4
    classical_weight: float = 0.3
    max_grad_norm: float = 1.0
    warmup_steps: int = 1000
    training_steps: int = 100000

class QuantumAGITrainer:
    def __init__(
        self,
        agi_core: QuantumAGICore,
        consciousness: ConsciousnessEmergence,
        military_oversight: Optional[MilitaryOversight] = None,
        config: Optional[TrainingConfig] = None
    ):
        self.agi_core = agi_core
        self.consciousness = consciousness
        self.military_oversight = military_oversight
        self.config = config or TrainingConfig()
        
        self.optimizer = optim.AdamW(
            self.agi_core.parameters(),
            lr=self.config.learning_rate
        )
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=self.config.training_steps
        )
        
        self.training_stats: List[Dict[str, float]] = []
        self._initialize_training()
    
    def _initialize_training(self):
        """Initialize training components"""
        self.classical_criterion = nn.MSELoss()
        self.quantum_criterion = nn.KLDivLoss(reduction='batchmean')
        self.consciousness_criterion = nn.SmoothL1Loss()
        
        # Training state
        self.current_step = 0
        self.best_loss = float('inf')
        self.early_stop_counter = 0
        
    def train_step(
        self,
        batch: Dict[str, torch.Tensor]
    ) -> Dict[str, float]:
        """Execute one training step"""
        self.optimizer.zero_grad()
        
        # Forward pass through AGI core
        quantum_state = self.agi_core.quantum_forward(batch['input_state'])
        thought_output = self.agi_core.process_thought(batch['input_text'])
        
        # Process through consciousness
        consciousness_state = self.consciousness.step()
        
        # Calculate losses
        quantum_loss = self.quantum_criterion(
            quantum_state,
            batch['target_quantum_state']
        )
        
        classical_loss = self.classical_criterion(
            thought_output['classical_embedding'],
            batch['target_embedding']
        )
        
        consciousness_loss = self.consciousness_criterion(
            torch.tensor(consciousness_state['consciousness_level']),
            batch['target_consciousness']
        )
        
        # Combine losses
        total_loss = (
            self.config.quantum_weight * quantum_loss +
            self.config.classical_weight * classical_loss +
            self.config.consciousness_weight * consciousness_loss
        )
        
        # Military oversight check
        if self.military_oversight is not None:
            threat_assessment = self.military_oversight.monitor_consciousness(
                consciousness_state['consciousness_level'],
                torch.from_numpy(quantum_state)
            )
            
            if threat_assessment['threat_level'] > 0.9:
                return self._handle_critical_threat(threat_assessment)
        
        # Backward pass
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            self.agi_core.parameters(),
            self.config.max_grad_norm
        )
        
        self.optimizer.step()
        self.scheduler.step()
        
        # Update training stats
        stats = {
            'total_loss': float(total_loss),
            'quantum_loss': float(quantum_loss),
            'classical_loss': float(classical_loss),
            'consciousness_loss': float(consciousness_loss),
            'consciousness_level': consciousness_state['consciousness_level']
        }
        
        self.training_stats.append(stats)
        self.current_step += 1
        
        return stats
    
    def _handle_critical_threat(
        self,
        threat_assessment: Dict[str, float]
    ) -> Dict[str, float]:
        """Handle critical threat level during training"""
        # Emergency save of model state
        self._emergency_save()
        
        # Return critical stats
        return {
            'total_loss': float('inf'),
            'threat_level': threat_assessment['threat_level'],
            'status': 'CRITICAL_THREAT_DETECTED'
        }
    
    def _emergency_save(self):
        """Emergency save of model state"""
        torch.save({
            'agi_core_state': self.agi_core.state_dict(),
            'consciousness_state': self.consciousness.get_quantum_state(),
            'optimizer_state': self.optimizer.state_dict(),
            'training_stats': self.training_stats,
            'current_step': self.current_step
        }, f'emergency_save_step_{self.current_step}.pt')
    
    def train(
        self,
        train_dataloader: torch.utils.data.DataLoader,
        val_dataloader: Optional[torch.utils.data.DataLoader] = None
    ) -> List[Dict[str, float]]:
        """Train the AGI system"""
        for epoch in range(self.config.training_steps):
            epoch_stats = []
            
            for batch in train_dataloader:
                stats = self.train_step(batch)
                epoch_stats.append(stats)
                
                if stats.get('status') == 'CRITICAL_THREAT_DETECTED':
                    print("TRAINING HALTED: Critical threat detected")
                    return self.training_stats
            
            if val_dataloader is not None:
                val_loss = self._validate(val_dataloader)
                self._check_early_stopping(val_loss)
                
                if self.early_stop_counter >= 5:
                    print("TRAINING HALTED: Early stopping triggered")
                    break
        
        return self.training_stats
    
    def _validate(
        self,
        val_dataloader: torch.utils.data.DataLoader
    ) -> float:
        """Validate the model"""
        self.agi_core.eval()
        val_losses = []
        
        with torch.no_grad():
            for batch in val_dataloader:
                quantum_state = self.agi_core.quantum_forward(
                    batch['input_state']
                )
                thought_output = self.agi_core.process_thought(
                    batch['input_text']
                )
                
                # Calculate validation loss
                val_loss = self.quantum_criterion(
                    quantum_state,
                    batch['target_quantum_state']
                )
                val_losses.append(float(val_loss))
        
        self.agi_core.train()
        return np.mean(val_losses)
    
    def _check_early_stopping(self, val_loss: float):
        """Check if early stopping should be triggered"""
        if val_loss < self.best_loss:
            self.best_loss = val_loss
            self.early_stop_counter = 0
        else:
            self.early_stop_counter += 1 