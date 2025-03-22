import pandas as pd
import numpy as np
import os
import json
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Union

@dataclass
class TrainingResultDTO:
    """
    Data Transfer Object (DTO) representing the results of a model training run.
    
    This object encapsulates all data produced during a training run, including:
    - Training metrics and history
    - Evaluation results
    - Model information and paths
    
    This allows for clean separation between the training logic and storage management.
    """
    # Basic information
    ticker: str
    run_id: Optional[str] = None
    
    # Training history and metrics
    history: Dict[str, List[float]] = field(default_factory=dict)
    metrics: Dict[str, Any] = field(default_factory=dict)
    baseline_metrics: Dict[str, Any] = field(default_factory=dict)
    
    # Evaluation results
    evaluation_df: Optional[pd.DataFrame] = None
    
    # Paths and references
    model_dir: Optional[str] = None
    run_dir: Optional[str] = None
    best_model_path: Optional[str] = None
    
    # TensorFlow model instance (transient, not serialized)
    model: Any = field(default=None)  # This will be a TensorFlow model
    
    def get_train_loss(self) -> List[float]:
        """Get the training loss history."""
        return self.history.get('loss', [])
    
    def get_val_loss(self) -> List[float]:
        """Get the validation loss history."""
        return self.history.get('val_loss', [])
    
    def get_final_train_loss(self) -> Optional[float]:
        """Get the final training loss."""
        return self.metrics.get('final_train_loss')
    
    def get_final_val_loss(self) -> Optional[float]:
        """Get the final validation loss."""
        return self.metrics.get('final_val_loss')
    
    def get_best_val_loss(self) -> Optional[float]:
        """Get the best validation loss."""
        return self.metrics.get('best_val_loss')
    
    def get_mse(self) -> Optional[float]:
        """Get the MSE from evaluation metrics if available."""
        if 'evaluation' in self.metrics:
            return self.metrics['evaluation'].get('Overall_MSE')
        return None
    
    def get_best_epoch(self) -> Optional[int]:
        """Get the best epoch index."""
        return self.metrics.get('best_epoch')
        
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TrainingResultDTO':
        """
        Create a TrainingResultDTO instance from a dictionary.
        
        Args:
            data: Dictionary containing the DTO attributes
            
        Returns:
            A new TrainingResultDTO instance
        """
        # Create a copy to avoid modifying the input dictionary
        data_copy = data.copy()
        
        # Handle special case for evaluation_df which needs to be converted from dict to DataFrame
        if 'evaluation_df' in data_copy and data_copy['evaluation_df'] is not None:
            if isinstance(data_copy['evaluation_df'], dict):
                data_copy['evaluation_df'] = pd.DataFrame(data_copy['evaluation_df'])
        
        # Remove model from dictionary since it's not serializable
        if 'model' in data_copy:
            del data_copy['model']
            
        # Convert dictionary to TrainingResultDTO
        return cls(**data_copy)