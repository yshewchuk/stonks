from typing import Dict, Any, Optional, List, Set
import json
import uuid
import copy
from datetime import datetime
from utils.logger import log_info, log_warning, log_error, log_debug

class Specimen:
    """
    Represents a model configuration (individual) in the genetic algorithm.
    
    A specimen contains:
    - Genetics (model parameters)
    - Fitness metrics
    - Results from training
    - Lineage information
    """
    
    def __init__(
        self,
        model_parameters: Dict[str, Any],
        training_parameters: Dict[str, Any],
        feature_indexes: Set[int],
        generation: int = 0,
        parent_ids: Optional[List[str]] = None,
        specimen_id: Optional[str] = None,
        model_id: Optional[str] = None
    ):
        """
        Initialize a new specimen.
        
        Args:
            model_parameters: Dictionary of model hyperparameters
            training_parameters: Dictionary of training parameters
            feature_indexes: Set of feature indexes to use
            generation: Generation number in the evolution process
            parent_ids: List of parent specimen IDs (for crossover)
            specimen_id: Unique identifier (generated if not provided)
            model_id: Model identifier encoding the genetic parameters
        """
        # Core genetics
        self.model_parameters = model_parameters
        self.training_parameters = training_parameters
        self.feature_indexes = feature_indexes
        self.genetic_model_id = model_id  # Store the model ID for genetic operations
        
        # Evolutionary metadata
        self.generation = generation
        self.parent_ids = parent_ids or []
        self.specimen_id = specimen_id or f"specimen_{uuid.uuid4().hex[:8]}"
        self.created_at = datetime.now().isoformat()
        
        # Training results (filled after evaluation)
        self.model_id = None  # This is the trained model ID
        self.best_mse = None
        self.is_evaluated = False
        self.training_results = {}
        self.training_time = None
        self.fitness_score = None  # Lower is better for MSE
        self.captured_results = 0
        self.average_best_mse = 0
    
    def set_evaluation_results(
        self,
        model_id: str,
        best_mse: float,
        training_results: Dict[str, Any],
        training_time: float
    ):
        """
        Set the evaluation results for this specimen.
        
        Args:
            model_id: The model identifier from training
            best_mse: The best MSE achieved during training
            training_results: Full training results dictionary
            training_time: Time taken for training in seconds
        """
        self.model_id = model_id
        self.best_mse = best_mse if self.best_mse is None else min(self.best_mse, best_mse)
        self.training_results = training_results
        self.training_time = training_time
        self.average_best_mse = (self.average_best_mse * self.captured_results + best_mse) / (self.captured_results + 1)
        self.captured_results = self.captured_results + 1
        self.is_evaluated = self.captured_results >= 3
        
        # Set fitness score (we use negative MSE since we maximize fitness)
        self.fitness_score = -best_mse if best_mse is not None else float('-inf')
    
    def clone(self) -> 'Specimen':
        """
        Create a deep copy of this specimen.
        
        Returns:
            A new Specimen instance with the same properties
        """
        new_specimen = Specimen(
            model_parameters=copy.deepcopy(self.model_parameters),
            training_parameters=copy.deepcopy(self.training_parameters),
            feature_indexes=copy.deepcopy(self.feature_indexes),
            generation=self.generation,
            parent_ids=self.parent_ids.copy() if self.parent_ids else None,
            specimen_id=f"clone_{self.specimen_id}",
            model_id=self.genetic_model_id
        )
        
        # Copy evaluation results if present
        if self.is_evaluated:
            new_specimen.set_evaluation_results(
                model_id=self.model_id,
                best_mse=self.best_mse,
                training_results=copy.deepcopy(self.training_results),
                training_time=self.training_time
            )
        
        return new_specimen
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the specimen to a dictionary for serialization.
        
        Returns:
            Dictionary representation of the specimen
        """
        return {
            "specimen_id": self.specimen_id,
            "generation": self.generation,
            "parent_ids": self.parent_ids,
            "created_at": self.created_at,
            "model_parameters": self.model_parameters,
            "training_parameters": self.training_parameters,
            "feature_indexes": list(self.feature_indexes),
            "genetic_model_id": self.genetic_model_id,
            "model_id": self.model_id,
            "best_mse": self.best_mse,
            "is_evaluated": self.is_evaluated,
            "training_results": self.training_results,
            "training_time": self.training_time,
            "fitness_score": self.fitness_score
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Specimen':
        """
        Create a specimen from a dictionary.
        
        Args:
            data: Dictionary containing specimen data
            
        Returns:
            A new Specimen instance
        """
        # Extract the core parameters
        specimen = cls(
            model_parameters=data["model_parameters"],
            training_parameters=data["training_parameters"],
            feature_indexes=set(data["feature_indexes"]),
            generation=data["generation"],
            parent_ids=data["parent_ids"],
            specimen_id=data["specimen_id"],
            model_id=data.get("genetic_model_id")
        )
        
        # Set additional properties
        specimen.created_at = data["created_at"]
        
        # Set evaluation results if the specimen was evaluated
        if data["is_evaluated"]:
            specimen.set_evaluation_results(
                model_id=data["model_id"],
                best_mse=data["best_mse"],
                training_results=data["training_results"],
                training_time=data["training_time"]
            )
        
        return specimen
    
    def __str__(self) -> str:
        """String representation of the specimen."""
        fitness_str = f"{self.fitness_score:.6f}" if self.fitness_score is not None else "Not evaluated"
        return f"Specimen {self.specimen_id} (Gen {self.generation}): Fitness={fitness_str}"
    
    def __repr__(self) -> str:
        """Detailed representation of the specimen."""
        return self.__str__() 