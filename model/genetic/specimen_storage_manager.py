import os
import json
import csv
from typing import Dict, List, Any, Optional, Tuple, Set
import pandas as pd
from datetime import datetime
import glob
import shutil

from .specimen import Specimen
from utils.logger import log_info, log_warning, log_error, log_debug, log_success

class SpecimenStorageManager:
    """
    Manages the storage and retrieval of specimens and evolution progress.
    
    This class is responsible for:
    - Saving specimens to disk
    - Loading specimens from disk
    - Tracking the best specimens across generations
    - Saving evolution progress and statistics
    """
    
    def __init__(
        self,
        base_dir: str,
        experiment_name: Optional[str] = None,
        ticker: Optional[str] = None
    ):
        """
        Initialize the SpecimenStorageManager.
        
        Args:
            base_dir: Base directory for storing evolution data
            experiment_name: Name of the evolutionary experiment
            ticker: Ticker symbol for this experiment
        """
        self.base_dir = base_dir
        self.ticker = ticker
        
        # Create a unique experiment name if not provided
        if experiment_name:
            self.experiment_name = experiment_name
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            ticker_str = f"{ticker}_" if ticker else ""
            self.experiment_name = f"{ticker_str}evolution_{timestamp}"
        
        # Create the experiment directory
        self.experiment_dir = os.path.join(base_dir, self.experiment_name)
        os.makedirs(self.experiment_dir, exist_ok=True)
        
        # Create subdirectories
        self.specimens_dir = os.path.join(self.experiment_dir, "specimens")
        self.generations_dir = os.path.join(self.experiment_dir, "generations")
        self.stats_dir = os.path.join(self.experiment_dir, "stats")
        
        os.makedirs(self.specimens_dir, exist_ok=True)
        os.makedirs(self.generations_dir, exist_ok=True)
        os.makedirs(self.stats_dir, exist_ok=True)
        
        # Track best specimens
        self.best_specimen = None
        self.best_specimens_by_generation = {}
        
        # Create experiment log
        self.experiment_log_path = os.path.join(self.experiment_dir, "experiment_log.csv")
        if not os.path.exists(self.experiment_log_path):
            with open(self.experiment_log_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    "Generation", "Timestamp", "Best_Specimen_ID", "Best_MSE", 
                    "Avg_MSE", "Min_MSE", "Population_Size"
                ])
        
        log_info(f"SpecimenStorageManager initialized with experiment dir: {self.experiment_dir}")
    
    def save_specimen(self, specimen: Specimen) -> str:
        """
        Save a specimen to disk.
        
        Args:
            specimen: The specimen to save
            
        Returns:
            Path to the saved specimen file
        """
        # Create a filename based on specimen ID and generation
        filename = f"{specimen.specimen_id}_gen{specimen.generation}.json"
        specimen_path = os.path.join(self.specimens_dir, filename)
        
        # Convert to dictionary and save as JSON
        specimen_dict = specimen.to_dict()
        try:
            with open(specimen_path, 'w') as f:
                json.dump(specimen_dict, f, indent=2)
            
            # Update best specimen if applicable
            if self._is_better_specimen(specimen):
                self.best_specimen = specimen
                log_info(f"New best specimen: {specimen.specimen_id} with MSE={specimen.best_mse}")
            
            # Update best specimen for this generation
            if specimen.generation not in self.best_specimens_by_generation or \
               self._is_better_than(specimen, self.best_specimens_by_generation[specimen.generation]):
                self.best_specimens_by_generation[specimen.generation] = specimen
            
            return specimen_path
        except Exception as e:
            log_error(f"Error saving specimen {specimen.specimen_id}: {e}")
            return None
    
    def load_specimen(self, specimen_path: str) -> Optional[Specimen]:
        """
        Load a specimen from disk.
        
        Args:
            specimen_path: Path to the specimen file
            
        Returns:
            The loaded specimen or None if loading fails
        """
        try:
            with open(specimen_path, 'r') as f:
                specimen_dict = json.load(f)
            
            return Specimen.from_dict(specimen_dict)
        except Exception as e:
            log_error(f"Error loading specimen from {specimen_path}: {e}")
            return None
    
    def save_generation(self, generation: int, specimens: List[Specimen]) -> str:
        """
        Save a complete generation of specimens.
        
        Args:
            generation: Generation number
            specimens: List of specimens in this generation
            
        Returns:
            Path to the generation directory
        """
        # Create a directory for this generation
        generation_dir = os.path.join(self.generations_dir, f"generation_{generation}")
        os.makedirs(generation_dir, exist_ok=True)
        
        # Save each specimen
        for specimen in specimens:
            if specimen.generation != generation:
                log_warning(f"Specimen {specimen.specimen_id} has generation {specimen.generation} "
                           f"but is being saved in generation {generation}")
            
            self.save_specimen(specimen)
            
            # Also save a copy in the generation directory
            specimen_dict = specimen.to_dict()
            specimen_path = os.path.join(generation_dir, f"{specimen.specimen_id}.json")
            
            with open(specimen_path, 'w') as f:
                json.dump(specimen_dict, f, indent=2)
        
        # Create a summary of this generation
        evaluated_specimens = [s for s in specimens if s.is_evaluated]
        if evaluated_specimens:
            mse_values = [s.best_mse for s in evaluated_specimens if s.best_mse is not None]
            avg_mse = sum(mse_values) / len(mse_values) if mse_values else None
            min_mse = min(mse_values) if mse_values else None
            best_specimen = min(evaluated_specimens, key=lambda s: s.best_mse if s.best_mse is not None else float('inf'))
            
            # Update the experiment log
            with open(self.experiment_log_path, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    generation,
                    datetime.now().isoformat(),
                    best_specimen.specimen_id,
                    best_specimen.best_mse,
                    avg_mse,
                    min_mse,
                    len(specimens)
                ])
            
            # Create a generation summary
            summary = {
                "generation": generation,
                "timestamp": datetime.now().isoformat(),
                "population_size": len(specimens),
                "evaluated_specimens": len(evaluated_specimens),
                "avg_mse": avg_mse,
                "min_mse": min_mse,
                "best_specimen_id": best_specimen.specimen_id,
                "specimen_ids": [s.specimen_id for s in specimens]
            }
            
            summary_path = os.path.join(generation_dir, "summary.json")
            with open(summary_path, 'w') as f:
                json.dump(summary, f, indent=2)
                
            # Create a detailed CSV with all specimens
            specimens_df = pd.DataFrame([
                {
                    "specimen_id": s.specimen_id,
                    "best_mse": s.best_mse if s.is_evaluated else None,
                    "fitness_score": s.fitness_score,
                    "is_evaluated": s.is_evaluated,
                    "training_time": s.training_time,
                    "model_id": s.model_id,
                    "parent_ids": ','.join(s.parent_ids) if s.parent_ids else "",
                    "cnn_layers": s.model_parameters.get("cnn_layers"),
                    "lstm_layers": s.model_parameters.get("lstm_layers"),
                    "lstm_units": s.model_parameters.get("lstm_units"),
                    "dropout_rate": s.model_parameters.get("dropout_rate"),
                    "l2_reg": s.model_parameters.get("l2_reg"),
                    "learning_rate": s.model_parameters.get("learning_rate")
                }
                for s in specimens
            ])
            
            specimens_df.to_csv(os.path.join(generation_dir, "specimens.csv"), index=False)
        
        log_info(f"Saved generation {generation} with {len(specimens)} specimens to {generation_dir}")
        return generation_dir
    
    def load_generation(self, generation: int) -> List[Specimen]:
        """
        Load a complete generation of specimens.
        
        Args:
            generation: Generation number to load
            
        Returns:
            List of specimens in this generation
        """
        generation_dir = os.path.join(self.generations_dir, f"generation_{generation}")
        if not os.path.exists(generation_dir):
            log_error(f"Generation directory does not exist: {generation_dir}")
            return []
        
        specimens = []
        for filename in os.listdir(generation_dir):
            if filename.endswith('.json') and filename != "summary.json":
                specimen_path = os.path.join(generation_dir, filename)
                specimen = self.load_specimen(specimen_path)
                if specimen:
                    specimens.append(specimen)
        
        log_info(f"Loaded {len(specimens)} specimens from generation {generation}")
        return specimens
    
    def get_best_specimen(self) -> Optional[Specimen]:
        """
        Get the best specimen across all generations.
        
        Returns:
            The best specimen or None if no evaluated specimens
        """
        # If we've been tracking the best specimen, return it
        if self.best_specimen:
            return self.best_specimen
        
        # Otherwise, scan all specimen files to find the best one
        best_specimen = None
        best_mse = float('inf')
        
        for path in glob.glob(os.path.join(self.specimens_dir, "*.json")):
            specimen = self.load_specimen(path)
            if specimen and specimen.is_evaluated and specimen.best_mse is not None:
                if specimen.best_mse < best_mse:
                    best_specimen = specimen
                    best_mse = specimen.best_mse
        
        self.best_specimen = best_specimen
        return best_specimen
    
    def get_best_specimen_for_generation(self, generation: int) -> Optional[Specimen]:
        """
        Get the best specimen for a specific generation.
        
        Args:
            generation: Generation number
            
        Returns:
            The best specimen in that generation or None if no evaluated specimens
        """
        # If we've been tracking the best specimens by generation, check cache
        if generation in self.best_specimens_by_generation:
            return self.best_specimens_by_generation[generation]
        
        # Otherwise, load the generation and find the best specimen
        specimens = self.load_generation(generation)
        evaluated_specimens = [s for s in specimens if s.is_evaluated and s.best_mse is not None]
        
        if not evaluated_specimens:
            return None
            
        best_specimen = min(evaluated_specimens, key=lambda s: s.best_mse)
        self.best_specimens_by_generation[generation] = best_specimen
        return best_specimen
    
    def save_evolution_stats(self, stats_data: Dict[str, Any]) -> str:
        """
        Save statistics about the evolution process.
        
        Args:
            stats_data: Dictionary containing statistics
            
        Returns:
            Path to the stats file
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        stats_path = os.path.join(self.stats_dir, f"stats_{timestamp}.json")
        
        # Add timestamp to stats
        stats_data["timestamp"] = datetime.now().isoformat()
        
        with open(stats_path, 'w') as f:
            json.dump(stats_data, f, indent=2)
            
        return stats_path
    
    def create_experiment_summary(self) -> Dict[str, Any]:
        """
        Create a summary of the entire experiment.
        
        Returns:
            Dictionary with experiment summary
        """
        # Find all generation directories
        generation_dirs = glob.glob(os.path.join(self.generations_dir, "generation_*"))
        generations = sorted([int(os.path.basename(d).split('_')[1]) for d in generation_dirs])
        
        # Get best specimen overall
        best_specimen = self.get_best_specimen()
        
        # Get best specimens by generation
        best_by_gen = {}
        for gen in generations:
            best_in_gen = self.get_best_specimen_for_generation(gen)
            if best_in_gen:
                best_by_gen[gen] = {
                    "specimen_id": best_in_gen.specimen_id,
                    "best_mse": best_in_gen.best_mse,
                    "model_id": best_in_gen.model_id
                }
        
        # Create dataframe from experiment log
        experiment_log = None
        if os.path.exists(self.experiment_log_path):
            experiment_log = pd.read_csv(self.experiment_log_path)
            
            # Create a summary plot of MSE by generation
            try:
                import matplotlib.pyplot as plt
                plt.figure(figsize=(12, 6))
                plt.plot(experiment_log["Generation"], experiment_log["Min_MSE"], marker='o', label="Best MSE")
                plt.plot(experiment_log["Generation"], experiment_log["Avg_MSE"], marker='x', label="Average MSE")
                plt.title(f"Evolution Progress - {self.experiment_name}")
                plt.xlabel("Generation")
                plt.ylabel("MSE (lower is better)")
                plt.legend()
                plt.grid(True)
                
                plot_path = os.path.join(self.experiment_dir, "evolution_progress.png")
                plt.savefig(plot_path)
                plt.close()
            except Exception as e:
                log_error(f"Error creating evolution progress plot: {e}")
        
        # Compile summary
        summary = {
            "experiment_name": self.experiment_name,
            "ticker": self.ticker,
            "experiment_dir": self.experiment_dir,
            "generations": generations,
            "total_generations": len(generations),
            "best_specimen": best_specimen.to_dict() if best_specimen else None,
            "best_by_generation": best_by_gen,
            "creation_time": datetime.now().isoformat()
        }
        
        # Save summary
        summary_path = os.path.join(self.experiment_dir, "experiment_summary.json")
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
            
        return summary
    
    def _is_better_specimen(self, specimen: Specimen) -> bool:
        """
        Check if a specimen is better than the current best.
        
        Args:
            specimen: Specimen to check
            
        Returns:
            True if the specimen is better
        """
        if not specimen.is_evaluated or specimen.best_mse is None:
            return False
            
        if not self.best_specimen or not self.best_specimen.is_evaluated:
            return True
            
        return specimen.best_mse < self.best_specimen.best_mse
    
    def _is_better_than(self, specimen1: Specimen, specimen2: Specimen) -> bool:
        """
        Check if specimen1 is better than specimen2.
        
        Args:
            specimen1: First specimen
            specimen2: Second specimen
            
        Returns:
            True if specimen1 is better than specimen2
        """
        if not specimen1.is_evaluated or specimen1.best_mse is None:
            return False
            
        if not specimen2.is_evaluated or specimen2.best_mse is None:
            return True
            
        return specimen1.best_mse < specimen2.best_mse 