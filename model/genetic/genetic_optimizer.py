import os
import random
import numpy as np
import time
import concurrent.futures
from typing import List, Dict, Any, Optional, Set, Tuple, Callable
from datetime import datetime

# Set matplotlib backend to non-interactive to avoid tkinter issues in multithreading
import matplotlib
matplotlib.use('Agg')  # Must be before any other matplotlib imports

from model.model_definition import ModelDefinition
from model.model_training_manager import ModelTrainingManager
from model.model_identifier import ModelIdentifier
from model.model_identifier.bit_array import BitArray
from model.model_data import ModelData
from .specimen import Specimen
from .specimen_storage_manager import SpecimenStorageManager

# Import logger utility functions
from utils.logger import log_info, log_warning, log_error, log_debug, log_success

class GeneticOptimizer:
    """
    Implements a genetic algorithm for hyperparameter optimization of price prediction models.
    
    This class handles:
    - Creation of the initial population
    - Training and evaluation of specimens using ModelTrainingManager
    - Selection of the fittest specimens
    - Breeding new generations through crossover and mutation
    - Tracking evolution progress and identifying the best model
    """
    
    def __init__(
        self,
        ticker: str,
        train_data: List[ModelData],
        eval_data: List[ModelData],
        output_dir: str = "data/genetic",
        experiment_name: Optional[str] = None,
        population_size: int = 20,
        generations: int = 10,
        mutation_rate: float = 0.3,
        crossover_rate: float = 0.7,
        elitism_count: int = 2,
        tournament_size: int = 3,
        epochs: int = 500,
        early_stopping_patience: int = 20,
        max_workers: int = 1,
        logger = None
    ):
        """
        Initialize the GeneticOptimizer.
        
        Args:
            ticker: Ticker symbol for this optimization
            train_data: List of ModelData objects for training
            eval_data: List of ModelData objects for evaluation
            output_dir: Base directory for storing evolution data
            experiment_name: Name of the evolutionary experiment (auto-generated if None)
            population_size: Number of specimens per generation
            generations: Number of generations to evolve
            mutation_rate: Probability of mutation for each parameter (0.0-1.0)
            crossover_rate: Probability of crossover for breeding (0.0-1.0)
            elitism_count: Number of best specimens to carry forward unchanged
            tournament_size: Number of specimens in each selection tournament
            epochs: Maximum number of training epochs
            early_stopping_patience: Number of epochs with no improvement before stopping
            max_workers: Maximum number of concurrent training processes
            logger: Logger instance (optional, not used with new logging system)
        """
        self.ticker = ticker
        self.train_data = train_data
        self.eval_data = eval_data
        self.output_dir = output_dir
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.elitism_count = elitism_count
        self.tournament_size = tournament_size
        self.epochs = epochs
        self.early_stopping_patience = early_stopping_patience
        self.max_workers = max_workers
        
        # Initialize storage manager for tracking evolution
        self.storage_manager = SpecimenStorageManager(
            base_dir=output_dir,
            experiment_name=experiment_name,
            ticker=ticker
        )
        
        # Initialize current population and generation
        self.current_generation = 0
        self.current_population = []
        self.best_specimen = None
        
        # Parameter bounds and options
        self._initialize_parameter_space()
        
        # Initialize model identifier
        self.model_identifier = ModelIdentifier()
    
    def _initialize_parameter_space(self) -> None:
        """Initialize the parameter space with valid options for genetic operations."""
        # Get options from ModelDefinition
        self.model_param_options = ModelDefinition.MODEL_PARAMETER_OPTIONS
        self.training_param_options = ModelDefinition.TRAINING_PARAMETER_OPTIONS
        
        # Define continuous parameter ranges (min, max) for parameters not in discrete options
        self.continuous_params = {
            'dropout_rate': (0.0, 0.7),
            'recurrent_dropout_rate': (0.0, 0.5),
            'l2_reg': (0.0, 0.1)
        }
        
        # Define default values
        self.default_model_params = ModelDefinition.get_default_model_parameters()
        self.default_training_params = ModelDefinition.get_default_training_parameters()
    
    def _generate_random_model_id(self) -> str:
        """
        Generate a random model identifier by creating random parameters 
        and encoding them using ModelIdentifier.
        Ensures the generated model ID is valid.
        
        Returns:
            A model identifier string
        """
        # Try up to 5 times to create a valid model ID
        for attempt in range(5):
            try:
                # Generate random model parameters
                model_params = self._generate_random_model_parameters()
                
                # Generate random training parameters
                training_params = self._generate_random_training_parameters()
                
                # Generate random feature subset
                total_features = ModelDefinition.TOTAL_FEATURE_COUNT
                # Select between 1% and 80% of features randomly
                min_features = max(1, int(total_features * 0.01))
                max_features = int(total_features * 0.8)
                num_features = random.randint(min_features, max_features)
                feature_indexes = set(random.sample(range(total_features), num_features))
                
                # Create the model identifier
                model_id = self.model_identifier.create_model_identifier(
                    model_parameters=model_params,
                    training_parameters=training_params,
                    selected_feature_indexes=feature_indexes
                )
                
                # Validate by attempting to decode it
                self.model_identifier.decode_model_identifier(model_id)
                
                log_debug(f"Generated model ID with {len(feature_indexes)} features (attempt {attempt+1})")
                return model_id
                
            except ValueError as e:
                log_warning(f"Failed to generate valid model ID (attempt {attempt+1}): {e}")
        
        # If all attempts failed, use a simple fallback approach with defaults
        log_warning("All random model ID generation attempts failed, using fallback approach")
        model_params = ModelDefinition.get_default_model_parameters()
        training_params = ModelDefinition.get_default_training_parameters()
        # Use a small, safe set of features
        feature_indexes = set(range(10))  # Just use the first 10 features
        
        model_id = self.model_identifier.create_model_identifier(
            model_parameters=model_params,
            training_parameters=training_params,
            selected_feature_indexes=feature_indexes
        )
        
        return model_id
    
    def create_initial_population(self) -> List[Specimen]:
        """
        Create the initial population of specimens.
        
        Returns:
            List of randomly generated specimens
        """
        log_info(f"Creating initial population of {self.population_size} specimens for {self.ticker}")
        population = []
        
        for i in range(self.population_size):
            # Generate a random model ID
            model_id = self._generate_random_model_id()
            
            # Decode the model ID to get parameters
            model_config = self.model_identifier.decode_model_identifier(model_id)
            
            # Create the specimen
            specimen = Specimen(
                model_parameters=model_config['model_parameters'],
                training_parameters=model_config['training_parameters'],
                feature_indexes=model_config['feature_indexes'],
                generation=0,
                model_id=model_id  # Store the model ID for genetic operations
            )
            
            population.append(specimen)
            log_info(f"Created specimen {i+1}/{self.population_size}: {specimen.specimen_id}, features: {len(specimen.feature_indexes)}")
        
        self.current_population = population
        return population
    
    def _generate_random_model_parameters(self) -> Dict[str, Any]:
        """Generate random model parameters within allowed values."""
        model_params = {}
        
        # Add required parameters from options
        for param, options in self.model_param_options.items():
            if options:  # Only if we have options
                model_params[param] = random.choice(options)
        
        # Handle continuous parameters
        for param, (min_val, max_val) in self.continuous_params.items():
            if param not in model_params:
                model_params[param] = random.uniform(min_val, max_val)
        
        # Ensure required parameters are included
        for param, default_value in self.default_model_params.items():
            if param not in model_params:
                model_params[param] = default_value
        
        return model_params
    
    def _generate_random_training_parameters(self) -> Dict[str, Any]:
        """Generate random training parameters within allowed values."""
        training_params = {}
        
        # Add parameters from options
        for param, options in self.training_param_options.items():
            if options:  # Only if we have options
                training_params[param] = random.choice(options)
        
        # Ensure required parameters are included
        for param, default_value in self.default_training_params.items():
            if param not in training_params:
                training_params[param] = default_value
        
        return training_params
    
    def evaluate_population(self, population: List[Specimen]) -> List[Specimen]:
        """
        Evaluate a population of specimens by training models.
        
        Args:
            population: List of specimens to evaluate
            
        Returns:
            The evaluated population
        """
        if not population:
            log_warning("Empty population, nothing to evaluate")
            return []
            
        log_info(f"Evaluating {len(population)} specimens for generation {self.current_generation}")
        
        # Filter out already evaluated specimens
        unevaluated = [s for s in population if not s.is_evaluated]
        evaluated = [s for s in population if s.is_evaluated]
        
        if unevaluated:
            log_info(f"Training {len(unevaluated)} unevaluated specimens...")
            
            # Evaluate specimens in parallel or sequentially
            if self.max_workers > 1:
                evaluated_specimens = self._evaluate_population_parallel(unevaluated)
            else:
                evaluated_specimens = self._evaluate_population_sequential(unevaluated)
            
            # Combine with already evaluated specimens
            evaluated.extend(evaluated_specimens)
        else:
            log_info("All specimens already evaluated")
        
        # Update current population
        self.current_population = evaluated
        
        # Save this generation
        self.storage_manager.save_generation(self.current_generation, evaluated)
        
        # Track best specimen
        evaluated_specimens = [s for s in evaluated if s.captured_results > 0 and s.best_mse is not None]
        if evaluated_specimens:
            best_in_gen = min(evaluated_specimens, key=lambda s: s.best_mse)
            if not self.best_specimen or best_in_gen.best_mse < self.best_specimen.best_mse:
                self.best_specimen = best_in_gen
                log_success(f"New best specimen found: {best_in_gen.specimen_id} with MSE={best_in_gen.best_mse:.6f}")
        
        return evaluated
    
    def _evaluate_population_sequential(self, specimens: List[Specimen]) -> List[Specimen]:
        """Evaluate specimens sequentially."""
        evaluated_specimens = []
        
        for i, specimen in enumerate(specimens):
            log_info(f"Evaluating specimen {i+1}/{len(specimens)}: {specimen.specimen_id}")
            evaluated = self._evaluate_specimen(specimen)
            evaluated_specimens.append(evaluated)
            
        return evaluated_specimens
    
    def _evaluate_population_parallel(self, specimens: List[Specimen]) -> List[Specimen]:
        """Evaluate specimens in parallel using a thread pool."""
        evaluated_specimens = []
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_specimen = {
                executor.submit(self._evaluate_specimen, specimen): specimen
                for specimen in specimens
            }
            
            for i, future in enumerate(concurrent.futures.as_completed(future_to_specimen)):
                specimen = future_to_specimen[future]
                try:
                    evaluated = future.result()
                    evaluated_specimens.append(evaluated)
                    log_info(f"Completed evaluation {i+1}/{len(specimens)}: {specimen.specimen_id}")
                except Exception as e:
                    log_error(f"Error evaluating specimen {specimen.specimen_id}: {e}")
                    # Still add the specimen to avoid losing it
                    evaluated_specimens.append(specimen)
        
        return evaluated_specimens
    
    def _evaluate_specimen(self, specimen: Specimen) -> Specimen:
        """Train and evaluate a single specimen."""
        start_time = time.time()
        
        try:
            # Create a model training manager
            training_manager = ModelTrainingManager(
                ticker=self.ticker,
                output_dir=os.path.join(self.storage_manager.experiment_dir, "models"),
                epochs=self.epochs,
                early_stopping_patience=self.early_stopping_patience
            )
            
            log_debug(f"Training specimen {specimen.specimen_id} with model ID: {specimen.genetic_model_id}")
            
            # Train using model identifier
            training_results = training_manager.train_from_identifier(
                model_id=specimen.genetic_model_id,
                train_data_list=self.train_data,
                eval_data_list=self.eval_data
            )
            
            # Extract results
            if training_results['success']:
                model_id = training_results.get('model_id', '')
                best_mse = float(training_results.get('metrics', {}).get('Overall_MSE', float('inf')))
                training_time = float(training_results.get('training_time', 0))
                
                # Set evaluation results on the specimen
                specimen.set_evaluation_results(
                    model_id=model_id,
                    best_mse=best_mse,
                    training_results=training_results,
                    training_time=training_time
                )
                
                log_info(f"Specimen {specimen.specimen_id} trained successfully, MSE={best_mse:.6f}")
            else:
                error_msg = training_results.get('error', 'Unknown error')
                log_error(f"Specimen {specimen.specimen_id} training failed: {error_msg}")
        except Exception as e:
            log_error(f"Error evaluating specimen {specimen.specimen_id}: {str(e)}")
            import traceback
            traceback.print_exc()
        
        # Save the specimen even if evaluation failed
        self.storage_manager.save_specimen(specimen)
        
        return specimen
    
    def select_parents(self, population: List[Specimen], count: int) -> List[Specimen]:
        """
        Select parents for breeding using tournament selection.
        
        Args:
            population: List of specimens to select from
            count: Number of parents to select
            
        Returns:
            List of selected parent specimens
        """
        # Only consider evaluated specimens with valid fitness
        valid_specimens = [s for s in population if s.captured_results > 0 and s.fitness_score is not None]
        if not valid_specimens:
            log_warning("No valid specimens for selection")
            return []
        
        parents = []
        for _ in range(count):
            # Tournament selection
            tournament = random.sample(valid_specimens, min(self.tournament_size, len(valid_specimens)))
            winner = max(tournament, key=lambda s: s.fitness_score)
            parents.append(winner)
        
        return parents
    
    def crossover(self, parent1: Specimen, parent2: Specimen) -> Tuple[Specimen, Specimen]:
        """
        Perform crossover between two parent specimens to create two children.
        Uses the ModelIdentifier to perform bit-level crossover on the genetic material.
        Ensures that the resulting model IDs are always valid.
        
        Args:
            parent1: First parent specimen
            parent2: Second parent specimen
            
        Returns:
            Tuple of two child specimens
        """
        if random.random() > self.crossover_rate or not parent1.genetic_model_id or not parent2.genetic_model_id:
            # No crossover, just clone the parents
            child1 = parent1.clone()
            child2 = parent2.clone()
            child1.generation = self.current_generation + 1
            child2.generation = self.current_generation + 1
            return child1, child2
        
        # Try up to 5 times to create valid crossovers
        for attempt in range(5):
            try:
                # Perform crossover at the bit level using ModelIdentifier
                # Get bit arrays from the model IDs
                parent1_bits = BitArray.from_base32(parent1.genetic_model_id)
                parent2_bits = BitArray.from_base32(parent2.genetic_model_id)
                
                # Keep version bits (first 8 bits) unchanged
                version_bits = parent1_bits[:ModelIdentifier.VERSION_BITS]
                
                # Perform single-point crossover on the remaining bits
                remaining_bits1 = parent1_bits[ModelIdentifier.VERSION_BITS:]
                remaining_bits2 = parent2_bits[ModelIdentifier.VERSION_BITS:]
                
                # Choose crossover point
                crossover_point = random.randint(0, len(remaining_bits1))
                
                # Create offspring bit arrays
                child1_bits = BitArray(list(version_bits) + 
                                     list(remaining_bits1[:crossover_point]) + 
                                     list(remaining_bits2[crossover_point:]))
                
                child2_bits = BitArray(list(version_bits) + 
                                     list(remaining_bits2[:crossover_point]) + 
                                     list(remaining_bits1[crossover_point:]))
                
                # Convert bit arrays back to model IDs
                child1_id = child1_bits.to_base32()
                child2_id = child2_bits.to_base32()
                
                # Try to decode the model IDs to validate them
                child1_config = self.model_identifier.decode_model_identifier(child1_id)
                child2_config = self.model_identifier.decode_model_identifier(child2_id)
                
                # Create the child specimens
                child1 = Specimen(
                    model_parameters=child1_config['model_parameters'],
                    training_parameters=child1_config['training_parameters'],
                    feature_indexes=child1_config['feature_indexes'],
                    generation=self.current_generation + 1,
                    parent_ids=[parent1.specimen_id, parent2.specimen_id],
                    model_id=child1_id
                )
                
                child2 = Specimen(
                    model_parameters=child2_config['model_parameters'],
                    training_parameters=child2_config['training_parameters'],
                    feature_indexes=child2_config['feature_indexes'],
                    generation=self.current_generation + 1,
                    parent_ids=[parent1.specimen_id, parent2.specimen_id],
                    model_id=child2_id
                )
                
                # If we got here, crossover was successful
                log_debug(f"Successfully performed crossover (attempt {attempt+1}), child1: {child1.genetic_model_id}, child2: {child2.genetic_model_id}, parent1: {parent1.genetic_model_id}, parent2: {parent2.genetic_model_id}")
                return child1, child2
                
            except ValueError as e:
                # If decoding failed, log the error and try again or fall back to cloning
                log_warning(f"Crossover attempt {attempt+1} failed: {e}")
        
        # If all crossover attempts failed, clone the parents
        log_warning(f"All crossover attempts failed, using clones instead")
        child1 = parent1.clone()
        child2 = parent2.clone()
        child1.generation = self.current_generation + 1
        child2.generation = self.current_generation + 1
        return child1, child2
    
    def mutate(self, specimen: Specimen) -> Specimen:
        """
        Perform mutation on a specimen by randomly changing bits in its genetic encoding.
        Ensures that the resulting model ID is always valid.
        
        Args:
            specimen: The specimen to mutate
            
        Returns:
            The mutated specimen (or original if no mutation)
        """
        # If no model_id is available, return the specimen unchanged
        if not specimen.genetic_model_id:
            mutated = specimen.clone()
            mutated.generation = self.current_generation + 1
            mutated.parent_ids = [specimen.specimen_id]
            return mutated
        
        # Try up to 5 times to create a valid mutation
        for attempt in range(5):
            try:
                # Get bit array from the model ID
                bits = BitArray.from_base32(specimen.genetic_model_id)
                
                # Keep version bits (first 8 bits) unchanged
                version_bits = bits[:ModelIdentifier.VERSION_BITS]
                remaining_bits = bits[ModelIdentifier.VERSION_BITS:]
                
                # Perform mutation on remaining bits
                mutated_bits = BitArray(list(version_bits))
                for bit in remaining_bits:
                    # Each bit has a chance to flip based on mutation rate
                    if random.random() < self.mutation_rate:
                        mutated_bits.append(not bit)  # Flip the bit
                    else:
                        mutated_bits.append(bit)  # Keep the bit unchanged
                
                # Convert bit array back to model ID
                mutated_id = mutated_bits.to_base32()
                
                # Try to decode the model ID to validate it
                mutated_config = self.model_identifier.decode_model_identifier(mutated_id)
                
                # Create the mutated specimen
                mutated = Specimen(
                    model_parameters=mutated_config['model_parameters'],
                    training_parameters=mutated_config['training_parameters'],
                    feature_indexes=mutated_config['feature_indexes'],
                    generation=self.current_generation + 1,
                    parent_ids=[specimen.specimen_id],
                    model_id=mutated_id
                )
                
                # If we got here, mutation was successful
                log_debug(f"Successfully mutated specimen {specimen.specimen_id} (attempt {attempt+1}), mutated: {mutated.genetic_model_id}, original: {specimen.genetic_model_id}")
                return mutated
                
            except ValueError as e:
                # If decoding failed, log the error and try again or fall back to original
                log_warning(f"Mutation attempt {attempt+1} failed: {e}")
        
        # If all mutation attempts failed, clone the original specimen
        log_warning(f"All mutation attempts failed for specimen {specimen.specimen_id}, using clone instead")
        mutated = specimen.clone()
        mutated.generation = self.current_generation + 1
        mutated.parent_ids = [specimen.specimen_id]
        return mutated
    
    def breed_new_generation(self) -> List[Specimen]:
        """
        Create a new generation of specimens through selection, crossover, and mutation.
        
        Returns:
            List of new specimens for the next generation
        """
        new_population = []
        
        # Check current population
        if not self.current_population:
            log_error("Current population is empty, cannot breed new generation")
            return []
        
        # Apply elitism - carry over best specimens unchanged
        sorted_population = sorted(
            [s for s in self.current_population if s.captured_results > 0 and s.fitness_score is not None],
            key=lambda s: s.fitness_score,
            reverse=True  # Higher fitness (negative MSE) is better
        )
        
        elites = []
        for i in range(min(self.elitism_count, len(sorted_population))):
            elite = sorted_population[i].clone()
            elite.generation = self.current_generation + 1
            new_population.append(elite)
            elites.append(elite)
        
        log_info(f"Added {len(elites)} elite specimens to next generation")
        
        # Fill rest of population with offspring
        while len(new_population) < self.population_size:
            # Select parents
            parents = self.select_parents(self.current_population, 2)
            if len(parents) < 2:
                log_warning("Not enough parents for breeding")
                break
            
            # Perform crossover
            child1, child2 = self.crossover(parents[0], parents[1])
            
            # Perform mutation
            child1 = self.mutate(child1)
            child2 = self.mutate(child2)
            
            # Add to new population
            new_population.append(child1)
            if len(new_population) < self.population_size:
                new_population.append(child2)
        
        log_info(f"Created {len(new_population)} specimens for generation {self.current_generation + 1}")
        return new_population
    
    def evolve(self) -> Optional[Specimen]:
        """
        Run the genetic algorithm for the specified number of generations.
        
        Returns:
            The best specimen found
        """
        log_info(f"Starting evolution for {self.ticker} with {self.generations} generations")
        start_time = time.time()
        
        # Create initial population
        if not self.current_population:
            self.current_population = self.create_initial_population()
        
        # Run evolution for specified generations
        for generation in range(self.generations):
            self.current_generation = generation
            generation_start = time.time()
            
            log_info(f"\n--- Generation {generation} ---")
            
            # Evaluate current population
            self.evaluate_population(self.current_population)
            
            # Save evolution stats for this generation
            stats = {
                "generation": generation,
                "population_size": len(self.current_population),
                "evaluated": sum(1 for s in self.current_population if s.captured_results > 0),
                "best_specimen_id": self.best_specimen.specimen_id if self.best_specimen else None,
                "best_mse": self.best_specimen.best_mse if self.best_specimen else None,
                "generation_time": time.time() - generation_start
            }
            self.storage_manager.save_evolution_stats(stats)
            
            # Log generation results
            evaluated = [s for s in self.current_population if s.captured_results > 0 and s.best_mse is not None]
            if evaluated:
                best_in_gen = min(evaluated, key=lambda s: s.best_mse)
                avg_mse = sum(s.best_mse for s in evaluated) / len(evaluated)
                
                log_info(f"Generation {generation} results:")
                log_info(f"  Population size: {len(self.current_population)}")
                log_info(f"  Evaluated specimens: {len(evaluated)}")
                log_info(f"  Best MSE in generation: {best_in_gen.best_mse:.6f} (Specimen {best_in_gen.specimen_id})")
                log_info(f"  Average MSE: {avg_mse:.6f}")
                log_info(f"  Best MSE overall: {self.best_specimen.best_mse:.6f} (Specimen {self.best_specimen.specimen_id})")
            
            # Create next generation (if not the last one)
            if generation < self.generations - 1:
                self.current_population = self.breed_new_generation()
        
        # Create experiment summary
        log_info("Evolution complete, creating summary")
        summary = self.storage_manager.create_experiment_summary()
        
        total_time = time.time() - start_time
        log_info(f"Evolution completed in {total_time:.2f} seconds")
        log_success(f"Best specimen: {self.best_specimen.specimen_id} with MSE={self.best_specimen.best_mse:.6f}")
        
        return self.best_specimen
    
    def get_best_specimen(self) -> Optional[Specimen]:
        """Get the best specimen found so far."""
        if self.best_specimen:
            return self.best_specimen
        
        # If not cached, get from storage manager
        return self.storage_manager.get_best_specimen() 