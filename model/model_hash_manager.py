import os
import json
import hashlib
from datetime import datetime

class ModelHashManager:
    """
    A class that manages hash generation and collision resolution for models.
    
    This class is responsible for:
    - Generating unique hashes for model parameters
    - Resolving hash collisions when multiple models might have the same hash
    - Verifying model parameters match an existing hash
    """
    
    @staticmethod
    def get_model_hash(model_params):
        """
        Generate a unique hash for model parameters.
        
        Args:
            model_params (dict): Model parameters
            
        Returns:
            str: Unique hash ID for the model parameters
        """
        # Create a sorted, deterministic string representation of model parameters
        param_str = json.dumps(model_params, sort_keys=True)
        
        # Hash the string to create a unique identifier
        model_hash = hashlib.md5(param_str.encode('utf-8')).hexdigest()[:12]
        
        return model_hash
    
    @staticmethod
    def resolve_hash_collision(model_hash, model_params, base_output_dir):
        """
        Resolve hash collisions by checking if the hash already exists with different parameters.
        
        Args:
            model_hash (str): The original hash
            model_params (dict): The model parameters
            base_output_dir (str): Base directory for models
            
        Returns:
            str: A unique hash that doesn't collide with existing model hashes
        """
        # Check if model directory exists
        model_dir = os.path.join(base_output_dir, model_hash)
        
        if not os.path.exists(model_dir):
            return model_hash  # No collision
            
        # Check if parameters file exists and matches
        params_file = os.path.join(model_dir, "model_params.json")
        if os.path.exists(params_file):
            try:
                with open(params_file, 'r') as f:
                    existing_params = json.load(f)
                
                # If parameters match, we can use the same hash
                if existing_params == model_params:
                    return model_hash
            except:
                pass  # If any error occurs, generate a new hash
        
        # Handle collision: add a counter to the parameter string
        counter = 1
        while True:
            # Add counter to parameter string
            collision_params = model_params.copy()
            collision_params['_collision_counter'] = counter
            
            # Generate new hash
            new_param_str = json.dumps(collision_params, sort_keys=True)
            new_hash = hashlib.md5(new_param_str.encode('utf-8')).hexdigest()[:12]
            
            # Check if new hash already exists
            new_model_dir = os.path.join(base_output_dir, new_hash)
            if not os.path.exists(new_model_dir):
                return new_hash
                
            counter += 1
            if counter > 100:  # Prevent infinite loop
                # If we get too many collisions, just use timestamp as part of the hash
                timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
                final_hash = f"{model_hash}_{timestamp}"
                return final_hash
    
    @staticmethod
    def is_hash_valid(model_hash, model_params, base_output_dir):
        """
        Check if a hash is valid for the given model parameters.
        
        Args:
            model_hash (str): The model hash to check
            model_params (dict): The model parameters
            base_output_dir (str): Base directory for models
            
        Returns:
            bool: True if the hash is valid for the parameters, False otherwise
        """
        # Check if model directory exists
        model_dir = os.path.join(base_output_dir, model_hash)
        
        if not os.path.exists(model_dir):
            return False  # Directory doesn't exist
            
        # Check if parameters file exists and matches
        params_file = os.path.join(model_dir, "model_params.json")
        if not os.path.exists(params_file):
            return False  # Parameters file doesn't exist
            
        try:
            with open(params_file, 'r') as f:
                existing_params = json.load(f)
            
            # Check if parameters match
            return existing_params == model_params
        except:
            return False  # Error reading parameters file 