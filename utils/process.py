import json
import os
import shutil
import re
from datetime import datetime
from config import OUTPUT_DIR
import pandas as pd
from utils.logger import log_info, log_success, log_error, log_warning
import multiprocessing

class DateTimeEncoder(json.JSONEncoder):
    """
    Custom JSON encoder that handles datetime objects.
    """
    def default(self, obj):
        if isinstance(obj, datetime):
            return obj.strftime('%Y-%m-%d %H:%M:%S')
        return super().default(obj)

class Process:
    """
    Utility class for process-related operations.
    """
    
    @staticmethod
    def start_process(config):
        """
        Starts a new process:
        1. Creates a numbered backup of the existing content in the output directory
        2. Removes all existing files from the output directory (after backup)
        3. Writes a metadata.json file containing the provided configuration dictionary
           along with a human-readable timestamp to the OUTPUT_DIR
        
        Handles datetime objects by converting them to string format.
        
        Args:
            config (dict): Configuration dictionary containing at least an OUTPUT_DIR key
            
        Returns:
            bool: True if successful, False otherwise
        """
        if not isinstance(config, dict):
            log_error("config must be a dictionary")
            return False
            
        if OUTPUT_DIR not in config:
            log_error(f"config must contain an {OUTPUT_DIR} key")
            return False
            
        try:
            # Ensure the output directory exists
            output_dir = config[OUTPUT_DIR]
            os.makedirs(output_dir, exist_ok=True)
            
            # Create a backup of the existing content
            backup_created = Process._create_backup(output_dir)
            if backup_created:
                log_success(f"Backup of existing content created in {output_dir}")
                
                # After successful backup, clean up the output directory
                # by removing all non-backup files and folders
                backup_pattern = re.compile(r'backup_(\d+)$')
                items = os.listdir(output_dir)
                non_backup_items = [item for item in items if not backup_pattern.match(item)]
                
                for item in non_backup_items:
                    item_path = os.path.join(output_dir, item)
                    if os.path.isdir(item_path):
                        shutil.rmtree(item_path)
                    else:
                        os.remove(item_path)
                
                log_success(f"Cleaned output directory {output_dir}")
            
            # Create a copy of the config to avoid modifying the original
            metadata = config.copy()
            
            # Add a timestamp in human-readable format
            current_time = datetime.now()
            metadata['timestamp'] = current_time.strftime('%Y-%m-%d %H:%M:%S')
            
            # Write the metadata to a JSON file using the custom encoder
            metadata_path = os.path.join(output_dir, 'metadata.json')
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=4, cls=DateTimeEncoder)
                
            log_success(f"Metadata written to {metadata_path}")
            return True
            
        except Exception as e:
            log_error(f"Error starting process: {e}")
            return False
    
    @staticmethod
    def _create_backup(directory):
        """
        Creates a numbered backup of the contents in the specified directory.
        
        Args:
            directory (str): The directory to backup
            
        Returns:
            bool: True if backup was created, False if directory was empty or backup failed
        """
        try:
            # Get all items in the directory
            items = os.listdir(directory)
            
            # Filter out existing backup folders
            backup_pattern = re.compile(r'backup_(\d+)$')
            non_backup_items = [item for item in items if not backup_pattern.match(item)]
            
            # If directory is empty (excluding backup folders), no need for backup
            if not non_backup_items:
                log_info(f"No files to backup in {directory}")
                return False
                
            # Find the highest existing backup number
            backup_numbers = [int(backup_pattern.match(item).group(1)) 
                             for item in items if backup_pattern.match(item)]
            next_backup_num = max(backup_numbers, default=0) + 1
            
            # Create new backup folder
            backup_dir = os.path.join(directory, f"backup_{next_backup_num}")
            os.makedirs(backup_dir, exist_ok=True)
            
            # Copy non-backup items to the backup folder
            for item in non_backup_items:
                item_path = os.path.join(directory, item)
                backup_path = os.path.join(backup_dir, item)
                
                if os.path.isdir(item_path):
                    shutil.copytree(item_path, backup_path)
                else:
                    shutil.copy2(item_path, backup_path)
            
            log_success(f"Created backup #{next_backup_num} in {backup_dir}")
            return True
            
        except Exception as e:
            log_warning(f"Could not create backup: {e}")
            return False
            
    @staticmethod
    def save_execution_metadata(config, filename, metadata, start_time=None, time_markers=None):
        """
        Saves execution metadata to a JSON file in the output directory.
        
        Args:
            config (dict): Configuration dictionary containing OUTPUT_DIR key
            filename (str): Name of the JSON file to create (without path)
            metadata (dict): Dictionary containing the metadata to save
            start_time (float, optional): Start time of the process (from time.time())
            time_markers (dict, optional): Dictionary of named time markers (from time.time())
                Example: {'load': load_time, 'process': process_time, 'save': save_time}
        
        Returns:
            bool: True if successful, False otherwise
        """
        if not isinstance(config, dict) or OUTPUT_DIR not in config:
            log_error(f"Config must be a dictionary containing {OUTPUT_DIR} key")
            return False
            
        if not isinstance(metadata, dict):
            log_error("Metadata must be a dictionary")
            return False
            
        try:
            # Ensure output directory exists
            output_dir = config[OUTPUT_DIR]
            os.makedirs(output_dir, exist_ok=True)
            
            # Create a copy of the metadata to avoid modifying the original
            result_metadata = metadata.copy()
            
            # Add common execution information
            if 'multiprocessing_used' not in result_metadata:
                result_metadata['multiprocessing_used'] = True
                
            if 'workers_used' not in result_metadata and 'max_workers' in config:
                result_metadata['workers_used'] = config['max_workers']
                
            if 'cpu_cores_available' not in result_metadata:
                result_metadata['cpu_cores_available'] = multiprocessing.cpu_count()
            
            # Add time measurements if provided
            if start_time is not None and time_markers is not None:
                processing_times = {}
                
                # Sort time markers chronologically
                sorted_markers = sorted(time_markers.items(), key=lambda x: x[1])
                
                # Calculate time differences between markers
                prev_time, prev_name = start_time, "start"
                for name, marker_time in sorted_markers:
                    processing_times[f"{prev_name}_to_{name}"] = round(marker_time - prev_time, 2)
                    prev_time, prev_name = marker_time, name
                
                # Add total time
                if sorted_markers:
                    last_time = sorted_markers[-1][1]
                    processing_times["total"] = round(last_time - start_time, 2)
                
                result_metadata['processing_time_seconds'] = processing_times
            
            # Write metadata to file
            metadata_path = os.path.join(output_dir, filename)
            with open(metadata_path, 'w') as f:
                json.dump(result_metadata, f, indent=2, default=str)
            
            log_success(f"Saved execution metadata to {metadata_path}")
            return True
            
        except Exception as e:
            log_error(f"Error saving execution metadata: {e}")
            return False
