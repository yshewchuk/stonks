import json
import os
import shutil
import re
from datetime import datetime
from config import OUTPUT_DIR
import pandas as pd

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
            print("❌ Error: config must be a dictionary")
            return False
            
        if OUTPUT_DIR not in config:
            print(f"❌ Error: config must contain an {OUTPUT_DIR} key")
            return False
            
        try:
            # Ensure the output directory exists
            output_dir = config[OUTPUT_DIR]
            os.makedirs(output_dir, exist_ok=True)
            
            # Create a backup of the existing content
            backup_created = Process._create_backup(output_dir)
            if backup_created:
                print(f"✅ Backup of existing content created in {output_dir}")
                
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
                
                print(f"✅ Cleaned output directory {output_dir}")
            
            # Create a copy of the config to avoid modifying the original
            metadata = config.copy()
            
            # Add a timestamp in human-readable format
            current_time = datetime.now()
            metadata['timestamp'] = current_time.strftime('%Y-%m-%d %H:%M:%S')
            
            # Write the metadata to a JSON file using the custom encoder
            metadata_path = os.path.join(output_dir, 'metadata.json')
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=4, cls=DateTimeEncoder)
                
            print(f"✅ Metadata written to {metadata_path}")
            return True
            
        except Exception as e:
            print(f"❌ Error starting process: {e}")
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
                print(f"ℹ️ No files to backup in {directory}")
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
            
            print(f"✅ Created backup #{next_backup_num} in {backup_dir}")
            return True
            
        except Exception as e:
            print(f"⚠️ Warning: Could not create backup: {e}")
            return False
