"""
Test script for the Process utility class
"""
import os
import sys
import shutil
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json

# Add the parent directory to sys.path to allow imports from the project root
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.process import Process
from config import OUTPUT_DIR

def setup_test_directory():
    """Set up a clean test directory"""
    test_dir = 'tests/test_output'
    # Clean up existing test directory if it exists
    if os.path.exists(test_dir):
        shutil.rmtree(test_dir)
    os.makedirs(test_dir, exist_ok=True)
    return test_dir

def create_test_files(directory):
    """Create some test files in the directory"""
    # Create a text file
    with open(os.path.join(directory, 'test_file.txt'), 'w') as f:
        f.write('Test content')
    
    # Create a small subdirectory with a file in it
    subdir = os.path.join(directory, 'test_subdir')
    os.makedirs(subdir, exist_ok=True)
    with open(os.path.join(subdir, 'sub_file.txt'), 'w') as f:
        f.write('Subdirectory file content')
    
    return [
        os.path.join(directory, 'test_file.txt'),
        os.path.join(directory, 'test_subdir', 'sub_file.txt')
    ]

def test_start_process():
    """
    Test the Process.start_process method
    """
    # Set up test directory
    test_dir = setup_test_directory()
    
    # Create a sample configuration dictionary
    config = {
        'START_DATE': '2022-01-01',
        'END_DATE': '2023-01-01',
        OUTPUT_DIR: test_dir,
        'TICKERS': ['AAPL', 'MSFT', 'GOOG']
    }
    
    # Create some test files
    test_files = create_test_files(test_dir)
    
    # Start the process
    result = Process.start_process(config)
    
    assert result, "Process start failed"
    assert os.path.exists(f"{test_dir}/metadata.json"), "Metadata file not found"
    
    # Check that backup folder was created
    backup_folder = os.path.join(test_dir, "backup_1")
    assert os.path.exists(backup_folder), "Backup folder not created"
    
    # Check that files were backed up
    for file_path in test_files:
        backup_path = file_path.replace(test_dir, backup_folder)
        assert os.path.exists(backup_path), f"Backup of {file_path} not found"
    
    print("✅ Test start_process successful: Metadata file written and backup created")
    
    # Test second run to check incrementing backup numbers
    result = Process.start_process(config)
    
    assert result, "Second process start failed"
    backup_folder_2 = os.path.join(test_dir, "backup_2")
    assert os.path.exists(backup_folder_2), "Second backup folder not created"
    
    print("✅ Test start_process with incremental backups successful")

def test_start_process_with_datetime():
    """
    Test the Process.start_process method with datetime objects
    """
    # Set up test directory
    test_dir = setup_test_directory()
    
    # Create a sample configuration dictionary with datetime objects
    now = datetime.now()
    config = {
        'START_DATE': now - timedelta(days=30),
        'END_DATE': now,
        OUTPUT_DIR: test_dir,
        'TICKERS': ['AAPL', 'MSFT', 'GOOG']
    }
    
    # Start the process
    result = Process.start_process(config)
    
    assert result, "Process start with datetime failed"
    assert os.path.exists(f"{test_dir}/metadata.json"), "Metadata file not found"
    
    # Read back the JSON file to verify it contains date values
    with open(f"{test_dir}/metadata.json", 'r') as f:
        data = json.load(f)
    
    assert 'START_DATE' in data, "START_DATE missing in JSON file"
    assert 'END_DATE' in data, "END_DATE missing in JSON file"
    assert isinstance(data['START_DATE'], str), "START_DATE not converted to string"
    assert isinstance(data['END_DATE'], str), "END_DATE not converted to string"
    
    print("✅ Test start_process_with_datetime successful: Metadata with datetime objects written correctly")

def test_write_dataframes_to_parquet():
    """
    Test the Process.write_dataframes_to_parquet method
    """
    # Set up test directory
    test_dir = setup_test_directory()
    
    # Create a sample configuration dictionary
    config = {
        'START_DATE': '2022-01-01',
        'END_DATE': '2023-01-01',
        OUTPUT_DIR: test_dir,
        'TICKERS': ['AAPL', 'MSFT', 'GOOG']
    }
    
    # Create sample dataframes
    dates = pd.date_range(start='2022-01-01', end='2022-01-10')
    
    # Sample dataframe for AAPL
    aapl_data = pd.DataFrame({
        'Open': np.random.rand(10) * 100 + 150,
        'High': np.random.rand(10) * 100 + 160,
        'Low': np.random.rand(10) * 100 + 140,
        'Close': np.random.rand(10) * 100 + 155,
        'Volume': np.random.randint(1000000, 10000000, 10)
    }, index=dates)
    
    # Sample dataframe for MSFT
    msft_data = pd.DataFrame({
        'Open': np.random.rand(10) * 100 + 250,
        'High': np.random.rand(10) * 100 + 260,
        'Low': np.random.rand(10) * 100 + 240,
        'Close': np.random.rand(10) * 100 + 255,
        'Volume': np.random.randint(1000000, 10000000, 10)
    }, index=dates)
    
    # Create dictionary of dataframes
    dataframes_dict = {
        'AAPL': aapl_data,
        'MSFT': msft_data
    }
    
    # Write the dataframes to parquet
    result = Process.write_dataframes_to_parquet(dataframes_dict, config)
    
    assert result, "Dataframes writing failed"
    assert os.path.exists(f"{test_dir}/AAPL.parquet"), "AAPL parquet file not found"
    assert os.path.exists(f"{test_dir}/MSFT.parquet"), "MSFT parquet file not found"
    
    # Try to read the parquet files to validate
    read_aapl_df = pd.read_parquet(f"{test_dir}/AAPL.parquet")
    read_msft_df = pd.read_parquet(f"{test_dir}/MSFT.parquet")
    
    assert read_aapl_df.equals(aapl_data), "Read AAPL dataframe does not match original"
    assert read_msft_df.equals(msft_data), "Read MSFT dataframe does not match original"
    
    print("✅ Test write_dataframes_to_parquet successful: All dataframes saved and verified")

if __name__ == "__main__":
    print("🚀 Testing Process utility class...")
    
    # Run tests
    test_start_process()
    test_start_process_with_datetime()
    test_write_dataframes_to_parquet()
    
    print("🎉 All tests completed successfully!") 