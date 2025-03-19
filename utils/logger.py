"""
Logging utility for standardized console output across all processing steps.

This module provides consistent log formatting and level-based logging functions
for the data processing pipeline.
"""

import time
from datetime import datetime
from config import OUTPUT_DIR, STEP_NAME, DESCRIPTION

# ANSI color codes for terminal output
class Colors:
    RESET = "\033[0m"
    BOLD = "\033[1m"
    BLUE = "\033[94m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    CYAN = "\033[96m"
    PURPLE = "\033[95m"

def log_step_start(config):
    """
    Log the start of a processing step with standardized formatting.
    
    Args:
        config (dict): Configuration dictionary containing step information
    """
    step_name = config.get(STEP_NAME, "Processing Step")
    output_dir = config.get(OUTPUT_DIR, "Unknown")
    
    print(f"\n{Colors.BOLD}{Colors.BLUE}{'='*80}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.BLUE}🚀 STARTING: {step_name}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.BLUE}📂 OUTPUT: {output_dir}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.BLUE}⏱️ TIME: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}{Colors.RESET}")
    if DESCRIPTION in config:
        print(f"{Colors.BOLD}{Colors.BLUE}📝 {config[DESCRIPTION]}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.BLUE}{'='*80}{Colors.RESET}\n")

def log_step_complete(start_time=None):
    """
    Log the completion of a processing step with standardized formatting.
    
    Args:
        start_time (float, optional): Start time as returned by time.time()
                                     If provided, duration will be calculated
    """
    duration_str = ""
    if start_time is not None:
        duration = time.time() - start_time
        if duration < 60:
            duration_str = f" in {duration:.2f} seconds"
        else:
            minutes = int(duration // 60)
            seconds = duration % 60
            duration_str = f" in {minutes} minutes and {seconds:.2f} seconds"
    
    print(f"\n{Colors.BOLD}{Colors.GREEN}{'='*80}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.GREEN}✅ COMPLETED SUCCESSFULLY{duration_str}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.GREEN}⏱️ TIME: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.GREEN}{'='*80}{Colors.RESET}\n")

def log_info(message):
    """
    Log an informational message.
    
    Args:
        message (str): The message to log
    """
    print(f"{Colors.CYAN}ℹ️ {message}{Colors.RESET}")

def log_success(message):
    """
    Log a success message.
    
    Args:
        message (str): The message to log
    """
    print(f"{Colors.GREEN}✅ {message}{Colors.RESET}")

def log_warning(message):
    """
    Log a warning message.
    
    Args:
        message (str): The message to log
    """
    print(f"{Colors.YELLOW}⚠️ {message}{Colors.RESET}")

def log_error(message):
    """
    Log an error message.
    
    Args:
        message (str): The message to log
    """
    print(f"{Colors.RED}❌ {message}{Colors.RESET}")

def log_progress(current, total, message="Progress", frequency=10):
    """
    Log a progress message at specified intervals.
    
    Args:
        current (int): Current item number
        total (int): Total number of items
        message (str): Progress message prefix
        frequency (int): How often to log (e.g., every 10 items)
                        If total < frequency*2, will log every item
    """
    if total < frequency*2 or current % frequency == 0 or current == total:
        percentage = (current / total) * 100
        print(f"{Colors.PURPLE}🔄 {message}: {current}/{total} ({percentage:.1f}%){Colors.RESET}")

def log_section(title):
    """
    Log a section header.
    
    Args:
        title (str): Section title
    """
    print(f"\n{Colors.BOLD}{Colors.PURPLE}▓▓▓▓ {title} ▓▓▓▓{Colors.RESET}")

def log_debug(message):
    """
    Log a debug message.
    
    Args:
        message (str): The message to log
    """
    print(f"{Colors.RESET}🔍 {message}{Colors.RESET}") 