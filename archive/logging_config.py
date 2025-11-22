"""
Centralized logging configuration for the entire system
"""

import logging
import sys
from pathlib import Path


def setup_logging(level=logging.INFO, quiet_mode=False, log_file=None):
    """
    Setup unified logging configuration.
    
    Args:
        level: Logging level (logging.DEBUG, INFO, WARNING, ERROR)
        quiet_mode: If True, suppress most library logs
        log_file: Optional file path to save logs
    """
    
    # Root logger configuration
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    
    # Clear existing handlers
    root_logger.handlers.clear()
    
    # Console handler with clean format
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    
    # Simplified format
    formatter = logging.Formatter(
        '%(levelname)s | %(name)s | %(message)s'
    )
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    
    # Optional file handler
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)  # Always capture full detail in file
        
        file_formatter = logging.Formatter(
            '%(asctime)s | %(levelname)s | %(name)s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(file_formatter)
        root_logger.addHandler(file_handler)
    
    if quiet_mode:
        # Suppress verbose third-party libraries
        logging.getLogger('urllib3').setLevel(logging.WARNING)
        logging.getLogger('requests').setLevel(logging.WARNING)
        logging.getLogger('matplotlib').setLevel(logging.WARNING)
        logging.getLogger('PIL').setLevel(logging.WARNING)
        logging.getLogger('xgboost').setLevel(logging.WARNING)
        logging.getLogger('sklearn').setLevel(logging.WARNING)
        logging.getLogger('pandas').setLevel(logging.WARNING)
        logging.getLogger('yfinance').setLevel(logging.WARNING)
        logging.getLogger('fredapi').setLevel(logging.WARNING)
        
        # Keep core modules visible but clean
        logging.getLogger('__main__').setLevel(level)
        logging.getLogger('core').setLevel(level)
        logging.getLogger('export').setLevel(level)


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger with the given name.
    
    Args:
        name: Logger name (usually __name__)
        
    Returns:
        Configured logger instance
    """
    return logging.getLogger(name)


# Default configuration when imported
setup_logging(level=logging.INFO, quiet_mode=True)
