# AGENTIC_MIRAI/shared/log.py
import logging
import sys
from logging.handlers import TimedRotatingFileHandler
import os
from .config import settings # Use the instantiated settings

# Ensure log directory exists
log_dir = os.path.dirname(settings.LOG_FILE_PATH)
if log_dir and not os.path.exists(log_dir):
    os.makedirs(log_dir)

# Formatter
LOG_FORMAT = "%(asctime)s - %(levelname)s - %(name)s - %(filename)s:%(lineno)d - %(message)s"
formatter = logging.Formatter(LOG_FORMAT)

# Console Handler
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setFormatter(formatter)

# File Handler (Rotating)
file_handler = TimedRotatingFileHandler(
    filename=settings.LOG_FILE_PATH,
    when=settings.LOG_ROTATION_WHEN,
    interval=settings.LOG_ROTATION_INTERVAL,
    backupCount=settings.LOG_ROTATION_BACKUP_COUNT,
    encoding='utf-8'
)
file_handler.setFormatter(formatter)

def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(settings.LOG_LEVEL.upper())
    
    # Add handlers only if they haven't been added yet
    if not logger.handlers:
        logger.addHandler(console_handler)
        logger.addHandler(file_handler)
        logger.propagate = False # Prevent duplicate logs in parent loggers
        
    return logger

# Example usage:
# from shared.log import get_logger
# logger = get_logger(__name__)
# logger.info("This is an info message.")