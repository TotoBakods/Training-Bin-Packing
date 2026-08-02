import os
import logging
from logging.handlers import RotatingFileHandler

LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'logs')
LOG_FILE = os.path.join(LOG_DIR, 'system.log')

def _ensure_log_dir():
    if not os.path.exists(LOG_DIR):
        os.makedirs(LOG_DIR, exist_ok=True)

def setup_root_logger():
    _ensure_log_dir()
    log_level_str = os.environ.get('LOG_LEVEL', 'INFO').upper()
    log_level = getattr(logging, log_level_str, logging.INFO)

    root_logger = logging.getLogger()
    if root_logger.handlers:
        # Root logger already configured
        return root_logger

    root_logger.setLevel(log_level)

    formatter = logging.Formatter(
        '[%(asctime)s] [%(levelname)s] [%(name)s] [%(process)d:%(threadName)s] - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # Rotating File Handler - 10MB per file, 5 backups
    file_handler = RotatingFileHandler(LOG_FILE, maxBytes=10 * 1024 * 1024, backupCount=5, encoding='utf-8')
    file_handler.setLevel(log_level)
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)

    # Console Stream Handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    return root_logger

def get_logger(name: str) -> logging.Logger:
    setup_root_logger()
    return logging.getLogger(name)
