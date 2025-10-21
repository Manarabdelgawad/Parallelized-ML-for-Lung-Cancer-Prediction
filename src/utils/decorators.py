import time
import logging
import os
import threading
from functools import wraps

def timeit(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        execution_time = time.time() - start
        
        logger = logging.getLogger(func.__module__)
        logger.info(
            f"{func.__name__} took {execution_time:.2f}s "
            f"(Thread: {threading.current_thread().name}, "
            f"Process ID: {os.getpid()})"
        )
        return result
    return wrapper

def log_process_info(func):
    """Decorator to log process and thread information"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        logger = logging.getLogger(func.__module__)
        logger.debug(
            f"Executing {func.__name__} "
            f"(Thread: {threading.current_thread().name}, "
            f"Process ID: {os.getpid()})"
        )
        return func(*args, **kwargs)
    return wrapper