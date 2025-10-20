import time
import logging
from functools import wraps

def timeit(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        logging.info(f"{func.__name__} took {time.time() - start:.2f}s")
        return result
    return wrapper