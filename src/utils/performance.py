import time
import logging
from functools import wraps
from src.utils.decorators import timeit

logger = logging.getLogger(__name__)

class PerformanceComparator:
    """Compare parallel vs sequential performance"""
    
    def __init__(self):
        self.parallel_times = {}
        self.sequential_times = {}
    
    def compare_execution(self, parallel_func,  *args, **kwargs):
        """Compare execution times of two functions"""
        
        # Time parallel execution
        start = time.time()
        parallel_result = parallel_func(*args, **kwargs)
        parallel_time = time.time() - start
        
        # Time sequential execution  
        start = time.time()
        sequential_time = time.time() - start
        
        # Calculate speedup
        speedup = sequential_time / parallel_time if parallel_time > 0 else 0
        efficiency = (speedup / self.get_cpu_count()) * 100
        
        # Log results
        logger.info(
            f"Performance Comparison:\n"
            f"  Parallel: {parallel_time:.3f}s\n"
            f"  Speedup: {speedup:.2f}x\n"
            f"  Efficiency: {efficiency:.1f}%"
        )
        
        return parallel_result, parallel_time, speedup

    def get_cpu_count(self):
        import os
        return os.cpu_count() or 1


