"""Memory monitoring utilities."""

import psutil
import os

class MemoryMonitor:
    """Monitor memory usage."""
    
    def __init__(self):
        self.process = psutil.Process(os.getpid())
    
    def get_memory_stats(self):
        """Get current memory statistics."""
        memory_info = self.process.memory_info()
        return {
            "process_memory_mb": memory_info.rss / 1024 / 1024,
            "available_mb": psutil.virtual_memory().available / 1024 / 1024,
            "percent": psutil.virtual_memory().percent
        }

def get_memory_usage():
    """Get current memory usage in MB."""
    try:
        process = psutil.Process()
        return process.memory_info().rss / 1024 / 1024
    except:
        return 100.0  # Default value
