"""
Memory monitoring utilities.
"""

import psutil
import os
from typing import Dict, Any

class MemoryMonitor:
    """Monitor memory usage."""
    
    def __init__(self):
        """Initialize memory monitor."""
        pass
    
    def get_memory_usage(self) -> Dict[str, float]:
        """Get current memory usage."""
        try:
            process = psutil.Process(os.getpid())
            memory_info = process.memory_info()
            return {
                "rss_mb": memory_info.rss / 1024 / 1024,
                "vms_mb": memory_info.vms / 1024 / 1024,
                "percent": process.memory_percent(),
                "available_mb": psutil.virtual_memory().available / 1024 / 1024,
                "process_memory_mb": memory_info.rss / 1024 / 1024
            }
        except Exception:
            return {
                "rss_mb": 0,
                "vms_mb": 0,
                "percent": 0,
                "available_mb": 0,
                "process_memory_mb": 0
            }
    
    def check_memory_limit(self, limit_mb: float = 500) -> bool:
        """Check if memory usage is within limits."""
        usage = self.get_memory_usage()
        return usage.get("process_memory_mb", 0) < limit_mb

# Exportar a classe
__all__ = ["MemoryMonitor"]
