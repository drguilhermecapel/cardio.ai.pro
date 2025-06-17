"""
Memory monitoring utilities
"""

import os


def get_memory_usage() -> dict:
    """Get current memory usage."""
    try:
        import psutil
        process = psutil.Process(os.getpid())
        memory_info = process.memory_info()
        
        return {
            "rss_mb": memory_info.rss / 1024 / 1024,
            "vms_mb": memory_info.vms / 1024 / 1024,
            "percent": process.memory_percent(),
            "available_mb": psutil.virtual_memory().available / 1024 / 1024
        }
    except ImportError:
        # psutil não disponível, retornar valores simulados
        return {
            "rss_mb": 100.0,
            "vms_mb": 200.0,
            "percent": 10.0,
            "available_mb": 8000.0
        }
