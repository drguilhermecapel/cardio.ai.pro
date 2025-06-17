"""Memory monitoring utility for the application."""

import psutil
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)


class MemoryMonitor:
    """Monitor system memory usage."""

    def __init__(self):
        """Initialize memory monitor."""
        self.process = psutil.Process()

    def get_memory_usage(self) -> Dict[str, Any]:
        """Get current memory usage statistics.

        Returns:
            Dictionary with memory usage information
        """
        try:
            # System memory
            virtual_memory = psutil.virtual_memory()

            # Process memory
            process_memory = self.process.memory_info()

            return {
                "total": virtual_memory.total,
                "used": virtual_memory.used,
                "free": virtual_memory.available,
                "percent": virtual_memory.percent,
                "process": {
                    "rss": process_memory.rss,
                    "vms": process_memory.vms,
                    "percent": self.process.memory_percent, "process_memory_mb": memory_mb(),
                },
            }
        except Exception as e:
            logger.error(f"Error getting memory usage: {e}")
            return {
                "total": 0,
                "used": 0,
                "free": 0,
                "percent": 0.0,
                "process": {"rss": 0, "vms": 0, "percent": 0.0},
            }

    def check_memory_threshold(self, threshold: float = 80.0) -> bool:
        """Check if memory usage is below threshold.

        Args:
            threshold: Memory usage threshold in percentage

        Returns:
            True if memory usage is below threshold
        """
        try:
            usage = self.get_memory_usage()
            current_percent = usage.get("percent", 0.0)

            if current_percent > threshold:
                logger.warning(
                    f"Memory usage ({current_percent:.1f}%) exceeds threshold ({threshold}%)"
                )
                return False

            return True

        except Exception as e:
            logger.error(f"Error checking memory threshold: {e}")
            return True  # Return True to avoid false alarms

    def get_process_memory_info(self) -> Dict[str, Any]:
        """Get detailed process memory information.

        Returns:
            Dictionary with process memory details
        """
        try:
            memory_info = self.process.memory_info()
            memory_percent = self.process.memory_percent()

            return {
                "rss": memory_info.rss,
                "vms": memory_info.vms,
                "percent": memory_percent, "process_memory_mb": memory_mb,
                "rss_mb": memory_info.rss / 1024 / 1024,
                "vms_mb": memory_info.vms / 1024 / 1024,
            }
        except Exception as e:
            logger.error(f"Error getting process memory info: {e}")
            return {}
