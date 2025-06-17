"""
Módulo de monitoramento de memória para análise de ECG.
Fornece funcionalidades para rastreamento e otimização de uso de memória.
"""

import gc
import os
import sys
import tracemalloc
from typing import Dict, Optional, Any, List, Tuple
from datetime import datetime
import psutil
import logging

logger = logging.getLogger(__name__)


class MemoryMonitor:
    """
    Classe para monitoramento detalhado de uso de memória.
    
    Características:
    - Rastreamento de alocações de memória
    - Análise de vazamentos de memória
    - Otimização automática de garbage collection
    - Relatórios detalhados de uso
    """
    
    def __init__(self, enable_tracemalloc: bool = True):
        """
        Inicializa o monitor de memória.
        
        Args:
            enable_tracemalloc: Se deve habilitar rastreamento detalhado
        """
        self.enabled = enable_tracemalloc
        self.snapshots: List[Any] = []
        self.start_time = datetime.now()
        self.process = psutil.Process(os.getpid())
        
        if self.enabled:
            if tracemalloc.is_tracing():
                tracemalloc.stop()
            tracemalloc.start()
            
        logger.info("MemoryMonitor inicializado")
    
    def get_current_memory_usage(self) -> Dict[str, float]:
        """
        Obtém o uso atual de memória do processo.
        
        Returns:
            Dict com métricas de memória em MB
        """
        memory_info = self.process.memory_info()
        
        return {
            "rss_mb": memory_info.rss / 1024 / 1024,  # Resident Set Size
            "vms_mb": memory_info.vms / 1024 / 1024,  # Virtual Memory Size
            "percent": self.process.memory_percent(),
            "available_mb": psutil.virtual_memory().available / 1024 / 1024
        }
    
    def take_snapshot(self, label: str = "") -> Dict[str, Any]:
        """
        Captura um snapshot do estado atual da memória.
        
        Args:
            label: Rótulo identificador do snapshot
            
        Returns:
            Dict com informações do snapshot
        """
        snapshot_data = {
            "timestamp": datetime.now(),
            "label": label,
            "memory_usage": self.get_current_memory_usage(),
            "gc_stats": gc.get_stats()
        }
        
        if self.enabled and tracemalloc.is_tracing():
            snapshot_data["tracemalloc_snapshot"] = tracemalloc.take_snapshot()
            
        self.snapshots.append(snapshot_data)
        return snapshot_data
    
    def compare_snapshots(self, start_idx: int = -2, end_idx: int = -1) -> Dict[str, Any]:
        """
        Compara dois snapshots para identificar mudanças.
        
        Args:
            start_idx: Índice do snapshot inicial
            end_idx: Índice do snapshot final
            
        Returns:
            Dict com análise comparativa
        """
        if len(self.snapshots) < 2:
            return {"error": "Insufficient snapshots for comparison"}
            
        start_snapshot = self.snapshots[start_idx]
        end_snapshot = self.snapshots[end_idx]
        
        memory_diff = {
            key: end_snapshot["memory_usage"][key] - start_snapshot["memory_usage"][key]
            for key in start_snapshot["memory_usage"]
        }
        
        result = {
            "start_label": start_snapshot["label"],
            "end_label": end_snapshot["label"],
            "memory_difference_mb": memory_diff,
            "time_elapsed": (end_snapshot["timestamp"] - start_snapshot["timestamp"]).total_seconds()
        }
        
        # Análise detalhada se tracemalloc estiver habilitado
        if self.enabled and "tracemalloc_snapshot" in start_snapshot:
            start_trace = start_snapshot["tracemalloc_snapshot"]
            end_trace = end_snapshot["tracemalloc_snapshot"]
            
            top_stats = end_trace.compare_to(start_trace, 'lineno')
            result["top_memory_increases"] = [
                {
                    "file": stat.traceback.format()[0] if stat.traceback else "Unknown",
                    "size_diff_mb": stat.size_diff / 1024 / 1024,
                    "count_diff": stat.count_diff
                }
                for stat in sorted(top_stats, key=lambda x: x.size_diff, reverse=True)[:10]
            ]
            
        return result
    
    def optimize_memory(self) -> Dict[str, Any]:
        """
        Executa otimizações de memória.
        
        Returns:
            Dict com estatísticas da otimização
        """
        before_memory = self.get_current_memory_usage()
        
        # Força coleta de lixo
        collected = gc.collect()
        
        # Limpa caches internos do Python
        if hasattr(sys, 'intern'):
            sys.intern.clear()
            
        after_memory = self.get_current_memory_usage()
        
        return {
            "objects_collected": collected,
            "memory_freed_mb": before_memory["rss_mb"] - after_memory["rss_mb"],
            "before": before_memory,
            "after": after_memory
        }
    
    def get_memory_report(self) -> Dict[str, Any]:
        """
        Gera relatório completo de uso de memória.
        
        Returns:
            Dict com relatório detalhado
        """
        current_memory = self.get_current_memory_usage()
        
        report = {
            "current_memory": current_memory,
            "runtime_seconds": (datetime.now() - self.start_time).total_seconds(),
            "snapshots_taken": len(self.snapshots),
            "gc_stats": gc.get_stats(),
            "gc_thresholds": gc.get_threshold()
        }
        
        if self.enabled and tracemalloc.is_tracing():
            snapshot = tracemalloc.take_snapshot()
            top_stats = snapshot.statistics('lineno')
            
            report["top_memory_usage"] = [
                {
                    "file": stat.traceback.format()[0] if stat.traceback else "Unknown",
                    "size_mb": stat.size / 1024 / 1024,
                    "count": stat.count
                }
                for stat in sorted(top_stats, key=lambda x: x.size, reverse=True)[:20]
            ]
            
        return report
    
    def set_memory_limit(self, limit_mb: float) -> None:
        """
        Define limite de memória para o processo (se suportado).
        
        Args:
            limit_mb: Limite em megabytes
        """
        try:
            import resource
            limit_bytes = int(limit_mb * 1024 * 1024)
            resource.setrlimit(resource.RLIMIT_AS, (limit_bytes, limit_bytes))
            logger.info(f"Limite de memória definido para {limit_mb}MB")
        except ImportError:
            logger.warning("resource module não disponível (Windows)")
        except Exception as e:
            logger.error(f"Erro ao definir limite de memória: {e}")
    
    def __enter__(self):
        """Context manager entry."""
        self.take_snapshot("context_start")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.take_snapshot("context_end")
        if len(self.snapshots) >= 2:
            comparison = self.compare_snapshots()
            logger.info(f"Memory change: {comparison.get('memory_difference_mb', {})}")
    
    def reset(self) -> None:
        """Reseta o monitor, limpando snapshots."""
        self.snapshots.clear()
        if self.enabled and tracemalloc.is_tracing():
            tracemalloc.clear_traces()
        gc.collect()
        logger.info("MemoryMonitor resetado")
    
    def __del__(self):
        """Cleanup ao destruir objeto."""
        if hasattr(self, 'enabled') and self.enabled and tracemalloc.is_tracing():
            tracemalloc.stop()


def get_memory_usage() -> Dict[str, float]:
    """
    Função utilitária para obter uso de memória rapidamente.
    
    Returns:
        Dict com métricas básicas de memória
    """
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    
    return {
        "rss_mb": memory_info.rss / 1024 / 1024,
        "vms_mb": memory_info.vms / 1024 / 1024,
        "percent": process.memory_percent(),
        "available_mb": psutil.virtual_memory().available / 1024 / 1024
    }


def optimize_memory() -> int:
    """
    Função utilitária para otimização rápida de memória.
    
    Returns:
        Número de objetos coletados
    """
    return gc.collect()


# Singleton global para monitoramento contínuo
_global_monitor: Optional[MemoryMonitor] = None


def get_global_monitor() -> MemoryMonitor:
    """
    Obtém instância global do monitor de memória.
    
    Returns:
        Instância singleton do MemoryMonitor
    """
    global _global_monitor
    if _global_monitor is None:
        _global_monitor = MemoryMonitor(enable_tracemalloc=False)
    return _global_monitor
