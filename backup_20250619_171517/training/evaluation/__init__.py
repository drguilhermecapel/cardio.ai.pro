"""
Módulo de avaliação para modelos de ECG
"""

from .metrics import calculate_metrics
from .visualizations import plot_confusion_matrix, plot_roc_curve

__all__ = [
    "calculate_metrics",
    "plot_confusion_matrix",
    "plot_roc_curve"
]


