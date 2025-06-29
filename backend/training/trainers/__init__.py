"""
Módulo de treinadores para modelos ECG
"""

from .base_trainer import BaseTrainer
from .classification_trainer import ClassificationTrainer
from .ecg_trainer import ECGTrainer

__all__ = [
    "BaseTrainer",
    "ClassificationTrainer",
    "ECGTrainer"
]

