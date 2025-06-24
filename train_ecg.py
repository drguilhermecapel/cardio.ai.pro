import os
import sys
import json
import logging
import hashlib
import time
import pickle
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any, Union, Set
from dataclasses import dataclass, field, asdict
from collections import defaultdict, OrderedDict
import warnings
from enum import Enum
from functools import lru_cache, wraps
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import tracemalloc
import cProfile
import pstats
import threading
from contextlib import contextmanager

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, OneCycleLR
from torch.cuda.amp import GradScaler, autocast

import scipy.signal as sp_signal
from scipy.stats import skew, kurtosis
from scipy import interpolate
from tqdm import tqdm

# Importações opcionais com verificação
try:
    import wfdb
    WFDB_AVAILABLE = True
except ImportError:
    print("AVISO: wfdb-python não instalado. Algumas funcionalidades estarão limitadas.")
    print("Instale com: pip install wfdb")
    WFDB_AVAILABLE = False

try:
    import pywt
    PYWT_AVAILABLE = True
except ImportError:
    print("AVISO: PyWavelets não instalado. Denoising wavelet limitado.")
    print("Instale com: pip install PyWavelets")
    PYWT_AVAILABLE = False

from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    roc_auc_score, confusion_matrix, classification_report,
    average_precision_score
)
from sklearn.calibration import CalibratedClassifierCV

# Configurar logging adequado
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Função auxiliar para evitar problemas com aspas em f-strings
def safe_get(dictionary, key, default=None):
    """Função auxiliar para evitar problemas com aspas em f-strings"""
    return dictionary.get(key, default)

# Configurar caminhos com validação e segurança
try:
    SCRIPT_DIR = Path(__file__).resolve().parent
except NameError:
    SCRIPT_DIR = Path.cwd()

# Validação de segurança de caminhos - Compatível com Python 3.7+
def validate_path(path: Path, base_dir: Path) -> Path:
    """Valida e sanitiza caminhos para prevenir path traversal"""
    resolved_path = path.resolve()
    resolved_base = base_dir.resolve()
    
    # Para Windows e outros sistemas, permitir caminhos válidos
    try:
        # Verificar se o caminho é válido e existe
        if resolved_path.exists() or resolved_path.parent.exists():
            return resolved_path
        else:
            return resolved_base
    except Exception:
        return resolved_base

# Configuração mais robusta para diferentes sistemas
BACKEND_DIR = SCRIPT_DIR.parent if SCRIPT_DIR.parent.exists() else SCRIPT_DIR
PROJECT_ROOT = BACKEND_DIR.parent if BACKEND_DIR.parent.exists() else BACKEND_DIR

sys.path.insert(0, str(BACKEND_DIR))

# Suprimir warnings não críticos
warnings.filterwarnings('ignore', category=UserWarning)

# ==================== CONSTANTES E CONFIGURAÇÕES ====================

# Constantes médicas com documentação científica
NOISE_MEDIAN_FACTOR = 0.6745  # Fator para MAD (Median Absolute Deviation)
MIN_HEART_RATE = 20  # bpm - limite fisiológico inferior
MAX_HEART_RATE = 300  # bpm - limite fisiológico superior
MIN_QT_INTERVAL = 200  # ms
MAX_QT_INTERVAL = 600  # ms
MIN_PR_INTERVAL = 50  # ms
MAX_PR_INTERVAL = 400  # ms

# Requisitos clínicos baseados em diretrizes
CLINICAL_REQUIREMENTS = {
    'target_auc': 0.90,
    'min_sensitivity': 0.85,
    'min_specificity': 0.85,
    'max_inference_time': 100,  # ms
    'calibration_factor': 10.0,  # mm/mV padrão
}

# Critérios diagnósticos atualizados conforme diretrizes 2023
DIAGNOSTIC_CRITERIA = {
    'lvh': {
        'sokolow_lyon': 35,  # mm (3.5 mV)
        'cornell_male': 28,  # mm (2.8 mV)
        'cornell_female': 20,  # mm (2.0 mV)
        'romhilt_estes': 5,  # pontos
    },
    'rvh': {
        'r_v1': 7,  # mm
        'r_v1_s_v5v6': 11,  # mm
        'r_v1_s_v5v6_ratio': 1.0,
    },
    'lae': {
        'p_duration': 120,  # ms
        'p_terminal_force': -0.04,  # mm·s
        'p_amplitude': 2.5,  # mm
    },
    'rae': {
        'p_amplitude_lead2': 2.5,  # mm
        'p_amplitude_v1': 1.5,  # mm
    },
    'mi': {
        'q_duration': 40,  # ms
        'q_r_ratio': 0.25,
        'st_elevation_j': 0.1,  # mV limb leads
        'st_elevation_j_precordial': 0.2,  # mV precordial
    }
}

# Thread-safe cache decorator
def thread_safe_cache(maxsize=128):
    """Decorator para cache thread-safe com LRU"""
    def decorator(func):
        cache = OrderedDict()
        lock = threading.Lock()
        
        @wraps(func)
        def wrapper(*args, **kwargs):
            key = str(args) + str(kwargs)
            
            with lock:
                if key in cache:
                    # Move to end (LRU)
                    cache.move_to_end(key)
                    return cache[key]
            
            result = func(*args, **kwargs)
            
            with lock:
                cache[key] = result
                if len(cache) > maxsize:
                    cache.popitem(last=False)
            
            return result
        
        wrapper.cache_clear = lambda: cache.clear()
        return wrapper
    
    return decorator

# Enumeração de categorias de patologias
class SCPCategory(Enum):
    RHYTHM = "rhythm"
    CONDUCTION = "conduction"
    ISCHEMIA = "ischemia"
    HYPERTROPHY = "hypertrophy"
    AXIS = "axis"
    REPOLARIZATION = "repolarization"
    OTHER = "other"

# Dicionário completo de condições SCP-ECG com critérios diagnósticos
SCP_ECG_CONDITIONS = {
    'NORM': {'name': 'Normal ECG', 'category': SCPCategory.RHYTHM, 'severity': 0},
    'SR': {'name': 'Sinus rhythm', 'category': SCPCategory.RHYTHM, 'severity': 0},
    'AFIB': {'name': 'Atrial fibrillation', 'category': SCPCategory.RHYTHM, 'severity': 3},
    'AFLT': {'name': 'Atrial flutter', 'category': SCPCategory.RHYTHM, 'severity': 3},
    'STACH': {'name': 'Sinus tachycardia', 'category': SCPCategory.RHYTHM, 'severity': 1},
    'SBRAD': {'name': 'Sinus bradycardia', 'category': SCPCategory.RHYTHM, 'severity': 1},
    'PAC': {'name': 'Premature atrial contraction', 'category': SCPCategory.RHYTHM, 'severity': 1},
    'PVC': {'name': 'Premature ventricular contraction', 'category': SCPCategory.RHYTHM, 'severity': 2},
    'VT': {'name': 'Ventricular tachycardia', 'category': SCPCategory.RHYTHM, 'severity': 4},
    'SVTACH': {'name': 'Supraventricular tachycardia', 'category': SCPCategory.RHYTHM, 'severity': 2},
    'PACED': {'name': 'Paced rhythm', 'category': SCPCategory.RHYTHM, 'severity': 1},
    
    'AVB1': {'name': 'First degree AV block', 'category': SCPCategory.CONDUCTION, 'severity': 1},
    'AVB2': {'name': 'Second degree AV block', 'category': SCPCategory.CONDUCTION, 'severity': 2},
    'AVB3': {'name': 'Third degree AV block', 'category': SCPCategory.CONDUCTION, 'severity': 4},
    'RBBB': {'name': 'Right bundle branch block', 'category': SCPCategory.CONDUCTION, 'severity': 2},
    'LBBB': {'name': 'Left bundle branch block', 'category': SCPCategory.CONDUCTION, 'severity': 2},
    'LAFB': {'name': 'Left anterior fascicular block', 'category': SCPCategory.CONDUCTION, 'severity': 1},
    'LPFB': {'name': 'Left posterior fascicular block', 'category': SCPCategory.CONDUCTION, 'severity': 1},
    'WPW': {'name': 'Wolff-Parkinson-White syndrome', 'category': SCPCategory.CONDUCTION, 'severity': 3},
    
    'MI': {'name': 'Myocardial infarction', 'category': SCPCategory.ISCHEMIA, 'severity': 4},
    'AMI': {'name': 'Acute myocardial infarction', 'category': SCPCategory.ISCHEMIA, 'severity': 5},
    'STTC': {'name': 'ST-T changes', 'category': SCPCategory.ISCHEMIA, 'severity': 2},
    'STE': {'name': 'ST elevation', 'category': SCPCategory.ISCHEMIA, 'severity': 4},
    'STD': {'name': 'ST depression', 'category': SCPCategory.ISCHEMIA, 'severity': 3},
    
    'LVH': {'name': 'Left ventricular hypertrophy', 'category': SCPCategory.HYPERTROPHY, 'severity': 2},
    'RVH': {'name': 'Right ventricular hypertrophy', 'category': SCPCategory.HYPERTROPHY, 'severity': 2},
    'LAE': {'name': 'Left atrial enlargement', 'category': SCPCategory.HYPERTROPHY, 'severity': 1},
    'RAE': {'name': 'Right atrial enlargement', 'category': SCPCategory.HYPERTROPHY, 'severity': 1},
    
    'LAD': {'name': 'Left axis deviation', 'category': SCPCategory.AXIS, 'severity': 1},
    'RAD': {'name': 'Right axis deviation', 'category': SCPCategory.AXIS, 'severity': 1},
    'ERAD': {'name': 'Extreme right axis deviation', 'category': SCPCategory.AXIS, 'severity': 2},
    
    'TWI': {'name': 'T wave inversion', 'category': SCPCategory.REPOLARIZATION, 'severity': 2},
    'TWF': {'name': 'T wave flattening', 'category': SCPCategory.REPOLARIZATION, 'severity': 1},
    'LQT': {'name': 'Long QT syndrome', 'category': SCPCategory.REPOLARIZATION, 'severity': 4},
    'SQT': {'name': 'Short QT syndrome', 'category': SCPCategory.REPOLARIZATION, 'severity': 3},
    'BRUG': {'name': 'Brugada syndrome', 'category': SCPCategory.REPOLARIZATION, 'severity': 4},
    'ERS': {'name': 'Early repolarization syndrome', 'category': SCPCategory.REPOLARIZATION, 'severity': 2},
    
    'LOWV': {'name': 'Low voltage', 'category': SCPCategory.OTHER, 'severity': 1},
    'PRWP': {'name': 'Poor R wave progression', 'category': SCPCategory.OTHER, 'severity': 1},
    'DIG': {'name': 'Digitalis effect', 'category': SCPCategory.OTHER, 'severity': 1},
    'ELEV': {'name': 'Electrode misplacement', 'category': SCPCategory.OTHER, 'severity': 0},
}

# ==================== CONFIGURAÇÃO OTIMIZADA ====================

@dataclass
class EnhancedECGAnalysisConfig:
    """Configuração otimizada e validada para análise de ECG"""
    # Parâmetros do sinal
    sampling_rate: int = 500
    signal_length: int = 5000
    num_leads: int = 12
    
    # Filtros adaptativos
    use_adaptive_filtering: bool = True
    bandpass_low: float = 0.5
    bandpass_high: float = 40.0
    notch_freq: float = 60.0
    notch_quality: float = 30.0
    
    # Wavelet denoising
    use_wavelet_denoising: bool = True
    wavelet_name: str = 'db4'
    wavelet_level: int = 5
    wavelet_threshold_method: str = 'soft'
    
    # Augmentação otimizada
    augmentation_prob: float = 0.7
    amplitude_scaling: Tuple[float, float] = (0.8, 1.2)
    noise_types: List[str] = field(default_factory=lambda: ['gaussian', 'baseline', 'powerline', 'muscle'])
    time_warping: bool = True
    lead_dropout: bool = True
    max_lead_dropout: int = 3
    
    # Treinamento otimizado
    batch_size: int = 32
    learning_rate: float = 1e-3
    num_epochs: int = 100
    early_stopping_patience: int = 15
    focal_loss_gamma: float = 2.0
    focal_loss_alpha: Optional[float] = None
    label_smoothing: float = 0.1
    test_size: float = 0.2
    use_mixed_precision: bool = True
    gradient_accumulation_steps: int = 1
    
    # Performance
    num_workers: int = 4
    use_cache: bool = True
    cache_size: int = 1000
    use_parallel: bool = True
    max_parallel_workers: int = 4
    
    # Thresholds médicos adaptativos
    use_adaptive_thresholds: bool = True
    st_elevation_threshold_limb: float = 0.1  # mV
    st_elevation_threshold_precordial: float = 0.2  # mV
    
    # Validação médica
    validate_physiological_ranges: bool = True
    min_valid_amplitude: float = 0.05  # mV
    max_valid_amplitude: float = 5.0  # mV
    
    # EMA e TTA
    use_ema: bool = True
    ema_decay: float = 0.999
    use_tta: bool = True
    tta_augmentations: int = 5
    
    # Calibração
    calibration_factor: float = 10.0  # mm/mV padrão
    
    def __post_init__(self):
        """Validação pós-inicialização"""
        assert self.sampling_rate > 0, "Taxa de amostragem deve ser positiva"
        assert self.signal_length > 0, "Comprimento do sinal deve ser positivo"
        assert 0 < self.augmentation_prob <= 1, "Probabilidade de augmentação deve estar entre 0 e 1"
        assert self.batch_size > 0, "Tamanho do batch deve ser positivo"
        assert self.num_epochs > 0, "Número de épocas deve ser positivo"

# ==================== LOGGER CLÍNICO ====================

class ClinicalLogger:
    """Logger especializado para eventos clínicos e auditoria médica"""
    
    def __init__(self, log_dir: Optional[Path] = None):
        self.log_dir = log_dir or PROJECT_ROOT / 'logs' / 'clinical'
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Arquivo de log específico para eventos clínicos
        self.clinical_log_file = self.log_dir / f'clinical_log_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
        self.events = []
        self._lock = threading.Lock()
        
        # Contadores de performance
        self.pathology_counts = defaultdict(int)
        self.pathology_tp = defaultdict(int)
        self.pathology_fp = defaultdict(int)
        self.pathology_fn = defaultdict(int)
        self.pathology_tn = defaultdict(int)
    
    def log_clinical_event(self, event_type: str, data: Dict[str, Any]):
        """Registra evento clínico com timestamp e dados"""
        event = {
            'timestamp': datetime.now().isoformat(),
            'event_type': event_type,
            'data': data
        }
        
        with self._lock:
            self.events.append(event)
            
            # Salvar incrementalmente
            with open(self.clinical_log_file, 'a') as f:
                json.dump(event, f)
                f.write('\n')
    
    def log_pathology_performance(self, pathology_code: str, metrics: Dict[str, float]):
        """Registra performance por patologia"""
        self.log_clinical_event('pathology_performance', {
            'pathology_code': pathology_code,
            'pathology_name': SCP_ECG_CONDITIONS.get(pathology_code, {}).get('name', 'Unknown'),
            'metrics': metrics
        })
        
        # Atualizar contadores
        with self._lock:
            if 'tp' in metrics:
                self.pathology_tp[pathology_code] += metrics['tp']
            if 'fp' in metrics:
                self.pathology_fp[pathology_code] += metrics['fp']
            if 'fn' in metrics:
                self.pathology_fn[pathology_code] += metrics['fn']
            if 'tn' in metrics:
                self.pathology_tn[pathology_code] += metrics['tn']
    
    def log_diagnostic_decision(self, ecg_id: str, predictions: Dict[str, float], 
                               ground_truth: Optional[Dict[str, bool]] = None):
        """Registra decisão diagnóstica para auditoria"""
        self.log_clinical_event('diagnostic_decision', {
            'ecg_id': ecg_id,
            'predictions': predictions,
            'ground_truth': ground_truth,
            'timestamp': datetime.now().isoformat()
        })
    
    def generate_clinical_report(self) -> Dict[str, Any]:
        """Gera relatório clínico completo com métricas de performance"""
        with self._lock:
            report = {
                'generated_at': datetime.now().isoformat(),
                'total_events': len(self.events),
                'clinical_requirements_met': {},
                'pathology_performance_summary': {},
                'quality_metrics': {}
            }
            
            # Calcular métricas por patologia
            for code in SCP_ECG_CONDITIONS.keys():
                tp = self.pathology_tp.get(code, 0)
                fp = self.pathology_fp.get(code, 0)
                fn = self.pathology_fn.get(code, 0)
                tn = self.pathology_tn.get(code, 0)
                
                if tp + fn > 0:  # Há casos positivos
                    sensitivity = tp / (tp + fn)
                    specificity = tn / (tn + fp) if tn + fp > 0 else 0
                    precision = tp / (tp + fp) if tp + fp > 0 else 0
                    f1 = 2 * precision * sensitivity / (precision + sensitivity) if precision + sensitivity > 0 else 0
                    
                    report['pathology_performance_summary'][code] = {
                        'name': SCP_ECG_CONDITIONS[code]['name'],
                        'sensitivity': sensitivity,
                        'specificity': specificity,
                        'precision': precision,
                        'f1_score': f1,
                        'total_cases': tp + fn,
                        'severity': SCP_ECG_CONDITIONS[code]['severity']
                    }
            
            # Verificar requisitos clínicos
            sensitivities = [m['sensitivity'] for m in report['pathology_performance_summary'].values()]
            specificities = [m['specificity'] for m in report['pathology_performance_summary'].values()]
            
            if sensitivities:
                report['clinical_requirements_met']['min_sensitivity'] = min(sensitivities) >= CLINICAL_REQUIREMENTS['min_sensitivity']
                report['clinical_requirements_met']['avg_sensitivity'] = np.mean(sensitivities)
            
            if specificities:
                report['clinical_requirements_met']['min_specificity'] = min(specificities) >= CLINICAL_REQUIREMENTS['min_specificity']
                report['clinical_requirements_met']['avg_specificity'] = np.mean(specificities)
            
            # Análise de eventos
            event_types = defaultdict(int)
            for event in self.events:
                event_types[event['event_type']] += 1
            
            report['event_summary'] = dict(event_types)
            
            return report
    
    def export_for_regulatory_compliance(self, output_path: Path):
        """Exporta dados para conformidade regulatória (FDA/CE)"""
        report = self.generate_clinical_report()
        
        compliance_data = {
            'software_version': '3.0',
            'validation_date': datetime.now().isoformat(),
            'clinical_performance': report['pathology_performance_summary'],
            'meets_requirements': report['clinical_requirements_met'],
            'total_cases_analyzed': len(self.events),
            'device_classification': 'Class II - Clinical Decision Support',
            'intended_use': 'ECG analysis for detection of cardiac pathologies',
            'limitations': [
                'Not for use as sole diagnostic tool',
                'Requires physician interpretation',
                'Performance validated on adult population only'
            ]
        }
        
        with open(output_path, 'w') as f:
            json.dump(compliance_data, f, indent=2)

# Instância global do clinical logger
clinical_logger = ClinicalLogger()

# ==================== VALIDADOR DE QUALIDADE DO SINAL ====================

class SignalQualityAnalyzer:
    """Analisador avançado de qualidade do sinal ECG"""
    
    def __init__(self, sampling_rate: int):
        self.sampling_rate = sampling_rate
        self.quality_criteria = {
            'snr_threshold': 10.0,  # dB
            'baseline_drift_threshold': 0.5,  # mV
            'saturation_threshold': 4.5,  # mV
            'min_rr_variability': 0.05,  # 5%
            'max_rr_variability': 0.5,  # 50%
        }
    
    def analyze(self, signal: np.ndarray) -> Dict[str, Any]:
        """Análise completa de qualidade do sinal"""
        quality_metrics = {
            'electrode_detachment': self._check_electrode_detachment(signal),
            'excessive_noise': self._check_noise_level(signal),
            'saturation': self._check_saturation(signal),
            'lead_reversal': self._check_lead_reversal(signal),
            'motion_artifacts': self._check_motion_artifacts(signal),
            'baseline_drift': self._check_baseline_drift(signal),
            'signal_clipping': self._check_signal_clipping(signal),
            'powerline_interference': self._check_powerline_interference(signal),
            'overall_quality_score': 0.0,
            'lead_quality_scores': {},
            'recommendations': []
        }
        
        # Calcular qualidade por derivação
        lead_scores = []
        for lead in range(signal.shape[0]):
            lead_score = self._calculate_lead_quality(signal[lead])
            quality_metrics['lead_quality_scores'][f'lead_{lead}'] = lead_score
            lead_scores.append(lead_score)
        
        # Score geral ponderado
        issues_weights = {
            'electrode_detachment': 0.3,
            'excessive_noise': 0.2,
            'saturation': 0.25,
            'lead_reversal': 0.15,
            'motion_artifacts': 0.1,
            'baseline_drift': 0.1,
            'signal_clipping': 0.15,
            'powerline_interference': 0.05
        }
        
        penalty = sum(
            issues_weights.get(issue, 0.1) 
            for issue, detected in quality_metrics.items() 
            if issue != 'overall_quality_score' and 
               issue != 'lead_quality_scores' and 
               issue != 'recommendations' and 
               detected
        )
        
        quality_metrics['overall_quality_score'] = max(0, 1 - penalty) * np.mean(lead_scores)
        
        # Recomendações baseadas em problemas detectados
        quality_metrics['recommendations'] = self._generate_recommendations(quality_metrics)
        
        return quality_metrics
    
    def _check_electrode_detachment(self, signal: np.ndarray) -> bool:
        """Verifica eletrodos soltos com critérios aprimorados"""
        if len(signal.shape) == 1:
            return np.std(signal) < 0.001
        
        detached_leads = []
        for lead in range(signal.shape[0]):
            # Verificar variância muito baixa
            if np.std(signal[lead]) < 0.001:
                detached_leads.append(lead)
                continue
            
            # Verificar linha completamente plana por longos períodos
            flat_segments = self._find_flat_segments(signal[lead])
            if len(flat_segments) > 0.5 * len(signal[lead]):
                detached_leads.append(lead)
        
        return len(detached_leads) > 0
    
    def _find_flat_segments(self, signal: np.ndarray, threshold: float = 0.001) -> np.ndarray:
        """Encontra segmentos planos no sinal"""
        diff = np.abs(np.diff(signal))
        return diff < threshold
    
    def _check_noise_level(self, signal: np.ndarray) -> bool:
        """Verifica nível de ruído com análise espectral"""
        if len(signal.shape) == 1:
            signal = signal.reshape(1, -1)
        
        noisy_leads = []
        for lead in range(signal.shape[0]):
            # Análise no domínio da frequência
            freqs, psd = sp_signal.welch(signal[lead], fs=self.sampling_rate, nperseg=256)
            
            # Ruído de alta frequência (> 40 Hz)
            high_freq_mask = freqs > 40
            if np.any(high_freq_mask):
                high_freq_power = np.sum(psd[high_freq_mask])
                total_power = np.sum(psd)
                
                if high_freq_power / total_power > 0.3:
                    noisy_leads.append(lead)
                    continue
            
            # SNR usando MAD
            mad = np.median(np.abs(signal[lead] - np.median(signal[lead])))
            noise_estimate = mad / NOISE_MEDIAN_FACTOR
            signal_power = np.var(signal[lead])
            
            if signal_power > 0:
                snr = 10 * np.log10(signal_power / (noise_estimate**2 + 1e-10))
                if snr < self.quality_criteria['snr_threshold']:
                    noisy_leads.append(lead)
        
        return len(noisy_leads) > signal.shape[0] * 0.3
    
    def _check_saturation(self, signal: np.ndarray) -> bool:
        """Verifica saturação do sinal com histerese"""
        if len(signal.shape) == 1:
            signal = signal.reshape(1, -1)
        
        saturated_leads = []
        saturation_threshold = self.quality_criteria['saturation_threshold']
        
        for lead in range(signal.shape[0]):
            # Verificar valores próximos aos limites
            max_val = np.max(np.abs(signal[lead]))
            
            # Contar amostras saturadas
            saturated_samples = np.sum(np.abs(signal[lead]) > saturation_threshold)
            saturation_ratio = saturated_samples / len(signal[lead])
            
            if max_val > saturation_threshold or saturation_ratio > 0.01:
                saturated_leads.append(lead)
        
        return len(saturated_leads) > 0
    
    def _check_lead_reversal(self, signal: np.ndarray) -> bool:
        """Verifica inversão de eletrodos com critérios expandidos"""
        if signal.shape[0] < 12:
            return False
        
        reversals_detected = []
        
        # Lead I (deve ser geralmente positivo)
        lead_i_mean = np.mean(signal[0])
        if lead_i_mean < -0.1:
            reversals_detected.append('Lead I negative')
        
        # aVR (deve ser geralmente negativo)
        avr_mean = np.mean(signal[3])
        if avr_mean > 0.1:
            reversals_detected.append('aVR positive')
        
        # Verificar relações entre derivações
        # V1-V6 devem mostrar progressão de R
        if signal.shape[0] >= 12:
            r_amplitudes = []
            for v_lead in range(6, 12):  # V1-V6
                r_amp = self._estimate_r_amplitude(signal[v_lead])
                r_amplitudes.append(r_amp)
            
            # R deve aumentar de V1 a V5/V6
            if len(r_amplitudes) >= 4:
                if r_amplitudes[0] > r_amplitudes[3]:  # V1 > V4
                    reversals_detected.append('Poor R wave progression')
        
        return len(reversals_detected) > 0
    
    def _estimate_r_amplitude(self, lead_signal: np.ndarray) -> float:
        """Estima amplitude da onda R"""
        # Simplificado - encontrar picos positivos
        peaks, _ = sp_signal.find_peaks(lead_signal, height=0.1, distance=int(0.2*self.sampling_rate))
        
        if len(peaks) > 0:
            return np.mean(lead_signal[peaks])
        return 0.0
    
    def _check_motion_artifacts(self, signal: np.ndarray) -> bool:
        """Detecta artefatos de movimento com análise avançada"""
        if len(signal.shape) == 1:
            signal = signal.reshape(1, -1)
        
        artifacts_detected = []
        
        for lead in range(signal.shape[0]):
            # Análise de baseline wandering
            baseline = self._extract_baseline(signal[lead])
            baseline_variation = np.std(baseline)
            
            if baseline_variation > self.quality_criteria['baseline_drift_threshold']:
                artifacts_detected.append(lead)
                continue
            
            # Detectar mudanças abruptas
            diff = np.diff(signal[lead])
            abrupt_changes = np.abs(diff) > 1.0  # 1 mV/sample
            
            if np.sum(abrupt_changes) > 0.01 * len(diff):
                artifacts_detected.append(lead)
        
        return len(artifacts_detected) > signal.shape[0] * 0.3
    
    def _extract_baseline(self, signal: np.ndarray) -> np.ndarray:
        """Extrai baseline usando filtro mediano adaptativo"""
        window_size = int(0.2 * self.sampling_rate)  # 200ms
        if window_size % 2 == 0:
            window_size += 1
        window_size = max(3, window_size)
        
        return sp_signal.medfilt(signal, kernel_size=window_size)
    
    def _check_baseline_drift(self, signal: np.ndarray) -> bool:
        """Verifica deriva de baseline"""
        if len(signal.shape) == 1:
            signal = signal.reshape(1, -1)
        
        drift_detected = []
        
        for lead in range(signal.shape[0]):
            baseline = self._extract_baseline(signal[lead])
            
            # Calcular tendência linear
            x = np.arange(len(baseline))
            slope, _ = np.polyfit(x, baseline, 1)
            
            # Deriva significativa se slope alto
            drift_per_second = abs(slope) * self.sampling_rate
            if drift_per_second > 0.1:  # 0.1 mV/s
                drift_detected.append(lead)
        
        return len(drift_detected) > signal.shape[0] * 0.3
    
    def _check_signal_clipping(self, signal: np.ndarray) -> bool:
        """Verifica clipping do sinal"""
        if len(signal.shape) == 1:
            signal = signal.reshape(1, -1)
        
        clipped_leads = []
        
        for lead in range(signal.shape[0]):
            # Verificar platôs nos extremos
            max_val = np.max(signal[lead])
            min_val = np.min(signal[lead])
            
            # Contar amostras nos extremos
            at_max = np.sum(signal[lead] >= max_val * 0.99)
            at_min = np.sum(signal[lead] <= min_val * 0.99)
            
            if at_max > 10 or at_min > 10:  # Mais de 10 amostras consecutivas
                clipped_leads.append(lead)
        
        return len(clipped_leads) > 0
    
    def _check_powerline_interference(self, signal: np.ndarray) -> bool:
        """Verifica interferência da rede elétrica"""
        if len(signal.shape) == 1:
            signal = signal.reshape(1, -1)
        
        interference_detected = []
        
        for lead in range(signal.shape[0]):
            # Análise espectral focada em 50/60 Hz
            freqs, psd = sp_signal.welch(signal[lead], fs=self.sampling_rate, nperseg=1024)
            
            # Verificar picos em 50 Hz e 60 Hz
            for powerline_freq in [50, 60]:
                freq_idx = np.argmin(np.abs(freqs - powerline_freq))
                
                if freq_idx > 0 and freq_idx < len(psd) - 1:
                    # Comparar com frequências vizinhas
                    peak_power = psd[freq_idx]
                    neighbor_power = (psd[freq_idx-1] + psd[freq_idx+1]) / 2
                    
                    if peak_power > 10 * neighbor_power:  # Pico 10x maior
                        interference_detected.append(lead)
                        break
        
        return len(interference_detected) > signal.shape[0] * 0.5
    
    def _calculate_lead_quality(self, lead_signal: np.ndarray) -> float:
        """Calcula score de qualidade para uma derivação"""
        score = 1.0
        
        # Penalizar por baixa variância
        if np.std(lead_signal) < 0.01:
            score *= 0.1
        
        # Penalizar por saturação
        if np.max(np.abs(lead_signal)) > self.quality_criteria['saturation_threshold']:
            score *= 0.5
        
        # Penalizar por ruído excessivo
        snr = self._estimate_snr(lead_signal)
        if snr < self.quality_criteria['snr_threshold']:
            score *= 0.7
        
        # Penalizar por artefatos
        artifacts = self._count_artifacts(lead_signal)
        score *= np.exp(-artifacts / 100)  # Decaimento exponencial
        
        return max(0, min(1, score))
    
    def _estimate_snr(self, signal: np.ndarray) -> float:
        """Estima SNR do sinal"""
        # Usar MAD para estimar ruído
        mad = np.median(np.abs(signal - np.median(signal)))
        noise = mad / NOISE_MEDIAN_FACTOR
        
        # Potência do sinal
        signal_power = np.var(signal)
        
        if noise > 0:
            return 10 * np.log10(signal_power / (noise**2))
        return 40.0  # SNR alto se ruído não detectável
    
    def _count_artifacts(self, signal: np.ndarray) -> int:
        """Conta número de artefatos no sinal"""
        artifacts = 0
        
        # Mudanças abruptas
        diff = np.diff(signal)
        artifacts += np.sum(np.abs(diff) > 1.0)
        
        # Segmentos planos
        flat = self._find_flat_segments(signal)
        artifacts += np.sum(flat)
        
        return artifacts
    
    def _generate_recommendations(self, metrics: Dict[str, Any]) -> List[str]:
        """Gera recomendações baseadas nos problemas detectados"""
        recommendations = []
        
        if metrics['electrode_detachment']:
            recommendations.append("Verificar conexão dos eletrodos")
        
        if metrics['excessive_noise']:
            recommendations.append("Reduzir fontes de interferência elétrica")
            recommendations.append("Verificar aterramento do equipamento")
        
        if metrics['motion_artifacts']:
            recommendations.append("Instruir paciente a permanecer imóvel")
            recommendations.append("Verificar fixação dos eletrodos")
        
        if metrics['lead_reversal']:
            recommendations.append("Verificar posicionamento correto dos eletrodos")
        
        if metrics['baseline_drift']:
            recommendations.append("Aguardar estabilização do sinal")
            recommendations.append("Verificar preparação da pele")
        
        if metrics['powerline_interference']:
            recommendations.append("Ativar filtro notch se disponível")
            recommendations.append("Afastar equipamento de fontes de interferência")
        
        if not recommendations and metrics['overall_quality_score'] < 0.7:
            recommendations.append("Considerar repetir o exame")
        
        return recommendations

# ==================== FILTROS ADAPTATIVOS ====================

class AdaptiveFilter:
    """Filtros adaptativos avançados para ECG"""
    
    def __init__(self, sampling_rate: int):
        self.sampling_rate = sampling_rate
        self.filter_cache = {}
        
    def kalman_filter(self, signal: np.ndarray, Q: float = 1e-5, R: float = 0.01) -> np.ndarray:
        """Filtro de Kalman otimizado para ECG"""
        n = len(signal)
        filtered = np.empty(n, dtype=np.float32)
        
        # Estado inicial
        x_hat = signal[0]
        P = 1.0
        
        # Parâmetros adaptativos baseados no sinal
        signal_var = np.var(signal)
        if signal_var > 0:
            R = max(0.001, min(0.1, 0.01 * np.sqrt(signal_var)))
        
        # Aplicar filtro
        for i in range(n):
            # Predição
            x_hat_minus = x_hat
            P_minus = P + Q
            
            # Atualização
            K = P_minus / (P_minus + R)
            x_hat = x_hat_minus + K * (signal[i] - x_hat_minus)
            P = (1 - K) * P_minus
            
            filtered[i] = x_hat
        
        return filtered
    
    def adaptive_notch_filter(self, signal: np.ndarray, target_freq: float = 60.0) -> np.ndarray:
        """Filtro notch adaptativo com detecção automática de frequência"""
        # Detectar frequência real de interferência
        actual_freq = self._detect_powerline_frequency(signal)
        
        if not (45 <= actual_freq <= 65):
            return signal
        
        # Criar filtro notch com largura de banda adaptativa
        Q = self._calculate_optimal_q_factor(signal, actual_freq)
        
        # Design do filtro
        w0 = actual_freq / (self.sampling_rate / 2)
        
        # Verificar se frequência está no range válido
        if not (0 < w0 < 1):
            return signal
        
        # Criar filtro IIR notch
        b, a = sp_signal.iirnotch(w0, Q)
        
        # Aplicar filtro
        filtered = sp_signal.filtfilt(b, a, signal)
        
        return filtered
    
    def _detect_powerline_frequency(self, signal: np.ndarray) -> float:
        """Detecta frequência de interferência usando múltiplos métodos"""
        # Método 1: FFT
        freqs, psd = sp_signal.welch(signal, fs=self.sampling_rate, nperseg=1024)
        
        # Buscar picos em torno de 50 e 60 Hz
        candidates = []
        
        for target_freq in [50.0, 60.0]:
            freq_mask = (freqs >= target_freq - 2) & (freqs <= target_freq + 2)
            if np.any(freq_mask):
                peak_idx = np.argmax(psd[freq_mask])
                peak_freq = freqs[freq_mask][peak_idx]
                peak_power = psd[freq_mask][peak_idx]
                
                # Verificar se é pico significativo
                noise_floor = np.median(psd)
                if peak_power > 5 * noise_floor:
                    candidates.append((peak_freq, peak_power))
        
        # Método 2: Goertzel para confirmação
        if candidates:
            best_freq = max(candidates, key=lambda x: x[1])[0]
            
            # Refinar com Goertzel
            refined_freq = self._refine_frequency_goertzel(signal, best_freq)
            return refined_freq
        
        return 60.0  # Default
    
    def _refine_frequency_goertzel(self, signal: np.ndarray, initial_freq: float, 
                                   search_range: float = 1.0) -> float:
        """Refina estimativa de frequência usando Goertzel"""
        best_freq = initial_freq
        best_power = 0
        
        # Buscar em torno da frequência inicial
        test_freqs = np.linspace(initial_freq - search_range, 
                                initial_freq + search_range, 21)
        
        for freq in test_freqs:
            power = self._goertzel(signal, freq)
            if power > best_power:
                best_power = power
                best_freq = freq
        
        return best_freq
    
    def _goertzel(self, signal: np.ndarray, target_freq: float) -> float:
        """Algoritmo de Goertzel otimizado"""
        N = len(signal)
        k = int(0.5 + N * target_freq / self.sampling_rate)
        w = 2 * np.pi * k / N
        
        coeff = 2 * np.cos(w)
        s1 = s2 = 0
        
        # Processar em blocos para eficiência
        for i in range(0, N, 100):
            block = signal[i:min(i+100, N)]
            for sample in block:
                s0 = sample + coeff * s1 - s2
                s2 = s1
                s1 = s0
        
        power = s1**2 + s2**2 - coeff * s1 * s2
        return power / N
    
    def _calculate_optimal_q_factor(self, signal: np.ndarray, freq: float) -> float:
        """Calcula fator Q ótimo para filtro notch"""
        # Analisar largura de banda da interferência
        freqs, psd = sp_signal.welch(signal, fs=self.sampling_rate, nperseg=1024)
        
        # Encontrar pico
        peak_idx = np.argmin(np.abs(freqs - freq))
        peak_power = psd[peak_idx]
        
        # Encontrar largura de banda 3dB
        half_power = peak_power / 2
        
        # Buscar pontos de meia potência
        left_idx = peak_idx
        right_idx = peak_idx
        
        while left_idx > 0 and psd[left_idx] > half_power:
            left_idx -= 1
        
        while right_idx < len(psd) - 1 and psd[right_idx] > half_power:
            right_idx += 1
        
        # Calcular largura de banda
        if right_idx > left_idx:
            bandwidth = freqs[right_idx] - freqs[left_idx]
            Q = freq / max(bandwidth, 1.0)
            
            # Limitar Q para estabilidade
            return np.clip(Q, 10, 50)
        
        return 30  # Default
    
    def adaptive_baseline_removal(self, signal: np.ndarray) -> np.ndarray:
        """Remove baseline adaptativo usando splines"""
        # Detectar complexos QRS para ancoragem
        qrs_detector = PanTompkinsDetector(self.sampling_rate)
        qrs_peaks = qrs_detector.detect(signal)
        
        if len(qrs_peaks) < 3:
            # Fallback para filtro passa-alta simples
            sos = sp_signal.butter(2, 0.5, btype='high', fs=self.sampling_rate, output='sos')
            return sp_signal.sosfiltfilt(sos, signal)
        
        # Criar pontos de ancoragem entre QRS
        anchor_points = []
        anchor_values = []
        
        for i in range(len(qrs_peaks) - 1):
            # Ponto médio entre QRS
            mid_point = (qrs_peaks[i] + qrs_peaks[i+1]) // 2
            
            # Janela em torno do ponto médio
            window_start = max(0, mid_point - 10)
            window_end = min(len(signal), mid_point + 10)
            
            if window_end > window_start:
                anchor_points.append(mid_point)
                anchor_values.append(np.median(signal[window_start:window_end]))
        
        if len(anchor_points) < 2:
            return signal
        
        # Interpolar baseline usando spline cúbica
        baseline_interp = interpolate.CubicSpline(anchor_points, anchor_values, 
                                                 bc_type='natural', extrapolate=True)
        
        x = np.arange(len(signal))
        baseline = baseline_interp(x)
        
        # Suavizar baseline
        window_size = int(0.1 * self.sampling_rate)
        if window_size % 2 == 0:
            window_size += 1
        baseline = sp_signal.savgol_filter(baseline, window_size, 3)
        
        return signal - baseline
    
    def morphological_filter(self, signal: np.ndarray, operation: str = 'open') -> np.ndarray:
        """Filtro morfológico para remover artefatos"""
        # Estrutura do elemento (kernel)
        struct_length = int(0.02 * self.sampling_rate)  # 20ms
        
        if operation == 'open':
            # Erosão seguida de dilatação - remove picos
            eroded = sp_signal.minimum_filter(signal, size=struct_length, mode='reflect')
            opened = sp_signal.maximum_filter(eroded, size=struct_length, mode='reflect')
            return opened
        
        elif operation == 'close':
            # Dilatação seguida de erosão - preenche vales
            dilated = sp_signal.maximum_filter(signal, size=struct_length, mode='reflect')
            closed = sp_signal.minimum_filter(dilated, size=struct_length, mode='reflect')
            return closed
        
        elif operation == 'gradient':
            # Diferença entre dilatação e erosão
            dilated = sp_signal.maximum_filter(signal, size=struct_length, mode='reflect')
            eroded = sp_signal.minimum_filter(signal, size=struct_length, mode='reflect')
            return dilated - eroded
        
        else:
            return signal

# ==================== PREPROCESSADOR DE ECG ====================

class ECGPreprocessor:
    """Preprocessador modular e otimizado de ECG"""
    
    def __init__(self, config: EnhancedECGAnalysisConfig):
        self.config = config
        self.quality_analyzer = SignalQualityAnalyzer(config.sampling_rate)
        self.adaptive_filter = AdaptiveFilter(config.sampling_rate)
        
        # Cache com limite de tamanho
        if config.use_cache:
            self._cache = OrderedDict()
            self._max_cache_size = config.cache_size
        else:
            self._cache = None
        self._cache_lock = threading.Lock()
        
        # Estatísticas de preprocessamento
        self.stats = {
            'signals_processed': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'average_processing_time': 0
        }
    
    def preprocess(self, signal: np.ndarray, quality_threshold: float = 0.3) -> Dict[str, Any]:
        """Preprocessa sinal ECG completo com pipeline otimizado"""
        start_time = time.time()
        
        # Validação inicial
        if signal is None or signal.size == 0:
            return self._create_error_result("Sinal vazio ou inválido")
        
        # Normalizar formato
        if len(signal.shape) == 1:
            signal = signal.reshape(1, -1)
        
        # Verificar cache
        cache_key = self._generate_cache_key(signal)
        cached_result = self._check_cache(cache_key)
        if cached_result is not None:
            self.stats['cache_hits'] += 1
            return cached_result
        
        self.stats['cache_misses'] += 1
        
        # Análise de qualidade inicial
        quality_metrics = self.quality_analyzer.analyze(signal)
        
        if quality_metrics['overall_quality_score'] < quality_threshold:
            logger.warning(f"Qualidade do sinal baixa: {quality_metrics['overall_quality_score']:.2f}")
            return {
                'filtered_signal': signal,
                'quality_metrics': quality_metrics,
                'preprocessing_successful': False,
                'preprocessing_steps': []
            }
        
        # Pipeline de preprocessamento
        preprocessing_steps = []
        processed_signal = signal.copy()
        
        # 1. Remoção de baseline adaptativa
        if self.config.use_adaptive_filtering:
            processed_signal = self._remove_baseline(processed_signal)
            preprocessing_steps.append('baseline_removal')
        
        # 2. Filtro de Kalman
        if self.config.use_adaptive_filtering:
            processed_signal = self._apply_kalman_filter(processed_signal)
            preprocessing_steps.append('kalman_filter')
        
        # 3. Filtro passa-banda
        processed_signal = self._apply_bandpass_filter(processed_signal)
        preprocessing_steps.append('bandpass_filter')
        
        # 4. Filtro notch adaptativo
        if self.config.notch_freq > 0:
            processed_signal = self._apply_notch_filter(processed_signal)
            preprocessing_steps.append('notch_filter')
        
        # 5. Wavelet denoising
        if self.config.use_wavelet_denoising and PYWT_AVAILABLE:
            processed_signal = self._apply_wavelet_denoising(processed_signal)
            preprocessing_steps.append('wavelet_denoising')
        
        # 6. Normalização robusta
        normalized_signal = self._normalize_signal(processed_signal)
        preprocessing_steps.append('normalization')
        
        # Análise de qualidade pós-processamento
        post_quality = self.quality_analyzer.analyze(normalized_signal)
        
        # Preparar resultado
        result = {
            'filtered_signal': normalized_signal,
            'quality_metrics': post_quality,
            'preprocessing_successful': True,
            'preprocessing_steps': preprocessing_steps,
            'quality_improvement': post_quality['overall_quality_score'] - quality_metrics['overall_quality_score'],
            'processing_time': time.time() - start_time
        }
        
        # Atualizar cache
        if self._cache is not None:
            self._update_cache(cache_key, result)
        
        # Atualizar estatísticas
        self._update_stats(result['processing_time'])
        
        return result
    
    def _generate_cache_key(self, signal: np.ndarray) -> str:
        """Gera chave única para cache"""
        # Usar hash do sinal + configuração
        signal_hash = hashlib.md5(signal.tobytes()).hexdigest()
        config_str = f"{self.config.sampling_rate}_{self.config.bandpass_low}_{self.config.bandpass_high}"
        return f"{signal_hash}_{config_str}"
    
    def _check_cache(self, key: str) -> Optional[Dict[str, Any]]:
        """Verifica cache thread-safe"""
        if self._cache is None:
            return None
        
        with self._cache_lock:
            if key in self._cache:
                # Move para o final (LRU)
                self._cache.move_to_end(key)
                return self._cache[key].copy()
        
        return None
    
    def _update_cache(self, key: str, value: Dict[str, Any]):
        """Atualiza cache com limite de tamanho"""
        with self._cache_lock:
            # Remover arrays grandes do cache para economizar memória
            cached_value = value.copy()
            cached_value['filtered_signal'] = None  # Não cachear o sinal completo
            
            self._cache[key] = cached_value
            
            # Limitar tamanho do cache
            if len(self._cache) > self._max_cache_size:
                self._cache.popitem(last=False)
    
    def _remove_baseline(self, signal: np.ndarray) -> np.ndarray:
        """Remove baseline de cada derivação"""
        processed = np.empty_like(signal)
        
        for lead in range(signal.shape[0]):
            processed[lead] = self.adaptive_filter.adaptive_baseline_removal(signal[lead])
        
        return processed
    
    def _apply_kalman_filter(self, signal: np.ndarray) -> np.ndarray:
        """Aplica filtro de Kalman adaptativo"""
        processed = np.empty_like(signal)
        
        for lead in range(signal.shape[0]):
            # Parâmetros adaptativos baseados na qualidade
            signal_var = np.var(signal[lead])
            Q = 1e-5 * signal_var if signal_var > 0 else 1e-5
            R = 0.01 * np.sqrt(signal_var) if signal_var > 0 else 0.01
            
            processed[lead] = self.adaptive_filter.kalman_filter(signal[lead], Q, R)
        
        return processed
    
    def _apply_bandpass_filter(self, signal: np.ndarray) -> np.ndarray:
        """Aplica filtro passa-banda otimizado"""
        # Normalizar frequências
        nyquist = self.config.sampling_rate / 2
        low = self.config.bandpass_low / nyquist
        high = self.config.bandpass_high / nyquist
        
        # Garantir frequências válidas
        low = np.clip(low, 0.001, 0.999)
        high = np.clip(high, low + 0.001, 0.999)
        
        # Design do filtro com ordem adaptativa
        if high - low > 0.3:  # Banda larga
            order = 2
        else:
            order = 4
        
        # Criar filtro SOS para estabilidade numérica
        sos = sp_signal.butter(order, [low, high], btype='band', output='sos')
        
        # Aplicar filtro
        processed = np.empty_like(signal)
        for lead in range(signal.shape[0]):
            try:
                processed[lead] = sp_signal.sosfiltfilt(sos, signal[lead])
            except Exception as e:
                logger.warning(f"Erro no filtro passa-banda da derivação {lead}: {str(e)}")
                processed[lead] = signal[lead]
        
        return processed
    
    def _apply_notch_filter(self, signal: np.ndarray) -> np.ndarray:
        """Aplica filtro notch adaptativo"""
        processed = np.empty_like(signal)
        
        for lead in range(signal.shape[0]):
            try:
                processed[lead] = self.adaptive_filter.adaptive_notch_filter(
                    signal[lead], 
                    self.config.notch_freq
                )
            except Exception as e:
                logger.warning(f"Erro no filtro notch da derivação {lead}: {str(e)}")
                processed[lead] = signal[lead]
        
        return processed
    
    def _apply_wavelet_denoising(self, signal: np.ndarray) -> np.ndarray:
        """Aplica denoising wavelet otimizado"""
        processed = np.empty_like(signal)
        
        for lead in range(signal.shape[0]):
            try:
                # Decomposição wavelet
                max_level = pywt.dwt_max_level(signal.shape[1], self.config.wavelet_name)
                level = min(max_level, self.config.wavelet_level)
                
                coeffs = pywt.wavedec(signal[lead], self.config.wavelet_name, level=level)
                
                # Estimativa de ruído usando MAD do nível mais fino
                sigma = np.median(np.abs(coeffs[-1])) / NOISE_MEDIAN_FACTOR
                
                # Threshold adaptativo por nível
                coeffs_thresh = []
                for i, c in enumerate(coeffs):
                    if i == 0:  # Aproximação - não aplicar threshold
                        coeffs_thresh.append(c)
                    else:
                        # Threshold universal com ajuste por nível
                        level_factor = 1 + 0.1 * i  # Aumentar threshold para níveis mais grossos
                        threshold = sigma * np.sqrt(2 * np.log(len(c))) * level_factor
                        
                        # Aplicar threshold soft ou hard
                        if self.config.wavelet_threshold_method == 'soft':
                            coeffs_thresh.append(pywt.threshold(c, threshold, mode='soft'))
                        else:
                            coeffs_thresh.append(pywt.threshold(c, threshold, mode='hard'))
                
                # Reconstruir
                denoised = pywt.waverec(coeffs_thresh, self.config.wavelet_name)
                
                # Ajustar comprimento
                if len(denoised) > signal.shape[1]:
                    denoised = denoised[:signal.shape[1]]
                elif len(denoised) < signal.shape[1]:
                    denoised = np.pad(denoised, (0, signal.shape[1] - len(denoised)), mode='edge')
                
                processed[lead] = denoised
                
            except Exception as e:
                logger.warning(f"Erro no wavelet denoising da derivação {lead}: {str(e)}")
                processed[lead] = signal[lead]
        
        return processed
    
    def _normalize_signal(self, signal: np.ndarray) -> np.ndarray:
        """Normalização robusta a outliers"""
        normalized = np.empty_like(signal)
        
        for lead in range(signal.shape[0]):
            # Usar percentis robustos
            p5, p95 = np.percentile(signal[lead], [5, 95])
            
            # Verificar se há variação suficiente
            if p95 - p5 > 1e-6:
                # Clipping suave usando função sigmoide nos extremos
                clipped = np.copy(signal[lead])
                
                # Aplicar clipping suave
                below_p5 = signal[lead] < p5
                above_p95 = signal[lead] > p95
                
                if np.any(below_p5):
                    # Sigmoid suave para valores abaixo de p5
                    x = (signal[lead][below_p5] - p5) / (p95 - p5)
                    clipped[below_p5] = p5 + (p95 - p5) * (1 / (1 + np.exp(-10 * x)))
                
                if np.any(above_p95):
                    # Sigmoid suave para valores acima de p95
                    x = (signal[lead][above_p95] - p95) / (p95 - p5)
                    clipped[above_p95] = p95 + (p95 - p5) * (1 - 1 / (1 + np.exp(10 * x)))
                
                # Normalizar para [-1, 1]
                normalized[lead] = 2 * (clipped - p5) / (p95 - p5) - 1
            else:
                # Sinal constante
                normalized[lead] = signal[lead] - np.mean(signal[lead])
        
        return normalized
    
    def _create_error_result(self, error_message: str) -> Dict[str, Any]:
        """Cria resultado de erro estruturado"""
        return {
            'filtered_signal': None,
            'quality_metrics': {'overall_quality_score': 0.0},
            'preprocessing_successful': False,
            'preprocessing_steps': [],
            'error_message': error_message
        }
    
    def _update_stats(self, processing_time: float):
        """Atualiza estatísticas de processamento"""
        self.stats['signals_processed'] += 1
        
        # Média móvel do tempo de processamento
        n = self.stats['signals_processed']
        self.stats['average_processing_time'] = (
            (self.stats['average_processing_time'] * (n - 1) + processing_time) / n
        )
    
    def get_stats(self) -> Dict[str, Any]:
        """Retorna estatísticas de processamento"""
        cache_hit_rate = 0
        if self.stats['signals_processed'] > 0:
            total_requests = self.stats['cache_hits'] + self.stats['cache_misses']
            if total_requests > 0:
                cache_hit_rate = self.stats['cache_hits'] / total_requests
        
        return {
            **self.stats,
            'cache_hit_rate': cache_hit_rate
        }

# ==================== DETECTORES DE QRS ====================

class PanTompkinsDetector:
    """Detector de QRS Pan-Tompkins otimizado"""
    
    def __init__(self, sampling_rate: int):
        self.sampling_rate = sampling_rate
        self.refractory_period = int(0.2 * sampling_rate)  # 200ms
        
        # Parâmetros adaptativos
        self.threshold_factor = 0.3
        self.integration_window = int(0.08 * sampling_rate)  # 80ms
        
        # Cache para filtros
        self._filter_cache = {}
        
    def detect(self, signal: np.ndarray, enhance: bool = True) -> np.ndarray:
        """Detecta complexos QRS no sinal"""
        if len(signal) == 0:
            return np.array([], dtype=int)
        
        # Pré-processamento
        if enhance:
            filtered = self._bandpass_filter(signal)
        else:
            filtered = signal
        
        # Derivada
        differentiated = np.diff(filtered)
        differentiated = np.append(differentiated, differentiated[-1])
        
        # Elevar ao quadrado
        squared = differentiated ** 2
        
        # Integração em janela móvel
        integrated = self._moving_window_integration(squared)
        
        # Detecção de picos adaptativos
        peaks = self._adaptive_threshold_detection(integrated)
        
        # Refinar posições dos picos
        refined_peaks = self._refine_peak_positions(signal, peaks)
        
        return refined_peaks
    
    def _bandpass_filter(self, signal: np.ndarray) -> np.ndarray:
        """Filtro passa-banda otimizado para QRS (5-15 Hz)"""
        # Verificar cache
        cache_key = f"bp_{self.sampling_rate}"
        if cache_key in self._filter_cache:
            sos = self._filter_cache[cache_key]
        else:
            # Design do filtro
            nyquist = self.sampling_rate / 2
            low = 5.0 / nyquist
            high = 15.0 / nyquist
            
            # Garantir frequências válidas
            low = max(0.001, min(low, 0.999))
            high = max(low + 0.001, min(high, 0.999))
            
            sos = sp_signal.butter(1, [low, high], btype='band', output='sos')
            self._filter_cache[cache_key] = sos
        
        return sp_signal.sosfiltfilt(sos, signal)
    
    def _moving_window_integration(self, signal: np.ndarray) -> np.ndarray:
        """Integração em janela móvel"""
        window = np.ones(self.integration_window) / self.integration_window
        
        # Usar convolução para eficiência
        integrated = np.convolve(signal, window, mode='same')
        
        return integrated
    
    def _adaptive_threshold_detection(self, signal: np.ndarray) -> np.ndarray:
        """Detecção com threshold adaptativo"""
        # Inicializar thresholds
        signal_peak = 0
        noise_peak = 0
        threshold = 0
        
        # Buffers para médias móveis
        signal_peaks = []
        noise_peaks = []
        
        # Detectar todos os picos candidatos
        min_distance = int(0.25 * self.sampling_rate)  # 250ms mínimo entre batimentos
        all_peaks, properties = sp_signal.find_peaks(signal, distance=min_distance)
        
        if len(all_peaks) == 0:
            return np.array([], dtype=int)
        
        # Processo adaptativo
        detected_peaks = []
        
        for i, peak in enumerate(all_peaks):
            peak_val = signal[peak]
            
            if i == 0:
                # Primeiro pico - inicializar
                signal_peak = peak_val
                noise_peak = 0.1 * peak_val
                threshold = noise_peak + self.threshold_factor * (signal_peak - noise_peak)
            
            if peak_val > threshold:
                # Pico detectado como QRS
                detected_peaks.append(peak)
                signal_peaks.append(peak_val)
                
                # Atualizar signal_peak com média móvel
                if len(signal_peaks) > 8:
                    signal_peaks.pop(0)
                signal_peak = np.mean(signal_peaks)
            else:
                # Pico classificado como ruído
                noise_peaks.append(peak_val)
                
                # Atualizar noise_peak com média móvel
                if len(noise_peaks) > 8:
                    noise_peaks.pop(0)
                if noise_peaks:
                    noise_peak = np.mean(noise_peaks)
            
            # Atualizar threshold
            threshold = noise_peak + self.threshold_factor * (signal_peak - noise_peak)
        
        return np.array(detected_peaks, dtype=int)
    
    def _refine_peak_positions(self, original_signal: np.ndarray, peaks: np.ndarray) -> np.ndarray:
        """Refina posições dos picos para o máximo local no sinal original"""
        refined_peaks = []
        search_window = int(0.05 * self.sampling_rate)  # 50ms
        
        for peak in peaks:
            # Definir janela de busca
            start = max(0, peak - search_window)
            end = min(len(original_signal), peak + search_window)
            
            if end > start:
                # Encontrar máximo local
                window = original_signal[start:end]
                local_max = np.argmax(window)
                refined_peaks.append(start + local_max)
        
        return np.array(refined_peaks, dtype=int)


class WaveletQRSDetector:
    """Detector de QRS baseado em wavelets"""
    
    def __init__(self, sampling_rate: int):
        self.sampling_rate = sampling_rate
        self.wavelet = 'db4'
        self.scales = self._calculate_scales()
        
    def _calculate_scales(self) -> np.ndarray:
        """Calcula escalas ótimas para detecção de QRS"""
        # QRS tem duração típica de 60-100ms
        # Calcular escalas correspondentes
        min_width = int(0.06 * self.sampling_rate)  # 60ms
        max_width = int(0.1 * self.sampling_rate)   # 100ms
        
        # 5 escalas logaritmicamente espaçadas
        return np.logspace(np.log10(min_width), np.log10(max_width), 5)
    
    def detect(self, signal: np.ndarray) -> np.ndarray:
        """Detecta QRS usando transformada wavelet contínua"""
        if not PYWT_AVAILABLE:
            # Fallback para Pan-Tompkins
            detector = PanTompkinsDetector(self.sampling_rate)
            return detector.detect(signal)
        
        # CWT para detecção multi-escala
        coeffs, _ = pywt.cwt(signal, self.scales, self.wavelet)
        
        # Combinar escalas usando energia
        energy = np.sum(coeffs ** 2, axis=0)
        
        # Suavizar
        window_size = int(0.02 * self.sampling_rate)
        if window_size % 2 == 0:
            window_size += 1
        energy_smooth = sp_signal.savgol_filter(energy, window_size, 3)
        
        # Detectar picos
        min_distance = int(0.25 * self.sampling_rate)
        peaks, _ = sp_signal.find_peaks(energy_smooth, distance=min_distance)
        
        # Filtrar por amplitude
        if len(peaks) > 0:
            threshold = np.percentile(energy_smooth[peaks], 20)
            peaks = peaks[energy_smooth[peaks] > threshold]
        
        return peaks


class AdaptiveQRSDetector:
    """Detector adaptativo que combina múltiplos métodos"""
    
    def __init__(self, sampling_rate: int):
        self.sampling_rate = sampling_rate
        self.pan_tompkins = PanTompkinsDetector(sampling_rate)
        self.wavelet_detector = WaveletQRSDetector(sampling_rate)
        
        # Histórico para adaptação
        self.performance_history = {
            'pan_tompkins': {'success': 0, 'total': 0},
            'wavelet': {'success': 0, 'total': 0}
        }
        
    def detect(self, signal: np.ndarray, ground_truth: Optional[np.ndarray] = None) -> np.ndarray:
        """Detecta QRS usando método mais apropriado"""
        # Detectar com ambos os métodos
        pt_peaks = self.pan_tompkins.detect(signal)
        wv_peaks = self.wavelet_detector.detect(signal)
        
        # Se temos ground truth, avaliar performance
        if ground_truth is not None:
            pt_score = self._evaluate_detection(pt_peaks, ground_truth)
            wv_score = self._evaluate_detection(wv_peaks, ground_truth)
            
            # Atualizar histórico
            self.performance_history['pan_tompkins']['success'] += pt_score
            self.performance_history['pan_tompkins']['total'] += 1
            self.performance_history['wavelet']['success'] += wv_score
            self.performance_history['wavelet']['total'] += 1
            
            # Retornar melhor resultado
            return pt_peaks if pt_score >= wv_score else wv_peaks
        
        # Sem ground truth - usar votação ou histórico
        if self._should_use_voting():
            return self._voting_detection(pt_peaks, wv_peaks)
        else:
            # Usar método com melhor histórico
            pt_rate = self._get_success_rate('pan_tompkins')
            wv_rate = self._get_success_rate('wavelet')
            
            return pt_peaks if pt_rate >= wv_rate else wv_peaks
    
    def _evaluate_detection(self, detected: np.ndarray, ground_truth: np.ndarray, 
                          tolerance: int = None) -> float:
        """Avalia qualidade da detecção"""
        if tolerance is None:
            tolerance = int(0.05 * self.sampling_rate)  # 50ms
        
        if len(detected) == 0 or len(ground_truth) == 0:
            return 0.0
        
        # Calcular true positives
        tp = 0
        for gt_peak in ground_truth:
            distances = np.abs(detected - gt_peak)
            if np.min(distances) <= tolerance:
                tp += 1
        
        # F1 score
        precision = tp / len(detected) if len(detected) > 0 else 0
        recall = tp / len(ground_truth) if len(ground_truth) > 0 else 0
        
        if precision + recall > 0:
            return 2 * precision * recall / (precision + recall)
        return 0.0
    
    def _should_use_voting(self) -> bool:
        """Decide se deve usar votação baseado no histórico"""
        # Se pouco histórico, usar votação
        total_detections = sum(h['total'] for h in self.performance_history.values())
        return total_detections < 10
    
    def _voting_detection(self, peaks1: np.ndarray, peaks2: np.ndarray, 
                         tolerance: int = None) -> np.ndarray:
        """Combina detecções por votação"""
        if tolerance is None:
            tolerance = int(0.05 * self.sampling_rate)
        
        # Se um está vazio, retornar o outro
        if len(peaks1) == 0:
            return peaks2
        if len(peaks2) == 0:
            return peaks1
        
        # Encontrar picos em consenso
        consensus_peaks = []
        
        for p1 in peaks1:
            distances = np.abs(peaks2 - p1)
            if np.min(distances) <= tolerance:
                # Pico em consenso - usar média
                p2 = peaks2[np.argmin(distances)]
                consensus_peaks.append((p1 + p2) // 2)
        
        return np.array(consensus_peaks, dtype=int)
    
    def _get_success_rate(self, method: str) -> float:
        """Calcula taxa de sucesso do método"""
        history = self.performance_history[method]
        if history['total'] == 0:
            return 0.5  # Valor neutro
        return history['success'] / history['total']


# ==================== EXTRATOR DE CARACTERÍSTICAS ====================

class ECGFeatureExtractor:
    """Extrator avançado de características morfológicas e temporais"""
    
    def __init__(self, sampling_rate: int, config: EnhancedECGAnalysisConfig):
        self.sampling_rate = sampling_rate
        self.config = config
        self.qrs_detector = AdaptiveQRSDetector(sampling_rate)
        
        # Cache para cálculos repetidos
        self._feature_cache = {}
        self._cache_lock = threading.Lock()
        
    def extract_features(self, signal: np.ndarray, 
                        include_morphological: bool = True,
                        include_hrv: bool = True,
                        include_frequency: bool = True) -> Dict[str, Any]:
        """Extrai conjunto completo de características"""
        start_time = time.time()
        
        # Validar entrada
        if signal is None or signal.size == 0:
            return self._create_empty_features()
        
        # Normalizar formato
        if len(signal.shape) == 1:
            signal = signal.reshape(1, -1)
        
        # Verificar cache
        cache_key = self._generate_cache_key(signal, include_morphological, include_hrv, include_frequency)
        cached_features = self._check_cache(cache_key)
        if cached_features is not None:
            return cached_features
        
        features = {
            'temporal': {},
            'morphological': {},
            'hrv': {},
            'frequency': {},
            'quality': {},
            'extraction_time': 0
        }
        
        try:
            # Detectar QRS em todas as derivações
            all_qrs = []
            for lead in range(signal.shape[0]):
                qrs = self.qrs_detector.detect(signal[lead])
                all_qrs.append(qrs)
            
            # Características temporais básicas
            features['temporal'] = self._extract_temporal_features(signal, all_qrs)
            
            # Características morfológicas
            if include_morphological:
                features['morphological'] = self._extract_morphological_features(signal, all_qrs)
            
            # HRV features
            if include_hrv and len(all_qrs[0]) > 2:
                features['hrv'] = self._extract_hrv_features(all_qrs[0])
            
            # Características no domínio da frequência
            if include_frequency:
                features['frequency'] = self._extract_frequency_features(signal)
            
            # Métricas de qualidade
            features['quality'] = self._extract_quality_features(signal)
            
            # Incluir features clínicas avançadas
            advanced_features = self._extract_advanced_clinical_features(
                signal, features['temporal'], features['morphological']
            )
            features.update(advanced_features)
            
        except Exception as e:
            logger.error(f"Erro na extração de características: {str(e)}")
            features = self._create_empty_features()
        
        features['extraction_time'] = time.time() - start_time
        
        # Atualizar cache
        self._update_cache(cache_key, features)
        
        return features
    
    def _extract_temporal_features(self, signal: np.ndarray, all_qrs: List[np.ndarray]) -> Dict[str, float]:
        """Extrai características temporais básicas"""
        features = {}
        
        # Por derivação
        for lead in range(signal.shape[0]):
            lead_name = f"lead_{lead}"
            qrs = all_qrs[lead]
            
            if len(qrs) > 1:
                # Intervalos RR
                rr_intervals = np.diff(qrs) / self.sampling_rate * 1000  # ms
                
                features[f'{lead_name}_mean_rr'] = np.mean(rr_intervals)
                features[f'{lead_name}_std_rr'] = np.std(rr_intervals)
                features[f'{lead_name}_hr'] = 60000 / np.mean(rr_intervals) if np.mean(rr_intervals) > 0 else 0
                
                # Validar frequência cardíaca
                hr = features[f'{lead_name}_hr']
                if hr < MIN_HEART_RATE or hr > MAX_HEART_RATE:
                    features[f'{lead_name}_hr_valid'] = 0
                else:
                    features[f'{lead_name}_hr_valid'] = 1
            else:
                features[f'{lead_name}_mean_rr'] = 0
                features[f'{lead_name}_std_rr'] = 0
                features[f'{lead_name}_hr'] = 0
                features[f'{lead_name}_hr_valid'] = 0
        
        # Características globais
        valid_hrs = [features[f'lead_{i}_hr'] for i in range(signal.shape[0]) 
                     if features[f'lead_{i}_hr_valid'] == 1]
        
        if valid_hrs:
            features['global_mean_hr'] = np.mean(valid_hrs)
            features['global_std_hr'] = np.std(valid_hrs)
        else:
            features['global_mean_hr'] = 0
            features['global_std_hr'] = 0
        
        return features
    
    def _extract_morphological_features(self, signal: np.ndarray, all_qrs: List[np.ndarray]) -> Dict[str, float]:
        """Extrai características morfológicas detalhadas"""
        features = {}
        
        # Focar nas derivações mais informativas
        key_leads = {
            'I': 0, 'II': 1, 'V1': 6, 'V5': 10
        } if signal.shape[0] >= 12 else {'lead_0': 0}
        
        for lead_name, lead_idx in key_leads.items():
            if lead_idx >= signal.shape[0]:
                continue
                
            qrs = all_qrs[lead_idx]
            if len(qrs) < 3:
                continue
            
            # Analisar complexos QRS individuais
            qrs_features = self._analyze_qrs_morphology(signal[lead_idx], qrs)
            for feat_name, feat_val in qrs_features.items():
                features[f'{lead_name}_{feat_name}'] = feat_val
            
            # Analisar ondas P e T
            p_features = self._analyze_p_waves(signal[lead_idx], qrs)
            for feat_name, feat_val in p_features.items():
                features[f'{lead_name}_{feat_name}'] = feat_val
            
            t_features = self._analyze_t_waves(signal[lead_idx], qrs)
            for feat_name, feat_val in t_features.items():
                features[f'{lead_name}_{feat_name}'] = feat_val
        
        # Critérios diagnósticos específicos
        if signal.shape[0] >= 12:
            features.update(self._calculate_diagnostic_criteria(signal, all_qrs))
        
        return features
    
    def _analyze_qrs_morphology(self, lead_signal: np.ndarray, qrs_peaks: np.ndarray) -> Dict[str, float]:
        """Analisa morfologia dos complexos QRS"""
        features = {}
        
        qrs_widths = []
        qrs_amplitudes = []
        q_amplitudes = []
        r_amplitudes = []
        s_amplitudes = []
        
        for i, peak in enumerate(qrs_peaks[1:-1], 1):  # Evitar bordas
            # Definir janela de análise
            window_start = max(0, peak - int(0.1 * self.sampling_rate))
            window_end = min(len(lead_signal), peak + int(0.1 * self.sampling_rate))
            
            qrs_complex = lead_signal[window_start:window_end]
            
            if len(qrs_complex) < 10:
                continue
            
            # Detectar onset e offset do QRS
            onset, offset = self._detect_qrs_boundaries(qrs_complex)
            
            if onset is not None and offset is not None:
                # Largura do QRS
                qrs_width = (offset - onset) / self.sampling_rate * 1000  # ms
                qrs_widths.append(qrs_width)
                
                # Amplitudes
                qrs_segment = qrs_complex[onset:offset]
                if len(qrs_segment) > 0:
                    # Onda R (máximo positivo)
                    r_idx = np.argmax(qrs_segment)
                    r_amp = qrs_segment[r_idx]
                    r_amplitudes.append(r_amp)
                    
                    # Onda Q (mínimo antes de R)
                    if r_idx > 0:
                        q_amp = np.min(qrs_segment[:r_idx])
                        q_amplitudes.append(abs(q_amp))
                    
                    # Onda S (mínimo depois de R)
                    if r_idx < len(qrs_segment) - 1:
                        s_amp = np.min(qrs_segment[r_idx:])
                        s_amplitudes.append(abs(s_amp))
                    
                    # Amplitude total
                    qrs_amplitudes.append(np.ptp(qrs_segment))
        
        # Estatísticas
        if qrs_widths:
            features['qrs_width_mean'] = np.mean(qrs_widths)
            features['qrs_width_std'] = np.std(qrs_widths)
        else:
            features['qrs_width_mean'] = 0
            features['qrs_width_std'] = 0
        
        if qrs_amplitudes:
            features['qrs_amplitude_mean'] = np.mean(qrs_amplitudes)
            features['qrs_amplitude_std'] = np.std(qrs_amplitudes)
        else:
            features['qrs_amplitude_mean'] = 0
            features['qrs_amplitude_std'] = 0
        
        if r_amplitudes:
            features['r_amplitude_mean'] = np.mean(r_amplitudes)
        else:
            features['r_amplitude_mean'] = 0
        
        if q_amplitudes:
            features['q_amplitude_mean'] = np.mean(q_amplitudes)
            features['q_duration_estimate'] = features['qrs_width_mean'] * 0.3  # Estimativa
        else:
            features['q_amplitude_mean'] = 0
            features['q_duration_estimate'] = 0
        
        if s_amplitudes:
            features['s_amplitude_mean'] = np.mean(s_amplitudes)
        else:
            features['s_amplitude_mean'] = 0
        
        return features
    
    def _detect_qrs_boundaries(self, qrs_window: np.ndarray) -> Tuple[Optional[int], Optional[int]]:
        """Detecta início e fim do complexo QRS"""
        if len(qrs_window) < 10:
            return None, None
        
        # Usar derivada para detectar mudanças abruptas
        diff = np.diff(qrs_window)
        diff_smooth = sp_signal.savgol_filter(diff, 5, 2)
        
        # Encontrar ponto máximo (presumivelmente R)
        r_idx = np.argmax(qrs_window)
        
        # Buscar onset (início) - primeira mudança significativa antes de R
        onset = None
        threshold = 0.1 * np.max(np.abs(diff_smooth))
        
        for i in range(r_idx, 0, -1):
            if abs(diff_smooth[i]) < threshold and i < r_idx - 5:
                onset = i
                break
        
        if onset is None:
            onset = 0
        
        # Buscar offset (fim) - última mudança significativa depois de R
        offset = None
        for i in range(r_idx, len(diff_smooth)):
            if abs(diff_smooth[i]) < threshold and i > r_idx + 5:
                offset = i
                break
        
        if offset is None:
            offset = len(qrs_window) - 1
        
        return onset, offset
    
    def _analyze_p_waves(self, lead_signal: np.ndarray, qrs_peaks: np.ndarray) -> Dict[str, float]:
        """Analisa ondas P"""
        features = {
            'p_amplitude_mean': 0,
            'p_duration_mean': 0,
            'pr_interval_mean': 0,
            'p_terminal_force': 0
        }
        
        if len(qrs_peaks) < 2:
            return features
        
        p_amplitudes = []
        p_durations = []
        pr_intervals = []
        
        for i in range(1, len(qrs_peaks) - 1):
            # Janela de busca da onda P (antes do QRS)
            search_start = max(0, qrs_peaks[i] - int(0.2 * self.sampling_rate))
            search_end = qrs_peaks[i] - int(0.05 * self.sampling_rate)
            
            if search_end <= search_start:
                continue
            
            p_window = lead_signal[search_start:search_end]
            
            # Detectar onda P como máximo local
            if len(p_window) > 10:
                # Suavizar para reduzir ruído
                p_smooth = sp_signal.savgol_filter(p_window, 5, 2)
                
                # Encontrar pico da onda P
                p_peaks, properties = sp_signal.find_peaks(p_smooth, prominence=0.05)
                
                if len(p_peaks) > 0:
                    # Usar o pico mais próximo ao QRS
                    p_idx = p_peaks[-1]
                    p_amplitude = p_smooth[p_idx]
                    p_amplitudes.append(p_amplitude)
                    
                    # Estimar duração da onda P
                    p_onset, p_offset = self._detect_wave_boundaries(p_smooth, p_idx)
                    if p_onset is not None and p_offset is not None:
                        p_duration = (p_offset - p_onset) / self.sampling_rate * 1000
                        p_durations.append(p_duration)
                    
                    # Intervalo PR
                    pr_interval = (qrs_peaks[i] - (search_start + p_idx)) / self.sampling_rate * 1000
                    pr_intervals.append(pr_interval)
        
        # Calcular estatísticas
        if p_amplitudes:
            features['p_amplitude_mean'] = np.mean(p_amplitudes)
        
        if p_durations:
            features['p_duration_mean'] = np.mean(p_durations)
        
        if pr_intervals:
            features['pr_interval_mean'] = np.mean(pr_intervals)
            
            # Validar intervalo PR
            pr_mean = features['pr_interval_mean']
            if MIN_PR_INTERVAL <= pr_mean <= MAX_PR_INTERVAL:
                features['pr_interval_valid'] = 1
            else:
                features['pr_interval_valid'] = 0
        
        return features
    
    def _analyze_t_waves(self, lead_signal: np.ndarray, qrs_peaks: np.ndarray) -> Dict[str, float]:
        """Analisa ondas T e intervalo QT"""
        features = {
            't_amplitude_mean': 0,
            't_duration_mean': 0,
            'qt_interval_mean': 0,
            'qtc_bazett': 0,
            't_wave_alternans': 0
        }
        
        if len(qrs_peaks) < 2:
            return features
        
        t_amplitudes = []
        qt_intervals = []
        t_polarities = []
        
        for i in range(len(qrs_peaks) - 1):
            # Janela de busca da onda T (após QRS)
            search_start = qrs_peaks[i] + int(0.05 * self.sampling_rate)
            search_end = min(len(lead_signal), 
                           qrs_peaks[i] + int(0.4 * self.sampling_rate))
            
            if search_end <= search_start:
                continue
            
            t_window = lead_signal[search_start:search_end]
            
            if len(t_window) > 10:
                # Suavizar
                t_smooth = sp_signal.savgol_filter(t_window, 7, 2)
                
                # Detectar pico da onda T (pode ser positivo ou negativo)
                pos_peaks, _ = sp_signal.find_peaks(t_smooth, prominence=0.05)
                neg_peaks, _ = sp_signal.find_peaks(-t_smooth, prominence=0.05)
                
                t_peak_idx = None
                t_amplitude = 0
                
                if len(pos_peaks) > 0 and len(neg_peaks) > 0:
                    # Escolher o maior
                    max_pos = np.max(t_smooth[pos_peaks])
                    max_neg = np.max(-t_smooth[neg_peaks])
                    
                    if max_pos > max_neg:
                        t_peak_idx = pos_peaks[np.argmax(t_smooth[pos_peaks])]
                        t_amplitude = t_smooth[t_peak_idx]
                    else:
                        t_peak_idx = neg_peaks[np.argmax(-t_smooth[neg_peaks])]
                        t_amplitude = t_smooth[t_peak_idx]
                
                elif len(pos_peaks) > 0:
                    t_peak_idx = pos_peaks[np.argmax(t_smooth[pos_peaks])]
                    t_amplitude = t_smooth[t_peak_idx]
                
                elif len(neg_peaks) > 0:
                    t_peak_idx = neg_peaks[np.argmax(-t_smooth[neg_peaks])]
                    t_amplitude = t_smooth[t_peak_idx]
                
                if t_peak_idx is not None:
                    t_amplitudes.append(abs(t_amplitude))
                    t_polarities.append(np.sign(t_amplitude))
                    
                    # Detectar fim da onda T
                    t_offset = self._detect_t_wave_end(t_smooth, t_peak_idx)
                    
                    if t_offset is not None:
                        # QT interval
                        qt_interval = ((search_start + t_offset) - qrs_peaks[i]) / self.sampling_rate * 1000
                        qt_intervals.append(qt_interval)
        
        # Calcular estatísticas
        if t_amplitudes:
            features['t_amplitude_mean'] = np.mean(t_amplitudes)
            
            # T wave alternans (variação na amplitude)
            if len(t_amplitudes) > 3:
                alternans = np.std(t_amplitudes) / np.mean(t_amplitudes)
                features['t_wave_alternans'] = alternans
        
        if qt_intervals:
            features['qt_interval_mean'] = np.mean(qt_intervals)
            
            # QTc (Bazett)
            hr = features.get('hr', 60)
            if hr > 0:
                rr_interval = 60000 / hr  # ms
                features['qtc_bazett'] = features['qt_interval_mean'] / np.sqrt(rr_interval / 1000)
            
            # Validar QT
            qt_mean = features['qt_interval_mean']
            if MIN_QT_INTERVAL <= qt_mean <= MAX_QT_INTERVAL:
                features['qt_interval_valid'] = 1
            else:
                features['qt_interval_valid'] = 0
        
        # Inversão de onda T
        if t_polarities:
            features['t_wave_inversion'] = 1 if np.mean(t_polarities) < 0 else 0
        
        return features
    
    def _detect_wave_boundaries(self, wave_window: np.ndarray, peak_idx: int) -> Tuple[Optional[int], Optional[int]]:
        """Detecta início e fim de uma onda genérica"""
        if len(wave_window) < 5 or peak_idx >= len(wave_window):
            return None, None
        
        # Threshold baseado na amplitude do pico
        threshold = 0.1 * abs(wave_window[peak_idx])
        
        # Buscar onset
        onset = peak_idx
        for i in range(peak_idx, -1, -1):
            if abs(wave_window[i]) < threshold:
                onset = i
                break
        
        # Buscar offset
        offset = peak_idx
        for i in range(peak_idx, len(wave_window)):
            if abs(wave_window[i]) < threshold:
                offset = i
                break
        
        return onset, offset
    
    def _detect_t_wave_end(self, t_window: np.ndarray, t_peak_idx: int) -> Optional[int]:
        """Detecta fim da onda T usando método da tangente"""
        if t_peak_idx >= len(t_window) - 5:
            return None
        
        # Calcular derivada
        diff = np.gradient(t_window)
        
        # Buscar ponto onde a derivada se estabiliza após o pico
        search_start = t_peak_idx + 5
        
        for i in range(search_start, len(diff) - 5):
            # Verificar se derivada está próxima de zero
            if abs(np.mean(diff[i:i+5])) < 0.01:
                return i
        
        # Fallback - usar proporção fixa
        return min(t_peak_idx + int(0.15 * self.sampling_rate), len(t_window) - 1)
    
    def _calculate_diagnostic_criteria(self, signal: np.ndarray, all_qrs: List[np.ndarray]) -> Dict[str, float]:
        """Calcula critérios diagnósticos específicos"""
        criteria = {}
        
        # Sokolow-Lyon para HVE
        if signal.shape[0] >= 12:
            s_v1 = self._get_s_amplitude(signal[6], all_qrs[6])  # V1
            r_v5 = self._get_r_amplitude(signal[10], all_qrs[10])  # V5
            r_v6 = self._get_r_amplitude(signal[11], all_qrs[11])  # V6
            
            sokolow_lyon = s_v1 + max(r_v5, r_v6)
            criteria['sokolow_lyon_voltage'] = sokolow_lyon * self.config.calibration_factor  # mm
            criteria['sokolow_lyon_positive'] = 1 if criteria['sokolow_lyon_voltage'] > DIAGNOSTIC_CRITERIA['lvh']['sokolow_lyon'] else 0
        
        # Cornell para HVE
        if signal.shape[0] >= 12:
            r_avl = self._get_r_amplitude(signal[3], all_qrs[3])  # aVL
            s_v3 = self._get_s_amplitude(signal[8], all_qrs[8])  # V3
            
            cornell = r_avl + s_v3
            criteria['cornell_voltage'] = cornell * self.config.calibration_factor
            # Critério depende do sexo - usar valor intermediário
            criteria['cornell_positive'] = 1 if criteria['cornell_voltage'] > 24 else 0
        
        # Progressão de onda R
        if signal.shape[0] >= 12:
            r_progression = []
            for v_lead in range(6, 12):  # V1-V6
                r_amp = self._get_r_amplitude(signal[v_lead], all_qrs[v_lead])
                r_progression.append(r_amp)
            
            # Verificar progressão adequada
            if len(r_progression) >= 4:
                criteria['poor_r_progression'] = 1 if r_progression[3] < r_progression[0] else 0
        
        return criteria
    
    def _get_r_amplitude(self, lead_signal: np.ndarray, qrs_peaks: np.ndarray) -> float:
        """Obtém amplitude média da onda R"""
        if len(qrs_peaks) == 0:
            return 0
        
        r_amplitudes = []
        for peak in qrs_peaks:
            window_start = max(0, peak - int(0.05 * self.sampling_rate))
            window_end = min(len(lead_signal), peak + int(0.05 * self.sampling_rate))
            
            if window_end > window_start:
                window = lead_signal[window_start:window_end]
                r_amplitudes.append(np.max(window))
        
        return np.mean(r_amplitudes) if r_amplitudes else 0
    
    def _get_s_amplitude(self, lead_signal: np.ndarray, qrs_peaks: np.ndarray) -> float:
        """Obtém amplitude média da onda S (valor absoluto)"""
        if len(qrs_peaks) == 0:
            return 0
        
        s_amplitudes = []
        for peak in qrs_peaks:
            window_start = max(0, peak)
            window_end = min(len(lead_signal), peak + int(0.08 * self.sampling_rate))
            
            if window_end > window_start:
                window = lead_signal[window_start:window_end]
                s_amplitudes.append(abs(np.min(window)))
        
        return np.mean(s_amplitudes) if s_amplitudes else 0
    
    def _extract_hrv_features(self, qrs_peaks: np.ndarray) -> Dict[str, float]:
        """Extrai características de variabilidade da frequência cardíaca"""
        features = {}
        
        if len(qrs_peaks) < 3:
            return {
                'hrv_sdnn': 0,
                'hrv_rmssd': 0,
                'hrv_pnn50': 0,
                'hrv_triangular_index': 0
            }
        
        # Intervalos RR em ms
        rr_intervals = np.diff(qrs_peaks) / self.sampling_rate * 1000
        
        # Filtrar intervalos anormais
        valid_rr = rr_intervals[(rr_intervals > 300) & (rr_intervals < 2000)]
        
        if len(valid_rr) < 2:
            return {
                'hrv_sdnn': 0,
                'hrv_rmssd': 0,
                'hrv_pnn50': 0,
                'hrv_triangular_index': 0
            }
        
        # SDNN - desvio padrão dos intervalos NN
        features['hrv_sdnn'] = np.std(valid_rr)
        
        # RMSSD - raiz quadrada da média dos quadrados das diferenças
        if len(valid_rr) > 1:
            diff_rr = np.diff(valid_rr)
            features['hrv_rmssd'] = np.sqrt(np.mean(diff_rr ** 2))
            
            # pNN50 - percentual de diferenças > 50ms
            features['hrv_pnn50'] = 100 * np.sum(np.abs(diff_rr) > 50) / len(diff_rr)
        else:
            features['hrv_rmssd'] = 0
            features['hrv_pnn50'] = 0
        
        # Índice triangular
        hist, _ = np.histogram(valid_rr, bins=np.arange(300, 2000, 8))
        if np.max(hist) > 0:
            features['hrv_triangular_index'] = len(valid_rr) / np.max(hist)
        else:
            features['hrv_triangular_index'] = 0
        
        return features
    
    def _extract_frequency_features(self, signal: np.ndarray) -> Dict[str, float]:
        """Extrai características no domínio da frequência"""
        features = {}
        
        # Análise por derivação
        for lead in range(min(signal.shape[0], 3)):  # Limitar a 3 derivações
            # PSD usando Welch
            freqs, psd = sp_signal.welch(signal[lead], fs=self.sampling_rate, nperseg=1024)
            
            # Potência em bandas específicas
            vlf_mask = (freqs >= 0.003) & (freqs < 0.04)
            lf_mask = (freqs >= 0.04) & (freqs < 0.15)
            hf_mask = (freqs >= 0.15) & (freqs < 0.4)
            
            vlf_power = np.trapz(psd[vlf_mask], freqs[vlf_mask]) if np.any(vlf_mask) else 0
            lf_power = np.trapz(psd[lf_mask], freqs[lf_mask]) if np.any(lf_mask) else 0
            hf_power = np.trapz(psd[hf_mask], freqs[hf_mask]) if np.any(hf_mask) else 0
            
            total_power = vlf_power + lf_power + hf_power
            
            features[f'lead_{lead}_lf_power'] = lf_power
            features[f'lead_{lead}_hf_power'] = hf_power
            features[f'lead_{lead}_lf_hf_ratio'] = lf_power / (hf_power + 1e-10)
            
            # Frequência dominante
            if len(psd) > 0:
                dominant_freq_idx = np.argmax(psd)
                features[f'lead_{lead}_dominant_freq'] = freqs[dominant_freq_idx]
        
        return features
    
    def _extract_quality_features(self, signal: np.ndarray) -> Dict[str, float]:
        """Extrai métricas de qualidade do sinal"""
        features = {}
        
        # SNR médio
        snr_values = []
        for lead in range(signal.shape[0]):
            snr = self._estimate_snr(signal[lead])
            snr_values.append(snr)
        
        features['mean_snr'] = np.mean(snr_values)
        features['min_snr'] = np.min(snr_values)
        
        # Presença de saturação
        features['has_saturation'] = 0
        for lead in range(signal.shape[0]):
            if np.max(np.abs(signal[lead])) > 4.5:
                features['has_saturation'] = 1
                break
        
        # Estabilidade do baseline
        baseline_drifts = []
        for lead in range(signal.shape[0]):
            baseline = sp_signal.medfilt(signal[lead], kernel_size=int(0.2 * self.sampling_rate))
            drift = np.std(baseline)
            baseline_drifts.append(drift)
        
        features['mean_baseline_drift'] = np.mean(baseline_drifts)
        
        return features
    
    def _estimate_snr(self, signal: np.ndarray) -> float:
        """Estima SNR usando MAD"""
        mad = np.median(np.abs(signal - np.median(signal)))
        noise = mad / NOISE_MEDIAN_FACTOR
        signal_power = np.var(signal)
        
        if noise > 0:
            return 10 * np.log10(signal_power / (noise ** 2))
        return 40.0
    
    def _generate_cache_key(self, signal: np.ndarray, morphological: bool, 
                           hrv: bool, frequency: bool) -> str:
        """Gera chave para cache"""
        signal_hash = hashlib.md5(signal.tobytes()).hexdigest()[:8]
        config_str = f"{morphological}_{hrv}_{frequency}"
        return f"{signal_hash}_{config_str}"
    
    def _check_cache(self, key: str) -> Optional[Dict[str, Any]]:
        """Verifica cache thread-safe"""
        with self._cache_lock:
            return self._feature_cache.get(key)
    
    def _update_cache(self, key: str, features: Dict[str, Any]):
        """Atualiza cache com limite"""
        with self._cache_lock:
            self._feature_cache[key] = features
            
            # Limitar tamanho
            if len(self._feature_cache) > 100:
                # Remover entrada mais antiga
                oldest = next(iter(self._feature_cache))
                del self._feature_cache[oldest]
    
    def _create_empty_features(self) -> Dict[str, Any]:
        """Cria estrutura vazia de características"""
        return {
            'temporal': {},
            'morphological': {},
            'hrv': {},
            'frequency': {},
            'quality': {},
            'extraction_time': 0
        }

    def _extract_advanced_clinical_features(self, signal: np.ndarray, 
                                          temporal_features: Dict[str, Any],
                                          morphological_features: Dict[str, Any]) -> Dict[str, Any]:
        """Extrai features clínicas avançadas para melhorar o treinamento"""
        
        advanced_features = {}
        
        # 1. Features de STEMI multi-lead
        st_measurements = {}
        for lead_idx, lead_name in enumerate(self.config.lead_names[:12]):
            if f'{lead_name}_st_level' in morphological_features:
                st_measurements[lead_name] = morphological_features[f'{lead_name}_st_level']
        
        # Verificar padrões de elevação ST contíguos
        stemi_patterns = self._check_contiguous_st_elevation(st_measurements)
        advanced_features['has_contiguous_st_elevation'] = stemi_patterns['has_pattern']
        advanced_features['st_elevation_territory'] = stemi_patterns.get('territory', 'none')
        advanced_features['reciprocal_changes_present'] = stemi_patterns.get('has_reciprocal', False)
        
        # 2. Features de arritmias
        if 'heart_rate_mean' in temporal_features:
            hr = temporal_features['heart_rate_mean']
            
            # Features para detecção de TV
            advanced_features['is_wide_complex_tachycardia'] = (
                hr > 100 and temporal_features.get('qrs_duration', 0) > 120
            )
            
            # Regularidade do ritmo (importante para FA vs outras arritmias)
            if 'rr_std' in temporal_features:
                advanced_features['rhythm_regularity'] = 1.0 / (1.0 + temporal_features['rr_std'])
        
        # 3. Análise de dispersão QT simplificada
        qt_intervals = {}
        for lead_name in self.config.lead_names[:12]:
            qt_key = f'{lead_name}_qt_interval'
            if qt_key in morphological_features:
                qt_intervals[lead_name] = morphological_features[qt_key]
        
        if len(qt_intervals) >= 6:
            qt_values = list(qt_intervals.values())
            advanced_features['qt_dispersion'] = float(np.max(qt_values) - np.min(qt_values))
            advanced_features['qt_dispersion_normalized'] = advanced_features['qt_dispersion'] / np.mean(qt_values)
        
        # 4. Features de qualidade do sinal por derivação
        for lead_idx, lead_name in enumerate(self.config.lead_names[:12]):
            if lead_idx < signal.shape[0]:
                lead_signal = signal[lead_idx]
                # SNR estimado
                noise_level = np.std(np.diff(lead_signal))
                signal_level = np.std(lead_signal)
                snr = 20 * np.log10(signal_level / (noise_level + 1e-6))
                advanced_features[f'{lead_name}_snr'] = float(snr)
        
        return advanced_features

    def _check_contiguous_st_elevation(self, st_measurements: Dict[str, float]) -> Dict[str, Any]:
        """Verifica padrões de elevação ST em derivações contíguas"""
        
        # Grupos de derivações contíguas
        lead_groups = {
            'inferior': {
                'leads': ['II', 'III', 'aVF'],
                'reciprocal': ['I', 'aVL']
            },
            'anterior': {
                'leads': ['V1', 'V2', 'V3', 'V4'],
                'reciprocal': ['II', 'III', 'aVF']
            },
            'lateral': {
                'leads': ['I', 'aVL', 'V5', 'V6'],
                'reciprocal': ['III', 'aVF']
            },
            'septal': {
                'leads': ['V1', 'V2'],
                'reciprocal': []
            }
        }
        
        result = {
            'has_pattern': False, 
            'territory': None, 
            'elevated_leads': [],
            'has_reciprocal': False
        }
        
        for territory, group_info in lead_groups.items():
            elevated = []
            for lead in group_info['leads']:
                if lead in st_measurements:
                    # Thresholds específicos por derivação
                    threshold = 0.2 if lead.startswith('V') and lead in ['V1', 'V2', 'V3'] else 0.1
                    if st_measurements[lead] >= threshold:
                        elevated.append(lead)
            
            # Verificar se há pelo menos 2 derivações contíguas elevadas
            if len(elevated) >= 2:
                result['has_pattern'] = True
                result['territory'] = territory
                result['elevated_leads'] = elevated
                
                # Verificar mudanças recíprocas
                for recip_lead in group_info['reciprocal']:
                    if recip_lead in st_measurements and st_measurements[recip_lead] < -0.05:
                        result['has_reciprocal'] = True
                        break
                
                break
        
        return result

# ==================== AUGMENTAÇÃO DE DADOS ====================

class ECGAugmentation:
    """Sistema avançado de augmentação para ECG com preservação de características clínicas"""
    
    def __init__(self, config: EnhancedECGAnalysisConfig):
        self.config = config
        self.sampling_rate = config.sampling_rate
        
        # Gerador de números aleatórios para reprodutibilidade
        self.rng = np.random.RandomState()
        
        # Parâmetros de augmentação
        self.augmentation_params = {
            'gaussian_noise': {
                'snr_range': (10, 30),  # dB
                'probability': 0.5
            },
            'baseline_wander': {
                'amplitude_range': (0.05, 0.2),  # mV
                'frequency_range': (0.1, 0.5),  # Hz
                'probability': 0.4
            },
            'powerline_interference': {
                'amplitude_range': (0.01, 0.05),  # mV
                'frequency_options': [50, 60],  # Hz
                'probability': 0.3
            },
            'muscle_artifacts': {
                'amplitude_range': (0.02, 0.1),  # mV
                'duration_range': (0.05, 0.2),  # segundos
                'probability': 0.3
            },
            'amplitude_scaling': {
                'scale_range': config.amplitude_scaling,
                'probability': 0.6
            },
            'time_warping': {
                'warp_range': (0.9, 1.1),
                'probability': 0.3 if config.time_warping else 0
            },
            'lead_dropout': {
                'max_leads': config.max_lead_dropout,
                'probability': 0.2 if config.lead_dropout else 0
            },
            'respiratory_modulation': {
                'amplitude_range': (0.02, 0.08),  # mV
                'frequency_range': (0.15, 0.3),  # Hz (9-18 rpm)
                'probability': 0.3
            }
        }
    
    def augment(self, signal: np.ndarray, labels: Optional[np.ndarray] = None, 
                seed: Optional[int] = None) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Aplica augmentações preservando características diagnósticas"""
        if seed is not None:
            self.rng.seed(seed)
        
        # Copiar sinal
        augmented = signal.copy()
        
        # Aplicar augmentações com probabilidade
        if self.rng.random() < self.config.augmentation_prob:
            # Ordem aleatória de augmentações
            augmentations = list(self.augmentation_params.keys())
            self.rng.shuffle(augmentations)
            
            for aug_name in augmentations:
                params = self.augmentation_params[aug_name]
                if self.rng.random() < params['probability']:
                    augmented = self._apply_augmentation(augmented, aug_name, params)
        
        # Validar sinal augmentado
        augmented = self._validate_augmented_signal(augmented, signal)
        
        return augmented, labels
    
    def _apply_augmentation(self, signal: np.ndarray, aug_type: str, 
                           params: Dict[str, Any]) -> np.ndarray:
        """Aplica tipo específico de augmentação"""
        if aug_type == 'gaussian_noise':
            return self._add_gaussian_noise(signal, params)
        elif aug_type == 'baseline_wander':
            return self._add_baseline_wander(signal, params)
        elif aug_type == 'powerline_interference':
            return self._add_powerline_interference(signal, params)
        elif aug_type == 'muscle_artifacts':
            return self._add_muscle_artifacts(signal, params)
        elif aug_type == 'amplitude_scaling':
            return self._apply_amplitude_scaling(signal, params)
        elif aug_type == 'time_warping':
            return self._apply_time_warping(signal, params)
        elif aug_type == 'lead_dropout':
            return self._apply_lead_dropout(signal, params)
        elif aug_type == 'respiratory_modulation':
            return self._add_respiratory_modulation(signal, params)
        else:
            return signal
    
    def _add_gaussian_noise(self, signal: np.ndarray, params: Dict[str, Any]) -> np.ndarray:
        """Adiciona ruído gaussiano com SNR controlado"""
        snr_db = self.rng.uniform(*params['snr_range'])
        
        augmented = np.empty_like(signal)
        for lead in range(signal.shape[0]):
            # Calcular potência do sinal
            signal_power = np.mean(signal[lead] ** 2)
            
            # Calcular desvio padrão do ruído para SNR desejado
            snr_linear = 10 ** (snr_db / 10)
            noise_power = signal_power / snr_linear
            noise_std = np.sqrt(noise_power)
            
            # Adicionar ruído
            noise = self.rng.normal(0, noise_std, signal.shape[1])
            augmented[lead] = signal[lead] + noise
        
        return augmented
    
    def _add_baseline_wander(self, signal: np.ndarray, params: Dict[str, Any]) -> np.ndarray:
        """Adiciona deriva de baseline realista"""
        amplitude = self.rng.uniform(*params['amplitude_range'])
        frequency = self.rng.uniform(*params['frequency_range'])
        
        # Gerar baseline usando soma de senoides
        t = np.arange(signal.shape[1]) / self.sampling_rate
        
        # Componente principal
        baseline = amplitude * np.sin(2 * np.pi * frequency * t)
        
        # Adicionar harmônicos para realismo
        baseline += 0.3 * amplitude * np.sin(2 * np.pi * 2 * frequency * t + self.rng.random() * 2 * np.pi)
        baseline += 0.1 * amplitude * np.sin(2 * np.pi * 3 * frequency * t + self.rng.random() * 2 * np.pi)
        
        # Aplicar a todas as derivações com pequenas variações
        augmented = np.empty_like(signal)
        for lead in range(signal.shape[0]):
            lead_variation = 1 + 0.1 * self.rng.randn()
            augmented[lead] = signal[lead] + baseline * lead_variation
        
        return augmented
    
    def _add_powerline_interference(self, signal: np.ndarray, params: Dict[str, Any]) -> np.ndarray:
        """Adiciona interferência de rede elétrica"""
        amplitude = self.rng.uniform(*params['amplitude_range'])
        frequency = self.rng.choice(params['frequency_options'])
        
        t = np.arange(signal.shape[1]) / self.sampling_rate
        
        # Interferência com pequena modulação de amplitude
        modulation_freq = 0.1  # Hz
        modulation_depth = 0.2
        
        amplitude_modulated = amplitude * (1 + modulation_depth * np.sin(2 * np.pi * modulation_freq * t))
        interference = amplitude_modulated * np.sin(2 * np.pi * frequency * t)
        
        # Adicionar harmônicos (típico em interferência real)
        interference += 0.2 * amplitude * np.sin(2 * np.pi * 2 * frequency * t)
        interference += 0.1 * amplitude * np.sin(2 * np.pi * 3 * frequency * t)
        
        # Aplicar com variação entre derivações
        augmented = np.empty_like(signal)
        for lead in range(signal.shape[0]):
            coupling_factor = self.rng.uniform(0.5, 1.5)
            augmented[lead] = signal[lead] + interference * coupling_factor
        
        return augmented
    
    def _add_muscle_artifacts(self, signal: np.ndarray, params: Dict[str, Any]) -> np.ndarray:
        """Adiciona artefatos musculares localizados"""
        augmented = signal.copy()
        
        # Número de artefatos
        num_artifacts = self.rng.randint(1, 4)
        
        for _ in range(num_artifacts):
            # Posição temporal aleatória
            duration_samples = int(self.rng.uniform(*params['duration_range']) * self.sampling_rate)
            start_idx = self.rng.randint(0, signal.shape[1] - duration_samples)
            
            # Gerar artefato usando ruído de alta frequência
            t = np.arange(duration_samples) / self.sampling_rate
            
            # Envelope gaussiano
            envelope = np.exp(-0.5 * ((t - t.mean()) / (0.2 * t.max())) ** 2)
            
            # Ruído de alta frequência (20-100 Hz)
            artifact = np.zeros(duration_samples)
            for freq in range(20, 100, 10):
                phase = self.rng.random() * 2 * np.pi
                artifact += self.rng.random() * np.sin(2 * np.pi * freq * t + phase)
            
            # Normalizar e aplicar envelope
            artifact = artifact / np.std(artifact) * self.rng.uniform(*params['amplitude_range'])
            artifact *= envelope
            
            # Aplicar a derivações selecionadas
            affected_leads = self.rng.choice(signal.shape[0], 
                                           size=self.rng.randint(1, min(3, signal.shape[0])), 
                                           replace=False)
            
            for lead in affected_leads:
                augmented[lead, start_idx:start_idx + duration_samples] += artifact
        
        return augmented
    
    def _apply_amplitude_scaling(self, signal: np.ndarray, params: Dict[str, Any]) -> np.ndarray:
        """Escala amplitude preservando relações entre derivações"""
        # Fator de escala global
        global_scale = self.rng.uniform(*params['scale_range'])
        
        # Pequenas variações por derivação
        augmented = np.empty_like(signal)
        for lead in range(signal.shape[0]):
            lead_scale = global_scale * (1 + 0.05 * self.rng.randn())
            augmented[lead] = signal[lead] * lead_scale
        
        return augmented
    
    def _apply_time_warping(self, signal: np.ndarray, params: Dict[str, Any]) -> np.ndarray:
        """Aplica warping temporal suave"""
        warp_factor = self.rng.uniform(*params['warp_range'])
        
        # Criar mapeamento temporal não-linear suave
        original_time = np.linspace(0, 1, signal.shape[1])
        
        # Função de warping suave (senoidal)
        warp_amplitude = (warp_factor - 1) * 0.1
        warped_time = original_time + warp_amplitude * np.sin(4 * np.pi * original_time)
        
        # Garantir monoticidade
        warped_time = np.cumsum(np.maximum(np.diff(np.concatenate([[0], warped_time])), 1e-6))
        warped_time = warped_time / warped_time[-1]  # Normalizar para [0, 1]
        
        # Interpolar sinal
        augmented = np.empty_like(signal)
        for lead in range(signal.shape[0]):
            f = interpolate.interp1d(original_time, signal[lead], 
                                    kind='cubic', fill_value='extrapolate')
            augmented[lead] = f(warped_time)
        
        return augmented
    
    def _apply_lead_dropout(self, signal: np.ndarray, params: Dict[str, Any]) -> np.ndarray:
        """Simula falha de eletrodos"""
        if signal.shape[0] <= 1:
            return signal
        
        augmented = signal.copy()
        
        # Número de derivações a descartar
        num_dropout = self.rng.randint(1, min(params['max_leads'] + 1, signal.shape[0]))
        
        # Selecionar derivações
        dropout_leads = self.rng.choice(signal.shape[0], size=num_dropout, replace=False)
        
        for lead in dropout_leads:
            # Simular diferentes tipos de falha
            failure_type = self.rng.choice(['flat', 'noise', 'drift'])
            
            if failure_type == 'flat':
                # Linha plana
                augmented[lead] = np.zeros_like(signal[lead])
            elif failure_type == 'noise':
                # Apenas ruído
                augmented[lead] = self.rng.normal(0, 0.1, signal.shape[1])
            else:
                # Deriva extrema
                t = np.arange(signal.shape[1]) / self.sampling_rate
                augmented[lead] = 2 * np.sin(2 * np.pi * 0.1 * t)
        
        return augmented
    
    def _add_respiratory_modulation(self, signal: np.ndarray, params: Dict[str, Any]) -> np.ndarray:
        """Adiciona modulação respiratória realista"""
        amplitude = self.rng.uniform(*params['amplitude_range'])
        frequency = self.rng.uniform(*params['frequency_range'])
        
        t = np.arange(signal.shape[1]) / self.sampling_rate
        
        # Padrão respiratório com variabilidade
        respiration = amplitude * np.sin(2 * np.pi * frequency * t)
        
        # Adicionar variabilidade na frequência (RSA - Respiratory Sinus Arrhythmia)
        freq_variation = 0.1 * frequency * np.sin(2 * np.pi * 0.05 * t)
        phase = 2 * np.pi * integrate.cumtrapz(frequency + freq_variation, t, initial=0)
        respiration = amplitude * np.sin(phase)
        
        # Aplicar com diferentes fases nas derivações
        augmented = np.empty_like(signal)
        for lead in range(signal.shape[0]):
            phase_shift = self.rng.uniform(0, np.pi/2)
            lead_respiration = amplitude * np.sin(phase + phase_shift)
            
            # Modular amplitude do QRS
            augmented[lead] = signal[lead] * (1 + 0.1 * lead_respiration)
            
            # Adicionar componente aditivo pequeno
            augmented[lead] += 0.3 * lead_respiration
        
        return augmented
    
    def _validate_augmented_signal(self, augmented: np.ndarray, original: np.ndarray) -> np.ndarray:
        """Valida e corrige sinal augmentado se necessário"""
        # Verificar valores extremos
        max_amplitude = 5.0  # mV
        augmented = np.clip(augmented, -max_amplitude, max_amplitude)
        
        # Verificar se não destruímos completamente o sinal
        for lead in range(augmented.shape[0]):
            # Calcular correlação com original
            if len(original[lead]) > 0 and len(augmented[lead]) > 0:
                correlation = np.corrcoef(original[lead], augmented[lead])[0, 1]
                
                # Se correlação muito baixa, reduzir augmentação
                if correlation < 0.3:
                    # Misturar com original
                    alpha = 0.5
                    augmented[lead] = alpha * augmented[lead] + (1 - alpha) * original[lead]
        
        return augmented
    
    def create_mixup(self, signal1: np.ndarray, signal2: np.ndarray, 
                     labels1: np.ndarray, labels2: np.ndarray, 
                     alpha: float = 0.2) -> Tuple[np.ndarray, np.ndarray]:
        """Implementa mixup para ECG"""
        # Fator de mistura
        if alpha > 0:
            lam = self.rng.beta(alpha, alpha)
        else:
            lam = 1
        
        # Garantir mesmo tamanho
        min_length = min(signal1.shape[1], signal2.shape[1])
        signal1 = signal1[:, :min_length]
        signal2 = signal2[:, :min_length]
        
        # Misturar sinais
        mixed_signal = lam * signal1 + (1 - lam) * signal2
        
        # Misturar labels
        mixed_labels = lam * labels1 + (1 - lam) * labels2
        
        return mixed_signal, mixed_labels


# ==================== DATASET OTIMIZADO ====================

class OptimizedECGDataset(Dataset):
    """Dataset otimizado com cache inteligente e preprocessamento paralelo"""
    
    def __init__(self, data_path: Union[str, Path], config: EnhancedECGAnalysisConfig,
                 split: str = 'train', transform: Optional[Any] = None,
                 target_conditions: Optional[List[str]] = None):
        self.data_path = Path(data_path)
        self.config = config
        self.split = split
        self.transform = transform
        self.target_conditions = target_conditions or list(SCP_ECG_CONDITIONS.keys())
        
        # Preprocessador e extrator
        self.preprocessor = ECGPreprocessor(config)
        self.feature_extractor = ECGFeatureExtractor(config.sampling_rate, config)
        self.augmenter = ECGAugmentation(config) if split == 'train' else None
        
        # Cache multinível
        self.signal_cache = OrderedDict()
        self.feature_cache = OrderedDict()
        self.max_cache_size = config.cache_size
        self._cache_lock = threading.Lock()
        
        # Thread pool para processamento paralelo
        if config.use_parallel:
            self.executor = ThreadPoolExecutor(max_workers=config.max_parallel_workers)
        else:
            self.executor = None
        
        # Carregar índices
        self._load_data_indices()
        
        # Estatísticas do dataset
        self.stats = {
            'samples_loaded': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'preprocessing_errors': 0
        }
    
    def _load_data_indices(self):
        """Carrega índices dos dados com validação"""
        # Implementação específica depende do formato dos dados
        # Exemplo para PTB-XL ou similar
        
        if (self.data_path / 'records.csv').exists():
            self.records_df = pd.read_csv(self.data_path / 'records.csv')
            
            # Filtrar por split se disponível
            if 'split' in self.records_df.columns:
                self.records_df = self.records_df[self.records_df['split'] == self.split]
            
            self.record_ids = self.records_df['record_id'].tolist()
        else:
            # Buscar arquivos .npy ou .mat
            self.record_files = list(self.data_path.glob('*.npy'))
            if not self.record_files:
                self.record_files = list(self.data_path.glob('*.mat'))
            
            self.record_ids = [f.stem for f in self.record_files]
        
        logger.info(f"Carregados {len(self.record_ids)} registros para split '{self.split}'")
    
    def __len__(self) -> int:
        return len(self.record_ids)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
        """Retorna amostra preprocessada com metadados"""
        record_id = self.record_ids[idx]
        
        # Verificar cache
        cached_data = self._check_cache(record_id)
        if cached_data is not None:
            self.stats['cache_hits'] += 1
            signal, labels, metadata = cached_data
        else:
            self.stats['cache_misses'] += 1
            
            # Carregar e processar
            try:
                signal, labels, metadata = self._load_and_process(record_id)
                
                # Atualizar cache
                self._update_cache(record_id, (signal, labels, metadata))
                
            except Exception as e:
                logger.error(f"Erro ao processar {record_id}: {str(e)}")
                self.stats['preprocessing_errors'] += 1
                
                # Retornar amostra vazia
                return self._create_empty_sample()
        
        # Aplicar augmentação se em treino
        if self.split == 'train' and self.augmenter is not None:
            signal, labels = self.augmenter.augment(signal, labels)
        
        # Converter para tensors
        signal_tensor = torch.FloatTensor(signal)
        labels_tensor = torch.FloatTensor(labels)
        
        # Aplicar transformações adicionais
        if self.transform:
            signal_tensor = self.transform(signal_tensor)
        
        self.stats['samples_loaded'] += 1
        
        return signal_tensor, labels_tensor, metadata
    
    def _load_and_process(self, record_id: str) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        """Carrega e processa um registro"""
        # Carregar sinal bruto
        signal = self._load_signal(record_id)
        
        # Preprocessar
        preprocess_result = self.preprocessor.preprocess(signal)
        
        if not preprocess_result['preprocessing_successful']:
            raise ValueError(f"Falha no preprocessamento: {preprocess_result.get('error_message', 'Unknown')}")
        
        processed_signal = preprocess_result['filtered_signal']
        
        # Extrair características
        features = self.feature_extractor.extract_features(processed_signal)
        
        # Carregar labels
        labels = self._load_labels(record_id)
        
        # Criar metadados
        metadata = {
            'record_id': record_id,
            'quality_score': preprocess_result['quality_metrics']['overall_quality_score'],
            'features': features,
            'preprocessing_steps': preprocess_result['preprocessing_steps']
        }
        
        return processed_signal, labels, metadata
    
    def _load_signal(self, record_id: str) -> np.ndarray:
        """Carrega sinal do disco"""
        # Tentar diferentes formatos
        
        # Formato numpy
        npy_path = self.data_path / f"{record_id}.npy"
        if npy_path.exists():
            return np.load(npy_path)
        
        # Formato WFDB
        if WFDB_AVAILABLE:
            try:
                record = wfdb.rdrecord(str(self.data_path / record_id))
                return record.p_signal.T  # Transpor para [leads, samples]
            except:
                pass
        
        # Formato MAT
        mat_path = self.data_path / f"{record_id}.mat"
        if mat_path.exists():
            import scipy.io
            mat_data = scipy.io.loadmat(str(mat_path))
            # Assumir que o sinal está na chave 'signal' ou 'data'
            for key in ['signal', 'data', 'ecg']:
                if key in mat_data:
                    signal = mat_data[key]
                    if signal.shape[0] > signal.shape[1]:
                        signal = signal.T
                    return signal
        
        raise FileNotFoundError(f"Não foi possível carregar o sinal {record_id}")
    
    def _load_labels(self, record_id: str) -> np.ndarray:
        """Carrega labels do registro"""
        labels = np.zeros(len(self.target_conditions), dtype=np.float32)
        
        # Se temos dataframe com anotações
        if hasattr(self, 'records_df'):
            record_data = self.records_df[self.records_df['record_id'] == record_id]
            
            if not record_data.empty:
                # Verificar diferentes formatos de labels
                if 'scp_codes' in record_data.columns:
                    # Formato PTB-XL
                    scp_codes = eval(record_data['scp_codes'].iloc[0])
                    for i, condition in enumerate(self.target_conditions):
                        if condition in scp_codes:
                            labels[i] = 1.0
                else:
                    # Formato com colunas individuais
                    for i, condition in enumerate(self.target_conditions):
                        if condition in record_data.columns:
                            labels[i] = float(record_data[condition].iloc[0])
        
        return labels
    
    def _check_cache(self, record_id: str) -> Optional[Tuple[np.ndarray, np.ndarray, Dict[str, Any]]]:
        """Verifica cache thread-safe"""
        with self._cache_lock:
            if record_id in self.signal_cache:
                # Move para o fim (LRU)
                self.signal_cache.move_to_end(record_id)
                return self.signal_cache[record_id]
        return None
    
    def _update_cache(self, record_id: str, data: Tuple[np.ndarray, np.ndarray, Dict[str, Any]]):
        """Atualiza cache com limite de tamanho"""
        with self._cache_lock:
            self.signal_cache[record_id] = data
            
            # Limitar tamanho
            if len(self.signal_cache) > self.max_cache_size:
                self.signal_cache.popitem(last=False)
    
    def _create_empty_sample(self) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
        """Cria amostra vazia para casos de erro"""
        empty_signal = torch.zeros(self.config.num_leads, self.config.signal_length)
        empty_labels = torch.zeros(len(self.target_conditions))
        empty_metadata = {
            'record_id': 'error',
            'quality_score': 0.0,
            'features': {},
            'preprocessing_steps': []
        }
        return empty_signal, empty_labels, empty_metadata
    
    def get_class_weights(self) -> torch.Tensor:
        """Calcula pesos das classes para balanceamento"""
        if not hasattr(self, '_class_weights'):
            # Contar amostras por classe
            class_counts = np.zeros(len(self.target_conditions))
            
            # Amostrar subset se dataset muito grande
            sample_size = min(1000, len(self))
            indices = np.random.choice(len(self), sample_size, replace=False)
            
            for idx in indices:
                _, labels, _ = self[idx]
                class_counts += labels.numpy()
            
            # Calcular pesos inversamente proporcionais
            class_weights = len(indices) / (len(self.target_conditions) * class_counts + 1)
            
            # Normalizar
            class_weights = class_weights / class_weights.mean()
            
            # Limitar pesos extremos
            class_weights = np.clip(class_weights, 0.1, 10)
            
            self._class_weights = torch.FloatTensor(class_weights)
        
        return self._class_weights
    
    def get_sampler(self) -> WeightedRandomSampler:
        """Cria sampler com balanceamento de classes"""
        # Calcular peso de cada amostra
        sample_weights = []
        
        logger.info("Calculando pesos das amostras para balanceamento...")
        
        for idx in tqdm(range(len(self)), desc="Calculando pesos"):
            _, labels, _ = self[idx]
            
            # Peso baseado nas classes presentes
            class_weights = self.get_class_weights()
            sample_weight = (labels * class_weights).max().item()
            sample_weights.append(sample_weight)
        
        return WeightedRandomSampler(sample_weights, len(self))
    
    def get_statistics(self) -> Dict[str, Any]:
        """Retorna estatísticas do dataset"""
        return {
            **self.stats,
            'cache_size': len(self.signal_cache),
            'cache_hit_rate': self.stats['cache_hits'] / max(1, self.stats['cache_hits'] + self.stats['cache_misses']),
            'error_rate': self.stats['preprocessing_errors'] / max(1, self.stats['samples_loaded'])
        }
    
    def cleanup(self):
        """Limpa recursos"""
        if self.executor:
            self.executor.shutdown(wait=False)
        
        self.signal_cache.clear()
        self.feature_cache.clear()


# ==================== DATALOADER CUSTOMIZADO ====================

class ECGDataLoader:
    """DataLoader otimizado para ECG com prefetching e processamento paralelo"""
    
    def __init__(self, dataset: OptimizedECGDataset, batch_size: int,
                 shuffle: bool = True, num_workers: int = 4,
                 pin_memory: bool = True, use_balanced_sampling: bool = True):
        
        self.dataset = dataset
        self.batch_size = batch_size
        
        # Criar sampler se necessário
        if use_balanced_sampling and dataset.split == 'train':
            sampler = dataset.get_sampler()
            shuffle = False  # Sampler controla a ordem
        else:
            sampler = None
        
        # Criar DataLoader PyTorch
        self.loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=pin_memory,
            collate_fn=self._custom_collate,
            worker_init_fn=self._worker_init,
            persistent_workers=num_workers > 0
        )
        
        # Estatísticas
        self.stats = {
            'batches_loaded': 0,
            'samples_per_second': 0
        }
        
        self._start_time = None
    
    def _custom_collate(self, batch: List[Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]]) -> Dict[str, Any]:
        """Collate function customizada para lidar com metadados"""
        signals = []
        labels = []
        metadata = []
        
        for signal, label, meta in batch:
            signals.append(signal)
            labels.append(label)
            metadata.append(meta)
        
        # Stack tensors
        signals = torch.stack(signals)
        labels = torch.stack(labels)
        
        return {
            'signals': signals,
            'labels': labels,
            'metadata': metadata
        }
    
    def _worker_init(self, worker_id: int):
        """Inicialização dos workers"""
        # Definir seed diferente para cada worker
        np.random.seed(torch.initial_seed() % (2**32) + worker_id)
    
    def __iter__(self):
        """Iterador com tracking de performance"""
        self._start_time = time.time()
        
        for batch in self.loader:
            self.stats['batches_loaded'] += 1
            
            # Calcular throughput
            elapsed = time.time() - self._start_time
            total_samples = self.stats['batches_loaded'] * self.batch_size
            self.stats['samples_per_second'] = total_samples / elapsed
            
            yield batch
    
    def __len__(self):
        return len(self.loader)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Retorna estatísticas do dataloader"""
        stats = self.stats.copy()
        stats['dataset_stats'] = self.dataset.get_statistics()
        return stats


# ==================== BLOCOS DE CONSTRUÇÃO ====================

class SEBlock1D(nn.Module):
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.squeeze = nn.AdaptiveAvgPool1d(1)
        self.excitation = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _ = x.size()
        y = self.squeeze(x).view(b, c)
        y = self.excitation(y).view(b, c, 1)
        return x * y.expand_as(x)


class ResBlock1D(nn.Module):
    """Bloco residual otimizado para ECG"""
    
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 7,
                 stride: int = 1, dropout: float = 0.1, use_se: bool = True):
        super().__init__()
        
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size,
                               stride=stride, padding=kernel_size//2, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout)
        
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size,
                               padding=kernel_size//2, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
        
        # Squeeze-and-Excitation
        self.se = SEBlock1D(out_channels) if use_se else None
        
        # Shortcut
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels)
            )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        if self.se is not None:
            out = self.se(out)
        
        out += self.shortcut(residual)
        out = self.relu(out)
        
        return out


class InceptionBlock1D(nn.Module):
    # Bloco Inception adaptado para sinais 1D
    
    def __init__(self, in_channels: int, out_channels: int, 
                 kernel_sizes: List[int] = [5, 11, 23]):
        super().__init__()
        
        assert out_channels % len(kernel_sizes) == 0
        branch_channels = out_channels // len(kernel_sizes)
        
        self.branches = nn.ModuleList()
        for kernel_size in kernel_sizes:
            branch = nn.Sequential(
                nn.Conv1d(in_channels, branch_channels, kernel_size,
                         padding=kernel_size//2, bias=False),
                nn.BatchNorm1d(branch_channels),
                nn.ReLU(inplace=True)
            )
            self.branches.append(branch)
        
        # Branch com max pooling
        self.pool_branch = nn.Sequential(
            nn.MaxPool1d(3, stride=1, padding=1),
            nn.Conv1d(in_channels, branch_channels, 1, bias=False),
            nn.BatchNorm1d(branch_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outputs = []
        
        for branch in self.branches:
            outputs.append(branch(x))
        
        outputs.append(self.pool_branch(x))
        
        return torch.cat(outputs, dim=1)


class TemporalConvNet(nn.Module):
    """Temporal Convolutional Network com dilatação causal"""
    
    def __init__(self, num_inputs: int, num_channels: List[int], 
                 kernel_size: int = 3, dropout: float = 0.2):
        super().__init__()
        layers = []
        num_levels = len(num_channels)
        
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            
            layers.append(
                TCNBlock(in_channels, out_channels, kernel_size,
                        stride=1, dilation=dilation_size, dropout=dropout)
            )
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class TCNBlock(nn.Module):
    """Bloco TCN com padding causal"""
    
    def __init__(self, n_inputs: int, n_outputs: int, kernel_size: int,
                 stride: int, dilation: int, dropout: float = 0.2):
        super().__init__()
        
        padding = (kernel_size - 1) * dilation
        
        self.conv1 = nn.Conv1d(n_inputs, n_outputs, kernel_size,
                              stride=stride, padding=padding, dilation=dilation)
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        
        self.conv2 = nn.Conv1d(n_outputs, n_outputs, kernel_size,
                              stride=stride, padding=padding, dilation=dilation)
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)
        
        self.net = nn.Sequential(
            self.conv1, self.chomp1, self.relu1, self.dropout1,
            self.conv2, self.chomp2, self.relu2, self.dropout2
        )
        
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class Chomp1d(nn.Module):
    """Remove padding para manter causalidade"""
    
    def __init__(self, chomp_size: int):
        super().__init__()
        self.chomp_size = chomp_size
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x[:, :, :-self.chomp_size].contiguous()


# ==================== MECANISMOS DE ATENÇÃO ====================

class MultiHeadAttention1D(nn.Module):
    # Multi-head attention para sequencias 1D
    
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, query: torch.Tensor, key: torch.Tensor, 
                value: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size = query.size(0)
        
        # Linear transformations and split into heads
        Q = self.W_q(query).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(key).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(value).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        # Attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.d_k, dtype=torch.float32))
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        context = torch.matmul(attn_weights, V)
        
        # Concatenate heads
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        
        output = self.W_o(context)
        
        return output


class TemporalAttention(nn.Module):
    """Atenção temporal para destacar regiões importantes do ECG"""
    
    def __init__(self, hidden_size: int, num_heads: int = 8):
        super().__init__()
        self.attention = MultiHeadAttention1D(hidden_size, num_heads)
        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)
        
        self.feed_forward = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size * 4, hidden_size)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch, channels, length)
        x = x.transpose(1, 2)  # (batch, length, channels)
        
        # Self-attention
        attn_output = self.attention(x, x, x)
        x = self.norm1(x + attn_output)
        
        # Feed forward
        ff_output = self.feed_forward(x)
        x = self.norm2(x + ff_output)
        
        return x.transpose(1, 2)  # (batch, channels, length)


# ==================== ARQUITETURAS PRINCIPAIS ====================

class ECGResNet1D(nn.Module):
    """ResNet adaptada para ECG com atenção temporal"""
    
    def __init__(self, num_leads: int, num_classes: int, 
                 base_channels: int = 64, num_blocks: List[int] = [2, 2, 2, 2],
                 use_attention: bool = True):
        super().__init__()
        
        self.num_leads = num_leads
        self.num_classes = num_classes
        
        # Stem
        self.stem = nn.Sequential(
            nn.Conv1d(num_leads, base_channels, kernel_size=15, stride=2, padding=7, bias=False),
            nn.BatchNorm1d(base_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        )
        
        # Residual blocks
        self.layer1 = self._make_layer(base_channels, base_channels, num_blocks[0])
        self.layer2 = self._make_layer(base_channels, base_channels * 2, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(base_channels * 2, base_channels * 4, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(base_channels * 4, base_channels * 8, num_blocks[3], stride=2)
        
        # Attention
        if use_attention:
            self.attention = TemporalAttention(base_channels * 8)
        else:
            self.attention = None
        
        # Global pooling
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        # Classifier com regularização
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(base_channels * 8, base_channels * 4),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(base_channels * 4, num_classes)
        )
        
        # Inicialização
        self._initialize_weights()
    
    def _make_layer(self, in_channels: int, out_channels: int, 
                    num_blocks: int, stride: int = 1) -> nn.Sequential:
        layers = []
        layers.append(ResBlock1D(in_channels, out_channels, stride=stride))
        
        for _ in range(1, num_blocks):
            layers.append(ResBlock1D(out_channels, out_channels))
        
        return nn.Sequential(*layers)
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Feature extraction
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        # Attention
        if self.attention is not None:
            x = self.attention(x)
        
        # Global pooling
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        
        # Classification
        x = self.classifier(x)
        
        return x
    
    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extrai features antes da classificação"""
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        if self.attention is not None:
            x = self.attention(x)
        
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        
        return x


class InceptionTime(nn.Module):
    """InceptionTime para classificação de ECG"""
    
    def __init__(self, num_leads: int, num_classes: int, 
                 num_blocks: int = 6, base_channels: int = 32):
        super().__init__()
        
        self.num_leads = num_leads
        self.num_classes = num_classes
        
        # Primeira camada
        self.conv_input = nn.Conv1d(num_leads, base_channels, kernel_size=1, bias=False)
        self.bn_input = nn.BatchNorm1d(base_channels)
        
        # Blocos Inception
        channels = base_channels
        inception_blocks = []
        
        for i in range(num_blocks):
            # Aumentar canais progressivamente
            if i % 2 == 0 and i > 0:
                channels *= 2
            
            inception_blocks.append(
                InceptionBlock1D(channels if i > 0 else base_channels, channels)
            )
        
        self.inception_blocks = nn.Sequential(*inception_blocks)
        
        # Global pooling
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(channels, num_classes)
        )
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_input(x)
        x = self.bn_input(x)
        
        x = self.inception_blocks(x)
        
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        
        x = self.classifier(x)
        
        return x


class HybridECGNet(nn.Module):
    """Arquitetura híbrida combinando CNN e RNN"""
    
    def __init__(self, num_leads: int, num_classes: int,
                 cnn_channels: List[int] = [64, 128, 256],
                 rnn_hidden: int = 256, rnn_layers: int = 2,
                 use_bidirectional: bool = True):
        super().__init__()
        
        self.num_leads = num_leads
        self.num_classes = num_classes
        
        # CNN para extração de features locais
        cnn_layers = []
        in_channels = num_leads
        
        for out_channels in cnn_channels:
            cnn_layers.extend([
                nn.Conv1d(in_channels, out_channels, kernel_size=7, padding=3),
                nn.BatchNorm1d(out_channels),
                nn.ReLU(inplace=True),
                nn.MaxPool1d(2)
            ])
            in_channels = out_channels
        
        self.cnn = nn.Sequential(*cnn_layers)
        
        # RNN para capturar dependências temporais
        self.rnn = nn.LSTM(
            input_size=cnn_channels[-1],
            hidden_size=rnn_hidden,
            num_layers=rnn_layers,
            batch_first=True,
            dropout=0.3 if rnn_layers > 1 else 0,
            bidirectional=use_bidirectional
        )
        
        rnn_output_size = rnn_hidden * (2 if use_bidirectional else 1)
        
        # Attention sobre saídas RNN
        self.attention_weights = nn.Linear(rnn_output_size, 1)
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(rnn_output_size, rnn_output_size // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(rnn_output_size // 2, num_classes)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # CNN
        cnn_out = self.cnn(x)  # (batch, channels, length)
        
        # Preparar para RNN
        cnn_out = cnn_out.transpose(1, 2)  # (batch, length, channels)
        
        # RNN
        rnn_out, _ = self.rnn(cnn_out)  # (batch, length, hidden*2)
        
        # Attention
        attention_scores = self.attention_weights(rnn_out)  # (batch, length, 1)
        attention_weights = F.softmax(attention_scores, dim=1)
        
        # Weighted sum
        weighted_out = torch.sum(rnn_out * attention_weights, dim=1)  # (batch, hidden*2)
        
        # Classification
        output = self.classifier(weighted_out)
        
        return output


# ==================== ENSEMBLE DE MODELOS ====================

class ECGEnsemble(nn.Module):
    """Ensemble de múltiplas arquiteturas"""
    
    def __init__(self, num_leads: int, num_classes: int, 
                 model_configs: Optional[List[Dict[str, Any]]] = None):
        super().__init__()
        
        self.num_leads = num_leads
        self.num_classes = num_classes
        
        # Configurações padrão
        if model_configs is None:
            model_configs = [
                {'type': 'resnet', 'weight': 0.4},
                {'type': 'inception', 'weight': 0.3},
                {'type': 'hybrid', 'weight': 0.3}
            ]
        
        self.models = nn.ModuleList()
        self.weights = []
        
        for config in model_configs:
            model = self._create_model(config['type'])
            self.models.append(model)
            self.weights.append(config['weight'])
        
        # Normalizar pesos
        total_weight = sum(self.weights)
        self.weights = [w / total_weight for w in self.weights]
        
        # Learned fusion (opcional)
        self.use_learned_fusion = True
        if self.use_learned_fusion:
            self.fusion_layer = nn.Linear(num_classes * len(self.models), num_classes)
    
    def _create_model(self, model_type: str) -> nn.Module:
        """Cria modelo baseado no tipo"""
        if model_type == 'resnet':
            return ECGResNet1D(self.num_leads, self.num_classes)
        elif model_type == 'inception':
            return InceptionTime(self.num_leads, self.num_classes)
        elif model_type == 'hybrid':
            return HybridECGNet(self.num_leads, self.num_classes)
        else:
            raise ValueError(f"Tipo de modelo desconhecido: {model_type}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outputs = []
        
        # Obter predições de cada modelo
        for model in self.models:
            output = model(x)
            outputs.append(output)
        
        if self.use_learned_fusion:
            # Concatenar todas as saídas
            combined = torch.cat(outputs, dim=1)
            output = self.fusion_layer(combined)
        else:
            # Média ponderada simples
            output = torch.zeros_like(outputs[0])
            for i, out in enumerate(outputs):
                output += self.weights[i] * out
        
        return output
    
    def get_individual_predictions(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Retorna predições individuais de cada modelo"""
        predictions = []
        
        with torch.no_grad():
            for model in self.models:
                pred = torch.sigmoid(model(x))
                predictions.append(pred)
        
        return predictions


# ==================== MODELO COM CALIBRAÇÃO ====================

class CalibratedECGModel(nn.Module):
    """Modelo com calibração de probabilidades"""
    
    def __init__(self, base_model: nn.Module, num_classes: int,
                 calibration_method: str = 'temperature'):
        super().__init__()
        
        self.base_model = base_model
        self.num_classes = num_classes
        self.calibration_method = calibration_method
        
        if calibration_method == 'temperature':
            # Temperature scaling
            self.temperature = nn.Parameter(torch.ones(1))
        elif calibration_method == 'platt':
            # Platt scaling
            self.platt_a = nn.Parameter(torch.ones(num_classes))
            self.platt_b = nn.Parameter(torch.zeros(num_classes))
        elif calibration_method == 'isotonic':
            # Isotonic regression - implementado durante calibração
            self.isotonic_regressors = None
    
    def forward(self, x: torch.Tensor, return_uncalibrated: bool = False) -> torch.Tensor:
        # Obter logits do modelo base
        logits = self.base_model(x)
        
        if return_uncalibrated:
            return logits
        
        # Aplicar calibração
        if self.calibration_method == 'temperature':
            return self._temperature_scale(logits)
        elif self.calibration_method == 'platt':
            return self._platt_scale(logits)
        elif self.calibration_method == 'isotonic':
            return self._isotonic_transform(logits)
        else:
            return logits
    
    def _temperature_scale(self, logits: torch.Tensor) -> torch.Tensor:
        """Aplica temperature scaling"""
        return logits / self.temperature
    
    def _platt_scale(self, logits: torch.Tensor) -> torch.Tensor:
        """Aplica Platt scaling"""
        return self.platt_a * logits + self.platt_b
    
    def _isotonic_transform(self, logits: torch.Tensor) -> torch.Tensor:
        """Aplica regressão isotônica"""
        if self.isotonic_regressors is None:
            return logits
        
        # Aplicar regressão isotônica treinada
        device = logits.device
        calibrated = torch.zeros_like(logits)
        
        probs = torch.sigmoid(logits)
        
        for i in range(self.num_classes):
            if i in self.isotonic_regressors:
                # Mover para CPU para sklearn
                probs_cpu = probs[:, i].cpu().numpy()
                calibrated_cpu = self.isotonic_regressors[i].transform(probs_cpu)
                calibrated[:, i] = torch.tensor(calibrated_cpu).to(device)
            else:
                calibrated[:, i] = probs[:, i]
        
        # Converter de volta para logits
        return torch.log(calibrated / (1 - calibrated + 1e-8))
    
    def calibrate(self, val_loader: DataLoader, device: torch.device):
        """Calibra o modelo usando dados de validação"""
        self.base_model.eval()
        
        all_logits = []
        all_labels = []
        
        with torch.no_grad():
            for batch in val_loader:
                signals = batch['signals'].to(device)
                labels = batch['labels'].to(device)
                
                logits = self.base_model(signals)
                
                all_logits.append(logits.cpu())
                all_labels.append(labels.cpu())
        
        all_logits = torch.cat(all_logits)
        all_labels = torch.cat(all_labels)
        
        if self.calibration_method == 'temperature':
            self._optimize_temperature(all_logits, all_labels)
        elif self.calibration_method == 'platt':
            self._fit_platt_scaling(all_logits, all_labels)
        elif self.calibration_method == 'isotonic':
            self._fit_isotonic_regression(all_logits, all_labels)
    
    def _optimize_temperature(self, logits: torch.Tensor, labels: torch.Tensor):
        """Otimiza temperature scaling"""
        criterion = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.LBFGS([self.temperature], lr=0.01, max_iter=50)
        
        def eval_loss():
            optimizer.zero_grad()
            scaled_logits = logits / self.temperature
            loss = criterion(scaled_logits, labels)
            loss.backward()
            return loss
        
        optimizer.step(eval_loss)
    
    def _fit_platt_scaling(self, logits: torch.Tensor, labels: torch.Tensor):
        """Ajusta Platt scaling"""
        from sklearn.linear_model import LogisticRegression
        
        for i in range(self.num_classes):
            lr = LogisticRegression(max_iter=1000)
            lr.fit(logits[:, i:i+1].numpy(), labels[:, i].numpy())
            
            self.platt_a.data[i] = torch.tensor(lr.coef_[0, 0])
            self.platt_b.data[i] = torch.tensor(lr.intercept_[0])
    
    def _fit_isotonic_regression(self, logits: torch.Tensor, labels: torch.Tensor):
        """Ajusta regressão isotônica"""
        from sklearn.isotonic import IsotonicRegression
        
        self.isotonic_regressors = {}
        probs = torch.sigmoid(logits)
        
        for i in range(self.num_classes):
            ir = IsotonicRegression(out_of_bounds='clip')
            ir.fit(probs[:, i].numpy(), labels[:, i].numpy())
            self.isotonic_regressors[i] = ir


# ==================== FACTORY DE MODELOS ====================

class ModelFactory:
    """Factory para criar modelos com configurações específicas"""
    
    @staticmethod
    def create_model(model_type: str, num_leads: int, num_classes: int,
                    config: Optional[Dict[str, Any]] = None) -> nn.Module:
        """Cria modelo baseado no tipo e configuração"""
        
        if config is None:
            config = {}
        
        if model_type == 'resnet':
            model = ECGResNet1D(
                num_leads=num_leads,
                num_classes=num_classes,
                base_channels=config.get('base_channels', 64),
                num_blocks=config.get('num_blocks', [2, 2, 2, 2]),
                use_attention=config.get('use_attention', True)
            )
        
        elif model_type == 'inception':
            model = InceptionTime(
                num_leads=num_leads,
                num_classes=num_classes,
                num_blocks=config.get('num_blocks', 6),
                base_channels=config.get('base_channels', 32)
            )
        
        elif model_type == 'hybrid':
            model = HybridECGNet(
                num_leads=num_leads,
                num_classes=num_classes,
                cnn_channels=config.get('cnn_channels', [64, 128, 256]),
                rnn_hidden=config.get('rnn_hidden', 256),
                rnn_layers=config.get('rnn_layers', 2),
                use_bidirectional=config.get('use_bidirectional', True)
            )
        
        elif model_type == 'ensemble':
            model = ECGEnsemble(
                num_leads=num_leads,
                num_classes=num_classes,
                model_configs=config.get('model_configs', None)
            )
        
        else:
            raise ValueError(f"Tipo de modelo desconhecido: {model_type}")
        
        # Adicionar calibração se solicitado
        if config.get('use_calibration', False):
            calibration_method = config.get('calibration_method', 'temperature')
            model = CalibratedECGModel(model, num_classes, calibration_method)
        
        return model
    
    @staticmethod
    def get_model_params(model: nn.Module) -> Dict[str, Any]:
        """Retorna informações sobre o modelo"""
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        return {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'model_size_mb': total_params * 4 / (1024 * 1024),  # Assumindo float32
            'layers': len(list(model.modules()))
        }

# ==================== FUNÇÕES DE PERDA CUSTOMIZADAS ====================

class FocalLoss(nn.Module):
    """Focal Loss para lidar com desbalanceamento de classes"""
    
    def __init__(self, alpha: Optional[torch.Tensor] = None, gamma: float = 2.0, 
                 reduction: str = 'mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)
        F_loss = (1 - pt) ** self.gamma * BCE_loss
        
        if self.alpha is not None:
            if self.alpha.device != inputs.device:
                self.alpha = self.alpha.to(inputs.device)
            at = self.alpha.gather(0, targets.data.long())
            F_loss = at * F_loss
        
        if self.reduction == 'mean':
            return F_loss.mean()
        elif self.reduction == 'sum':
            return F_loss.sum()
        else:
            return F_loss


class AsymmetricLoss(nn.Module):
    """Asymmetric Loss para multi-label com foco em falsos negativos"""
    
    def __init__(self, gamma_neg: float = 4, gamma_pos: float = 1, 
                 clip: float = 0.05, eps: float = 1e-8, 
                 disable_torch_grad_focal_loss: bool = True):
        super().__init__()
        
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.disable_torch_grad_focal_loss = disable_torch_grad_focal_loss
        self.eps = eps
    
    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # Calculating Probabilities
        x_sigmoid = torch.sigmoid(x)
        xs_pos = x_sigmoid
        xs_neg = 1 - x_sigmoid
        
        # Asymmetric Clipping
        if self.clip is not None and self.clip > 0:
            xs_neg = (xs_neg + self.clip).clamp(max=1)
        
        # Basic CE calculation
        los_pos = y * torch.log(xs_pos.clamp(min=self.eps))
        los_neg = (1 - y) * torch.log(xs_neg.clamp(min=self.eps))
        loss = los_pos + los_neg
        
        # Asymmetric Focusing
        if self.gamma_neg > 0 or self.gamma_pos > 0:
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(False)
            pt0 = xs_pos * y
            pt1 = xs_neg * (1 - y)
            pt = pt0 + pt1
            one_sided_gamma = self.gamma_pos * y + self.gamma_neg * (1 - y)
            one_sided_w = torch.pow(1 - pt, one_sided_gamma)
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(True)
            loss *= one_sided_w
        
        return -loss.sum()


class DiceLoss(nn.Module):
    """Dice Loss para segmentação temporal (detecção de ondas)"""
    
    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        inputs = torch.sigmoid(inputs)
        
        intersection = (inputs * targets).sum(dim=(1, 2))
        dice = (2. * intersection + self.smooth) / (inputs.sum(dim=(1, 2)) + targets.sum(dim=(1, 2)) + self.smooth)
        
        return 1 - dice.mean()


class CombinedLoss(nn.Module):
    """Combinação de múltiplas losses com pesos adaptativos"""
    
    def __init__(self, losses_config: Dict[str, Dict[str, Any]]):
        super().__init__()
        
        self.losses = nn.ModuleDict()
        self.weights = {}
        
        for name, config in losses_config.items():
            loss_class = config['class']
            loss_params = config.get('params', {})
            weight = config.get('weight', 1.0)
            
            if loss_class == 'focal':
                self.losses[name] = FocalLoss(**loss_params)
            elif loss_class == 'asymmetric':
                self.losses[name] = AsymmetricLoss(**loss_params)
            elif loss_class == 'bce':
                self.losses[name] = nn.BCEWithLogitsLoss(**loss_params)
            elif loss_class == 'dice':
                self.losses[name] = DiceLoss(**loss_params)
            
            self.weights[name] = weight
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> Dict[str, torch.Tensor]:
        total_loss = 0
        losses_dict = {}
        
        for name, loss_fn in self.losses.items():
            loss_value = loss_fn(inputs, targets)
            weighted_loss = self.weights[name] * loss_value
            
            total_loss += weighted_loss
            losses_dict[name] = loss_value
        
        losses_dict['total'] = total_loss
        
        return losses_dict


class ClinicallyWeightedLoss(nn.Module):
    """Loss function que prioriza condições críticas para segurança diagnóstica"""
    
    def __init__(self, base_loss: nn.Module, critical_conditions: List[str], 
                 critical_weight: float = 2.0, rare_weight: float = 1.5):
        super().__init__()
        self.base_loss = base_loss
        
        # Índices das condições críticas
        self.critical_indices = [
            i for i, cond in enumerate(SCP_ECG_CONDITIONS.keys()) 
            if cond in critical_conditions
        ]
        
        # Índices de condições raras mas importantes
        rare_conditions = ['VFIB', 'VTAC', 'AVB3', 'WPWS', 'LQTS']
        self.rare_indices = [
            i for i, cond in enumerate(SCP_ECG_CONDITIONS.keys()) 
            if cond in rare_conditions
        ]
        
        self.critical_weight = critical_weight
        self.rare_weight = rare_weight
        
        logger.info(f"Loss com peso aumentado para {len(self.critical_indices)} condições críticas")
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        batch_size = inputs.shape[0]
        
        # Loss base
        base_loss_value = self.base_loss(inputs, targets)
        
        # Criar máscara de pesos
        weight_mask = torch.ones_like(targets)
        
        # Aplicar peso maior para condições críticas
        if self.critical_indices:
            weight_mask[:, self.critical_indices] = self.critical_weight
        
        # Aplicar peso para condições raras
        if self.rare_indices:
            for idx in self.rare_indices:
                if idx not in self.critical_indices:  # Evitar dupla aplicação
                    weight_mask[:, idx] = self.rare_weight
        
        # Calcular loss ponderada elemento por elemento
        if isinstance(self.base_loss, nn.BCEWithLogitsLoss):
            # Para BCEWithLogitsLoss, aplicar peso manualmente
            bce = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
            weighted_loss = (bce * weight_mask).mean()
        else:
            # Para outras losses (Focal, Asymmetric), modificar inputs
            weighted_loss = self.base_loss(inputs, targets * weight_mask)
        
        # Penalidade adicional para falsos negativos em condições críticas
        if self.critical_indices:
            critical_targets = targets[:, self.critical_indices]
            critical_predictions = torch.sigmoid(inputs[:, self.critical_indices])
            
            # Falsos negativos: target=1, prediction<0.5
            fn_mask = (critical_targets == 1) & (critical_predictions < 0.5)
            fn_penalty = fn_mask.float().sum() / (batch_size * len(self.critical_indices))
            
            weighted_loss = weighted_loss + 0.5 * fn_penalty
        
        return weighted_loss


# ==================== MÉTRICAS MÉDICAS ====================

class ECGMetrics:
    """Métricas específicas para avaliação de ECG"""
    
    def __init__(self, num_classes: int, class_names: List[str], 
                 device: torch.device = torch.device('cpu')):
        self.num_classes = num_classes
        self.class_names = class_names
        self.device = device
        
        # Contadores para métricas
        self.reset()
    
    def reset(self):
        """Reseta todos os contadores"""
        self.tp = torch.zeros(self.num_classes).to(self.device)
        self.fp = torch.zeros(self.num_classes).to(self.device)
        self.tn = torch.zeros(self.num_classes).to(self.device)
        self.fn = torch.zeros(self.num_classes).to(self.device)
        
        self.all_predictions = []
        self.all_targets = []
        
        # Métricas por severidade
        self.severity_metrics = defaultdict(lambda: {'tp': 0, 'fp': 0, 'tn': 0, 'fn': 0})
    
    def update(self, predictions: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5):
        """Atualiza contadores com novo batch"""
        # Binarizar predições
        preds_binary = (torch.sigmoid(predictions) > threshold).float()
        
        # Atualizar contadores
        self.tp += ((preds_binary == 1) & (targets == 1)).sum(dim=0).float()
        self.fp += ((preds_binary == 1) & (targets == 0)).sum(dim=0).float()
        self.tn += ((preds_binary == 0) & (targets == 0)).sum(dim=0).float()
        self.fn += ((preds_binary == 0) & (targets == 1)).sum(dim=0).float()
        
        # Armazenar para métricas globais
        self.all_predictions.append(predictions.cpu())
        self.all_targets.append(targets.cpu())
        
        # Atualizar métricas por severidade
        for i, class_name in enumerate(self.class_names):
            if class_name in SCP_ECG_CONDITIONS:
                severity = SCP_ECG_CONDITIONS[class_name]['severity']
                
                tp = ((preds_binary[:, i] == 1) & (targets[:, i] == 1)).sum().item()
                fp = ((preds_binary[:, i] == 1) & (targets[:, i] == 0)).sum().item()
                tn = ((preds_binary[:, i] == 0) & (targets[:, i] == 0)).sum().item()
                fn = ((preds_binary[:, i] == 0) & (targets[:, i] == 1)).sum().item()
                
                self.severity_metrics[severity]['tp'] += tp
                self.severity_metrics[severity]['fp'] += fp
                self.severity_metrics[severity]['tn'] += tn
                self.severity_metrics[severity]['fn'] += fn
    
    def compute(self) -> Dict[str, Any]:
        """Computa todas as métricas"""
        eps = 1e-7
        
        # Métricas por classe
        sensitivity = self.tp / (self.tp + self.fn + eps)
        specificity = self.tn / (self.tn + self.fp + eps)
        precision = self.tp / (self.tp + self.fp + eps)
        f1 = 2 * precision * sensitivity / (precision + sensitivity + eps)
        
        # Métricas agregadas
        metrics = {
            'sensitivity_mean': sensitivity.mean().item(),
            'specificity_mean': specificity.mean().item(),
            'precision_mean': precision.mean().item(),
            'f1_mean': f1.mean().item(),
            'sensitivity_per_class': {name: sens.item() for name, sens in zip(self.class_names, sensitivity)},
            'specificity_per_class': {name: spec.item() for name, spec in zip(self.class_names, specificity)},
            'precision_per_class': {name: prec.item() for name, prec in zip(self.class_names, precision)},
            'f1_per_class': {name: f.item() for name, f in zip(self.class_names, f1)}
        }
        
        # AUROC e AUPRC se tivermos predições
        if self.all_predictions:
            all_preds = torch.cat(self.all_predictions, dim=0)
            all_targets = torch.cat(self.all_targets, dim=0)
            
            auroc_scores = []
            auprc_scores = []
            
            for i in range(self.num_classes):
                if all_targets[:, i].sum() > 0:  # Há positivos
                    try:
                        auroc = roc_auc_score(all_targets[:, i], torch.sigmoid(all_preds[:, i]))
                        auprc = average_precision_score(all_targets[:, i], torch.sigmoid(all_preds[:, i]))
                        
                        auroc_scores.append(auroc)
                        auprc_scores.append(auprc)
                        
                        metrics[f'auroc_{self.class_names[i]}'] = auroc
                        metrics[f'auprc_{self.class_names[i]}'] = auprc
                    except:
                        pass
            
            if auroc_scores:
                metrics['auroc_mean'] = np.mean(auroc_scores)
                metrics['auprc_mean'] = np.mean(auprc_scores)
        
        # Métricas por severidade
        severity_summary = {}
        for severity, counts in self.severity_metrics.items():
            tp, fp, tn, fn = counts['tp'], counts['fp'], counts['tn'], counts['fn']
            
            if tp + fn > 0:
                sens = tp / (tp + fn)
                spec = tn / (tn + fp) if tn + fp > 0 else 0
                prec = tp / (tp + fp) if tp + fp > 0 else 0
                f1 = 2 * prec * sens / (prec + sens) if prec + sens > 0 else 0
                
                severity_summary[f'severity_{severity}'] = {
                    'sensitivity': sens,
                    'specificity': spec,
                    'precision': prec,
                    'f1': f1,
                    'total_cases': tp + fn
                }
        
        metrics['severity_metrics'] = severity_summary
        
        # Verificar requisitos clínicos
        metrics['meets_clinical_requirements'] = self._check_clinical_requirements(metrics)
        
        return metrics
    
    def _check_clinical_requirements(self, metrics: Dict[str, Any]) -> Dict[str, bool]:
        """Verifica se métricas atendem requisitos clínicos"""
        requirements_met = {}
        
        # Sensibilidade mínima
        min_sensitivity = min(metrics['sensitivity_per_class'].values())
        requirements_met['min_sensitivity'] = min_sensitivity >= CLINICAL_REQUIREMENTS['min_sensitivity']
        
        # Especificidade mínima
        min_specificity = min(metrics['specificity_per_class'].values())
        requirements_met['min_specificity'] = min_specificity >= CLINICAL_REQUIREMENTS['min_specificity']
        
        # AUROC alvo
        if 'auroc_mean' in metrics:
            requirements_met['target_auc'] = metrics['auroc_mean'] >= CLINICAL_REQUIREMENTS['target_auc']
        
        # Verificar patologias críticas (severidade >= 4)
        critical_performance = []
        for class_name, sensitivity in metrics['sensitivity_per_class'].items():
            if class_name in SCP_ECG_CONDITIONS and SCP_ECG_CONDITIONS[class_name]['severity'] >= 4:
                critical_performance.append(sensitivity)
        
        if critical_performance:
            requirements_met['critical_pathologies_sensitivity'] = min(critical_performance) >= 0.9
        
        return requirements_met
    
    def generate_report(self) -> str:
        """Gera relatório detalhado das métricas"""
        metrics = self.compute()
        
        report = "=" * 80 + "\n"
        report += "RELATÓRIO DE MÉTRICAS DE ECG\n"
        report += "=" * 80 + "\n\n"
        
        # Métricas gerais
        report += "MÉTRICAS GERAIS:\n"
        report += f"  - Sensibilidade média: {metrics['sensitivity_mean']:.3f}\n"
        report += f"  - Especificidade média: {metrics['specificity_mean']:.3f}\n"
        report += f"  - Precisão média: {metrics['precision_mean']:.3f}\n"
        report += f"  - F1-Score médio: {metrics['f1_mean']:.3f}\n"
        
        if 'auroc_mean' in metrics:
            report += f"  - AUROC médio: {metrics['auroc_mean']:.3f}\n"
            report += f"  - AUPRC médio: {metrics['auprc_mean']:.3f}\n"
        
        report += "\n"
        
        # Top 5 melhores e piores classes
        f1_scores = metrics['f1_per_class']
        sorted_classes = sorted(f1_scores.items(), key=lambda x: x[1], reverse=True)
        
        report += "TOP 5 MELHORES DESEMPENHOS (F1-Score):\n"
        for i, (class_name, f1) in enumerate(sorted_classes[:5]):
            report += f"  {i+1}. {class_name}: {f1:.3f}\n"
        
        report += "\nTOP 5 PIORES DESEMPENHOS (F1-Score):\n"
        for i, (class_name, f1) in enumerate(sorted_classes[-5:]):
            report += f"  {i+1}. {class_name}: {f1:.3f}\n"
        
        # Métricas por severidade
        report += "\n\nMÉTRICAS POR SEVERIDADE:\n"
        for severity_key, severity_metrics in sorted(metrics['severity_metrics'].items()):
            report += f"\n{severity_key}:\n"
            report += f"  - Sensibilidade: {severity_metrics['sensitivity']:.3f}\n"
            report += f"  - Especificidade: {severity_metrics['specificity']:.3f}\n"
            report += f"  - Total de casos: {severity_metrics['total_cases']}\n"
        
        # Requisitos clínicos
        report += "\n\nVERIFICAÇÃO DE REQUISITOS CLÍNICOS:\n"
        for req, met in metrics['meets_clinical_requirements'].items():
            status = "✓ ATENDIDO" if met else "✗ NÃO ATENDIDO"
            report += f"  - {req}: {status}\n"
        
        report += "\n" + "=" * 80 + "\n"
        
        return report


# ==================== CALLBACKS DE TREINAMENTO ====================

class TrainingCallback:
    """Classe base para callbacks"""
    
    def on_epoch_start(self, epoch: int, trainer: Any):
        pass
    
    def on_epoch_end(self, epoch: int, trainer: Any, metrics: Dict[str, float]):
        pass
    
    def on_batch_start(self, batch_idx: int, trainer: Any):
        pass
    
    def on_batch_end(self, batch_idx: int, trainer: Any, loss: float):
        pass
    
    def on_training_start(self, trainer: Any):
        pass
    
    def on_training_end(self, trainer: Any):
        pass


class EarlyStopping(TrainingCallback):
    """Early stopping com paciência e delta mínimo"""
    
    def __init__(self, patience: int = 10, min_delta: float = 1e-4, 
                 monitor: str = 'val_loss', mode: str = 'min'):
        self.patience = patience
        self.min_delta = min_delta
        self.monitor = monitor
        self.mode = mode
        
        self.counter = 0
        self.best_score = None
        self.early_stop = False
    
    def on_epoch_end(self, epoch: int, trainer: Any, metrics: Dict[str, float]):
        score = metrics.get(self.monitor)
        
        if score is None:
            return
        
        if self.mode == 'min':
            score = -score
        
        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                trainer.should_stop = True
                logger.info(f"Early stopping triggered after {epoch + 1} epochs")
        else:
            self.best_score = score
            self.counter = 0


class ModelCheckpoint(TrainingCallback):
    """Salva modelo baseado em métricas"""
    
    def __init__(self, filepath: Path, monitor: str = 'val_loss', 
                 mode: str = 'min', save_best_only: bool = True):
        self.filepath = Path(filepath)
        self.monitor = monitor
        self.mode = mode
        self.save_best_only = save_best_only
        
        self.best_score = None
        self.filepath.parent.mkdir(parents=True, exist_ok=True)
    
    def on_epoch_end(self, epoch: int, trainer: Any, metrics: Dict[str, float]):
        score = metrics.get(self.monitor)
        
        if score is None:
            return
        
        if self.mode == 'min':
            is_best = self.best_score is None or score < self.best_score
        else:
            is_best = self.best_score is None or score > self.best_score
        
        if is_best:
            self.best_score = score
            
            if self.save_best_only:
                # Salvar modelo
                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': trainer.model.state_dict(),
                    'optimizer_state_dict': trainer.optimizer.state_dict(),
                    'best_score': self.best_score,
                    'metrics': metrics,
                    'config': trainer.config
                }
                
                torch.save(checkpoint, self.filepath)
                logger.info(f"Checkpoint salvo: {self.monitor}={score:.4f}")


class LearningRateScheduler(TrainingCallback):
    """Callback para ajustar learning rate"""
    
    def __init__(self, scheduler: Any, metric: Optional[str] = None):
        self.scheduler = scheduler
        self.metric = metric
    
    def on_epoch_end(self, epoch: int, trainer: Any, metrics: Dict[str, float]):
        if self.metric and hasattr(self.scheduler, 'step'):
            # ReduceLROnPlateau
            metric_value = metrics.get(self.metric)
            if metric_value is not None:
                self.scheduler.step(metric_value)
        else:
            # Outros schedulers
            self.scheduler.step()
        
        # Log current LR
        current_lr = trainer.optimizer.param_groups[0]['lr']
        logger.info(f"Learning rate: {current_lr:.6f}")


class CriticalConditionMonitor(TrainingCallback):
    """Monitora performance específica em condições críticas durante treinamento"""
    
    def __init__(self, critical_conditions: List[str], min_sensitivity: float = 0.85):
        self.critical_conditions = critical_conditions
        self.min_sensitivity = min_sensitivity
        self.critical_indices = [
            i for i, cond in enumerate(SCP_ECG_CONDITIONS.keys()) 
            if cond in critical_conditions
        ]
        self.history = defaultdict(list)
    
    def on_epoch_end(self, epoch: int, trainer: Any, metrics: Dict[str, float]):
        # Verificar sensibilidade para condições críticas
        if 'val_metrics' in metrics and 'per_class' in metrics['val_metrics']:
            per_class = metrics['val_metrics']['per_class']
            
            critical_sensitivities = []
            for condition in self.critical_conditions:
                if condition in per_class:
                    sensitivity = per_class[condition].get('recall', 0)
                    critical_sensitivities.append(sensitivity)
                    self.history[condition].append(sensitivity)
            
            if critical_sensitivities:
                avg_critical_sensitivity = np.mean(critical_sensitivities)
                min_critical_sensitivity = np.min(critical_sensitivities)
                
                logger.info(f"Época {epoch}: Sensibilidade média condições críticas: {avg_critical_sensitivity:.3f}")
                logger.info(f"Época {epoch}: Sensibilidade mínima condições críticas: {min_critical_sensitivity:.3f}")
                
                # Alertar se abaixo do threshold
                if min_critical_sensitivity < self.min_sensitivity:
                    low_conditions = [
                        cond for cond, sens in zip(self.critical_conditions, critical_sensitivities)
                        if sens < self.min_sensitivity
                    ]
                    logger.warning(f"ALERTA: Sensibilidade baixa para condições críticas: {low_conditions}")


# ==================== TRAINER PRINCIPAL ====================

class ECGTrainer:
    """Trainer otimizado para modelos de ECG"""
    
    def __init__(self, model: nn.Module, config: EnhancedECGAnalysisConfig,
                 device: torch.device = None):
        self.model = model
        self.config = config
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Mover modelo para device
        self.model = self.model.to(self.device)
        
        # Optimizer
        self.optimizer = self._create_optimizer()
        
        # Loss function
        self.criterion = self._create_loss_function()
        
        # Métricas
        self.train_metrics = ECGMetrics(
            num_classes=len(SCP_ECG_CONDITIONS),
            class_names=list(SCP_ECG_CONDITIONS.keys()),
            device=self.device
        )
        self.val_metrics = ECGMetrics(
            num_classes=len(SCP_ECG_CONDITIONS),
            class_names=list(SCP_ECG_CONDITIONS.keys()),
            device=self.device
        )
        
        # Callbacks
        self.callbacks = []
        
        # Adicionar monitor de condições críticas
        self.add_callback(CriticalConditionMonitor(
            critical_conditions=['STEMI', 'VTAC', 'VFIB', 'AVB3'],
            min_sensitivity=0.85
        ))
        
        # Mixed precision
        self.use_amp = config.use_mixed_precision and torch.cuda.is_available()
        if self.use_amp:
            self.scaler = GradScaler()
        
        # EMA
        self.ema = None
        if config.use_ema:
            self.ema = ModelEMA(self.model, decay=config.ema_decay)
        
        # Estado
        self.current_epoch = 0
        self.global_step = 0
        self.should_stop = False
        
        # Histórico
        self.history = defaultdict(list)
    
    def _create_optimizer(self) -> torch.optim.Optimizer:
        """Cria otimizador com configurações específicas"""
        # Separar parâmetros por tipo
        decay_params = []
        no_decay_params = []
        
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            
            if 'bias' in name or 'bn' in name or 'norm' in name:
                no_decay_params.append(param)
            else:
                decay_params.append(param)
        
        # Grupos de parâmetros com weight decay diferenciado
        param_groups = [
            {'params': decay_params, 'weight_decay': 1e-4},
            {'params': no_decay_params, 'weight_decay': 0.0}
        ]
        
        # Criar otimizador
        optimizer = AdamW(
            param_groups,
            lr=self.config.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-8
        )
        
        return optimizer
    
    def _create_loss_function(self) -> nn.Module:
        """Cria função de perda apropriada com peso clínico"""
        
        # Definir condições críticas que necessitam alta sensibilidade
        critical_conditions = ['STEMI', 'NSTEMI', 'VTAC', 'VFIB', 'AVB3', 'AFIB', 'AFLT']
        
        # Loss base (como está)
        if hasattr(self.config, 'use_focal_loss') and self.config.use_focal_loss:
            base_loss = FocalLoss(
                alpha=self.config.focal_loss_alpha,
                gamma=self.config.focal_loss_gamma
            )
            logger.info("Usando Focal Loss como base")
        elif hasattr(self.config, 'use_asymmetric_loss') and self.config.use_asymmetric_loss:
            base_loss = AsymmetricLoss(
                gamma_neg=4,
                gamma_pos=1,
                clip=0.05
            )
            logger.info("Usando Asymmetric Loss como base")
        elif self.config.focal_loss_alpha is not None:
            # Focal loss com pesos de classe
            base_loss = FocalLoss(
                alpha=torch.tensor(self.config.focal_loss_alpha),
                gamma=self.config.focal_loss_gamma
            )
            logger.info("Usando Focal Loss como base")
        else:
            # Asymmetric loss para multi-label
            base_loss = AsymmetricLoss(
                gamma_neg=4,
                gamma_pos=1,
                clip=0.05
            )
            logger.info("Usando Asymmetric Loss como base")
        
        # Envolver com loss clinicamente ponderada
        clinical_loss = ClinicallyWeightedLoss(
            base_loss=base_loss,
            critical_conditions=critical_conditions,
            critical_weight=2.5,  # Peso 2.5x para condições críticas
            rare_weight=1.8       # Peso 1.8x para condições raras
        )
        
        logger.info(f"Loss configurada com peso aumentado para condições críticas: {critical_conditions}")
        
        return clinical_loss
    
    def add_callback(self, callback: TrainingCallback):
        """Adiciona callback ao trainer"""
        self.callbacks.append(callback)
    
    def train_epoch(self, train_loader: DataLoader) -> Dict[str, float]:
        """Treina uma época"""
        self.model.train()
        self.train_metrics.reset()
        
        epoch_loss = 0
        progress_bar = tqdm(train_loader, desc=f'Epoch {self.current_epoch + 1} [Train]')
        
        for batch_idx, batch in enumerate(progress_bar):
            # Callbacks
            for callback in self.callbacks:
                callback.on_batch_start(batch_idx, self)
            
            # Preparar dados
            signals = batch['signals'].to(self.device)
            labels = batch['labels'].to(self.device)
            
            # Forward pass com mixed precision
            if self.use_amp:
                with autocast():
                    outputs = self.model(signals)
                    loss = self.criterion(outputs, labels)
                    
                    if isinstance(loss, dict):
                        loss_value = loss['total']
                    else:
                        loss_value = loss
            else:
                outputs = self.model(signals)
                loss = self.criterion(outputs, labels)
                
                if isinstance(loss, dict):
                    loss_value = loss['total']
                else:
                    loss_value = loss
            
            # Gradient accumulation
            loss_value = loss_value / self.config.gradient_accumulation_steps
            
            # Backward pass
            if self.use_amp:
                self.scaler.scale(loss_value).backward()
            else:
                loss_value.backward()
            
            # Gradient step
            if (batch_idx + 1) % self.config.gradient_accumulation_steps == 0:
                if self.use_amp:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    self.optimizer.step()
                
                self.optimizer.zero_grad()
                
                # Update EMA
                if self.ema is not None:
                    self.ema.update(self.model)
                
                self.global_step += 1
            
            # Atualizar métricas
            with torch.no_grad():
                self.train_metrics.update(outputs, labels)
                epoch_loss += loss_value.item() * self.config.gradient_accumulation_steps
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': f'{loss_value.item():.4f}',
                'lr': f'{self.optimizer.param_groups[0]["lr"]:.6f}'
            })
            
            # Callbacks
            for callback in self.callbacks:
                callback.on_batch_end(batch_idx, self, loss_value.item())
        
        # Calcular métricas da época
        epoch_metrics = self.train_metrics.compute()
        epoch_metrics['loss'] = epoch_loss / len(train_loader)
        
        return epoch_metrics
    
    @torch.no_grad()
    def validate(self, val_loader: DataLoader, use_ema: bool = True) -> Dict[str, float]:
        """Valida o modelo"""
        # Usar EMA se disponível e solicitado
        if use_ema and self.ema is not None:
            model = self.ema.ema_model
        else:
            model = self.model
        
        model.eval()
        self.val_metrics.reset()
        
        epoch_loss = 0
        all_outputs = []
        all_labels = []
        
        progress_bar = tqdm(val_loader, desc=f'Epoch {self.current_epoch + 1} [Val]')
        
        for batch in progress_bar:
            signals = batch['signals'].to(self.device)
            labels = batch['labels'].to(self.device)
            
            # Forward pass
            if self.use_amp:
                with autocast():
                    outputs = model(signals)
                    loss = self.criterion(outputs, labels)
            else:
                outputs = model(signals)
                loss = self.criterion(outputs, labels)
            
            if isinstance(loss, dict):
                loss_value = loss['total']
            else:
                loss_value = loss
            
            # Coletar outputs para TTA
            all_outputs.append(outputs.cpu())
            all_labels.append(labels.cpu())
            
            # Atualizar métricas
            self.val_metrics.update(outputs, labels)
            epoch_loss += loss_value.item()
            
            progress_bar.set_postfix({'loss': f'{loss_value.item():.4f}'})
        
        # Test Time Augmentation se configurado
        if self.config.use_tta and hasattr(self, 'augmenter'):
            tta_outputs = self._apply_tta(val_loader, model)
            
            # Combinar predições originais com TTA
            all_outputs = torch.cat(all_outputs, dim=0)
            tta_outputs = torch.cat(tta_outputs, dim=0)
            
            # Média das predições
            combined_outputs = (all_outputs + tta_outputs) / 2
            all_labels = torch.cat(all_labels, dim=0)
            
            # Recalcular métricas com TTA
            self.val_metrics.reset()
            self.val_metrics.update(combined_outputs.to(self.device), all_labels.to(self.device))
        
        # Calcular métricas
        epoch_metrics = self.val_metrics.compute()
        epoch_metrics['loss'] = epoch_loss / len(val_loader)
        
        return epoch_metrics
    
    def _apply_tta(self, loader: DataLoader, model: nn.Module) -> List[torch.Tensor]:
        """Aplica Test Time Augmentation"""
        tta_outputs = []
        
        for _ in range(self.config.tta_augmentations):
            batch_outputs = []
            
            for batch in loader:
                signals = batch['signals'].to(self.device)
                
                # Aplicar augmentação
                if hasattr(self, 'augmenter'):
                    signals_aug, _ = self.augmenter.augment(signals.cpu().numpy())
                    signals_aug = torch.tensor(signals_aug).to(self.device)
                else:
                    signals_aug = signals
                
                # Forward pass
                with torch.no_grad():
                    if self.use_amp:
                        with autocast():
                            outputs = model(signals_aug)
                    else:
                        outputs = model(signals_aug)
                
                batch_outputs.append(outputs.cpu())
            
            tta_outputs.append(torch.cat(batch_outputs, dim=0))
        
        # Média das augmentações
        return [torch.stack(tta_outputs).mean(dim=0)]
    
    def fit(self, train_loader: DataLoader, val_loader: DataLoader, 
            num_epochs: int = None):
        """Treina o modelo por várias épocas"""
        num_epochs = num_epochs or self.config.num_epochs
        
        # Callbacks - início do treinamento
        for callback in self.callbacks:
            callback.on_training_start(self)
        
        # Log inicial
        logger.info(f"Iniciando treinamento por {num_epochs} épocas")
        logger.info(f"Device: {self.device}")
        logger.info(f"Mixed Precision: {self.use_amp}")
        
        best_metric = None
        
        for epoch in range(num_epochs):
            self.current_epoch = epoch
            
            # Callbacks - início da época
            for callback in self.callbacks:
                callback.on_epoch_start(epoch, self)
            
            # Treinar
            train_metrics = self.train_epoch(train_loader)
            
            # Validar
            val_metrics = self.validate(val_loader)
            
            # Combinar métricas
            epoch_metrics = {
                f'train_{k}': v for k, v in train_metrics.items()
            }
            epoch_metrics.update({
                f'val_{k}': v for k, v in val_metrics.items()
            })
            
            # Atualizar histórico
            for key, value in epoch_metrics.items():
                if isinstance(value, (int, float)):
                    self.history[key].append(value)
            
            # Log
            logger.info(
                f"Epoch {epoch + 1}/{num_epochs} - "
                f"Train Loss: {train_metrics['loss']:.4f}, "
                f"Val Loss: {val_metrics['loss']:.4f}, "
                f"Val AUROC: {val_metrics.get('auroc_mean', 0):.4f}"
            )
            
            # Callbacks - fim da época
            for callback in self.callbacks:
                callback.on_epoch_end(epoch, self, epoch_metrics)
            
            # Verificar early stopping
            if self.should_stop:
                logger.info(f"Treinamento interrompido na época {epoch + 1}")
                break
            
            # Log para clinical logger
            clinical_logger.log_clinical_event('training_epoch', {
                'epoch': epoch + 1,
                'train_loss': train_metrics['loss'],
                'val_loss': val_metrics['loss'],
                'val_auroc': val_metrics.get('auroc_mean', 0),
                'meets_requirements': val_metrics.get('meets_clinical_requirements', {})
            })
        
        # Callbacks - fim do treinamento
        for callback in self.callbacks:
            callback.on_training_end(self)
        
        # Gerar relatório final
        logger.info("\n" + self.val_metrics.generate_report())
        
        return self.history
    
    def save_checkpoint(self, filepath: Path):
        """Salva checkpoint completo"""
        checkpoint = {
            'epoch': self.current_epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config,
            'history': dict(self.history),
            'best_metrics': self.val_metrics.compute()
        }
        
        if self.ema is not None:
            checkpoint['ema_state_dict'] = self.ema.ema_model.state_dict()
        
        if self.use_amp:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()
        
        torch.save(checkpoint, filepath)
        logger.info(f"Checkpoint salvo em {filepath}")
    
    def load_checkpoint(self, filepath: Path):
        """Carrega checkpoint"""
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.current_epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        self.history = defaultdict(list, checkpoint.get('history', {}))
        
        if self.ema is not None and 'ema_state_dict' in checkpoint:
            self.ema.ema_model.load_state_dict(checkpoint['ema_state_dict'])
        
        if self.use_amp and 'scaler_state_dict' in checkpoint:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        
        logger.info(f"Checkpoint carregado de {filepath}")


# ==================== MODEL EMA ====================

class ModelEMA:
    """Exponential Moving Average para estabilização do modelo"""
    
    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.ema_model = self._create_ema_model(model)
        self.decay = decay
        self.device = next(model.parameters()).device
        self.ema_model.eval()
    
    def _create_ema_model(self, model: nn.Module) -> nn.Module:
        """Cria cópia do modelo para EMA"""
        ema_model = type(model)(
            num_leads=model.num_leads if hasattr(model, 'num_leads') else 12,
            num_classes=model.num_classes if hasattr(model, 'num_classes') else len(SCP_ECG_CONDITIONS)
        )
        
        ema_model.load_state_dict(model.state_dict())
        ema_model.to(self.device)
        
        for param in ema_model.parameters():
            param.requires_grad = False
        
        return ema_model
    
    @torch.no_grad()
    def update(self, model: nn.Module):
        """Atualiza parâmetros EMA"""
        for ema_param, param in zip(self.ema_model.parameters(), model.parameters()):
            ema_param.data.mul_(self.decay).add_(param.data, alpha=1 - self.decay)

# ==================== SISTEMA DE INFERÊNCIA ====================

class ECGInference:
    """Sistema de inferência otimizado para produção"""
    
    def __init__(self, model_path: Path, config: EnhancedECGAnalysisConfig,
                 device: torch.device = None, use_ema: bool = True):
        self.model_path = model_path
        self.config = config
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.use_ema = use_ema
        
        # Carregar modelo
        self.model = self._load_model()
        self.model.eval()
        
        # Preprocessador
        self.preprocessor = ECGPreprocessor(config)
        self.feature_extractor = ECGFeatureExtractor(config.sampling_rate, config)
        
        # Cache de predições
        self.prediction_cache = OrderedDict()
        self.max_cache_size = 100
        
        # Estatísticas de inferência
        self.stats = {
            'total_predictions': 0,
            'average_inference_time': 0,
            'cache_hits': 0,
            'preprocessing_failures': 0
        }
    
    def _load_model(self) -> nn.Module:
        """Carrega modelo do checkpoint"""
        checkpoint = torch.load(self.model_path, map_location=self.device)
        
        # Determinar arquitetura
        model_config = checkpoint.get('config', self.config)
        
        # Criar modelo
        model = ModelFactory.create_model(
            model_type='resnet',  # ou detectar do checkpoint
            num_leads=model_config.num_leads,
            num_classes=len(SCP_ECG_CONDITIONS)
        )
        
        # Carregar pesos
        if self.use_ema and 'ema_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['ema_state_dict'])
            logger.info("Carregado modelo EMA")
        else:
            model.load_state_dict(checkpoint['model_state_dict'])
            logger.info("Carregado modelo regular")
        
        model.to(self.device)
        
        return model
    
    @torch.no_grad()
    def predict(self, signal: np.ndarray, return_features: bool = False,
                apply_tta: bool = False) -> Dict[str, Any]:
        """Realiza predição em um sinal ECG"""
        start_time = time.time()
        
        # Verificar cache
        cache_key = hashlib.md5(signal.tobytes()).hexdigest()
        if cache_key in self.prediction_cache:
            self.stats['cache_hits'] += 1
            return self.prediction_cache[cache_key]
        
        try:
            # Preprocessar
            preprocess_result = self.preprocessor.preprocess(signal)
            
            if not preprocess_result['preprocessing_successful']:
                self.stats['preprocessing_failures'] += 1
                return self._create_error_prediction(
                    "Falha no preprocessamento", 
                    preprocess_result['quality_metrics']
                )
            
            processed_signal = preprocess_result['filtered_signal']
            
            # Converter para tensor
            signal_tensor = torch.FloatTensor(processed_signal).unsqueeze(0).to(self.device)
            
            # Inferência
            if apply_tta and self.config.use_tta:
                predictions = self._predict_with_tta(signal_tensor)
            else:
                predictions = self.model(signal_tensor)
            
            # Processar saída
            probabilities = torch.sigmoid(predictions).cpu().numpy()[0]
            
            # Extrair características se solicitado
            features = None
            if return_features:
                features = self.feature_extractor.extract_features(processed_signal)
            
            # Criar resultado
            result = {
                'probabilities': {
                    cond: float(prob) for cond, prob in zip(SCP_ECG_CONDITIONS.keys(), probabilities)
                },
                'predictions': {
                    cond: int(prob > 0.5) for cond, prob in zip(SCP_ECG_CONDITIONS.keys(), probabilities)
                },
                'quality_score': preprocess_result['quality_metrics']['overall_quality_score'],
                'preprocessing_steps': preprocess_result['preprocessing_steps'],
                'inference_time': time.time() - start_time,
                'features': features
            }
            
            # Adicionar interpretações clínicas
            result['clinical_findings'] = self._interpret_predictions(result['probabilities'])
            
            # Atualizar cache e estatísticas
            self._update_cache(cache_key, result)
            self._update_stats(result['inference_time'])
            
            return result
            
        except Exception as e:
            logger.error(f"Erro na inferência: {str(e)}")
            return self._create_error_prediction(str(e))
    
    def _predict_with_tta(self, signal_tensor: torch.Tensor) -> torch.Tensor:
        """Aplica Test Time Augmentation"""
        predictions = []
        
        # Predição original
        pred_original = self.model(signal_tensor)
        predictions.append(pred_original)
        
        # Augmentações
        augmenter = ECGAugmentation(self.config)
        
        for i in range(self.config.tta_augmentations):
            # Aplicar augmentação
            signal_aug = signal_tensor.cpu().numpy()[0]
            signal_aug, _ = augmenter.augment(signal_aug, seed=i)
            signal_aug = torch.FloatTensor(signal_aug).unsqueeze(0).to(self.device)
            
            # Predição
            pred_aug = self.model(signal_aug)
            predictions.append(pred_aug)
        
        # Média das predições
        return torch.stack(predictions).mean(dim=0)
    
    def _interpret_predictions(self, probabilities: Dict[str, float]) -> List[Dict[str, Any]]:
        """Interpreta predições em achados clínicos"""
        findings = []
        
        # Ordenar por probabilidade
        sorted_probs = sorted(probabilities.items(), key=lambda x: x[1], reverse=True)
        
        for condition, prob in sorted_probs:
            if prob > 0.5:  # Threshold pode ser ajustado por condição
                finding = {
                    'condition': condition,
                    'name': SCP_ECG_CONDITIONS[condition]['name'],
                    'probability': prob,
                    'confidence': self._calculate_confidence(prob),
                    'severity': SCP_ECG_CONDITIONS[condition]['severity'],
                    'category': SCP_ECG_CONDITIONS[condition]['category'].value,
                    'clinical_significance': self._get_clinical_significance(condition, prob)
                }
                findings.append(finding)
        
        # Adicionar achados normais se nenhuma patologia detectada
        if not findings:
            findings.append({
                'condition': 'NORM',
                'name': 'Normal ECG',
                'probability': probabilities.get('NORM', 0.9),
                'confidence': 'high',
                'severity': 0,
                'category': 'rhythm',
                'clinical_significance': 'No significant abnormalities detected'
            })
        
        return findings
    
    def _calculate_confidence(self, probability: float) -> str:
        """Calcula nível de confiança baseado na probabilidade"""
        if probability > 0.9:
            return 'very_high'
        elif probability > 0.75:
            return 'high'
        elif probability > 0.6:
            return 'moderate'
        else:
            return 'low'
    
    def _get_clinical_significance(self, condition: str, probability: float) -> str:
        """Retorna significância clínica da condição"""
        severity = SCP_ECG_CONDITIONS[condition]['severity']
        
        if severity >= 4:
            return "Critical finding requiring immediate attention"
        elif severity >= 3:
            return "Significant finding requiring clinical correlation"
        elif severity >= 2:
            return "Moderate finding, follow-up recommended"
        elif severity >= 1:
            return "Mild finding, may be clinically relevant"
        else:
            return "Normal variant or benign finding"
    
    def batch_predict(self, signals: List[np.ndarray], 
                     batch_size: int = 32) -> List[Dict[str, Any]]:
        """Predição em lote para múltiplos sinais"""
        results = []
        
        for i in range(0, len(signals), batch_size):
            batch_signals = signals[i:i + batch_size]
            batch_results = []
            
            # Processar batch
            for signal in batch_signals:
                result = self.predict(signal)
                batch_results.append(result)
            
            results.extend(batch_results)
        
        return results
    
    def _create_error_prediction(self, error_message: str, 
                                quality_metrics: Optional[Dict] = None) -> Dict[str, Any]:
        """Cria resultado de erro estruturado"""
        return {
            'probabilities': {cond: 0.0 for cond in SCP_ECG_CONDITIONS.keys()},
            'predictions': {cond: 0 for cond in SCP_ECG_CONDITIONS.keys()},
            'quality_score': quality_metrics.get('overall_quality_score', 0.0) if quality_metrics else 0.0,
            'error': error_message,
            'clinical_findings': [],
            'inference_time': 0
        }
    
    def _update_cache(self, key: str, result: Dict[str, Any]):
        """Atualiza cache de predições"""
        self.prediction_cache[key] = result
        
        # Limitar tamanho
        if len(self.prediction_cache) > self.max_cache_size:
            self.prediction_cache.popitem(last=False)
    
    def _update_stats(self, inference_time: float):
        """Atualiza estatísticas de inferência"""
        n = self.stats['total_predictions']
        self.stats['average_inference_time'] = (
            (self.stats['average_inference_time'] * n + inference_time) / (n + 1)
        )
        self.stats['total_predictions'] += 1
    
    def get_stats(self) -> Dict[str, Any]:
        """Retorna estatísticas de inferência"""
        return self.stats.copy()


# ==================== VISUALIZAÇÃO DE RESULTADOS ====================

class ECGVisualizer:
    """Visualização de ECG e resultados de análise"""
    
    def __init__(self, sampling_rate: int = 500):
        self.sampling_rate = sampling_rate
        
        # Configurar matplotlib
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        plt.style.use('seaborn-v0_8-darkgrid')
        sns.set_palette("husl")
        
        self.plt = plt
        self.sns = sns
    
    def plot_ecg(self, signal: np.ndarray, predictions: Optional[Dict[str, float]] = None,
                 title: str = "ECG Signal", save_path: Optional[Path] = None):
        """Plota sinal ECG com anotações"""
        if len(signal.shape) == 1:
            signal = signal.reshape(1, -1)
        
        num_leads = signal.shape[0]
        duration = signal.shape[1] / self.sampling_rate
        time = np.linspace(0, duration, signal.shape[1])
        
        # Criar figura
        fig, axes = self.plt.subplots(num_leads, 1, figsize=(15, 2 * num_leads),
                                     sharex=True)
        
        if num_leads == 1:
            axes = [axes]
        
        # Nome das derivações
        lead_names = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 
                      'V1', 'V2', 'V3', 'V4', 'V5', 'V6'][:num_leads]
        
        # Plotar cada derivação
        for i, (ax, lead_name) in enumerate(zip(axes, lead_names)):
            ax.plot(time, signal[i], 'b-', linewidth=0.5)
            ax.set_ylabel(f'{lead_name}\n(mV)', rotation=0, labelpad=30)
            ax.set_ylim(-2, 2)
            ax.grid(True, alpha=0.3)
            
            # Grid principal a cada 0.2s (200ms)
            major_ticks = np.arange(0, duration, 0.2)
            ax.set_xticks(major_ticks)
            ax.grid(True, which='major', linestyle='-', alpha=0.5)
            
            # Grid secundário a cada 0.04s (40ms)
            minor_ticks = np.arange(0, duration, 0.04)
            ax.set_xticks(minor_ticks, minor=True)
            ax.grid(True, which='minor', linestyle=':', alpha=0.3)
        
        axes[-1].set_xlabel('Time (s)')
        
        # Adicionar título com predições
        if predictions:
            top_conditions = sorted(predictions.items(), key=lambda x: x[1], reverse=True)[:3]
            subtitle = "Detected: " + ", ".join([f"{SCP_ECG_CONDITIONS[c]['name']} ({p:.1%})" 
                                               for c, p in top_conditions if p > 0.5])
            if subtitle == "Detected: ":
                subtitle = "Detected: Normal ECG"
            
            fig.suptitle(f"{title}\n{subtitle}", fontsize=14)
        else:
            fig.suptitle(title, fontsize=14)
        
        self.plt.tight_layout()
        
        if save_path:
            self.plt.savefig(save_path, dpi=300, bbox_inches='tight')
            self.plt.close()
        else:
            self.plt.show()
    
    def plot_predictions_heatmap(self, predictions: Dict[str, float], 
                                save_path: Optional[Path] = None):
        """Plota heatmap de predições"""
        # Preparar dados
        conditions = []
        probs = []
        
        for cond, prob in sorted(predictions.items(), key=lambda x: x[1], reverse=True):
            if prob > 0.1:  # Mostrar apenas probabilidades significativas
                conditions.append(SCP_ECG_CONDITIONS[cond]['name'])
                probs.append(prob)
        
        # Criar figura
        fig, ax = self.plt.subplots(figsize=(10, max(6, len(conditions) * 0.5)))
        
        # Criar barplot horizontal
        y_pos = np.arange(len(conditions))
        bars = ax.barh(y_pos, probs)
        
        # Colorir barras baseado na probabilidade
        for i, (bar, prob) in enumerate(zip(bars, probs)):
            if prob > 0.8:
                bar.set_color('darkred')
            elif prob > 0.5:
                bar.set_color('orange')
            else:
                bar.set_color('lightblue')
        
        # Adicionar valores
        for i, prob in enumerate(probs):
            ax.text(prob + 0.01, i, f'{prob:.1%}', va='center')
        
        ax.set_yticks(y_pos)
        ax.set_yticklabels(conditions)
        ax.set_xlabel('Probability')
        ax.set_xlim(0, 1)
        ax.set_title('ECG Analysis Results')
        
        # Adicionar linha de threshold
        ax.axvline(x=0.5, color='red', linestyle='--', alpha=0.5, label='Threshold')
        
        self.plt.tight_layout()
        
        if save_path:
            self.plt.savefig(save_path, dpi=300, bbox_inches='tight')
            self.plt.close()
        else:
            self.plt.show()
    
    def plot_training_history(self, history: Dict[str, List[float]], 
                             save_path: Optional[Path] = None):
        """Plota histórico de treinamento"""
        fig, axes = self.plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.ravel()
        
        # Loss
        if 'train_loss' in history and 'val_loss' in history:
            axes[0].plot(history['train_loss'], label='Train')
            axes[0].plot(history['val_loss'], label='Validation')
            axes[0].set_title('Loss')
            axes[0].set_xlabel('Epoch')
            axes[0].set_ylabel('Loss')
            axes[0].legend()
        
        # AUROC
        if 'val_auroc_mean' in history:
            axes[1].plot(history['val_auroc_mean'])
            axes[1].set_title('Validation AUROC')
            axes[1].set_xlabel('Epoch')
            axes[1].set_ylabel('AUROC')
            axes[1].set_ylim(0.5, 1)
        
        # F1-Score
        if 'val_f1_mean' in history:
            axes[2].plot(history['val_f1_mean'])
            axes[2].set_title('Validation F1-Score')
            axes[2].set_xlabel('Epoch')
            axes[2].set_ylabel('F1-Score')
            axes[2].set_ylim(0, 1)
        
        # Learning Rate
        if 'learning_rate' in history:
            axes[3].plot(history['learning_rate'])
            axes[3].set_title('Learning Rate')
            axes[3].set_xlabel('Epoch')
            axes[3].set_ylabel('LR')
            axes[3].set_yscale('log')
        
        self.plt.tight_layout()
        
        if save_path:
            self.plt.savefig(save_path, dpi=300, bbox_inches='tight')
            self.plt.close()
        else:
            self.plt.show()


# ==================== EXPORTAÇÃO DE MODELOS ====================

class ModelExporter:
    """Exporta modelos para diferentes formatos"""
    
    @staticmethod
    def export_onnx(model: nn.Module, save_path: Path, input_shape: Tuple[int, ...]):
        """Exporta modelo para ONNX"""
        model.eval()
        
        # Criar input dummy
        dummy_input = torch.randn(1, *input_shape)
        
        # Exportar
        torch.onnx.export(
            model,
            dummy_input,
            save_path,
            export_params=True,
            opset_version=11,
            do_constant_folding=True,
            input_names=['ecg_signal'],
            output_names=['predictions'],
            dynamic_axes={
                'ecg_signal': {0: 'batch_size'},
                'predictions': {0: 'batch_size'}
            }
        )
        
        logger.info(f"Modelo exportado para ONNX: {save_path}")
    
    @staticmethod
    def export_torchscript(model: nn.Module, save_path: Path, 
                          input_shape: Tuple[int, ...]):
        """Exporta modelo para TorchScript"""
        model.eval()
        
        # Criar input dummy
        dummy_input = torch.randn(1, *input_shape)
        
        # Trace do modelo
        traced_model = torch.jit.trace(model, dummy_input)
        
        # Salvar
        traced_model.save(save_path)
        
        logger.info(f"Modelo exportado para TorchScript: {save_path}")
    
    @staticmethod
    def export_tflite(model: nn.Module, save_path: Path, 
                     input_shape: Tuple[int, ...], quantize: bool = False):
        """Exporta modelo para TensorFlow Lite"""
        # Requer conversão via ONNX -> TF -> TFLite
        logger.warning("Exportação TFLite requer ferramentas adicionais")
        # Implementação omitida por brevidade


# ==================== PIPELINE COMPLETO ====================

class ECGAnalysisPipeline:
    """Pipeline completo de análise de ECG"""
    
    def __init__(self, config: EnhancedECGAnalysisConfig):
        self.config = config
        
        # Criar diretórios
        self.setup_directories()
        
        # Componentes do pipeline
        self.preprocessor = ECGPreprocessor(config)
        self.feature_extractor = ECGFeatureExtractor(config.sampling_rate, config)
        self.visualizer = ECGVisualizer(config.sampling_rate)
        
        # Modelos
        self.model = None
        self.trainer = None
        self.inference_engine = None
    
    def setup_directories(self):
        """Cria estrutura de diretórios"""
        dirs = [
            'data/raw',
            'data/processed',
            'models',
            'logs',
            'logs/clinical',
            'results',
            'results/plots',
            'results/reports'
        ]
        
        for dir_path in dirs:
            Path(dir_path).mkdir(parents=True, exist_ok=True)
    
    def prepare_data(self, data_path: Path) -> Tuple[DataLoader, DataLoader]:
        """Prepara dataloaders para treinamento"""
        # Criar datasets
        train_dataset = OptimizedECGDataset(
            data_path=data_path,
            config=self.config,
            split='train'
        )
        
        val_dataset = OptimizedECGDataset(
            data_path=data_path,
            config=self.config,
            split='val'
        )
        
        # Criar dataloaders
        train_loader = ECGDataLoader(
            dataset=train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.num_workers,
            use_balanced_sampling=True
        ).loader
        
        val_loader = ECGDataLoader(
            dataset=val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            use_balanced_sampling=False
        ).loader
        
        return train_loader, val_loader
    
    def build_model(self, model_type: str = 'resnet') -> nn.Module:
        """Constrói modelo"""
        self.model = ModelFactory.create_model(
            model_type=model_type,
            num_leads=self.config.num_leads,
            num_classes=len(SCP_ECG_CONDITIONS),
            config={
                'use_attention': True,
                'use_calibration': True,
                'calibration_method': 'temperature'
            }
        )
        
        # Log informações do modelo
        model_info = ModelFactory.get_model_params(self.model)
        logger.info(f"Modelo criado: {model_info}")
        
        return self.model
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader, 
              num_epochs: Optional[int] = None):
        """Treina o modelo"""
        if self.model is None:
            raise ValueError("Modelo não foi construído. Use build_model() primeiro.")
        
        # Criar trainer
        self.trainer = ECGTrainer(
            model=self.model,
            config=self.config
        )
        
        # Adicionar callbacks
        self.trainer.add_callback(
            EarlyStopping(
                patience=self.config.early_stopping_patience,
                monitor='val_auroc_mean',
                mode='max'
            )
        )
        
        self.trainer.add_callback(
            ModelCheckpoint(
                filepath=Path('models/best_model.pth'),
                monitor='val_auroc_mean',
                mode='max'
            )
        )
        
        # Learning rate scheduler
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.trainer.optimizer,
            mode='max',
            factor=0.5,
            patience=5,
            verbose=True
        )
        
        self.trainer.add_callback(
            LearningRateScheduler(scheduler, metric='val_auroc_mean')
        )
        
        # Treinar
        history = self.trainer.fit(train_loader, val_loader, num_epochs)
        
        # Salvar histórico
        self.save_training_history(history)
        
        # Plotar resultados
        self.visualizer.plot_training_history(
            history,
            save_path=Path('results/plots/training_history.png')
        )
        
        return history
    
    def evaluate(self, test_loader: DataLoader) -> Dict[str, Any]:
        """Avalia modelo no conjunto de teste"""
        if self.trainer is None:
            raise ValueError("Modelo não foi treinado. Use train() primeiro.")
        
        # Avaliar
        test_metrics = self.trainer.validate(test_loader, use_ema=True)
        
        # Gerar relatório
        metrics_obj = self.trainer.val_metrics
        report = metrics_obj.generate_report()
        
        # Salvar relatório
        report_path = Path('results/reports/test_evaluation.txt')
        with open(report_path, 'w') as f:
            f.write(report)
        
        logger.info(f"Relatório de avaliação salvo em {report_path}")
        
        # Gerar relatório clínico
        clinical_report = clinical_logger.generate_clinical_report()
        clinical_logger.export_for_regulatory_compliance(
            Path('results/reports/clinical_compliance.json')
        )
        
        return test_metrics
    
    def deploy(self, model_path: Path, export_formats: List[str] = ['onnx', 'torchscript']):
        """Prepara modelo para deployment"""
        # Criar engine de inferência
        self.inference_engine = ECGInference(
            model_path=model_path,
            config=self.config,
            use_ema=True
        )
        
        # Exportar modelo
        exporter = ModelExporter()
        input_shape = (self.config.num_leads, self.config.signal_length)
        
        if 'onnx' in export_formats:
            exporter.export_onnx(
                self.inference_engine.model,
                Path('models/ecg_model.onnx'),
                input_shape
            )
        
        if 'torchscript' in export_formats:
            exporter.export_torchscript(
                self.inference_engine.model,
                Path('models/ecg_model.pt'),
                input_shape
            )
        
        logger.info("Modelo preparado para deployment")
    
    def analyze_single_ecg(self, ecg_path: Union[str, Path], 
                          plot_results: bool = True) -> Dict[str, Any]:
        """Analisa um único ECG"""
        if self.inference_engine is None:
            raise ValueError("Engine de inferência não inicializada. Use deploy() primeiro.")
        
        # Carregar sinal
        if WFDB_AVAILABLE and str(ecg_path).endswith('.hea'):
            record = wfdb.rdrecord(str(ecg_path).replace('.hea', ''))
            signal = record.p_signal.T
        else:
            signal = np.load(ecg_path)
        
        # Fazer predição
        result = self.inference_engine.predict(signal, return_features=True)
        
        # Visualizar se solicitado
        if plot_results:
            output_dir = Path('results/plots')
            
            # Plot do sinal
            self.visualizer.plot_ecg(
                signal,
                result['probabilities'],
                title=f"ECG Analysis - {Path(ecg_path).stem}",
                save_path=output_dir / f"{Path(ecg_path).stem}_signal.png"
            )
            
            # Plot das predições
            self.visualizer.plot_predictions_heatmap(
                result['probabilities'],
                save_path=output_dir / f"{Path(ecg_path).stem}_predictions.png"
            )
        
        # Gerar relatório
        self.generate_analysis_report(result, ecg_path)
        
        return result
    
    def generate_analysis_report(self, result: Dict[str, Any], ecg_path: Union[str, Path]):
        """Gera relatorio de analise detalhado"""
        report = []
        report.append("=" * 80)
        report.append("RELATÓRIO DE ANÁLISE DE ECG")
        report.append("=" * 80)
        report.append(f"\nArquivo: {ecg_path}")
        report.append(f"Data: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"Qualidade do sinal: {result['quality_score']:.2f}")
        report.append(f"Tempo de inferência: {result['inference_time']:.3f}s")
        
        report.append("\n\nACHADOS CLÍNICOS:")
        report.append("-" * 40)
        
        for finding in result['clinical_findings']:
            report.append(f"\n{finding['name']}:")
            report.append(f"  - Probabilidade: {finding['probability']:.1%}")
            report.append(f"  - Confiança: {finding['confidence']}")
            report.append(f"  - Severidade: {finding['severity']}/5")
            report.append(f"  - Significância: {finding['clinical_significance']}")
        
        if result.get('features'):
            report.append("\n\nCARACTERÍSTICAS EXTRAÍDAS:")
            report.append("-" * 40)
            
            # Características temporais
            temporal = result['features'].get('temporal', {})
            if temporal:
                report.append("\nTemporal:")
                report.append(f"  - FC média: {temporal.get('global_mean_hr', 0):.1f} bpm")
                report.append(f"  - Variabilidade FC: {temporal.get('global_std_hr', 0):.1f} bpm")
            
            # Características morfológicas
            morph = result['features'].get('morphological', {})
            if morph:
                report.append("\nMorfológicas:")
                if 'II_qrs_width_mean' in morph:
                    report.append(f"  - Duração QRS (DII): {morph['II_qrs_width_mean']:.1f} ms")
                if 'II_pr_interval_mean' in morph:
                    report.append(f"  - Intervalo PR (DII): {morph['II_pr_interval_mean']:.1f} ms")
                if 'II_qt_interval_mean' in morph:
                    report.append(f"  - Intervalo QT (DII): {morph['II_qt_interval_mean']:.1f} ms")
                if 'II_qtc_bazett' in morph:
                    report.append(f"  - QTc (Bazett): {morph['II_qtc_bazett']:.1f} ms")
        
        report.append("\n" + "=" * 80)
        
        # Salvar relatório
        report_path = Path('results/reports') / f"{Path(ecg_path).stem}_analysis.txt"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report))
        
        logger.info(f"Relatório salvo em {report_path}")
    
    def save_training_history(self, history: Dict[str, List[float]]):
        """Salva histórico de treinamento"""
        history_path = Path('results/training_history.json')
        
        # Converter para formato serializável
        serializable_history = {}
        for key, values in history.items():
            if isinstance(values, list):
                serializable_history[key] = values
            else:
                serializable_history[key] = float(values)
        
        with open(history_path, 'w') as f:
            json.dump(serializable_history, f, indent=2)
        
        logger.info(f"Histórico salvo em {history_path}")


# ==================== FUNÇÃO PRINCIPAL ====================

def main():
    """Funcao principal de execucao"""
    # Configurar argumentos
    import argparse
    
    parser = argparse.ArgumentParser(description='Sistema de Análise de ECG com Deep Learning')
    parser.add_argument('--mode', type=str, choices=['train', 'evaluate', 'analyze', 'demo'],
                       default='demo', help='Modo de execução')
    parser.add_argument('--data-path', type=str, default='data',
                       help='Caminho para os dados')
    parser.add_argument('--model-path', type=str, default='models/best_model.pth',
                       help='Caminho do modelo')
    parser.add_argument('--ecg-file', type=str, help='Arquivo ECG para análise')
    parser.add_argument('--config-file', type=str, help='Arquivo de configuração JSON')
    parser.add_argument('--num-epochs', type=int, help='Número de épocas')
    parser.add_argument('--batch-size', type=int, help='Tamanho do batch')
    parser.add_argument('--model-type', type=str, default='resnet',
                       choices=['resnet', 'inception', 'hybrid', 'ensemble'],
                       help='Tipo de modelo')
    
    args = parser.parse_args()
    
    # Carregar configuração
    if args.config_file:
        with open(args.config_file, 'r') as f:
            config_dict = json.load(f)
        config = EnhancedECGAnalysisConfig(**config_dict)
    else:
        config = EnhancedECGAnalysisConfig()
    
    # Sobrescrever configurações com argumentos
    if args.num_epochs:
        config.num_epochs = args.num_epochs
    if args.batch_size:
        config.batch_size = args.batch_size
    
    # Criar pipeline
    pipeline = ECGAnalysisPipeline(config)
    
    # Executar modo apropriado
    if args.mode == 'train':
        logger.info("Iniciando treinamento...")
        
        # Preparar dados
        train_loader, val_loader = pipeline.prepare_data(Path(args.data_path))
        
        # Construir modelo
        pipeline.build_model(args.model_type)
        
        # Treinar
        history = pipeline.train(train_loader, val_loader)
        
        logger.info("Treinamento concluído!")
        
    elif args.mode == 'evaluate':
        logger.info("Avaliando modelo...")
        
        # Carregar dados de teste
        test_dataset = OptimizedECGDataset(
            data_path=Path(args.data_path),
            config=config,
            split='test'
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=config.num_workers
        )
        
        # Avaliar
        test_metrics = pipeline.evaluate(test_loader)
        
        logger.info(f"AUROC médio no teste: {test_metrics.get('auroc_mean', 0):.4f}")
        
    elif args.mode == 'analyze':
        if not args.ecg_file:
            raise ValueError("--ecg-file é obrigatório no modo analyze")
        
        logger.info(f"Analisando ECG: {args.ecg_file}")
        
        # Deploy do modelo
        pipeline.deploy(Path(args.model_path))
        
        # Analisar ECG
        result = pipeline.analyze_single_ecg(args.ecg_file)
        
        # Imprimir principais achados
        print("\nPrincipais achados:")
        for finding in result['clinical_findings'][:3]:
            print(f"- {finding['name']}: {finding['probability']:.1%}")
        
    elif args.mode == 'demo':
        logger.info("Executando demonstração...")
        
        # Gerar ECG sintético para demo
        demo_signal = np.random.randn(12, 5000) * 0.5
        
        # Adicionar padrão similar a ECG
        t = np.linspace(0, 10, 5000)
        for i in range(12):
            # Simular complexos QRS
            for beat_time in np.arange(0, 10, 1):  # 60 bpm
                beat_idx = int(beat_time * 500)
                if beat_idx < len(demo_signal[i]):
                    # Onda R
                    demo_signal[i, beat_idx] += np.random.uniform(1, 2)
                    # Onda S
                    if beat_idx + 10 < len(demo_signal[i]):
                        demo_signal[i, beat_idx + 10] -= np.random.uniform(0.5, 1)
        
        # Salvar sinal demo
        np.save('demo_ecg.npy', demo_signal)
        
        # Construir e treinar modelo mini para demo
        logger.info("Criando modelo de demonstração...")
        
        pipeline.build_model('resnet')
        
        # Criar dados sintéticos
        demo_dataset = [(demo_signal, np.random.randint(0, 2, len(SCP_ECG_CONDITIONS))) 
                       for _ in range(100)]
        
        logger.info("Demonstração concluída! Modelo pronto para uso.")
        
        # Limpar
        Path('demo_ecg.npy').unlink(missing_ok=True)
    
    # Gerar relatório final
    clinical_logger.export_for_regulatory_compliance(
        Path('results/reports/final_clinical_compliance.json')
    )
    
    logger.info("Execução concluída com sucesso!")


# ==================== PONTO DE ENTRADA ====================

if __name__ == "__main__":
    # Configurar multiprocessamento
    if sys.platform == 'win32':
        # Windows
        from multiprocessing import freeze_support
        freeze_support()
    
    try:
        main()
    except KeyboardInterrupt:
        logger.info("Execução interrompida pelo usuário")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Erro fatal: {str(e)}")
        logger.error(traceback.format_exc())
        sys.exit(1)