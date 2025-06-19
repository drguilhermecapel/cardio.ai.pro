
"""
Configurações e metadados dos datasets públicos de ECG
"""

from dataclasses import dataclass
from typing import List, Dict, Optional
from pathlib import Path


@dataclass
class DatasetConfig:
    """Configuração base para datasets"""
    name: str
    url: str
    download_size: str
    num_samples: int
    sampling_rate: int
    num_leads: int
    duration: float  # em segundos
    classes: List[str]
    format: str  # WFDB, MAT, CSV, etc
    description: str
    citation: str
    

# Configurações dos datasets públicos
DATASET_CONFIGS = {
    "mitbih": DatasetConfig(
        name="MIT-BIH Arrhythmia Database",
        url="https://physionet.org/content/mitdb/1.0.0/",
        download_size="104.3 MB",
        num_samples=48,
        sampling_rate=360,
        num_leads=2,
        duration=1800.0,  # 30 minutos
        classes=["N", "S", "V", "F", "Q"],  # Normal, Supraventricular, Ventricular, Fusion, Unknown
        format="WFDB",
        description="48 registros de 30 minutos com anotações de arritmias",
        citation="Moody GB, Mark RG. MIT-BIH Arrhythmia Database. 1980."
    ),
    
    "ptbxl": DatasetConfig(
        name="PTB-XL ECG Database", 
        url="https://physionet.org/content/ptb-xl/1.0.3/",
        download_size="3.0 GB",
        num_samples=21837,
        sampling_rate=500,
        num_leads=12,
        duration=10.0,
        classes=["NORM", "MI", "STTC", "CD", "HYP"],  # 5 superclasses
        format="WFDB",
        description="21.837 ECGs de 12 derivações com 71 diagnósticos",
        citation="Wagner P, et al. PTB-XL, a large publicly available ECG dataset. 2020."
    ),
    
    "cpsc2018": DatasetConfig(
        name="China Physiological Signal Challenge 2018",
        url="http://2018.icbeb.org/Challenge.html",
        download_size="1.2 GB",
        num_samples=6877,
        sampling_rate=500,
        num_leads=12,
        duration=10.0,
        classes=["AF", "I-AVB", "LBBB", "RBBB", "PAC", "PVC", "STD", "STE", "Normal"],
        format="MAT",
        description="6.877 ECGs de 11 hospitais chineses",
        citation="Liu F, et al. China Physiological Signal Challenge 2018."
    ),
    
    "mimic_ecg": DatasetConfig(
        name="MIMIC-IV ECG Database",
        url="https://physionet.org/content/mimic-iv-ecg/1.0/",
        download_size="~50 GB",
        num_samples=800000,
        sampling_rate=500,
        num_leads=12,
        duration=10.0,
        classes=["Multiple diagnostic codes"],
        format="WFDB",
        description="~800k ECGs de pacientes de UTI com dados clínicos",
        citation="Gow B, et al. MIMIC-IV-ECG: Diagnostic ECG Database. 2023."
    ),
    
    "icentia11k": DatasetConfig(
        name="Icentia11k Single Lead ECG",
        url="https://physionet.org/content/icentia11k-continuous-ecg/1.0/",
        download_size="150 GB",
        num_samples=11000,
        sampling_rate=250,
        num_leads=1,
        duration=604800.0,  # ~7 dias
        classes=["Normal", "AF", "AFL", "Others"],
        format="HDF5",
        description="11k pacientes com ~2 bilhões de batimentos anotados",
        citation="Tan S, et al. Icentia11k ECG Dataset. 2022."
    ),
    
    "physionet2017": DatasetConfig(
        name="PhysioNet Challenge 2017",
        url="https://physionet.org/content/challenge-2017/1.0.0/",
        download_size="150 MB",
        num_samples=8528,
        sampling_rate=300,
        num_leads=1,
        duration=30.0,
        classes=["Normal", "AF", "Other", "Noise"],
        format="MAT",
        description="ECGs single-lead para classificação de ritmo",
        citation="Clifford GD, et al. PhysioNet Challenge 2017."
    )
}


def get_dataset_config(dataset_name: str) -> DatasetConfig:
    """Retorna configuração do dataset"""
    if dataset_name not in DATASET_CONFIGS:
        raise ValueError(f"Dataset {dataset_name} não encontrado. "
                        f"Disponíveis: {list(DATASET_CONFIGS.keys())}")
    return DATASET_CONFIGS[dataset_name]


# Links diretos para download
DOWNLOAD_LINKS = {
    "mitbih": {
        "data": "https://physionet.org/files/mitdb/1.0.0/",
        "annotations": "https://physionet.org/files/mitdb/1.0.0/mitdbdir",
        "requires_auth": False
    },
    "ptbxl": {
        "data": "https://physionet.org/files/ptb-xl/1.0.3/",
        "metadata": "https://physionet.org/files/ptb-xl/1.0.3/ptbxl_database.csv",
        "requires_auth": False
    },
    "cpsc2018": {
        "data": "http://2018.icbeb.org/file/REFERENCE.csv",
        "training": "http://hhbucket.oss-cn-hongkong.aliyuncs.com/train_ecg.zip",
        "requires_auth": False
    },
    "mimic_ecg": {
        "data": "https://physionet.org/files/mimic-iv-ecg/1.0/",
        "requires_auth": True,  # Requer credenciamento PhysioNet
        "instructions": "Necessário criar conta e aceitar DUA em physionet.org"
    }
}


