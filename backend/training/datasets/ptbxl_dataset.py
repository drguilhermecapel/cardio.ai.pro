
"""
Implementação do dataset PTB-XL ECG Database
"""

import wfdb
import numpy as np
import pandas as pd
import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
import logging

from .base_dataset import BaseECGDataset
from ..config.dataset_configs import get_dataset_config

logger = logging.getLogger(__name__)


class PTBXLDataset(BaseECGDataset):
    """Dataset para o PTB-XL ECG Database"""
    
    def __init__(self, data_path: Union[str, Path], **kwargs):
        super().__init__(data_path, **kwargs)
        self.dataset_config = get_dataset_config("ptbxl")
        self.sampling_rate = kwargs.get("sampling_rate", self.dataset_config.sampling_rate)
        self.download_data() # Adiciona esta linha para baixar os dados se necessário
        self.load_metadata() # Garante que metadados sejam carregados após o download
        
    def load_metadata(self):
        """Carrega metadados do dataset PTB-XL"""
        # O PTB-XL possui um arquivo CSV com metadados e os sinais em formato WFDB
        # O arquivo CSV principal é `ptbxl_database.csv`
        
        metadata_path = self.data_path / "ptbxl_database.csv"
        if not metadata_path.exists():
            raise FileNotFoundError(f"Arquivo de metadados ptbxl_database.csv não encontrado em {self.data_path}")
            
        self.raw_metadata = pd.read_csv(metadata_path, index_col='ecg_id')
        self.raw_metadata.scp_codes = self.raw_metadata.scp_codes.apply(lambda x: eval(x))
        
        # Filtrar por taxa de amostragem se especificado
        # self.metadata = self.raw_metadata[self.raw_metadata["sampling_frequency"] == self.sampling_rate]
        self.metadata = self.raw_metadata.copy()
        
        # Mapeamento de diagnósticos para superclasses (5 superclasses)
        # Ref: https://physionet.org/content/ptb-xl/1.0.3/doc/scp_statements.html
        agg_df = pd.read_csv(self.data_path / "scp_statements.csv", index_col=0)
        agg_df = agg_df[agg_df.diagnostic == 1]
        
        def aggregate_diagnostic(scp_codes):
            diagnosis_map = {
                "NORM": "NORM",
                "MI": "MI",
                "STTC": "STTC",
                "CD": "CD",
                "HYP": "HYP"
            }
            
            for key, value in scp_codes.items():
                if key in agg_df.index and agg_df.loc[key].diagnostic_class in diagnosis_map:
                    return diagnosis_map[agg_df.loc[key].diagnostic_class]
            return "UNKNOWN" # Fallback

        self.metadata["diagnostic_superclass"] = self.metadata.scp_codes.apply(aggregate_diagnostic)
        
        # Mapear superclasses para inteiros
        self.class_to_idx = {cls: i for i, cls in enumerate(self.dataset_config.classes)}
        self.metadata["label"] = self.metadata["diagnostic_superclass"].map(self.class_to_idx)
        
        # Remover amostras com labels desconhecidas ou nulas
        self.metadata.dropna(subset=["label"], inplace=True)
        self.metadata["label"] = self.metadata["label"].astype(int)
        
        self.samples = self.metadata.index.tolist()
        self.labels = self.metadata["label"].tolist()
        
        logger.info(f"PTB-XL metadata loaded. Found {len(self.samples)} ECGs at {self.sampling_rate} Hz.")

    def load_signal(self, idx: int) -> Tuple[np.ndarray, Dict]:
        """Carrega sinal ECG do registro especificado"""
        ecg_id = self.samples[idx]
        row = self.metadata.loc[ecg_id]
        
        # Caminho para o arquivo WFDB
        # Ex: records100/00001_hr.dat
        path_parts = list(row["filename_hr"].split(os.sep))
        record_path = self.data_path / "physionet.org" / "files" / "ptb-xl" / "1.0.3" / Path(*path_parts)
        
        try:
            record = wfdb.rdrecord(str(record_path))
            signal = record.p_signal.T  # Transpõe para (num_leads, num_samples)
            
            metadata = {
                "ecg_id": ecg_id,
                "sampling_rate": record.fs,
                "num_leads": record.n_sig,
                "signal_length": record.sig_len,
                "units": record.units,
                "lead_names": record.sig_name,
                "age": row["age"],
                "sex": row["sex"],
                "diagnostic_superclass": row["diagnostic_superclass"]
            }
            
            # Assegura que o número de derivações corresponde ao esperado
            if signal.shape[0] != self.num_leads:
                # Se o dataset tem menos derivações, preenche com zeros
                if signal.shape[0] < self.num_leads:
                    padding = np.zeros((self.num_leads - signal.shape[0], signal.shape[1]))
                    signal = np.vstack((signal, padding))
                # Se o dataset tem mais derivações, seleciona as primeiras
                else:
                    signal = signal[:self.num_leads, :]
            
            return signal, metadata
        
        except Exception as e:
            logger.error(f"Erro ao carregar sinal para ECG ID {ecg_id}: {e}")
            # Retorna um sinal de zeros e metadados vazios em caso de erro
            return np.zeros((self.num_leads, self.target_length)), {"error": str(e)}




    def download_data(self):
        """Baixa o dataset PTB-XL se não estiver presente."""
        if not (self.data_path / "ptbxl_database.csv").exists():
            logger.info(f"Baixando dataset PTB-XL para {self.data_path}...")
            try:
                wfdb.dl_database(
                    "ptb-xl",
                    dl_dir=str(self.data_path)
                )
                logger.info("Download do dataset PTB-XL concluído.")
            except Exception as e:
                logger.error(f"Erro ao baixar o dataset PTB-XL: {e}")
                raise
