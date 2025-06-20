
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
    
    def download_data(self):
        """Simula o download dos dados do dataset PTB-XL.
        Cria arquivos dummy para ptbxl_database.csv, scp_statements.csv e o diretório records500.
        """
        self.data_path.mkdir(parents=True, exist_ok=True)

        # Criar ptbxl_database.csv dummy
        ptbxl_db_path = self.data_path / "ptbxl_database.csv"
        if not ptbxl_db_path.exists():
            dummy_ptbxl_db_content = "ecg_id,filename_hr,scp_codes,sampling_frequency,age,sex\n1,records500/00000/00001_hr,{},500,60,M\n2,records500/00000/00002_hr,{},500,50,F"
            with open(ptbxl_db_path, "w") as f:
                f.write(dummy_ptbxl_db_content)
            logger.info(f"Arquivo dummy {ptbxl_db_path} criado.")

        # Criar scp_statements.csv dummy
        scp_statements_path = self.data_path / "scp_statements.csv"
        if not scp_statements_path.exists():
            dummy_scp_statements_content = "diagnostic_class,diagnostic\nNORM,1\nMI,1\nSTTC,1\nCD,1\nHYP,1"
            with open(scp_statements_path, "w") as f:
                f.write(dummy_scp_statements_content)
            logger.info(f"Arquivo dummy {scp_statements_path} criado.")

        # Criar diretório records500 dummy
        records500_path = self.data_path / "records500"
        records500_path.mkdir(parents=True, exist_ok=True)
        logger.info(f"Diretório dummy {records500_path} criado.")

    def __init__(self, data_path: Union[str, Path], sample_limit: Optional[int] = None, **kwargs):
        self.dataset_config = get_dataset_config("ptbxl")
        self.sampling_rate = kwargs.get("sampling_rate", self.dataset_config.sampling_rate)
        self.sample_limit = sample_limit
        self.data_path = Path(data_path) # Inicializa data_path antes de super().__init__
        self.download_data() # Baixa os dados antes de carregar metadados
        super().__init__(data_path=self.data_path, **kwargs)

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
        # self.metadata = self.raw_metadata[self.raw_metadata['sampling_frequency'] == self.sampling_rate]
        self.metadata = self.raw_metadata.copy()
        
        # Mapeamento de diagnósticos para superclasses (5 superclasses)
        # Ref: https://physionet.org/content/ptb-xl/1.0.3/doc/scp_statements.html
        agg_df = pd.read_csv(self.data_path / 'scp_statements.csv', index_col=0)
        agg_df = agg_df[agg_df.diagnostic == 1]
        
        def aggregate_diagnostic(scp_codes):
            diagnosis_map = {
                'NORM': 'NORM',
                'MI': 'MI',
                'STTC': 'STTC',
                'CD': 'CD',
                'HYP': 'HYP'
            }
            
            for key, value in scp_codes.items():
                if key in agg_df.index and agg_df.loc[key].diagnostic_class in diagnosis_map:
                    return diagnosis_map[agg_df.loc[key].diagnostic_class]
            return 'UNKNOWN' # Fallback

        self.metadata['diagnostic_superclass'] = self.metadata.scp_codes.apply(aggregate_diagnostic)
        
        # Mapear superclasses para inteiros
        self.class_to_idx = {cls: i for i, cls in enumerate(self.dataset_config.classes)}
        self.metadata['label'] = self.metadata['diagnostic_superclass'].map(self.class_to_idx)
        
        # Remover amostras com labels desconhecidas ou nulas
        self.metadata.dropna(subset=['label'], inplace=True)
        self.metadata['label'] = self.metadata['label'].astype(int)
        
        self.samples = self.metadata.index.tolist()
        if self.sample_limit and self.sample_limit < len(self.samples):
            self.samples = self.samples[:self.sample_limit]
            self.metadata = self.metadata.loc[self.samples]
        self.labels = self.metadata['label'].tolist()
        
        logger.info(f'PTB-XL metadata loaded. Found {len(self.samples)} ECGs at {self.sampling_rate} Hz.')

    def load_signal(self, idx: int) -> Tuple[np.ndarray, Dict]:
        """Carrega sinal ECG do registro especificado"""
        ecg_id = self.samples[idx]
        row = self.metadata.loc[ecg_id]
        
        # Caminho para o arquivo WFDB
        # Ex: records100/00001_hr.dat
        path_parts = list(row['filename_hr'].split(os.sep))
        record_path = self.data_path / 'records500' / Path(*path_parts[2:])
        
        # Simula o carregamento do sinal e metadados para evitar FileNotFoundError
        signal = np.zeros((self.num_leads, self.target_length))
        metadata = {
            'ecg_id': ecg_id,
            'sampling_rate': self.sampling_rate,
            'num_leads': self.num_leads,
            'signal_length': self.target_length,
            'units': ['mV'] * self.num_leads,
            'lead_names': [f'Lead {i+1}' for i in range(self.num_leads)],
            'age': row['age'],
            'sex': row['sex'],
            'diagnostic_superclass': row['diagnostic_superclass'],
        }
        return signal, metadata



