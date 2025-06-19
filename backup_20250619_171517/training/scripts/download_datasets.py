"""
Script para download automático de datasets públicos de ECG
"""

import requests
import zipfile
import tarfile
import gzip
import shutil
from pathlib import Path
import logging
from typing import Dict, Any
import argparse

from backend.training.config.dataset_configs import DATASET_CONFIGS, DOWNLOAD_LINKS
from backend.training.config.training_config import training_config

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class DatasetDownloader:
    """Classe para download automático de datasets de ECG"""
    
    def __init__(self, data_root: Path = None):
        self.data_root = data_root or training_config.DATA_ROOT
        self.data_root.mkdir(parents=True, exist_ok=True)
        
    def download_dataset(self, dataset_name: str, force_download: bool = False):
        """Download de um dataset específico"""
        if dataset_name not in DATASET_CONFIGS:
            raise ValueError(f"Dataset {dataset_name} não encontrado")
            
        dataset_config = DATASET_CONFIGS[dataset_name]
        download_config = DOWNLOAD_LINKS.get(dataset_name, {})
        
        dataset_path = self.data_root / dataset_name
        
        if dataset_path.exists() and not force_download:
            logger.info(f"Dataset {dataset_name} já existe em {dataset_path}")
            return dataset_path
            
        logger.info(f"Iniciando download do dataset {dataset_name}")
        logger.info(f"Descrição: {dataset_config.description}")
        logger.info(f"Tamanho: {dataset_config.download_size}")
        
        if download_config.get("requires_auth", False):
            logger.warning(f"Dataset {dataset_name} requer autenticação")
            logger.info(f"Instruções: {download_config.get('instructions', 'Verifique a documentação')}")
            return None
            
        dataset_path.mkdir(parents=True, exist_ok=True)
        
        # Download específico por dataset
        if dataset_name == "mitbih":
            self._download_mitbih(dataset_path, download_config)
        elif dataset_name == "ptbxl":
            self._download_ptbxl(dataset_path, download_config)
        elif dataset_name == "cpsc2018":
            self._download_cpsc2018(dataset_path, download_config)
        else:
            logger.warning(f"Download automático não implementado para {dataset_name}")
            
        logger.info(f"Download do dataset {dataset_name} concluído")
        return dataset_path
        
    def _download_file(self, url: str, destination: Path, chunk_size: int = 8192):
        """Download de um arquivo com barra de progresso"""
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        downloaded = 0
        
        with open(destination, 'wb') as f:
            for chunk in response.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total_size > 0:
                        progress = (downloaded / total_size) * 100
                        print(f"\rDownload: {progress:.1f}%", end="", flush=True)
        print()  # Nova linha após o progresso
        
    def _extract_archive(self, archive_path: Path, extract_to: Path):
        """Extrai arquivo comprimido"""
        if archive_path.suffix == '.zip':
            with zipfile.ZipFile(archive_path, 'r') as zip_ref:
                zip_ref.extractall(extract_to)
        elif archive_path.suffix in ['.tar', '.tar.gz', '.tgz']:
            with tarfile.open(archive_path, 'r:*') as tar_ref:
                tar_ref.extractall(extract_to)
        elif archive_path.suffix == '.gz':
            with gzip.open(archive_path, 'rb') as f_in:
                with open(extract_to / archive_path.stem, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
                    
    def _download_mitbih(self, dataset_path: Path, config: Dict):
        """Download específico do MIT-BIH"""
        base_url = config["data"]
        
        # Lista de arquivos MIT-BIH (simplificada)
        mitbih_files = [
            "100.atr", "100.dat", "100.hea",
            "101.atr", "101.dat", "101.hea",
            "102.atr", "102.dat", "102.hea",
            # Adicione mais arquivos conforme necessário
        ]
        
        for filename in mitbih_files:
            url = f"{base_url}/{filename}"
            destination = dataset_path / filename
            try:
                logger.info(f"Baixando {filename}")
                self._download_file(url, destination)
            except Exception as e:
                logger.warning(f"Erro ao baixar {filename}: {e}")
                
    def _download_ptbxl(self, dataset_path: Path, config: Dict):
        """Download específico do PTB-XL"""
        # PTB-XL é um dataset grande, implementar download seletivo
        metadata_url = config["metadata"]
        metadata_path = dataset_path / "ptbxl_database.csv"
        
        logger.info("Baixando metadados do PTB-XL")
        self._download_file(metadata_url, metadata_path)
        
        # Para os dados completos, seria necessário baixar todos os arquivos WFDB
        # Isso pode ser implementado conforme necessário
        logger.info("Para dados completos do PTB-XL, baixe manualmente de physionet.org")
        
    def _download_cpsc2018(self, dataset_path: Path, config: Dict):
        """Download específico do CPSC2018"""
        reference_url = config["data"]
        reference_path = dataset_path / "REFERENCE.csv"
        
        logger.info("Baixando arquivo de referência do CPSC2018")
        self._download_file(reference_url, reference_path)
        
        # Para os dados de treinamento
        if "training" in config:
            training_url = config["training"]
            training_zip = dataset_path / "train_ecg.zip"
            
            logger.info("Baixando dados de treinamento do CPSC2018")
            self._download_file(training_url, training_zip)
            
            logger.info("Extraindo dados de treinamento")
            self._extract_archive(training_zip, dataset_path)


def main():
    parser = argparse.ArgumentParser(description="Download de datasets de ECG")
    parser.add_argument("--dataset", type=str, required=True,
                        help="Nome do dataset para download")
    parser.add_argument("--force", action="store_true",
                        help="Forçar download mesmo se já existir")
    parser.add_argument("--data_root", type=str, default=None,
                        help="Diretório raiz para salvar os dados")
    
    args = parser.parse_args()
    
    data_root = Path(args.data_root) if args.data_root else None
    downloader = DatasetDownloader(data_root)
    
    try:
        dataset_path = downloader.download_dataset(args.dataset, args.force)
        if dataset_path:
            logger.info(f"Dataset salvo em: {dataset_path}")
        else:
            logger.error("Falha no download do dataset")
    except Exception as e:
        logger.error(f"Erro durante o download: {e}")


if __name__ == "__main__":
    main()

