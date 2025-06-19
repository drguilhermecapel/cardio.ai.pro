"""
Configuração de integração da plataforma de treinamento
"""

import os
from pathlib import Path
from typing import Optional

# Configurações específicas para a plataforma de treinamento
TRAINING_ENABLED = os.getenv("TRAINING_ENABLED", "true").lower() == "true"
TRAINING_API_PORT = int(os.getenv("TRAINING_API_PORT", "8001"))
TRAINING_DATA_PATH = os.getenv("TRAINING_DATA_PATH", "/data/training")
TRAINING_MODELS_PATH = os.getenv("TRAINING_MODELS_PATH", "/models/training")

# Configurações de recursos
TRAINING_MAX_CONCURRENT_JOBS = int(os.getenv("TRAINING_MAX_CONCURRENT_JOBS", "2"))
TRAINING_GPU_ENABLED = os.getenv("TRAINING_GPU_ENABLED", "auto").lower()
TRAINING_MAX_EPOCHS_DEFAULT = int(os.getenv("TRAINING_MAX_EPOCHS_DEFAULT", "100"))

# Configurações de segurança
TRAINING_ALLOWED_ROLES = os.getenv("TRAINING_ALLOWED_ROLES", "admin,researcher").split(",")
TRAINING_REQUIRE_APPROVAL = os.getenv("TRAINING_REQUIRE_APPROVAL", "false").lower() == "true"

# Configurações de notificação
TRAINING_NOTIFY_ON_COMPLETION = os.getenv("TRAINING_NOTIFY_ON_COMPLETION", "true").lower() == "true"
TRAINING_NOTIFY_ON_FAILURE = os.getenv("TRAINING_NOTIFY_ON_FAILURE", "true").lower() == "true"

class TrainingIntegrationConfig:
    """Configuração de integração da plataforma de treinamento"""
    
    def __init__(self):
        self.enabled = TRAINING_ENABLED
        self.api_port = TRAINING_API_PORT
        self.data_path = Path(TRAINING_DATA_PATH)
        self.models_path = Path(TRAINING_MODELS_PATH)
        self.max_concurrent_jobs = TRAINING_MAX_CONCURRENT_JOBS
        self.gpu_enabled = self._parse_gpu_config(TRAINING_GPU_ENABLED)
        self.max_epochs_default = TRAINING_MAX_EPOCHS_DEFAULT
        self.allowed_roles = TRAINING_ALLOWED_ROLES
        self.require_approval = TRAINING_REQUIRE_APPROVAL
        self.notify_on_completion = TRAINING_NOTIFY_ON_COMPLETION
        self.notify_on_failure = TRAINING_NOTIFY_ON_FAILURE
        
        # Criar diretórios se não existirem
        self.data_path.mkdir(parents=True, exist_ok=True)
        self.models_path.mkdir(parents=True, exist_ok=True)
    
    def _parse_gpu_config(self, gpu_config: str) -> bool:
        """Parse da configuração de GPU"""
        if gpu_config == "auto":
            try:
                import torch
                return torch.cuda.is_available()
            except ImportError:
                return False
        return gpu_config == "true"
    
    def get_training_config(self) -> dict:
        """Retorna configuração para a plataforma de treinamento"""
        return {
            "enabled": self.enabled,
            "data_path": str(self.data_path),
            "models_path": str(self.models_path),
            "gpu_enabled": self.gpu_enabled,
            "max_concurrent_jobs": self.max_concurrent_jobs,
            "max_epochs_default": self.max_epochs_default,
            "allowed_roles": self.allowed_roles,
            "require_approval": self.require_approval
        }

# Instância global
training_integration_config = TrainingIntegrationConfig()

