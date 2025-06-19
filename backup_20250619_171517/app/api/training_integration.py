"""
Integração da plataforma de treinamento com o sistema principal CardioAI Pro
"""

from fastapi import APIRouter, HTTPException, Depends
from typing import List, Dict, Any, Optional
import logging

from backend.training.api import app as training_app
from backend.training.config.training_config import training_config
from backend.training.config.dataset_configs import DATASET_CONFIGS
from backend.training.config.model_configs import MODEL_CONFIGS

logger = logging.getLogger(__name__)

# Router para integração com a API principal
training_router = APIRouter(prefix="/training", tags=["AI Training"])

@training_router.get("/info")
async def get_training_info():
    """Informações sobre a plataforma de treinamento"""
    return {
        "platform": "CardioAI Pro Training Platform",
        "version": "1.0.0",
        "available_models": list(MODEL_CONFIGS.keys()),
        "available_datasets": list(DATASET_CONFIGS.keys()),
        "status": "active"
    }

@training_router.get("/models/available")
async def get_available_models():
    """Lista modelos disponíveis para treinamento"""
    models = []
    for name, config_class in MODEL_CONFIGS.items():
        config = config_class()
        models.append({
            "name": name,
            "description": f"Modelo {name} para classificação de ECG",
            "input_channels": config.input_channels,
            "num_classes": config.num_classes,
            "recommended_for": _get_model_recommendations(name)
        })
    return {"models": models}

@training_router.get("/datasets/available")
async def get_available_datasets():
    """Lista datasets disponíveis para treinamento"""
    datasets = []
    for name, config in DATASET_CONFIGS.items():
        datasets.append({
            "name": name,
            "description": config.description,
            "num_samples": config.num_samples,
            "sampling_rate": config.sampling_rate,
            "num_leads": config.num_leads,
            "classes": config.classes,
            "download_size": config.download_size,
            "citation": config.citation
        })
    return {"datasets": datasets}

def _get_model_recommendations(model_name: str) -> List[str]:
    """Retorna recomendações de uso para cada modelo"""
    recommendations = {
        "heartbeit": [
            "Análise de ECG de alta complexidade",
            "Detecção de padrões sutis",
            "Pesquisa e desenvolvimento"
        ],
        "cnn_lstm": [
            "Classificação geral de arritmias",
            "Análise temporal de sinais",
            "Produção com recursos limitados"
        ],
        "se_resnet1d": [
            "Classificação rápida e eficiente",
            "Deployment em dispositivos móveis",
            "Análise em tempo real"
        ],
        "ecg_transformer": [
            "Análise interpretável",
            "Pesquisa clínica",
            "Modelos explicáveis"
        ]
    }
    return recommendations.get(model_name, ["Uso geral"])

# Função para integrar com o sistema de autenticação existente
def get_current_user():
    """Placeholder para integração com autenticação"""
    # Esta função deve ser implementada para usar o sistema de auth existente
    return {"user_id": "admin", "role": "admin"}

@training_router.post("/jobs/create")
async def create_training_job(
    model_name: str,
    dataset_name: str,
    config: Dict[str, Any],
    current_user: Dict = Depends(get_current_user)
):
    """Cria um novo job de treinamento"""
    if current_user["role"] not in ["admin", "researcher"]:
        raise HTTPException(status_code=403, detail="Permissão insuficiente")
    
    # Validações
    if model_name not in MODEL_CONFIGS:
        raise HTTPException(status_code=400, detail="Modelo não encontrado")
    
    if dataset_name not in DATASET_CONFIGS:
        raise HTTPException(status_code=400, detail="Dataset não encontrado")
    
    # Aqui seria feita a integração com o sistema de jobs do training_app
    # Por enquanto, retornamos uma resposta simulada
    
    return {
        "job_id": "job_123456",
        "status": "created",
        "message": "Job de treinamento criado com sucesso",
        "user_id": current_user["user_id"],
        "model_name": model_name,
        "dataset_name": dataset_name,
        "config": config
    }

# Função para incluir o router na aplicação principal
def include_training_routes(app):
    """Inclui as rotas de treinamento na aplicação principal"""
    app.include_router(training_router)
    logger.info("Rotas de treinamento de IA integradas com sucesso")

