"""
Aplicação principal CardioAI Pro
"""
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from contextlib import asynccontextmanager
from typing import Dict, Any
import logging
import os
from pathlib import Path

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Gerenciador de ciclo de vida da aplicação."""
    # Startup
    logger.info("Starting up CardioAI Pro...")
    yield
    # Shutdown
    logger.info("Shutting down CardioAI Pro...")


# Criar aplicação FastAPI
app = FastAPI(
    title="CardioAI Pro",
    description="Sistema Avançado de Análise de ECG",
    version="1.0.0",
    lifespan=lifespan
)


async def get_app_info() -> Dict[str, Any]:
    """Retorna informações da aplicação."""
    return {
        "name": "CardioAI Pro",
        "version": "1.0.0",
        "description": "Sistema Avançado de Análise de ECG",
        "status": "running",
        "features": [
            "Análise automática de ECG",
            "Detecção de arritmias",
            "Validação médica",
            "Relatórios detalhados"
        ]
    }


async def health_check() -> Dict[str, str]:
    """Endpoint de health check."""
    return {
        "status": "healthy",
        "service": "CardioAI Pro",
        "version": "1.0.0"
    }


class CardioAIApp:
    """Classe principal da aplicação CardioAI."""
    
    def __init__(self):
        self.name = "CardioAI Pro"
        self.version = "1.0.0"
        self.description = "Sistema Avançado de Análise de ECG"
        self.status = "initialized"
        self.modules = []
        
    def get_info(self) -> Dict[str, str]:
        """Retorna informações da aplicação."""
        return {
            "name": self.name,
            "version": self.version,
            "description": self.description,
            "status": self.status,
            "modules": self.modules
        }
    
    def start(self):
        """Inicia a aplicação."""
        self.status = "running"
        logger.info(f"{self.name} v{self.version} iniciado com sucesso")
        
    def stop(self):
        """Para a aplicação."""
        self.status = "stopped"
        logger.info(f"{self.name} parado")
        
    def add_module(self, module_name: str):
        """Adiciona módulo à aplicação."""
        self.modules.append(module_name)
        logger.info(f"Módulo {module_name} adicionado")
        return True


# Instância global da aplicação
cardio_app = CardioAIApp()


# Endpoints da API
@app.get("/")
async def root():
    """Endpoint raiz."""
    return await get_app_info()


@app.get("/health")
async def health():
    """Endpoint de health check."""
    return await health_check()


@app.get("/info")
async def info():
    """Endpoint de informações da aplicação."""
    return await get_app_info()


@app.get("/api/v1/health")
async def api_health():
    """Health check da API v1."""
    return {"status": "healthy", "api_version": "v1"}


# Incluir routers da API
try:
    from app.api.v1.api import api_router
    app.include_router(api_router, prefix="/api/v1")
    logger.info("API v1 router incluído com sucesso")
    
    # Adicionar rotas de compatibilidade para o frontend
    # Isso permite que o frontend acesse as rotas em /api/auth/ em vez de /api/v1/auth/
    from app.api.v1.endpoints import auth as auth_endpoints
    
    # Criar router de compatibilidade
    compat_router = APIRouter()
    compat_router.include_router(auth_endpoints.router, prefix="/auth", tags=["auth-compat"])
    app.include_router(compat_router, prefix="/api")
    logger.info("Rotas de compatibilidade adicionadas com sucesso")
except ImportError as e:
    logger.warning(f"API v1 router não encontrado: {e}")

# Configurar arquivos estáticos
static_dir = Path(__file__).parent / "static"
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")
    logger.info(f"Arquivos estáticos configurados em {static_dir}")

# Rota para o favicon
@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    """Serve o favicon."""
    favicon_path = static_dir / "favicon.ico"
    if favicon_path.exists():
        return FileResponse(favicon_path)
    logger.warning("Favicon não encontrado")


if __name__ == "__main__":
    import uvicorn
    cardio_app.start()
    uvicorn.run(app, host="0.0.0.0", port=8000)
