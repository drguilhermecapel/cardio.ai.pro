from fastapi import FastAPI
from contextlib import asynccontextmanager
from typing import Dict, Any

# Substituir @app.on_event deprecated
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    print("Starting up CardioAI...")
    yield
    # Shutdown
    print("Shutting down CardioAI...")

app = FastAPI(
    title="CardioAI",
    description="Sistema de Análise de ECG",
    version="1.0.0",
    lifespan=lifespan
)

# Adicionar funções faltantes
async def get_app_info() -> Dict[str, Any]:
    """Retorna informações da aplicação"""
    return {
        "name": "CardioAI",
        "version": "1.0.0",
        "description": "ECG Analysis System",
        "status": "running"
    }

async def health_check() -> Dict[str, str]:
    """Endpoint de health check"""
    return {"status": "healthy", "service": "CardioAI"}

class CardioAIApp:
    """Classe principal da aplicação CardioAI"""
    
    def __init__(self):
        self.name = "CardioAI"
        self.version = "1.0.0"
        self.description = "Sistema de Análise de ECG"
        self.status = "initialized"
    
    def get_info(self) -> Dict[str, str]:
        """Retorna informações da aplicação"""
        return {
            "name": self.name,
            "version": self.version,
            "description": self.description,
            "status": self.status
        }
    
    def start(self):
        """Inicia a aplicação"""
        self.status = "running"
        print(f"{self.name} v{self.version} iniciado com sucesso")
    
    def stop(self):
        """Para a aplicação"""
        self.status = "stopped"
        print(f"{self.name} parado")

# Endpoints da API
@app.get("/")
async def root():
    """Endpoint raiz"""
    return await get_app_info()

@app.get("/health")
async def health():
    """Endpoint de health check"""
    return await health_check()

@app.get("/info")
async def info():
    """Endpoint de informações da aplicação"""
    return await get_app_info()

# Instância global da aplicação
cardio_app = CardioAIApp()

if __name__ == "__main__":
    import uvicorn
    cardio_app.start()
    uvicorn.run(app, host="0.0.0.0", port=8000)

