#!/usr/bin/env python3
import subprocess
import threading
import time
import os
from pathlib import Path

def start_backend():
    """Inicia o backend FastAPI"""
    backend_dir = Path(__file__).parent / "backend"
    os.chdir(backend_dir)
    subprocess.run(["poetry", "run", "uvicorn", "app.main:app", "--reload", "--host", "0.0.0.0", "--port", "8000"])

def start_frontend():
    """Inicia o frontend Next.js"""
    frontend_dir = Path(__file__).parent / "frontend"
    os.chdir(frontend_dir)
    subprocess.run(["npm", "run", "dev"])

def start_celery():
    """Inicia o Celery worker"""
    backend_dir = Path(__file__).parent / "backend"
    os.chdir(backend_dir)
    subprocess.run(["poetry", "run", "celery", "-A", "app.core.celery", "worker", "--loglevel=info"])

if __name__ == "__main__":
    print("🚀 Iniciando cardio.ai.pro em modo desenvolvimento...")
    
    backend_thread = threading.Thread(target=start_backend, daemon=True)
    frontend_thread = threading.Thread(target=start_frontend, daemon=True)
    celery_thread = threading.Thread(target=start_celery, daemon=True)
    
    backend_thread.start()
    time.sleep(3)
    frontend_thread.start()
    time.sleep(2)
    celery_thread.start()
    
    print("✅ Serviços iniciados!")
    print("🌐 Frontend: http://localhost:3000")
    print("🔧 API: http://localhost:8000")
    print("📚 Docs: http://localhost:8000/docs")
    
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n🛑 Parando serviços...")
