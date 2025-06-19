#!/bin/bash

# Script de inicialização da plataforma de treinamento de IA
# Este script configura e inicia os serviços necessários

set -e

echo "=== Inicializando Plataforma de Treinamento CardioAI Pro ==="

# Verificar se estamos no diretório correto
if [ ! -f "backend/training/requirements.txt" ]; then
    echo "Erro: Execute este script a partir do diretório raiz do projeto"
    exit 1
fi

# Função para verificar se um comando existe
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Verificar dependências do sistema
echo "Verificando dependências do sistema..."

if ! command_exists python3; then
    echo "Erro: Python 3 não encontrado"
    exit 1
fi

if ! command_exists pip3; then
    echo "Erro: pip3 não encontrado"
    exit 1
fi

# Verificar versão do Python
PYTHON_VERSION=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
echo "Python version: $PYTHON_VERSION"

# Criar ambiente virtual se não existir
if [ ! -d "venv_training" ]; then
    echo "Criando ambiente virtual para treinamento..."
    python3 -m venv venv_training
fi

# Ativar ambiente virtual
echo "Ativando ambiente virtual..."
source venv_training/bin/activate

# Atualizar pip
echo "Atualizando pip..."
pip install --upgrade pip

# Instalar dependências principais
echo "Instalando dependências principais..."
pip install -r backend/requirements.txt

# Instalar dependências específicas de treinamento
echo "Instalando dependências de treinamento..."
pip install -r backend/training/requirements.txt

# Criar diretórios necessários
echo "Criando estrutura de diretórios..."
mkdir -p data/training
mkdir -p models/training
mkdir -p logs/training
mkdir -p checkpoints/training
mkdir -p exports/training

# Configurar variáveis de ambiente
echo "Configurando variáveis de ambiente..."
export TRAINING_ENABLED=true
export TRAINING_DATA_PATH="$(pwd)/data/training"
export TRAINING_MODELS_PATH="$(pwd)/models/training"
export PYTHONPATH="$(pwd):$PYTHONPATH"

# Verificar instalação do PyTorch
echo "Verificando instalação do PyTorch..."
python3 -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"

# Verificar outras dependências críticas
echo "Verificando dependências críticas..."
python3 -c "
import sys
try:
    import numpy
    import pandas
    import sklearn
    import scipy
    import matplotlib
    print('✓ Dependências científicas OK')
except ImportError as e:
    print(f'✗ Erro nas dependências científicas: {e}')
    sys.exit(1)

try:
    import fastapi
    import uvicorn
    import pydantic
    print('✓ Dependências de API OK')
except ImportError as e:
    print(f'✗ Erro nas dependências de API: {e}')
    sys.exit(1)
"

# Testar importação dos módulos de treinamento
echo "Testando módulos de treinamento..."
python3 -c "
import sys
sys.path.append('$(pwd)')
try:
    from backend.training.config.training_config import training_config
    from backend.training.models.model_factory import ModelFactory
    from backend.training.datasets.dataset_factory import DatasetFactory
    print('✓ Módulos de treinamento OK')
except ImportError as e:
    print(f'✗ Erro nos módulos de treinamento: {e}')
    sys.exit(1)
"

# Criar arquivo de configuração local se não existir
if [ ! -f ".env.training" ]; then
    echo "Criando arquivo de configuração local..."
    cat > .env.training << EOF
# Configurações da Plataforma de Treinamento
TRAINING_ENABLED=true
TRAINING_API_PORT=8001
TRAINING_DATA_PATH=$(pwd)/data/training
TRAINING_MODELS_PATH=$(pwd)/models/training
TRAINING_MAX_CONCURRENT_JOBS=2
TRAINING_GPU_ENABLED=auto
TRAINING_MAX_EPOCHS_DEFAULT=100
TRAINING_ALLOWED_ROLES=admin,researcher
TRAINING_REQUIRE_APPROVAL=false
TRAINING_NOTIFY_ON_COMPLETION=true
TRAINING_NOTIFY_ON_FAILURE=true

# Configurações de Logging
TRAINING_LOG_LEVEL=INFO
TRAINING_LOG_EVERY_N_STEPS=50
TRAINING_SAVE_TOP_K=3
TRAINING_TENSORBOARD=true

# Configurações de Dados
TRAINING_BATCH_SIZE=32
TRAINING_LEARNING_RATE=1e-4
TRAINING_EPOCHS=100
TRAINING_EARLY_STOPPING_PATIENCE=10
TRAINING_VAL_SPLIT=0.2
TRAINING_TEST_SPLIT=0.1
EOF
fi

echo ""
echo "=== Inicialização Concluída ==="
echo ""
echo "Para usar a plataforma de treinamento:"
echo "1. Ative o ambiente virtual: source venv_training/bin/activate"
echo "2. Configure as variáveis: source .env.training"
echo "3. Inicie a API: python backend/training/api.py"
echo "4. Ou execute treinamento: python backend/training/main.py --help"
echo ""
echo "Documentação completa: backend/training/README.md"
echo ""

# Verificar se deve iniciar automaticamente
if [ "$1" = "--start" ]; then
    echo "Iniciando API de treinamento..."
    source .env.training
    python backend/training/api.py
fi

