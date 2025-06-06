# 🫀 Cardio.AI.Pro - Guia de Instalação Completo

Sistema ECG Híbrido com IA Avançada - Instalação e Configuração

## 📋 Visão Geral

O cardio.ai.pro é um sistema completo de análise de ECG que utiliza inteligência artificial híbrida para detectar arritmias e outras condições cardíacas com precisão médica. Este guia fornece instruções detalhadas para instalação e configuração do sistema.

## 🎯 Características Principais

- **Análise ECG em Tempo Real**: Processamento de 12 derivações com latência <5ms
- **IA Híbrida**: Combina CNN, RNN, Transformers e arquitetura Mamba
- **Zero-Shot Learning**: Detecção de condições raras sem treinamento prévio
- **Edge Computing**: Otimizado para NVIDIA Jetson e dispositivos edge
- **Conformidade Regulatória**: FDA, CE, ANVISA, NMSA
- **Interface Futurística**: Dashboard 3D/AR/VR com visualizações holográficas
- **Segurança Avançada**: Criptografia quântica e blockchain

## 🔧 Requisitos do Sistema

### Requisitos Mínimos
- **SO**: Linux (Ubuntu 20.04+), macOS (10.15+), Windows 10/11
- **CPU**: Intel i5 ou AMD Ryzen 5 (4+ cores)
- **RAM**: 8GB (16GB recomendado)
- **Armazenamento**: 20GB livres
- **GPU**: NVIDIA GTX 1060+ (opcional, para aceleração)

### Requisitos de Software
- **Python**: 3.11+ 
- **Node.js**: 18.0+
- **Docker**: 20.10+
- **Docker Compose**: 2.0+
- **Git**: 2.30+

## 🚀 Instalação Rápida

### Opção 1: Instalador Python (Recomendado)

```bash
# Clone o repositório
git clone https://github.com/drguilhermecapel/cardio.ai.pro.git
cd cardio.ai.pro

# Execute o instalador
python3 install.py
```

### Opção 2: Instalador Shell Script

```bash
# Clone o repositório
git clone https://github.com/drguilhermecapel/cardio.ai.pro.git
cd cardio.ai.pro

# Torne o script executável e execute
chmod +x install.sh
./install.sh
```

### Opção 3: Docker Compose (Produção)

```bash
# Clone o repositório
git clone https://github.com/drguilhermecapel/cardio.ai.pro.git
cd cardio.ai.pro

# Inicie todos os serviços
docker-compose up -d --build
```

## 📦 Instalação Manual Detalhada

### 1. Preparação do Ambiente

```bash
# Atualize o sistema
sudo apt update && sudo apt upgrade -y

# Instale dependências básicas
sudo apt install -y python3 python3-pip python3-venv nodejs npm docker.io docker-compose git curl wget build-essential libpq-dev
```

### 2. Configuração do Docker

```bash
# Inicie e habilite Docker
sudo systemctl start docker
sudo systemctl enable docker

# Adicione usuário ao grupo docker
sudo usermod -aG docker $USER

# Faça logout/login para aplicar mudanças
```

### 3. Instalação do Poetry

```bash
# Instale Poetry
curl -sSL https://install.python-poetry.org | python3 -

# Adicione ao PATH
export PATH="$HOME/.local/bin:$PATH"
echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc
```

### 4. Configuração do Backend

```bash
cd backend

# Instale dependências Python
poetry install --with dev

# Instale dependências de tipos para MyPy
poetry add --group dev pandas-stubs types-numpy types-scipy types-scikit-learn types-requests

# Configure arquivo de ambiente
cp .env.example .env
# Edite .env com suas configurações
```

### 5. Configuração do Frontend

```bash
cd frontend

# Instale dependências Node.js
npm install

# Configure arquivo de ambiente
cp .env.local.example .env.local
# Edite .env.local com suas configurações
```

### 6. Configuração do Banco de Dados

```bash
# Inicie PostgreSQL
docker run -d \
  --name cardio-postgres \
  -e POSTGRES_USER=cardio_user \
  -e POSTGRES_PASSWORD=cardio_pass \
  -e POSTGRES_DB=cardio_db \
  -p 5432:5432 \
  postgres:15

# Inicie Redis
docker run -d \
  --name cardio-redis \
  -p 6379:6379 \
  redis:7-alpine

# Execute migrações
cd backend
poetry run alembic upgrade head
```

## 🏃‍♂️ Executando o Sistema

### Modo Desenvolvimento

```bash
# Opção 1: Script automático
./start_dev.sh

# Opção 2: Manual
# Terminal 1 - Backend
cd backend
poetry run uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# Terminal 2 - Frontend
cd frontend
npm run dev

# Terminal 3 - Celery Worker
cd backend
poetry run celery -A app.core.celery worker --loglevel=info
```

### Modo Produção

```bash
# Docker Compose
docker-compose up -d --build

# Ou script automático
./start_prod.sh
```

## 🌐 URLs de Acesso

Após a instalação, o sistema estará disponível em:

- **Frontend**: http://localhost:3000
- **API Backend**: http://localhost:8000
- **Documentação Swagger**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc
- **Monitoramento**: http://localhost:8000/health

## 🧪 Executando Testes

### Testes do Backend

```bash
cd backend

# Todos os testes
poetry run pytest

# Testes com cobertura
poetry run pytest --cov=app --cov-report=html

# Testes específicos
poetry run pytest tests/test_hybrid_ecg_service.py -v
```

### Testes do Frontend

```bash
cd frontend

# Todos os testes
npm test

# Testes em modo watch
npm run test:watch

# Testes de cobertura
npm run test:coverage
```

### Linting e Type Checking

```bash
cd backend

# Linting com Ruff
poetry run ruff check app/

# Type checking com MyPy
poetry run mypy app/

# Formatação com Black
poetry run black app/
```

## 🔧 Configuração Avançada

### Variáveis de Ambiente

#### Backend (.env)
```env
# Banco de Dados
DATABASE_URL=postgresql://cardio_user:cardio_pass@localhost:5432/cardio_db
REDIS_URL=redis://localhost:6379/0

# Segurança
SECRET_KEY=your-secret-key-here
ENCRYPTION_KEY=your-encryption-key-here

# Machine Learning
MODEL_PATH=./models
TENSORRT_ENABLED=true
CUDA_VISIBLE_DEVICES=0

# Features
BLOCKCHAIN_ENABLED=true
FHIR_ENABLED=true
EDGE_AI_ENABLED=true
```

#### Frontend (.env.local)
```env
# API
NEXT_PUBLIC_API_URL=http://localhost:8000
NEXT_PUBLIC_WS_URL=ws://localhost:8000

# Features
NEXT_PUBLIC_ENABLE_3D=true
NEXT_PUBLIC_ENABLE_AR_VR=true
NEXT_PUBLIC_ENABLE_VOICE=true
NEXT_PUBLIC_ENABLE_BLOCKCHAIN=true
```

### Configuração GPU/CUDA

```bash
# Instale NVIDIA Container Toolkit
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list

sudo apt-get update && sudo apt-get install -y nvidia-docker2
sudo systemctl restart docker

# Teste GPU no Docker
docker run --rm --gpus all nvidia/cuda:11.8-base-ubuntu20.04 nvidia-smi
```

## 🐳 Deployment com Kubernetes

```bash
# Aplique configurações Kubernetes
kubectl apply -f k8s/

# Verifique status
kubectl get pods -n cardio-ai

# Acesse logs
kubectl logs -f deployment/cardio-api -n cardio-ai
```

## 📊 Monitoramento e Logs

### Logs do Sistema

```bash
# Docker Compose
docker-compose logs -f

# Logs específicos
docker-compose logs -f api
docker-compose logs -f frontend
docker-compose logs -f celery

# Kubernetes
kubectl logs -f deployment/cardio-api -n cardio-ai
```

### Métricas de Performance

O sistema inclui métricas integradas acessíveis em:
- **Health Check**: http://localhost:8000/health
- **Métricas**: http://localhost:8000/metrics
- **Status**: http://localhost:8000/status

## 🔒 Segurança e Conformidade

### Certificados SSL

```bash
# Gere certificados para HTTPS
openssl req -x509 -newkey rsa:4096 -keyout key.pem -out cert.pem -days 365 -nodes
```

### Backup do Banco de Dados

```bash
# Backup
docker exec cardio-postgres pg_dump -U cardio_user cardio_db > backup.sql

# Restore
docker exec -i cardio-postgres psql -U cardio_user cardio_db < backup.sql
```

## 🆘 Solução de Problemas

### Problemas Comuns

#### 1. Erro de Conexão com Banco
```bash
# Verifique se PostgreSQL está rodando
docker ps | grep postgres

# Reinicie se necessário
docker restart cardio-postgres
```

#### 2. Erro de Dependências Python
```bash
# Limpe cache e reinstale
cd backend
poetry cache clear pypi --all
poetry install --with dev
```

#### 3. Erro de Build do Frontend
```bash
# Limpe cache e reinstale
cd frontend
rm -rf node_modules package-lock.json
npm install
npm run build
```

#### 4. Erro de Permissões Docker
```bash
# Adicione usuário ao grupo docker
sudo usermod -aG docker $USER
# Faça logout/login
```

### Logs de Debug

```bash
# Habilite logs detalhados
export LOG_LEVEL=DEBUG

# Backend com logs verbosos
cd backend
poetry run uvicorn app.main:app --log-level debug

# Frontend com logs de desenvolvimento
cd frontend
npm run dev -- --verbose
```

## 📞 Suporte

### Documentação
- **README Principal**: [README.md](README.md)
- **Documentação API**: http://localhost:8000/docs
- **Guias Técnicos**: [docs/](docs/)

### Contato
- **Issues**: https://github.com/drguilhermecapel/cardio.ai.pro/issues
- **Email**: drguilhermecapel@gmail.com
- **Discussões**: https://github.com/drguilhermecapel/cardio.ai.pro/discussions

### Contribuição
- **Guia de Contribuição**: [CONTRIBUTING.md](CONTRIBUTING.md)
- **Código de Conduta**: [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md)

## 📄 Licença

Este projeto está licenciado sob a Licença MIT. Veja [LICENSE](LICENSE) para detalhes.

---

**🫀 Cardio.AI.Pro** - Sistema ECG Híbrido com IA Avançada
*Desenvolvido com ❤️ para salvar vidas através da tecnologia*
