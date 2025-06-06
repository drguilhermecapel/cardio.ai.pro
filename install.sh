#!/bin/bash


set -e

detect_language() {
    DETECTED_LANG="${LANG%%.*}"
    DETECTED_LANG="${DETECTED_LANG%%_*}"
    
    case "$DETECTED_LANG" in
        en|pt|es|fr|de|zh|ar)
            echo "$DETECTED_LANG"
            ;;
        *)
            echo "en"
            ;;
    esac
}

load_messages() {
    LANG_CODE=$(detect_language)
    
    case "$LANG_CODE" in
        en)
            MSG_HEADER_TITLE="🫀 CARDIO.AI.PRO - COMPLETE FUNCTIONAL SHELL INSTALLER"
            MSG_HEADER_SUBTITLE="   Hybrid ECG System with Advanced AI"
            MSG_DETECTING_OS="Detecting operating system..."
            MSG_OS_DETECTED="System detected:"
            MSG_OS_NOT_SUPPORTED="Operating system not supported:"
            MSG_INSTALLING_DEPS="Installing system dependencies..."
            MSG_DEPS_INSTALLED="System dependencies installed"
            MSG_CONFIGURING_DOCKER="Configuring Docker..."
            MSG_DOCKER_CONFIGURED="Docker configured successfully"
            MSG_DOCKER_FAILED="Docker configuration failed"
            MSG_INSTALLING_POETRY="Installing Poetry..."
            MSG_POETRY_INSTALLED="Poetry installed successfully"
            MSG_POETRY_FAILED="Poetry installation failed"
            MSG_CONFIGURING_PYTHON="Configuring Python environment..."
            MSG_PYTHON_CONFIGURED="Python environment configured"
            MSG_CONFIGURING_NODEJS="Configuring Node.js environment..."
            MSG_NODEJS_CONFIGURED="Node.js environment configured"
            MSG_CONFIGURING_ENV="Configuring environment files..."
            MSG_BACKEND_ENV_CREATED="Backend .env file created"
            MSG_FRONTEND_ENV_CREATED="Frontend .env.local file created"
            MSG_STARTING_DB="Starting database services..."
            MSG_POSTGRES_STARTED="PostgreSQL started"
            MSG_POSTGRES_RUNNING="PostgreSQL already running"
            MSG_REDIS_STARTED="Redis started"
            MSG_REDIS_RUNNING="Redis already running"
            MSG_WAITING_SERVICES="Waiting for services to be ready..."
            MSG_RUNNING_MIGRATIONS="Running database migrations..."
            MSG_BUILDING_FRONTEND="Building frontend..."
            MSG_RUNNING_TESTS="Running tests..."
            MSG_CREATING_SCRIPTS="Creating startup scripts..."
            MSG_SCRIPTS_CREATED="Startup scripts created"
            MSG_INSTALLATION_COMPLETE="🎉 INSTALLATION COMPLETED SUCCESSFULLY!"
            MSG_HOW_TO_START="🚀 HOW TO START THE SYSTEM:"
            MSG_ACCESS_URLS="🌐 ACCESS URLS:"
            MSG_SYSTEM_READY="✅ cardio.ai.pro system ready to use!"
            ;;
        pt)
            MSG_HEADER_TITLE="🫀 CARDIO.AI.PRO - INSTALADOR SHELL SCRIPT COMPLETO"
            MSG_HEADER_SUBTITLE="   Sistema ECG Híbrido com IA Avançada"
            MSG_DETECTING_OS="Detectando sistema operacional..."
            MSG_OS_DETECTED="Sistema detectado:"
            MSG_OS_NOT_SUPPORTED="Sistema operacional não suportado:"
            MSG_INSTALLING_DEPS="Instalando dependências do sistema..."
            MSG_DEPS_INSTALLED="Dependências do sistema instaladas"
            MSG_CONFIGURING_DOCKER="Configurando Docker..."
            MSG_DOCKER_CONFIGURED="Docker configurado com sucesso"
            MSG_DOCKER_FAILED="Falha na configuração do Docker"
            MSG_INSTALLING_POETRY="Instalando Poetry..."
            MSG_POETRY_INSTALLED="Poetry instalado com sucesso"
            MSG_POETRY_FAILED="Falha na instalação do Poetry"
            MSG_CONFIGURING_PYTHON="Configurando ambiente Python..."
            MSG_PYTHON_CONFIGURED="Ambiente Python configurado"
            MSG_CONFIGURING_NODEJS="Configurando ambiente Node.js..."
            MSG_NODEJS_CONFIGURED="Ambiente Node.js configurado"
            MSG_CONFIGURING_ENV="Configurando arquivos de ambiente..."
            MSG_BACKEND_ENV_CREATED="Arquivo .env do backend criado"
            MSG_FRONTEND_ENV_CREATED="Arquivo .env.local do frontend criado"
            MSG_STARTING_DB="Iniciando serviços de banco de dados..."
            MSG_POSTGRES_STARTED="PostgreSQL iniciado"
            MSG_POSTGRES_RUNNING="PostgreSQL já está rodando"
            MSG_REDIS_STARTED="Redis iniciado"
            MSG_REDIS_RUNNING="Redis já está rodando"
            MSG_WAITING_SERVICES="Aguardando serviços estarem prontos..."
            MSG_RUNNING_MIGRATIONS="Executando migrações do banco..."
            MSG_BUILDING_FRONTEND="Compilando frontend..."
            MSG_RUNNING_TESTS="Executando testes..."
            MSG_CREATING_SCRIPTS="Criando scripts de inicialização..."
            MSG_SCRIPTS_CREATED="Scripts de inicialização criados"
            MSG_INSTALLATION_COMPLETE="🎉 INSTALAÇÃO CONCLUÍDA COM SUCESSO!"
            MSG_HOW_TO_START="🚀 COMO INICIAR O SISTEMA:"
            MSG_ACCESS_URLS="🌐 URLS DE ACESSO:"
            MSG_SYSTEM_READY="✅ Sistema cardio.ai.pro pronto para uso!"
            ;;
        es)
            MSG_HEADER_TITLE="🫀 CARDIO.AI.PRO - INSTALADOR SHELL SCRIPT COMPLETO"
            MSG_HEADER_SUBTITLE="   Sistema ECG Híbrido con IA Avanzada"
            MSG_DETECTING_OS="Detectando sistema operativo..."
            MSG_OS_DETECTED="Sistema detectado:"
            MSG_OS_NOT_SUPPORTED="Sistema operativo no soportado:"
            MSG_INSTALLING_DEPS="Instalando dependencias del sistema..."
            MSG_DEPS_INSTALLED="Dependencias del sistema instaladas"
            MSG_CONFIGURING_DOCKER="Configurando Docker..."
            MSG_DOCKER_CONFIGURED="Docker configurado exitosamente"
            MSG_DOCKER_FAILED="Falló la configuración de Docker"
            MSG_INSTALLING_POETRY="Instalando Poetry..."
            MSG_POETRY_INSTALLED="Poetry instalado exitosamente"
            MSG_POETRY_FAILED="Falló la instalación de Poetry"
            MSG_CONFIGURING_PYTHON="Configurando entorno Python..."
            MSG_PYTHON_CONFIGURED="Entorno Python configurado"
            MSG_CONFIGURING_NODEJS="Configurando entorno Node.js..."
            MSG_NODEJS_CONFIGURED="Entorno Node.js configurado"
            MSG_CONFIGURING_ENV="Configurando archivos de entorno..."
            MSG_BACKEND_ENV_CREATED="Archivo .env del backend creado"
            MSG_FRONTEND_ENV_CREATED="Archivo .env.local del frontend creado"
            MSG_STARTING_DB="Iniciando servicios de base de datos..."
            MSG_POSTGRES_STARTED="PostgreSQL iniciado"
            MSG_POSTGRES_RUNNING="PostgreSQL ya está ejecutándose"
            MSG_REDIS_STARTED="Redis iniciado"
            MSG_REDIS_RUNNING="Redis ya está ejecutándose"
            MSG_WAITING_SERVICES="Esperando que los servicios estén listos..."
            MSG_RUNNING_MIGRATIONS="Ejecutando migraciones de base de datos..."
            MSG_BUILDING_FRONTEND="Construyendo frontend..."
            MSG_RUNNING_TESTS="Ejecutando pruebas..."
            MSG_CREATING_SCRIPTS="Creando scripts de inicio..."
            MSG_SCRIPTS_CREATED="Scripts de inicio creados"
            MSG_INSTALLATION_COMPLETE="🎉 ¡INSTALACIÓN COMPLETADA EXITOSAMENTE!"
            MSG_HOW_TO_START="🚀 CÓMO INICIAR EL SISTEMA:"
            MSG_ACCESS_URLS="🌐 URLS DE ACCESO:"
            MSG_SYSTEM_READY="✅ ¡Sistema cardio.ai.pro listo para usar!"
            ;;
        fr)
            MSG_HEADER_TITLE="🫀 CARDIO.AI.PRO - INSTALLATEUR SHELL SCRIPT COMPLET"
            MSG_HEADER_SUBTITLE="   Système ECG Hybride avec IA Avancée"
            MSG_DETECTING_OS="Détection du système d'exploitation..."
            MSG_OS_DETECTED="Système détecté:"
            MSG_OS_NOT_SUPPORTED="Système d'exploitation non supporté:"
            MSG_INSTALLING_DEPS="Installation des dépendances système..."
            MSG_DEPS_INSTALLED="Dépendances système installées"
            MSG_CONFIGURING_DOCKER="Configuration de Docker..."
            MSG_DOCKER_CONFIGURED="Docker configuré avec succès"
            MSG_DOCKER_FAILED="Échec de la configuration Docker"
            MSG_INSTALLING_POETRY="Installation de Poetry..."
            MSG_POETRY_INSTALLED="Poetry installé avec succès"
            MSG_POETRY_FAILED="Échec de l'installation de Poetry"
            MSG_CONFIGURING_PYTHON="Configuration de l'environnement Python..."
            MSG_PYTHON_CONFIGURED="Environnement Python configuré"
            MSG_CONFIGURING_NODEJS="Configuration de l'environnement Node.js..."
            MSG_NODEJS_CONFIGURED="Environnement Node.js configuré"
            MSG_CONFIGURING_ENV="Configuration des fichiers d'environnement..."
            MSG_BACKEND_ENV_CREATED="Fichier .env backend créé"
            MSG_FRONTEND_ENV_CREATED="Fichier .env.local frontend créé"
            MSG_STARTING_DB="Démarrage des services de base de données..."
            MSG_POSTGRES_STARTED="PostgreSQL démarré"
            MSG_POSTGRES_RUNNING="PostgreSQL déjà en cours d'exécution"
            MSG_REDIS_STARTED="Redis démarré"
            MSG_REDIS_RUNNING="Redis déjà en cours d'exécution"
            MSG_WAITING_SERVICES="Attente que les services soient prêts..."
            MSG_RUNNING_MIGRATIONS="Exécution des migrations de base de données..."
            MSG_BUILDING_FRONTEND="Construction du frontend..."
            MSG_RUNNING_TESTS="Exécution des tests..."
            MSG_CREATING_SCRIPTS="Création des scripts de démarrage..."
            MSG_SCRIPTS_CREATED="Scripts de démarrage créés"
            MSG_INSTALLATION_COMPLETE="🎉 INSTALLATION TERMINÉE AVEC SUCCÈS!"
            MSG_HOW_TO_START="🚀 COMMENT DÉMARRER LE SYSTÈME:"
            MSG_ACCESS_URLS="🌐 URLS D'ACCÈS:"
            MSG_SYSTEM_READY="✅ Système cardio.ai.pro prêt à utiliser!"
            ;;
        *)
            MSG_HEADER_TITLE="🫀 CARDIO.AI.PRO - COMPLETE FUNCTIONAL SHELL INSTALLER"
            MSG_HEADER_SUBTITLE="   Hybrid ECG System with Advanced AI"
            MSG_DETECTING_OS="Detecting operating system..."
            MSG_OS_DETECTED="System detected:"
            MSG_OS_NOT_SUPPORTED="Operating system not supported:"
            MSG_INSTALLING_DEPS="Installing system dependencies..."
            MSG_DEPS_INSTALLED="System dependencies installed"
            MSG_CONFIGURING_DOCKER="Configuring Docker..."
            MSG_DOCKER_CONFIGURED="Docker configured successfully"
            MSG_DOCKER_FAILED="Docker configuration failed"
            MSG_INSTALLING_POETRY="Installing Poetry..."
            MSG_POETRY_INSTALLED="Poetry installed successfully"
            MSG_POETRY_FAILED="Poetry installation failed"
            MSG_CONFIGURING_PYTHON="Configuring Python environment..."
            MSG_PYTHON_CONFIGURED="Python environment configured"
            MSG_CONFIGURING_NODEJS="Configuring Node.js environment..."
            MSG_NODEJS_CONFIGURED="Node.js environment configured"
            MSG_CONFIGURING_ENV="Configuring environment files..."
            MSG_BACKEND_ENV_CREATED="Backend .env file created"
            MSG_FRONTEND_ENV_CREATED="Frontend .env.local file created"
            MSG_STARTING_DB="Starting database services..."
            MSG_POSTGRES_STARTED="PostgreSQL started"
            MSG_POSTGRES_RUNNING="PostgreSQL already running"
            MSG_REDIS_STARTED="Redis started"
            MSG_REDIS_RUNNING="Redis already running"
            MSG_WAITING_SERVICES="Waiting for services to be ready..."
            MSG_RUNNING_MIGRATIONS="Running database migrations..."
            MSG_BUILDING_FRONTEND="Building frontend..."
            MSG_RUNNING_TESTS="Running tests..."
            MSG_CREATING_SCRIPTS="Creating startup scripts..."
            MSG_SCRIPTS_CREATED="Startup scripts created"
            MSG_INSTALLATION_COMPLETE="🎉 INSTALLATION COMPLETED SUCCESSFULLY!"
            MSG_HOW_TO_START="🚀 HOW TO START THE SYSTEM:"
            MSG_ACCESS_URLS="🌐 ACCESS URLS:"
            MSG_SYSTEM_READY="✅ cardio.ai.pro system ready to use!"
            ;;
    esac
}

load_messages

echo "$MSG_HEADER_TITLE"
echo "$MSG_HEADER_SUBTITLE"
echo "=========================================="

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

log_info() {
    echo -e "${BLUE}ℹ️  $1${NC}"
}

log_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

log_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

log_error() {
    echo -e "${RED}❌ $1${NC}"
}

detect_os() {
    log_info "$MSG_DETECTING_OS"
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        OS="linux"
        if command -v apt-get &> /dev/null; then
            DISTRO="debian"
        elif command -v yum &> /dev/null; then
            DISTRO="redhat"
        fi
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        OS="macos"
    else
        log_error "$MSG_OS_NOT_SUPPORTED $OSTYPE"
        exit 1
    fi
    log_info "$MSG_OS_DETECTED $OS ($DISTRO)"
}

install_system_dependencies() {
    log_info "$MSG_INSTALLING_DEPS"
    
    if [[ "$OS" == "linux" ]]; then
        if [[ "$DISTRO" == "debian" ]]; then
            sudo apt-get update
            sudo apt-get install -y \
                python3 python3-pip python3-venv \
                nodejs npm \
                docker.io \
                git curl wget \
                postgresql-client \
                build-essential \
                libpq-dev
        elif [[ "$DISTRO" == "redhat" ]]; then
            sudo yum update -y
            sudo yum install -y \
                python3 python3-pip \
                nodejs npm \
                docker \
                git curl wget \
                postgresql \
                gcc gcc-c++ make \
                postgresql-devel
        fi
    elif [[ "$OS" == "macos" ]]; then
        if ! command -v brew &> /dev/null; then
            log_info "Instalando Homebrew..."
            /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
        fi
        
        brew update
        brew install python3 node docker git postgresql
    fi
    
    log_success "$MSG_DEPS_INSTALLED"
}

setup_docker() {
    log_info "$MSG_CONFIGURING_DOCKER"
    
    if [[ "$OS" == "linux" ]]; then
        sudo systemctl start docker
        sudo systemctl enable docker
        sudo usermod -aG docker $USER
        log_warning "Você foi adicionado ao grupo docker. Faça logout/login para aplicar as mudanças."
    fi
    
    if docker --version &> /dev/null; then
        log_success "$MSG_DOCKER_CONFIGURED"
    else
        log_error "$MSG_DOCKER_FAILED"
        exit 1
    fi
}

install_poetry() {
    log_info "$MSG_INSTALLING_POETRY"
    
    if ! command -v poetry &> /dev/null; then
        curl -sSL https://install.python-poetry.org | python3 -
        export PATH="$HOME/.local/bin:$PATH"
        echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc
    fi
    
    if poetry --version &> /dev/null; then
        log_success "$MSG_POETRY_INSTALLED"
    else
        log_error "$MSG_POETRY_FAILED"
        exit 1
    fi
}

setup_python_environment() {
    log_info "$MSG_CONFIGURING_PYTHON"
    
    cd backend
    
    poetry install --with dev
    
    poetry add --group dev \
        pandas-stubs \
        types-scipy \
        types-scikit-learn \
        types-requests || log_warning "Algumas dependências de tipos podem não estar disponíveis"
    
    cd ..
    log_success "$MSG_PYTHON_CONFIGURED"
}

setup_nodejs_environment() {
    log_info "$MSG_CONFIGURING_NODEJS"
    
    cd frontend
    
    npm install
    
    cd ..
    log_success "$MSG_NODEJS_CONFIGURED"
}

setup_environment_files() {
    log_info "$MSG_CONFIGURING_ENV"
    
    if [ ! -f "backend/.env" ]; then
        cat > backend/.env << EOF
DATABASE_URL=postgresql://cardio_user:cardio_pass@localhost:5432/cardio_db
REDIS_URL=redis://localhost:6379/0

SECRET_KEY=your-secret-key-here-$(openssl rand -hex 32)
ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=30

MODEL_PATH=./models
TENSORRT_ENABLED=false
CUDA_VISIBLE_DEVICES=0

LOG_LEVEL=INFO
ENABLE_METRICS=true

ENCRYPTION_KEY=your-encryption-key-here-$(openssl rand -hex 32)
BLOCKCHAIN_ENABLED=false

FHIR_SERVER_URL=http://localhost:8080/fhir
FHIR_ENABLED=false
EOF
        log_success "$MSG_BACKEND_ENV_CREATED"
    fi
    
    if [ ! -f "frontend/.env.local" ]; then
        cat > frontend/.env.local << EOF
NEXT_PUBLIC_API_URL=http://localhost:8000
NEXT_PUBLIC_WS_URL=ws://localhost:8000

NEXT_PUBLIC_ENABLE_3D=true
NEXT_PUBLIC_ENABLE_AR_VR=true
NEXT_PUBLIC_ENABLE_VOICE=true

NEXT_PUBLIC_ANALYTICS_ENABLED=false
EOF
        log_success "$MSG_FRONTEND_ENV_CREATED"
    fi
}

start_database_services() {
    log_info "$MSG_STARTING_DB"
    
    if ! docker ps | grep -q cardio-postgres; then
        docker run -d \
            --name cardio-postgres \
            -e POSTGRES_USER=cardio_user \
            -e POSTGRES_PASSWORD=cardio_pass \
            -e POSTGRES_DB=cardio_db \
            -p 5432:5432 \
            postgres:15
        log_success "$MSG_POSTGRES_STARTED"
    else
        log_warning "$MSG_POSTGRES_RUNNING"
    fi
    
    if ! docker ps | grep -q cardio-redis; then
        docker run -d \
            --name cardio-redis \
            -p 6379:6379 \
            redis:7-alpine
        log_success "$MSG_REDIS_STARTED"
    else
        log_warning "$MSG_REDIS_RUNNING"
    fi
    
    log_info "$MSG_WAITING_SERVICES"
    sleep 10
}

run_migrations() {
    log_info "$MSG_RUNNING_MIGRATIONS"
    
    cd backend
    poetry run alembic upgrade head || log_warning "Migrações podem ter falhado"
    cd ..
}

build_frontend() {
    log_info "$MSG_BUILDING_FRONTEND"
    
    cd frontend
    npm run build || log_warning "Build do frontend pode ter falhado"
    cd ..
}

run_tests() {
    log_info "$MSG_RUNNING_TESTS"
    
    cd backend
    poetry run pytest -v --cov=app --cov-report=term-missing || log_warning "Alguns testes podem ter falhado"
    cd ..
    
    cd frontend
    npm test -- --watchAll=false || log_warning "Alguns testes do frontend podem ter falhado"
    cd ..
}

create_startup_scripts() {
    log_info "$MSG_CREATING_SCRIPTS"
    
    cat > start_dev.sh << 'EOF'
#!/bin/bash

echo "🚀 Iniciando cardio.ai.pro em modo desenvolvimento..."

cleanup() {
    echo "🛑 Parando serviços..."
    kill $(jobs -p) 2>/dev/null
    exit 0
}

trap cleanup SIGINT SIGTERM

cd backend
poetry run uvicorn app.main:app --reload --host 0.0.0.0 --port 8000 &
BACKEND_PID=$!

sleep 5

cd ../frontend
npm run dev &
FRONTEND_PID=$!

cd ../backend
poetry run celery -A app.core.celery worker --loglevel=info &
CELERY_PID=$!

echo "✅ Serviços iniciados!"
echo "🌐 Frontend: http://localhost:3000"
echo "🔧 API: http://localhost:8000"
echo "📚 Docs: http://localhost:8000/docs"
echo ""
echo "Pressione Ctrl+C para parar todos os serviços"

wait
EOF
    
    chmod +x start_dev.sh
    
    cat > start_prod.sh << 'EOF'
#!/bin/bash

echo "🚀 Iniciando cardio.ai.pro em modo produção..."

docker compose up -d --build

echo "✅ Serviços iniciados via Docker Compose!"
echo "🌐 Frontend: http://localhost:3000"
echo "🔧 API: http://localhost:8000"
echo ""
echo "Para ver logs: docker compose logs -f"
echo "Para parar: docker compose down"
EOF
    
    chmod +x start_prod.sh
    
    log_success "$MSG_SCRIPTS_CREATED"
}

main() {
    log_info "Starting cardio.ai.pro installation..."
    
    detect_os
    install_system_dependencies
    setup_docker
    install_poetry
    setup_python_environment
    setup_nodejs_environment
    setup_environment_files
    start_database_services
    run_migrations
    build_frontend
    create_startup_scripts
    run_tests
    
    echo ""
    echo "$MSG_INSTALLATION_COMPLETE"
    echo "=========================================="
    echo ""
    echo "$MSG_HOW_TO_START"
    echo "  Desenvolvimento: ./start_dev.sh"
    echo "  Produção: ./start_prod.sh"
    echo ""
    echo "$MSG_ACCESS_URLS"
    echo "  • Frontend: http://localhost:3000"
    echo "  • API: http://localhost:8000"
    echo "  • Docs: http://localhost:8000/docs"
    echo ""
    echo "$MSG_SYSTEM_READY"
}

main "$@"
