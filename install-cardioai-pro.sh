#!/bin/bash


set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
WHITE='\033[1;37m'
NC='\033[0m' # No Color

DEFAULT_DB_PASSWORD="cardioai_secure_$(date +%s)"
DEFAULT_JWT_SECRET="cardioai_jwt_secret_$(date +%s)"
DEFAULT_API_PORT="8000"
DEFAULT_WEB_PORT="3000"

show_banner() {
    clear
    echo -e "${CYAN}"
    echo "╔══════════════════════════════════════════════════════════════════════════════╗"
    echo "║                          ${WHITE}CardioAI Pro v1.0.0${CYAN}                              ║"
    echo "║                   ${WHITE}Sistema de Análise de ECG com IA${CYAN}                       ║"
    echo "║                                                                              ║"
    echo "║  ${GREEN}✓${CYAN} Análise automática de ECG com IA                                      ║"
    echo "║  ${GREEN}✓${CYAN} Compliance médico ANVISA/FDA                                          ║"
    echo "║  ${GREEN}✓${CYAN} Interface web responsiva                                              ║"
    echo "║  ${GREEN}✓${CYAN} API REST completa                                                     ║"
    echo "║  ${GREEN}✓${CYAN} Segurança LGPD/HIPAA                                                  ║"
    echo "╚══════════════════════════════════════════════════════════════════════════════╝"
    echo -e "${NC}"
}

show_progress() {
    local current=$1
    local total=$2
    local description=$3
    local percent=$((current * 100 / total))
    local filled=$((percent / 5))
    local empty=$((20 - filled))
    
    printf "\r${BLUE}[${GREEN}"
    printf "%*s" $filled | tr ' ' '█'
    printf "${WHITE}"
    printf "%*s" $empty | tr ' ' '░'
    printf "${BLUE}] ${percent}%% - ${WHITE}${description}${NC}"
}

check_prerequisites() {
    echo -e "\n${YELLOW}🔍 Verificando pré-requisitos...${NC}\n"
    
    local errors=0
    
    if ! command -v docker &> /dev/null; then
        echo -e "${RED}❌ Docker não encontrado${NC}"
        echo -e "${YELLOW}   Instale com: sudo apt install docker.io${NC}"
        errors=$((errors + 1))
    else
        echo -e "${GREEN}✓ Docker encontrado: $(docker --version | cut -d' ' -f3)${NC}"
    fi
    
    if ! command -v docker-compose &> /dev/null && ! docker compose version &> /dev/null; then
        echo -e "${RED}❌ Docker Compose não encontrado${NC}"
        echo -e "${YELLOW}   Instale com: sudo apt install docker-compose${NC}"
        errors=$((errors + 1))
    else
        if command -v docker-compose &> /dev/null; then
            echo -e "${GREEN}✓ Docker Compose encontrado: $(docker-compose --version | cut -d' ' -f3)${NC}"
        else
            echo -e "${GREEN}✓ Docker Compose encontrado: $(docker compose version | cut -d' ' -f4)${NC}"
        fi
    fi
    
    if ! docker ps &> /dev/null; then
        echo -e "${YELLOW}⚠️  Usuário não tem permissão para Docker${NC}"
        echo -e "${YELLOW}   Execute: sudo usermod -aG docker \$USER${NC}"
        echo -e "${YELLOW}   Depois faça logout/login${NC}"
    else
        echo -e "${GREEN}✓ Permissões Docker OK${NC}"
    fi
    
    if netstat -tuln 2>/dev/null | grep -q ":${DEFAULT_API_PORT} "; then
        echo -e "${YELLOW}⚠️  Porta ${DEFAULT_API_PORT} em uso${NC}"
    else
        echo -e "${GREEN}✓ Porta ${DEFAULT_API_PORT} disponível${NC}"
    fi
    
    if netstat -tuln 2>/dev/null | grep -q ":${DEFAULT_WEB_PORT} "; then
        echo -e "${YELLOW}⚠️  Porta ${DEFAULT_WEB_PORT} em uso${NC}"
    else
        echo -e "${GREEN}✓ Porta ${DEFAULT_WEB_PORT} disponível${NC}"
    fi
    
    local available_space=$(df . | awk 'NR==2 {print $4}')
    if [ "$available_space" -lt 2097152 ]; then # 2GB em KB
        echo -e "${YELLOW}⚠️  Pouco espaço em disco (recomendado: 2GB+)${NC}"
    else
        echo -e "${GREEN}✓ Espaço em disco suficiente${NC}"
    fi
    
    if [ $errors -gt 0 ]; then
        echo -e "\n${RED}❌ Corrija os erros acima antes de continuar${NC}"
        read -p "Pressione Enter para sair..."
        exit 1
    fi
    
    echo -e "\n${GREEN}✅ Todos os pré-requisitos atendidos!${NC}"
    sleep 2
}

configure_environment() {
    show_banner
    echo -e "${BLUE}⚙️  Configuração do Ambiente${NC}\n"
    
    echo -e "${WHITE}📊 Configuração do Banco de Dados PostgreSQL:${NC}"
    read -p "Senha do banco (Enter para padrão seguro): " db_password
    db_password=${db_password:-$DEFAULT_DB_PASSWORD}
    
    echo -e "\n${WHITE}🔐 Configuração de Segurança:${NC}"
    read -p "Chave secreta JWT (Enter para gerar automaticamente): " jwt_secret
    jwt_secret=${jwt_secret:-$DEFAULT_JWT_SECRET}
    
    echo -e "\n${WHITE}🌐 Configuração de Rede:${NC}"
    read -p "Porta da API (padrão: ${DEFAULT_API_PORT}): " api_port
    api_port=${api_port:-$DEFAULT_API_PORT}
    
    read -p "Porta do Frontend (padrão: ${DEFAULT_WEB_PORT}): " web_port
    web_port=${web_port:-$DEFAULT_WEB_PORT}
    
    echo -e "\n${WHITE}📧 Configuração de Email (opcional para notificações):${NC}"
    read -p "Servidor SMTP (Enter para pular): " smtp_host
    if [ ! -z "$smtp_host" ]; then
        read -p "Porta SMTP (587): " smtp_port
        smtp_port=${smtp_port:-587}
        read -p "Usuário SMTP: " smtp_user
        read -s -p "Senha SMTP: " smtp_password
        echo
    fi
    
    cat > .env << EOF

ENVIRONMENT=production
DEBUG=false

DATABASE_URL=postgresql+asyncpg://cardioai:${db_password}@postgres:5432/cardioai_pro
POSTGRES_USER=cardioai
POSTGRES_PASSWORD=${db_password}
POSTGRES_DB=cardioai_pro

SECRET_KEY=${jwt_secret}
JWT_SECRET_KEY=${jwt_secret}
JWT_ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=30

REDIS_URL=redis://:${db_password}@redis:6379/0

CELERY_BROKER_URL=redis://:${db_password}@redis:6379/0
CELERY_RESULT_BACKEND=redis://:${db_password}@redis:6379/0

API_PORT=${api_port}
WEB_PORT=${web_port}

ALLOWED_ORIGINS=http://localhost:${web_port},http://127.0.0.1:${web_port}

MAX_UPLOAD_SIZE=50MB
UPLOAD_PATH=/app/uploads

LOG_LEVEL=INFO
LOG_FORMAT=json

MEDICAL_COMPLIANCE_MODE=true
AUDIT_LOGGING=true
DATA_RETENTION_DAYS=2555

REACT_APP_API_URL=http://localhost:${api_port}
REACT_APP_WS_URL=ws://localhost:${api_port}

EOF

    if [ ! -z "$smtp_host" ]; then
        cat >> .env << EOF
SMTP_HOST=${smtp_host}
SMTP_PORT=${smtp_port}
SMTP_USER=${smtp_user}
SMTP_PASSWORD=${smtp_password}
SMTP_TLS=true
SMTP_SSL=false

EOF
    fi
    
    echo -e "\n${GREEN}✅ Arquivo .env criado com sucesso!${NC}"
    sleep 1
}

install_system() {
    show_banner
    echo -e "${BLUE}🚀 Instalando CardioAI Pro...${NC}\n"
    
    local steps=8
    local current=0
    
    current=$((current + 1))
    show_progress $current $steps "Baixando imagens Docker..."
    docker-compose pull --quiet
    echo
    
    current=$((current + 1))
    show_progress $current $steps "Construindo aplicação..."
    docker-compose build --quiet
    echo
    
    current=$((current + 1))
    show_progress $current $steps "Criando volumes de dados..."
    docker volume create cardioai_postgres_data 2>/dev/null || true
    docker volume create cardioai_redis_data 2>/dev/null || true
    echo
    
    current=$((current + 1))
    show_progress $current $steps "Iniciando banco de dados..."
    docker-compose up -d postgres redis
    sleep 5
    echo
    
    current=$((current + 1))
    show_progress $current $steps "Executando migrações do banco..."
    docker-compose run --rm api alembic upgrade head 2>/dev/null || true
    echo
    
    current=$((current + 1))
    show_progress $current $steps "Criando usuário administrador..."
    docker-compose run --rm api python -c "
from app.scripts.create_admin import create_admin_user
import asyncio
asyncio.run(create_admin_user())
" 2>/dev/null || true
    echo
    
    current=$((current + 1))
    show_progress $current $steps "Iniciando todos os serviços..."
    docker-compose up -d
    echo
    
    current=$((current + 1))
    show_progress $current $steps "Verificando saúde do sistema..."
    sleep 10
    
    echo -e "\n\n${YELLOW}⏳ Aguardando serviços ficarem prontos...${NC}"
    local max_attempts=30
    local attempt=0
    
    while [ $attempt -lt $max_attempts ]; do
        if curl -s http://localhost:${api_port}/health > /dev/null 2>&1; then
            break
        fi
        attempt=$((attempt + 1))
        echo -n "."
        sleep 2
    done
    echo
    
    if [ $attempt -eq $max_attempts ]; then
        echo -e "${YELLOW}⚠️  Serviços ainda inicializando. Pode levar alguns minutos.${NC}"
    else
        echo -e "${GREEN}✅ Sistema iniciado com sucesso!${NC}"
    fi
}

verify_installation() {
    show_banner
    echo -e "${BLUE}🔍 Verificando Instalação...${NC}\n"
    
    local all_ok=true
    
    echo -e "${WHITE}📦 Status dos Containers:${NC}"
    if docker-compose ps | grep -q "Up"; then
        docker-compose ps | while read line; do
            if echo "$line" | grep -q "Up"; then
                service=$(echo "$line" | awk '{print $1}')
                echo -e "${GREEN}✓ $service${NC}"
            fi
        done
    else
        echo -e "${RED}❌ Containers não estão rodando${NC}"
        all_ok=false
    fi
    
    echo -e "\n${WHITE}🔌 Verificando API:${NC}"
    if curl -s http://localhost:${api_port}/health | grep -q "healthy"; then
        echo -e "${GREEN}✓ API respondendo em http://localhost:${api_port}${NC}"
    else
        echo -e "${RED}❌ API não está respondendo${NC}"
        all_ok=false
    fi
    
    echo -e "\n${WHITE}🌐 Verificando Frontend:${NC}"
    if curl -s http://localhost:${web_port} > /dev/null; then
        echo -e "${GREEN}✓ Frontend disponível em http://localhost:${web_port}${NC}"
    else
        echo -e "${RED}❌ Frontend não está disponível${NC}"
        all_ok=false
    fi
    
    echo -e "\n${WHITE}🗄️  Verificando Banco de Dados:${NC}"
    if docker-compose exec -T postgres pg_isready -U postgres > /dev/null 2>&1; then
        echo -e "${GREEN}✓ PostgreSQL conectado${NC}"
    else
        echo -e "${RED}❌ PostgreSQL não está conectado${NC}"
        all_ok=false
    fi
    
    echo -e "\n${WHITE}⚡ Verificando Cache:${NC}"
    if docker-compose exec -T redis redis-cli ping | grep -q "PONG"; then
        echo -e "${GREEN}✓ Redis conectado${NC}"
    else
        echo -e "${RED}❌ Redis não está conectado${NC}"
        all_ok=false
    fi
    
    if $all_ok; then
        echo -e "\n${GREEN}🎉 INSTALAÇÃO CONCLUÍDA COM SUCESSO! 🎉${NC}"
        echo -e "\n${WHITE}📋 Informações de Acesso:${NC}"
        echo -e "${CYAN}   • Frontend: ${WHITE}http://localhost:${web_port}${NC}"
        echo -e "${CYAN}   • API: ${WHITE}http://localhost:${api_port}${NC}"
        echo -e "${CYAN}   • Documentação: ${WHITE}http://localhost:${api_port}/docs${NC}"
        echo -e "${CYAN}   • Admin: ${WHITE}admin@cardioai.pro / admin123${NC}"
        
        echo -e "\n${WHITE}🔧 Comandos Úteis:${NC}"
        echo -e "${YELLOW}   • Ver logs: ${WHITE}docker-compose logs -f${NC}"
        echo -e "${YELLOW}   • Parar: ${WHITE}docker-compose down${NC}"
        echo -e "${YELLOW}   • Reiniciar: ${WHITE}docker-compose restart${NC}"
        echo -e "${YELLOW}   • Backup: ${WHITE}docker-compose exec postgres pg_dump -U postgres cardioai_pro > backup.sql${NC}"
    else
        echo -e "\n${RED}❌ Alguns serviços apresentaram problemas${NC}"
        echo -e "${YELLOW}Execute: docker-compose logs para ver detalhes${NC}"
    fi
}

maintenance_menu() {
    while true; do
        show_banner
        echo -e "${BLUE}🔧 Menu de Manutenção${NC}\n"
        echo -e "${WHITE}1.${NC} Ver status dos serviços"
        echo -e "${WHITE}2.${NC} Ver logs do sistema"
        echo -e "${WHITE}3.${NC} Reiniciar serviços"
        echo -e "${WHITE}4.${NC} Parar sistema"
        echo -e "${WHITE}5.${NC} Fazer backup do banco"
        echo -e "${WHITE}6.${NC} Atualizar sistema"
        echo -e "${WHITE}7.${NC} Limpar dados (CUIDADO!)"
        echo -e "${WHITE}0.${NC} Voltar ao menu principal"
        
        echo -e "\n${YELLOW}Escolha uma opção:${NC} "
        read choice
        
        case $choice in
            1)
                echo -e "\n${BLUE}📊 Status dos Serviços:${NC}"
                docker-compose ps
                read -p "Pressione Enter para continuar..."
                ;;
            2)
                echo -e "\n${BLUE}📋 Logs do Sistema:${NC}"
                echo "Pressione Ctrl+C para sair dos logs"
                sleep 2
                docker-compose logs -f
                ;;
            3)
                echo -e "\n${YELLOW}🔄 Reiniciando serviços...${NC}"
                docker-compose restart
                echo -e "${GREEN}✅ Serviços reiniciados!${NC}"
                sleep 2
                ;;
            4)
                echo -e "\n${YELLOW}⏹️  Parando sistema...${NC}"
                docker-compose down
                echo -e "${GREEN}✅ Sistema parado!${NC}"
                sleep 2
                ;;
            5)
                echo -e "\n${BLUE}💾 Fazendo backup...${NC}"
                backup_file="cardioai_backup_$(date +%Y%m%d_%H%M%S).sql"
                docker-compose exec -T postgres pg_dump -U postgres cardioai_pro > "$backup_file"
                echo -e "${GREEN}✅ Backup salvo em: $backup_file${NC}"
                read -p "Pressione Enter para continuar..."
                ;;
            6)
                echo -e "\n${BLUE}🔄 Atualizando sistema...${NC}"
                docker-compose pull
                docker-compose up -d --build
                echo -e "${GREEN}✅ Sistema atualizado!${NC}"
                sleep 2
                ;;
            7)
                echo -e "\n${RED}⚠️  ATENÇÃO: Isso apagará TODOS os dados!${NC}"
                read -p "Digite 'CONFIRMAR' para continuar: " confirm
                if [ "$confirm" = "CONFIRMAR" ]; then
                    docker-compose down -v
                    docker system prune -f
                    echo -e "${GREEN}✅ Dados limpos!${NC}"
                else
                    echo -e "${YELLOW}Operação cancelada${NC}"
                fi
                sleep 2
                ;;
            0)
                break
                ;;
            *)
                echo -e "${RED}Opção inválida!${NC}"
                sleep 1
                ;;
        esac
    done
}

main_menu() {
    while true; do
        show_banner
        echo -e "${WHITE}Selecione uma opção:${NC}\n"
        echo -e "${GREEN}1.${NC} 🚀 Instalação Completa (Recomendado)"
        echo -e "${BLUE}2.${NC} ⚙️  Apenas Configurar Ambiente"
        echo -e "${YELLOW}3.${NC} 🔍 Verificar Pré-requisitos"
        echo -e "${PURPLE}4.${NC} 📊 Verificar Instalação Existente"
        echo -e "${CYAN}5.${NC} 🔧 Menu de Manutenção"
        echo -e "${WHITE}6.${NC} 📖 Ajuda e Documentação"
        echo -e "${RED}0.${NC} ❌ Sair"
        
        echo -e "\n${YELLOW}Digite sua escolha [0-6]:${NC} "
        read choice
        
        case $choice in
            1)
                check_prerequisites
                configure_environment
                install_system
                verify_installation
                read -p "Pressione Enter para continuar..."
                ;;
            2)
                configure_environment
                read -p "Pressione Enter para continuar..."
                ;;
            3)
                check_prerequisites
                read -p "Pressione Enter para continuar..."
                ;;
            4)
                verify_installation
                read -p "Pressione Enter para continuar..."
                ;;
            5)
                maintenance_menu
                ;;
            6)
                show_help
                read -p "Pressione Enter para continuar..."
                ;;
            0)
                echo -e "\n${GREEN}Obrigado por usar CardioAI Pro! 🏥⚡${NC}"
                exit 0
                ;;
            *)
                echo -e "${RED}Opção inválida! Tente novamente.${NC}"
                sleep 1
                ;;
        esac
    done
}

show_help() {
    show_banner
    echo -e "${BLUE}📖 Ajuda e Documentação${NC}\n"
    
    echo -e "${WHITE}🎯 Sobre o CardioAI Pro:${NC}"
    echo -e "Sistema completo de análise de ECG com Inteligência Artificial"
    echo -e "Desenvolvido para compliance médico ANVISA/FDA/LGPD/HIPAA"
    echo
    
    echo -e "${WHITE}🔧 Requisitos do Sistema:${NC}"
    echo -e "• Docker 20.10+ e Docker Compose 2.0+"
    echo -e "• 4GB RAM mínimo (8GB recomendado)"
    echo -e "• 10GB espaço em disco"
    echo -e "• Portas 3000 e 8000 disponíveis"
    echo
    
    echo -e "${WHITE}🚀 Instalação Rápida:${NC}"
    echo -e "1. Execute este instalador"
    echo -e "2. Escolha 'Instalação Completa'"
    echo -e "3. Configure as opções básicas"
    echo -e "4. Aguarde a instalação"
    echo -e "5. Acesse http://localhost:3000"
    echo
    
    echo -e "${WHITE}🔐 Credenciais Padrão:${NC}"
    echo -e "• Usuário: admin@cardioai.pro"
    echo -e "• Senha: admin123"
    echo -e "• (Altere após primeiro login)"
    echo
    
    echo -e "${WHITE}📞 Suporte:${NC}"
    echo -e "• GitHub: https://github.com/drguilhermecapel/cardio.ai.pro"
    echo -e "• Documentação: http://localhost:8000/docs"
    echo -e "• Issues: https://github.com/drguilhermecapel/cardio.ai.pro/issues"
    echo
    
    echo -e "${WHITE}⚠️  Solução de Problemas:${NC}"
    echo -e "• Portas ocupadas: sudo netstat -tulpn | grep :3000"
    echo -e "• Logs de erro: docker-compose logs"
    echo -e "• Reiniciar: docker-compose restart"
    echo -e "• Limpar cache: docker system prune"
}

if [ ! -f "docker-compose.yml" ]; then
    echo -e "${RED}❌ Erro: docker-compose.yml não encontrado!${NC}"
    echo -e "${YELLOW}Execute este script no diretório do CardioAI Pro${NC}"
    exit 1
fi

if [ "$EUID" -eq 0 ]; then
    echo -e "${YELLOW}⚠️  Não execute como root (sudo)${NC}"
    echo -e "${YELLOW}Execute como usuário normal${NC}"
    exit 1
fi

main_menu
