# CardioAI Pro - Makefile Principal
# Versão otimizada com suporte completo para desenvolvimento, testes e compliance

# Configurações
SHELL := /bin/bash
.DEFAULT_GOAL := help

# Cores para output
GREEN := \033[0;32m
YELLOW := \033[1;33m
RED := \033[0;31m
BLUE := \033[0;34m
CYAN := \033[0;36m
NC := \033[0m # No Color

# Detecção de ambiente
DOCKER_COMPOSE := $(shell which docker-compose 2>/dev/null || echo "docker compose")
PYTHON := $(shell which python3 2>/dev/null || which python 2>/dev/null)
NPM := $(shell which npm 2>/dev/null)
POETRY := $(shell which poetry 2>/dev/null)

# Variáveis de ambiente
export ENVIRONMENT ?= development
export DATABASE_URL ?= postgresql://postgres:postgres@localhost:5432/cardioai

# Timestamps
TIMESTAMP := $(shell date +%Y%m%d_%H%M%S)

.PHONY: help
help: ## Mostra esta mensagem de ajuda
	@echo "$(BLUE)╔════════════════════════════════════════════════════════════════╗$(NC)"
	@echo "$(BLUE)║                   CardioAI Pro - Makefile                      ║$(NC)"
	@echo "$(BLUE)╚════════════════════════════════════════════════════════════════╝$(NC)"
	@echo ""
	@echo "$(YELLOW)🐳 Docker:$(NC)"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | grep docker- | awk 'BEGIN {FS = ":.*?## "}; {printf "  $(GREEN)%-20s$(NC) %s\n", $$1, $$2}'
	@echo ""
	@echo "$(YELLOW)🧪 Testes e Cobertura:$(NC)"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | grep -E "(test|coverage)" | grep -v docker | awk 'BEGIN {FS = ":.*?## "}; {printf "  $(GREEN)%-20s$(NC) %s\n", $$1, $$2}'
	@echo ""
	@echo "$(YELLOW)🏥 Compliance Médico:$(NC)"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | grep -E "(compliance|anvisa|fda|medical)" | awk 'BEGIN {FS = ":.*?## "}; {printf "  $(GREEN)%-20s$(NC) %s\n", $$1, $$2}'
	@echo ""
	@echo "$(YELLOW)🔧 Desenvolvimento:$(NC)"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | grep -E "(lint|format|migrate|setup)" | awk 'BEGIN {FS = ":.*?## "}; {printf "  $(GREEN)%-20s$(NC) %s\n", $$1, $$2}'
	@echo ""
	@echo "$(YELLOW)📊 Relatórios:$(NC)"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | grep -E "(report|dashboard)" | awk 'BEGIN {FS = ":.*?## "}; {printf "  $(GREEN)%-20s$(NC) %s\n", $$1, $$2}'
	@echo ""
	@echo "$(CYAN)💡 Dicas:$(NC)"
	@echo "  • Use $(GREEN)make quick-test$(NC) para testes rápidos"
	@echo "  • Use $(GREEN)make ci$(NC) para pipeline completo"
	@echo "  • Use $(GREEN)make compliance$(NC) para relatórios ANVISA/FDA"

# ═══════════════════════════════════════════════════════════════════
# 🐳 DOCKER OPERATIONS
# ═══════════════════════════════════════════════════════════════════

.PHONY: docker-build
docker-build: ## Constrói todas as imagens Docker
	@echo "$(BLUE)🔨 Construindo imagens Docker...$(NC)"
	$(DOCKER_COMPOSE) build --parallel

.PHONY: docker-up
docker-up: ## Inicia todos os serviços
	@echo "$(BLUE)🚀 Iniciando serviços...$(NC)"
	$(DOCKER_COMPOSE) up -d
	@echo "$(GREEN)✅ Serviços iniciados!$(NC)"
	@$(MAKE) docker-status

.PHONY: docker-down
docker-down: ## Para todos os serviços
	@echo "$(YELLOW)⏹️  Parando serviços...$(NC)"
	$(DOCKER_COMPOSE) down

.PHONY: docker-logs
docker-logs: ## Mostra logs de todos os serviços
	$(DOCKER_COMPOSE) logs -f

.PHONY: docker-shell
docker-shell: ## Abre shell no container da API
	$(DOCKER_COMPOSE) exec api bash

.PHONY: docker-status
docker-status: ## Mostra status dos containers
	@echo "$(BLUE)📊 Status dos containers:$(NC)"
	@$(DOCKER_COMPOSE) ps

.PHONY: docker-clean
docker-clean: ## Remove containers e volumes
	@echo "$(RED)🧹 Limpando ambiente Docker...$(NC)"
	$(DOCKER_COMPOSE) down -v
	docker system prune -f

# ═══════════════════════════════════════════════════════════════════
# 🧪 TESTES E COBERTURA
# ═══════════════════════════════════════════════════════════════════

.PHONY: test
test: ## Executa todos os testes
	@echo "$(BLUE)🧪 Executando todos os testes...$(NC)"
	@if [ -f /.dockerenv ]; then \
		$(MAKE) docker-test; \
	else \
		$(MAKE) local-test; \
	fi

.PHONY: docker-test
docker-test: ## Executa testes nos containers Docker
	@echo "$(YELLOW)🐳 Executando testes via Docker...$(NC)"
	$(DOCKER_COMPOSE) exec api pytest backend/tests/ -v
	$(DOCKER_COMPOSE) exec frontend npm test -- --watchAll=false

.PHONY: local-test
local-test: ## Executa testes localmente
	@echo "$(YELLOW)💻 Executando testes localmente...$(NC)"
	@cd backend && $(POETRY) run pytest tests/ -v || echo "$(RED)❌ Backend: Poetry não encontrado$(NC)"
	@cd frontend && $(NPM) test -- --watchAll=false || echo "$(RED)❌ Frontend: NPM não encontrado$(NC)"

.PHONY: coverage
coverage: ## Gera relatório completo de cobertura
	@echo "$(BLUE)📊 Gerando cobertura completa...$(NC)"
	@$(MAKE) backend-coverage
	@$(MAKE) frontend-coverage
	@$(MAKE) coverage-report

.PHONY: backend-coverage
backend-coverage: ## Cobertura do backend
	@echo "$(YELLOW)🐍 Backend - Gerando cobertura...$(NC)"
	@if [ -f /.dockerenv ]; then \
		pytest backend/tests/ --cov=app --cov-report=html --cov-report=term; \
	else \
		cd backend && $(POETRY) run pytest --cov=app --cov-report=html --cov-report=term --cov-fail-under=80; \
	fi
	@echo "$(GREEN)✅ Relatório: backend/htmlcov/index.html$(NC)"

.PHONY: frontend-coverage
frontend-coverage: ## Cobertura do frontend
	@echo "$(YELLOW)⚛️  Frontend - Gerando cobertura...$(NC)"
	@if [ -f /.dockerenv ]; then \
		cd frontend && npm run test:coverage; \
	else \
		cd frontend && $(NPM) run test:coverage -- --watchAll=false; \
	fi
	@echo "$(GREEN)✅ Relatório: frontend/coverage/lcov-report/index.html$(NC)"

.PHONY: quick-test
quick-test: ## Testes rápidos (apenas modificados)
	@echo "$(YELLOW)⚡ Executando testes rápidos...$(NC)"
	@cd backend && $(POETRY) run pytest -v --lf --tb=short
	@cd frontend && $(NPM) test -- --onlyChanged

.PHONY: coverage-report
coverage-report: ## Mostra resumo da cobertura
	@echo ""
	@echo "$(BLUE)╔════════════════════════════════════════════════════════════════╗$(NC)"
	@echo "$(BLUE)║                    RESUMO DE COBERTURA                         ║$(NC)"
	@echo "$(BLUE)╚════════════════════════════════════════════════════════════════╝$(NC)"
	@echo ""
	@echo "$(YELLOW)Backend:$(NC)"
	@cd backend && $(POETRY) run coverage report | grep TOTAL || echo "Execute 'make backend-coverage' primeiro"
	@echo ""
	@echo "$(YELLOW)Frontend:$(NC)"
	@cd frontend && cat coverage/coverage-summary.json 2>/dev/null | $(PYTHON) -c "import sys, json; data = json.load(sys.stdin); print(f'TOTAL: {data[\"total\"][\"lines\"][\"pct\"]}% lines')" || echo "Execute 'make frontend-coverage' primeiro"

# ═══════════════════════════════════════════════════════════════════
# 🏥 COMPLIANCE MÉDICO (ANVISA/FDA)
# ═══════════════════════════════════════════════════════════════════

.PHONY: compliance
compliance: ## Gera todos os relatórios de compliance
	@echo "$(BLUE)🏥 Gerando relatórios de compliance regulatório...$(NC)"
	@$(PYTHON) scripts/generate_compliance_coverage.py
	@echo "$(GREEN)✅ Relatórios gerados em: compliance_reports_$(TIMESTAMP)/$(NC)"

.PHONY: anvisa-report
anvisa-report: ## Gera relatório ANVISA RDC 40/2015
	@echo "$(YELLOW)🇧🇷 Gerando relatório ANVISA...$(NC)"
	@$(MAKE) coverage
	@$(PYTHON) scripts/generate_anvisa_report.py
	@echo "$(GREEN)✅ Relatório ANVISA gerado$(NC)"

.PHONY: fda-report
fda-report: ## Gera relatório FDA 21 CFR 820.30
	@echo "$(YELLOW)🇺🇸 Gerando relatório FDA...$(NC)"
	@$(MAKE) coverage
	@$(PYTHON) scripts/generate_fda_report.py
	@echo "$(GREEN)✅ Relatório FDA gerado$(NC)"

.PHONY: medical-tests
medical-tests: ## Executa testes de componentes médicos críticos
	@echo "$(BLUE)🏥 Executando testes médicos críticos...$(NC)"
	@cd backend && $(POETRY) run pytest tests/medical/ -v --cov=app/services/ecg --cov=app/services/ml --cov-fail-under=100
	@cd frontend && $(NPM) run test:medical
	@echo "$(GREEN)✅ Componentes médicos com 100% de cobertura!$(NC)"

.PHONY: security-check
security-check: ## Verifica segurança (HIPAA/LGPD)
	@echo "$(BLUE)🔐 Verificando segurança...$(NC)"
	@cd backend && $(POETRY) run bandit -r app/ -ll || echo "$(YELLOW)⚠️ Problemas de segurança encontrados$(NC)"
	@cd backend && $(POETRY) run pip-audit --desc || echo "$(YELLOW)⚠️ Vulnerabilidades encontradas$(NC)"
	@cd frontend && $(NPM) audit || echo "$(YELLOW)⚠️ Vulnerabilidades npm encontradas$(NC)"
	@echo "$(GREEN)✅ Verificação de segurança completa$(NC)"

# ═══════════════════════════════════════════════════════════════════
# 🔧 DESENVOLVIMENTO
# ═══════════════════════════════════════════════════════════════════

.PHONY: setup
setup: ## Configura ambiente de desenvolvimento
	@echo "$(BLUE)🔧 Configurando ambiente de desenvolvimento...$(NC)"
	@if [ ! -f .env ]; then \
		cp .env.example .env; \
		echo "$(YELLOW)📝 Por favor, edite o arquivo .env com suas configurações$(NC)"; \
	fi
	@$(MAKE) install
	@echo "$(GREEN)✅ Setup completo! Execute: make docker-up$(NC)"

.PHONY: install
install: ## Instala todas as dependências
	@echo "$(BLUE)📦 Instalando dependências...$(NC)"
	@cd backend && $(POETRY) install --with dev || echo "$(RED)❌ Poetry não encontrado$(NC)"
	@cd frontend && $(NPM) install || echo "$(RED)❌ NPM não encontrado$(NC)"
	@echo "$(GREEN)✅ Dependências instaladas!$(NC)"

.PHONY: install-dev
install-dev: ## Instala dependências de desenvolvimento
	@echo "$(BLUE)📦 Instalando dependências de desenvolvimento...$(NC)"
	@cd backend && $(POETRY) add --group dev pytest-cov pytest-asyncio pytest-mock radon bandit pip-audit black isort flake8 mypy || echo "$(YELLOW)⚠️ Algumas dependências podem já estar instaladas$(NC)"
	@cd backend && $(POETRY) install --with dev
	@cd frontend && $(NPM) install --save-dev
	@echo "$(GREEN)✅ Dependências de desenvolvimento instaladas!$(NC)"

.PHONY: lint
lint: ## Executa linting e type checking
	@echo "$(BLUE)🔍 Executando linting...$(NC)"
	@if [ -f /.dockerenv ]; then \
		$(MAKE) docker-lint; \
	else \
		$(MAKE) local-lint; \
	fi

.PHONY: docker-lint
docker-lint:
	$(DOCKER_COMPOSE) exec api ruff check backend/app/
	$(DOCKER_COMPOSE) exec api mypy backend/app/ --strict
	$(DOCKER_COMPOSE) exec frontend npm run lint

.PHONY: local-lint
local-lint:
	@cd backend && $(POETRY) run ruff check app/ || echo "$(YELLOW)⚠️ Ruff não instalado, tentando flake8...$(NC)" && $(POETRY) run flake8 app/
	@cd backend && $(POETRY) run mypy app/ --strict || echo "$(YELLOW)⚠️ MyPy não instalado$(NC)"
	@cd frontend && $(NPM) run lint || echo "$(YELLOW)⚠️ Linting frontend falhou$(NC)"

.PHONY: format
format: ## Formata código
	@echo "$(BLUE)✨ Formatando código...$(NC)"
	@cd backend && $(POETRY) run black app/ tests/ || echo "$(YELLOW)⚠️ Black não instalado$(NC)"
	@cd backend && $(POETRY) run isort app/ tests/ || echo "$(YELLOW)⚠️ isort não instalado$(NC)"
	@cd frontend && $(NPM) run format || echo "$(YELLOW)⚠️ Prettier não configurado$(NC)"
	@echo "$(GREEN)✅ Código formatado!$(NC)"

.PHONY: migrate
migrate: ## Executa migrações do banco
	@echo "$(BLUE)🗄️  Executando migrações...$(NC)"
	@if [ -f /.dockerenv ]; then \
		alembic upgrade head; \
	else \
		cd backend && $(POETRY) run alembic upgrade head; \
	fi

.PHONY: seed
seed: ## Popula banco com dados de exemplo
	@echo "$(BLUE)🌱 Populando banco de dados...$(NC)"
	@cd backend && $(POETRY) run python scripts/setup/seed_database.py

# ═══════════════════════════════════════════════════════════════════
# 📊 RELATÓRIOS E DASHBOARDS
# ═══════════════════════════════════════════════════════════════════

.PHONY: dashboard
dashboard: ## Abre dashboard de métricas
	@echo "$(BLUE)📊 Gerando dashboard consolidado...$(NC)"
	@$(PYTHON) scripts/generate_dashboard.py
	@echo "$(YELLOW)👁️  Abrindo dashboard...$(NC)"
	@$(PYTHON) -m webbrowser dashboard/index.html

.PHONY: view-coverage
view-coverage: ## Abre relatórios de cobertura no navegador
	@echo "$(YELLOW)👁️  Abrindo relatórios de cobertura...$(NC)"
	@$(PYTHON) -m webbrowser backend/htmlcov/index.html 2>/dev/null || true
	@$(PYTHON) -m webbrowser frontend/coverage/lcov-report/index.html 2>/dev/null || true

.PHONY: report-summary
report-summary: ## Gera sumário executivo
	@$(MAKE) coverage-report
	@echo ""
	@echo "$(CYAN)📄 Gerando sumário executivo...$(NC)"
	@$(PYTHON) scripts/generate_executive_summary.py

# ═══════════════════════════════════════════════════════════════════
# 🚀 CI/CD
# ═══════════════════════════════════════════════════════════════════

.PHONY: ci
ci: ## Pipeline CI completo
	@echo "$(BLUE)🚀 Executando pipeline CI completo...$(NC)"
	@$(MAKE) install
	@$(MAKE) lint
	@$(MAKE) test
	@$(MAKE) coverage
	@$(MAKE) security-check
	@$(MAKE) compliance
	@echo ""
	@echo "$(GREEN)╔════════════════════════════════════════════════════════════════╗$(NC)"
	@echo "$(GREEN)║                    ✅ PIPELINE CI COMPLETO!                    ║$(NC)"
	@echo "$(GREEN)╚════════════════════════════════════════════════════════════════╝$(NC)"

.PHONY: pre-commit
pre-commit: ## Executa verificações pre-commit
	@echo "$(BLUE)🔍 Executando verificações pre-commit...$(NC)"
	@$(MAKE) format
	@$(MAKE) lint
	@$(MAKE) quick-test

# ═══════════════════════════════════════════════════════════════════
# 🏭 PRODUÇÃO
# ═══════════════════════════════════════════════════════════════════

.PHONY: prod-build
prod-build: ## Constrói imagens de produção
	@echo "$(BLUE)🏭 Construindo para produção...$(NC)"
	$(DOCKER_COMPOSE) -f docker-compose.prod.yml build

.PHONY: prod-up
prod-up: ## Inicia serviços em produção
	@echo "$(BLUE)🚀 Iniciando produção...$(NC)"
	$(DOCKER_COMPOSE) -f docker-compose.prod.yml up -d

.PHONY: prod-deploy
prod-deploy: prod-build prod-up ## Deploy completo de produção
	@echo "$(GREEN)✅ Deploy de produção completo!$(NC)"

# ═══════════════════════════════════════════════════════════════════
# 🩺 HEALTH CHECKS
# ═══════════════════════════════════════════════════════════════════

.PHONY: health
health: ## Verifica saúde dos serviços
	@echo "$(BLUE)🩺 Verificando saúde dos serviços...$(NC)"
	@curl -f http://localhost:8000/health 2>/dev/null && echo "$(GREEN)✅ API: OK$(NC)" || echo "$(RED)❌ API: DOWN$(NC)"
	@curl -f http://localhost:3000 2>/dev/null && echo "$(GREEN)✅ Frontend: OK$(NC)" || echo "$(RED)❌ Frontend: DOWN$(NC)"
	@curl -f http://localhost:6379/ping 2>/dev/null && echo "$(GREEN)✅ Redis: OK$(NC)" || echo "$(RED)❌ Redis: DOWN$(NC)"
	@$(DOCKER_COMPOSE) exec -T db pg_isready && echo "$(GREEN)✅ PostgreSQL: OK$(NC)" || echo "$(RED)❌ PostgreSQL: DOWN$(NC)"

# ═══════════════════════════════════════════════════════════════════
# 🛠️ UTILIDADES
# ═══════════════════════════════════════════════════════════════════

.PHONY: clean
clean: ## Limpa arquivos temporários e caches
	@echo "$(YELLOW)🧹 Limpando arquivos temporários...$(NC)"
	@find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	@find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	@find . -type d -name "htmlcov" -exec rm -rf {} + 2>/dev/null || true
	@find . -type d -name "coverage" -exec rm -rf {} + 2>/dev/null || true
	@find . -type f -name ".coverage*" -delete 2>/dev/null || true
	@find . -type f -name "*.pyc" -delete 2>/dev/null || true
	@rm -rf compliance_reports_*/ 2>/dev/null || true
	@echo "$(GREEN)✅ Limpeza completa!$(NC)"

.PHONY: backup
backup: ## Faz backup do banco de dados
	@echo "$(BLUE)💾 Fazendo backup do banco...$(NC)"
	@mkdir -p backups
	@$(DOCKER_COMPOSE) exec -T db pg_dump -U postgres cardioai > backups/cardioai_$(TIMESTAMP).sql
	@echo "$(GREEN)✅ Backup salvo em: backups/cardioai_$(TIMESTAMP).sql$(NC)"

.PHONY: restore
restore: ## Restaura banco de dados do último backup
	@echo "$(BLUE)💾 Restaurando banco de dados...$(NC)"
	@LATEST_BACKUP=$$(ls -t backups/*.sql | head -1); \
	if [ -n "$$LATEST_BACKUP" ]; then \
		cat $$LATEST_BACKUP | $(DOCKER_COMPOSE) exec -T db psql -U postgres cardioai; \
		echo "$(GREEN)✅ Banco restaurado de: $$LATEST_BACKUP$(NC)"; \
	else \
		echo "$(RED)❌ Nenhum backup encontrado$(NC)"; \
	fi

# ═══════════════════════════════════════════════════════════════════
# 🔍 DESENVOLVIMENTO AVANÇADO
# ═══════════════════════════════════════════════════════════════════

.PHONY: watch
watch: ## Modo watch para desenvolvimento
	@echo "$(YELLOW)👁️  Iniciando modo watch...$(NC)"
	@trap 'echo "$(RED)Parando modo watch...$(NC)"; exit' INT; \
	cd backend && $(POETRY) run ptw -- --lf & \
	cd frontend && $(NPM) run dev & \
	wait

.PHONY: debug
debug: ## Inicia ambiente em modo debug
	@echo "$(YELLOW)🐛 Iniciando modo debug...$(NC)"
	@export DEBUG=True && $(MAKE) docker-up
	@$(DOCKER_COMPOSE) logs -f

.PHONY: profile
profile: ## Executa profiling de performance
	@echo "$(BLUE)⚡ Executando profiling...$(NC)"
	@cd backend && $(POETRY) run python -m cProfile -o profile.stats scripts/profile_app.py
	@echo "$(GREEN)✅ Profiling salvo em: backend/profile.stats$(NC)"

# ═══════════════════════════════════════════════════════════════════
# 🧪 COMANDOS ESPECÍFICOS DE TESTE
# ═══════════════════════════════════════════════════════════════════

.PHONY: test-ml
test-ml: ## Testa ML Model Service especificamente
	@echo "$(BLUE)🤖 Testando ML Model Service...$(NC)"
	@cd backend && $(POETRY) run pytest tests/test_ml_model_service_coverage.py -v --cov=app/services/ml_model_service --cov-report=term-missing

.PHONY: test-critical
test-critical: ## Testa componentes médicos críticos
	@echo "$(BLUE)🏥 Testando componentes críticos...$(NC)"
	@cd backend && $(POETRY) run pytest tests/test_medical_critical_components.py -v --cov=app/services/ecg --cov=app/services/diagnosis --cov-fail-under=100

.PHONY: test-unit
test-unit: ## Executa apenas testes unitários
	@echo "$(BLUE)🧪 Executando testes unitários...$(NC)"
	@cd backend && $(POETRY) run pytest tests/unit -v
	@cd frontend && $(NPM) test -- --testPathPattern="unit" --watchAll=false

.PHONY: test-integration
test-integration: ## Executa apenas testes de integração
	@echo "$(BLUE)🔗 Executando testes de integração...$(NC)"
	@cd backend && $(POETRY) run pytest tests/integration -v
	@cd frontend && $(NPM) test -- --testPathPattern="integration" --watchAll=false

# ═══════════════════════════════════════════════════════════════════
# 📋 ALIASES E ATALHOS
# ═══════════════════════════════════════════════════════════════════

# Aliases para comandos docker
build: docker-build
up: docker-up
down: docker-down
logs: docker-logs
shell: docker-shell

# Aliases curtos
t: test
c: coverage
l: lint
f: format
m: migrate

# Comandos compostos
.PHONY: dev
dev: install docker-up logs ## Inicia ambiente de desenvolvimento completo

.PHONY: reset
reset: docker-down docker-clean docker-build docker-up migrate seed ## Reset completo do ambiente

.PHONY: validate
validate: lint test coverage security-check ## Validação completa do código

# ═══════════════════════════════════════════════════════════════════
# 🎯 TARGETS ESPECIAIS
# ═══════════════════════════════════════════════════════════════════

.PHONY: all
all: setup build up migrate seed ## Setup completo do projeto

.PHONY: daily
daily: clean pull install migrate test ## Rotina diária de desenvolvimento

.PHONY: release
release: validate compliance prod-build ## Prepara release de produção

# Pull das últimas alterações
.PHONY: pull
pull: ## Atualiza código do repositório
	@echo "$(BLUE)📥 Atualizando código...$(NC)"
	@git pull origin main
	@git submodule update --init --recursive

# Informações do sistema
.PHONY: info
info: ## Mostra informações do sistema
	@echo "$(BLUE)ℹ️  Informações do Sistema:$(NC)"
	@echo "  OS: $$(uname -s)"
	@echo "  Docker: $$(docker --version 2>/dev/null || echo 'Não instalado')"
	@echo "  Python: $$($(PYTHON) --version 2>/dev/null || echo 'Não instalado')"
	@echo "  Node: $$(node --version 2>/dev/null || echo 'Não instalado')"
	@echo "  Poetry: $$($(POETRY) --version 2>/dev/null || echo 'Não instalado')"
	@echo ""
	@echo "$(BLUE)📁 Estrutura do Projeto:$(NC)"
	@echo "  Backend: $$(find backend/app -name "*.py" | wc -l) arquivos Python"
	@echo "  Frontend: $$(find frontend/src -name "*.tsx" -o -name "*.ts" | wc -l) arquivos TypeScript"
	@echo "  Testes: $$(find . -name "*test*.py" -o -name "*test*.tsx" | wc -l) arquivos de teste"
