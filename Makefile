.PHONY: help build up down logs shell test lint format clean migrate seed

# Default target
help:
	@echo "CardioAI Pro - Available commands:"
	@echo "  build     - Build all Docker images"
	@echo "  up        - Start all services"
	@echo "  down      - Stop all services"
	@echo "  logs      - Show logs for all services"
	@echo "  shell     - Open shell in API container"
	@echo "  test      - Run all tests"
	@echo "  lint      - Run linting and type checking"
	@echo "  format    - Format code"
	@echo "  clean     - Clean up containers and volumes"
	@echo "  migrate   - Run database migrations"
	@echo "  seed      - Seed database with sample data"

# Docker operations
build:
	docker-compose build

up:
	docker-compose up -d

down:
	docker-compose down

logs:
	docker-compose logs -f

shell:
	docker-compose exec api bash

# Development operations
test:
	docker-compose exec api pytest backend/tests/ --cov=app --cov-report=html
	docker-compose exec frontend npm test -- --coverage --watchAll=false

lint:
	docker-compose exec api ruff check backend/app/
	docker-compose exec api mypy backend/app/ --strict
	docker-compose exec frontend npm run lint

format:
	docker-compose exec api ruff format backend/app/
	docker-compose exec frontend npm run format

# Database operations
migrate:
	docker-compose exec api alembic upgrade head

seed:
	docker-compose exec api python scripts/setup/seed_database.py

# Cleanup
clean:
	docker-compose down -v
	docker system prune -f

# Setup operations
setup:
	@echo "Setting up CardioAI Pro development environment..."
	cp .env.example .env
	@echo "Please edit .env file with your configuration"
	@echo "Then run: make build && make up && make migrate"

# Production operations
prod-build:
	docker-compose -f docker-compose.prod.yml build

prod-up:
	docker-compose -f docker-compose.prod.yml up -d

# Health checks
health:
	@echo "Checking service health..."
	@curl -f http://localhost:8000/health || echo "API: DOWN"
	@curl -f http://localhost:3000 || echo "Frontend: DOWN"
