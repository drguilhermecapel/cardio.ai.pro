#!/bin/bash

echo "🚀 Executando Coverage Boost..."

cd backend

poetry install

poetry run pytest tests/test_coverage_boost_mega.py tests/test_boost_*.py \
    --cov=app \
    --cov-report=term-missing \
    --cov-report=html \
    -v

echo ""
echo "📊 Resumo da Cobertura:"
poetry run coverage report | grep TOTAL

echo ""
echo "🎯 Para visualizar relatório detalhado:"
echo "   → Abra backend/htmlcov/index.html no navegador"
