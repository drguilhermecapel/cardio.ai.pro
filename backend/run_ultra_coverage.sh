#!/bin/bash

echo "🚀 ULTRA COVERAGE BOOST 80% - CardioAI Pro"
echo "=========================================="

pip install pytest pytest-asyncio pytest-cov pytest-mock pytest-timeout black

mkdir -p tests

echo ""
echo "📊 Fase 1: NotificationService (16% → 80%)"
pytest tests/test_ultra_notification_service.py -v --cov=app.services.notification_service --cov-report=term

echo ""
echo "📊 Fase 2: Repositories (24% → 80%)"
pytest tests/test_ultra_repositories.py -v --cov=app.repositories --cov-report=term

echo ""
echo "📊 Fase 3: API Endpoints"
pytest tests/test_ultra_api_endpoints.py -v --cov=app.api --cov-report=term

echo ""
echo "📊 Fase 4: Async Services"
pytest tests/test_ultra_async_services.py -v --cov=app.services --cov-report=term

echo ""
echo "📊 RESULTADO FINAL:"
echo "=================="
pytest tests/test_ultra_*.py --cov=app --cov-report=term --cov-report=html

echo ""
echo "🎯 COBERTURA TOTAL:"
coverage report | grep TOTAL

echo ""
echo "📈 Relatório detalhado disponível em: htmlcov/index.html"

COVERAGE=$(coverage report | grep TOTAL | awk '{print $4}' | sed 's/%//')
if (( $(echo "$COVERAGE >= 80" | bc -l) )); then
    echo ""
    echo "✅ META ATINGIDA! Cobertura: ${COVERAGE}%"
    echo "🎉 Pronto para merge!"
else
    echo ""
    echo "❌ Ainda falta: $((80 - ${COVERAGE%.*}))% para 80%"
    echo "💡 Execute: python ultra_coverage_boost.py --extreme"
fi
