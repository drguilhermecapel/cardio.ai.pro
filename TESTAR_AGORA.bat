@echo off
echo.
echo ============================================
echo INSTALANDO DEPENDENCIAS DE TESTE...
echo ============================================
pip install pytest pytest-cov pytest-asyncio aiosqlite

echo.
echo ============================================
echo EXECUTANDO TESTES...
echo ============================================
cd backend
set PYTHONPATH=%CD%
set ENVIRONMENT=test

pytest --cov=app --cov-report=term-missing -v --tb=short

echo.
echo ============================================
echo TESTES CONCLUIDOS!
echo ============================================
pause
