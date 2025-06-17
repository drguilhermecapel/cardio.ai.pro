@echo off
REM Script para executar pytest com cobertura completa
set PYTHONPATH=%CD%
python -m pytest ^
    --cov=app ^
    --cov-branch ^
    --cov-report=term-missing ^
    --cov-report=html ^
    --cov-report=xml ^
    --cov-fail-under=80 ^
    -v ^
    --tb=short ^
    --maxfail=5
