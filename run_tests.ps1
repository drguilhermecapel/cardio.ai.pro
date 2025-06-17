# Script para executar pytest com cobertura completa
$env:PYTHONPATH = "$PWD"
python -m pytest `
    --cov=app `
    --cov-branch `
    --cov-report=term-missing `
    --cov-report=html `
    --cov-report=xml `
    --cov-fail-under=80 `
    -v `
    --tb=short `
    --maxfail=5
