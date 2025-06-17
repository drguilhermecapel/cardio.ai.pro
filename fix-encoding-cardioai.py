#!/usr/bin/env python3
"""
Script corrigido para resolver problemas de teste do CardioAI Pro.
Corrige o erro de encoding ao criar arquivos no Windows.
"""

import os
import sys
import subprocess
from pathlib import Path

def main():
    print("=" * 70)
    print("CARDIOAI PRO - CORRECAO DE TESTES (v2)".center(70))
    print("=" * 70)
    
    # Verificar diretório
    if not Path("backend").exists():
        print("\nERRO: Execute este script no diretorio raiz do CardioAI Pro!")
        print(f"Diretorio atual: {os.getcwd()}")
        return
        
    print("\nDiretorio correto detectado!")
    
    # Completar as correções que faltaram
    print("\nCompletando correcoes...")
    
    # Criar scripts de execução
    create_run_scripts()
    
    print("\n" + "=" * 70)
    print("CORRECOES CONCLUIDAS COM SUCESSO!".center(70))
    print("=" * 70)
    
    print("\nPROXIMOS PASSOS:")
    print("\n1. Execute os testes:")
    print("   > cd backend")
    print("   > run_tests.bat")
    print("\n2. Ou use PowerShell:")
    print("   > cd backend")
    print("   > .\\run_tests.ps1")
    print("\n3. Para ver a cobertura HTML:")
    print("   > Abra backend\\htmlcov\\index.html")

def create_run_scripts():
    """Cria scripts de execução sem emojis."""
    print("\nCriando scripts de execucao...")
    
    backend_dir = Path("backend")
    
    # Script Python seguro
    run_script = backend_dir / "run_tests_safe.py"
    
    script_content = '''#!/usr/bin/env python3
"""Script seguro para executar testes do CardioAI Pro."""

import subprocess
import sys
import os

def run_tests():
    """Executa testes com configuracao segura."""
    os.environ["ENVIRONMENT"] = "test"
    os.environ["PYTHONPATH"] = str(os.getcwd())
    
    print("Executando testes do CardioAI Pro...")
    print("=" * 60)
    
    # Comandos de teste em ordem de prioridade
    commands = [
        # Testes sem pyedflib primeiro
        ["pytest", "tests/test_exceptions_coverage.py", "-v"],
        ["pytest", "tests/test_config_coverage.py", "-v"],
        
        # Testes principais com coverage
        ["pytest", "--cov=app", "--cov-report=term-missing", "-v", "--tb=short"],
        
        # Relatorio final
        ["coverage", "report"],
        ["coverage", "html"],
    ]
    
    for cmd in commands:
        print(f"\\nExecutando: {' '.join(cmd)}")
        try:
            result = subprocess.run(cmd, capture_output=False, text=True)
            if result.returncode != 0:
                print(f"Aviso: Comando falhou com codigo {result.returncode}")
        except Exception as e:
            print(f"Erro ao executar comando: {e}")
    
    print("\\nTestes concluidos!")
    print("Relatorio HTML disponivel em: htmlcov/index.html")

if __name__ == "__main__":
    run_tests()
'''
    
    with open(run_script, 'w', encoding='utf-8') as f:
        f.write(script_content)
        
    # Script batch para Windows (sem emojis)
    batch_script = backend_dir / "run_tests.bat"
    
    batch_content = '''@echo off
echo ============================================================
echo Executando testes do CardioAI Pro...
echo ============================================================

set ENVIRONMENT=test
set PYTHONPATH=%CD%

REM Testes basicos primeiro
python -m pytest tests/test_exceptions_coverage.py -v
python -m pytest tests/test_config_coverage.py -v

REM Testes completos com cobertura
python -m pytest --cov=app --cov-report=term-missing -v --tb=short

REM Relatorios
coverage report
coverage html

echo.
echo Testes concluidos!
echo Relatorio HTML disponivel em: htmlcov\\index.html
pause
'''
    
    with open(batch_script, 'w', encoding='utf-8') as f:
        f.write(batch_content)
        
    # Script PowerShell
    ps_script = backend_dir / "run_tests.ps1"
    
    ps_content = '''# Script PowerShell para executar testes
$env:ENVIRONMENT = "test"
$env:PYTHONPATH = $PWD

Write-Host "============================================================"
Write-Host "Executando testes do CardioAI Pro..."
Write-Host "============================================================"

# Testes basicos
python -m pytest tests/test_exceptions