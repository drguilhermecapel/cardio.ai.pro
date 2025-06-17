#!/usr/bin/env python3
"""
Script para completar as correções do CardioAI Pro.
Cria os scripts de execução que faltaram devido ao erro de encoding.
"""

import os
from pathlib import Path

def main():
    print("COMPLETANDO CORRECOES CARDIOAI PRO")
    print("=" * 50)
    
    backend_dir = Path("backend")
    
    if not backend_dir.exists():
        print("ERRO: Execute no diretorio raiz do projeto!")
        return
        
    # Criar script batch simples
    print("\nCriando script de execucao...")
    
    batch_content = """@echo off
echo ============================================================
echo CARDIOAI PRO - EXECUTANDO TESTES
echo ============================================================

cd backend
set ENVIRONMENT=test
set PYTHONPATH=%CD%

echo.
echo [1/4] Testando configuracao basica...
python -m pytest tests/test_exceptions_coverage.py -v --tb=short

echo.
echo [2/4] Testando configuracoes...
python -m pytest tests/test_config_coverage.py -v --tb=short

echo.
echo [3/4] Executando todos os testes com cobertura...
python -m pytest --cov=app --cov-report=term-missing --cov-report=html -v --tb=short --maxfail=10

echo.
echo [4/4] Gerando relatorio...
coverage report

echo.
echo ============================================================
echo TESTES CONCLUIDOS!
echo.
echo Relatorio de cobertura: backend\\htmlcov\\index.html
echo ============================================================
pause
"""
    
    # Salvar script
    script_path = Path("run_cardioai_tests.bat")
    with open(script_path, 'w', encoding='ascii') as f:
        f.write(batch_content)
        
    print(f"Script criado: {script_path}")
    
    # Criar script Python alternativo
    py_content = """import subprocess
import os

os.environ['ENVIRONMENT'] = 'test'
os.environ['PYTHONPATH'] = 'backend'

print('CARDIOAI PRO - TESTES')
print('=' * 50)

os.chdir('backend')

# Executar testes
commands = [
    ['python', '-m', 'pytest', 'tests/test_exceptions_coverage.py', '-v'],
    ['python', '-m', 'pytest', 'tests/test_config_coverage.py', '-v'],
    ['python', '-m', 'pytest', '--cov=app', '--cov-report=html', '-v'],
]

for cmd in commands:
    print(f"\\nExecutando: {' '.join(cmd)}")
    subprocess.run(cmd)

print('\\nTestes concluidos! Veja: backend/htmlcov/index.html')
"""
    
    py_path = Path("run_tests.py")
    with open(py_path, 'w', encoding='utf-8') as f:
        f.write(py_content)
        
    print(f"Script Python criado: {py_path}")
    
    # Verificar se os testes foram criados
    test_files = [
        backend_dir / "tests" / "test_exceptions_coverage.py",
        backend_dir / "tests" / "test_config_coverage.py",
        backend_dir / "tests" / "conftest.py"
    ]
    
    print("\nVerificando arquivos de teste:")
    for test_file in test_files:
        if test_file.exists():
            print(f"  OK: {test_file.name}")
        else:
            print(f"  FALTANDO: {test_file.name}")
            
    # Instruções finais
    print("\n" + "=" * 50)
    print("CORRECOES COMPLETADAS!")
    print("=" * 50)
    
    print("\nPARA EXECUTAR OS TESTES:")
    print("\nOpcao 1 - Use o arquivo batch:")
    print("  > run_cardioai_tests.bat")
    
    print("\nOpcao 2 - Use o script Python:")
    print("  > python run_tests.py")
    
    print("\nOpcao 3 - Execute manualmente:")
    print("  > cd backend")
    print("  > python -m pytest -v")
    
    print("\nDICA: Se houver erros de importacao:")
    print("  1. Verifique se esta no diretorio correto")
    print("  2. Confirme que as excecoes foram adicionadas")
    print("  3. Execute: pip install pytest pytest-cov")

if __name__ == "__main__":
    main()
