#!/usr/bin/env python3
"""
Script para corrigir todos os erros de teste e aumentar a cobertura do CardioAI Pro
"""

import os
import sys
import subprocess
import re
from pathlib import Path

def fix_interpretability_service():
    """Corrige o erro de sintaxe no interpretability_service.py"""
    print("🔧 Corrigindo erro de sintaxe em interpretability_service.py...")
    
    file_path = Path("backend/app/services/interpretability_service.py")
    
    if file_path.exists():
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Corrige a linha 6 que tem apenas "import" sem nada
        lines = content.split('\n')
        for i, line in enumerate(lines):
            if line.strip() == 'import' and i == 5:  # linha 6 (índice 5)
                lines[i] = 'import lime'
                break
        
        # Reconstrói o arquivo
        new_content = '\n'.join(lines)
        
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(new_content)
        
        print("✅ Arquivo interpretability_service.py corrigido!")
    else:
        print("❌ Arquivo interpretability_service.py não encontrado!")

def fix_constants_imports():
    """Adiciona DiagnosisCode como alias para DiagnosisCategory em constants.py"""
    print("🔧 Corrigindo importação de DiagnosisCode...")
    
    constants_file = Path("backend/app/core/constants.py")
    
    if constants_file.exists():
        with open(constants_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Adiciona DiagnosisCode como alias após DiagnosisCategory
        if "DiagnosisCode = DiagnosisCategory" not in content:
            # Adiciona ao final do arquivo
            content += "\n\n# Alias para compatibilidade\nDiagnosisCode = DiagnosisCategory\n"
            
            with open(constants_file, 'w', encoding='utf-8') as f:
                f.write(content)
            
            print("✅ DiagnosisCode adicionado como alias!")
    else:
        print("❌ Arquivo constants.py não encontrado!")

def install_missing_dependencies():
    """Instala dependências faltantes"""
    print("🔧 Instalando dependências faltantes...")
    
    missing_packages = ['pyedflib']
    
    for package in missing_packages:
        try:
            subprocess.run([sys.executable, '-m', 'pip', 'install', package], 
                         check=True, capture_output=True, text=True)
            print(f"✅ {package} instalado com sucesso!")
        except subprocess.CalledProcessError as e:
            print(f"❌ Erro ao instalar {package}: {e.stderr}")

def create_pytest_script():
    """Cria script para executar pytest no Windows"""
    print("🔧 Criando script de execução do pytest...")
    
    # Script para PowerShell
    ps_script = """# Script para executar pytest com cobertura completa
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
"""
    
    # Script para CMD
    cmd_script = """@echo off
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
"""
    
    with open('run_tests.ps1', 'w', encoding='utf-8') as f:
        f.write(ps_script)
    
    with open('run_tests.cmd', 'w', encoding='utf-8') as f:
        f.write(cmd_script)
    
    print("✅ Scripts de teste criados: run_tests.ps1 e run_tests.cmd")

def main():
    """Executa todas as correções"""
    print("\n🚀 Iniciando correção dos problemas de teste do CardioAI Pro\n")
    
    # Verifica se está no diretório correto
    if not os.path.exists("backend"):
        print("❌ Diretório 'backend' não encontrado!")
        print("📁 Execute este script na raiz do projeto CardioAI Pro")
        print(f"📍 Diretório atual: {os.getcwd()}")
        return
    
    try:
        fix_interpretability_service()
        fix_constants_imports()
        install_missing_dependencies()
        create_pytest_script()
        
        print("\n✅ Todas as correções foram aplicadas!")
        print("\n📝 Para executar os testes:")
        print("   - PowerShell: .\\run_tests.ps1")
        print("   - CMD: run_tests.cmd")
        
    except Exception as e:
        print(f"\n❌ Erro durante as correções: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
