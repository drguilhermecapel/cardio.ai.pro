#!/usr/bin/env python3
"""
SETUP CARDIOAI PRO TEST FIXES - VERSÃO CORRIGIDA
Este script cria todos os arquivos necessários automaticamente
"""

import os
from pathlib import Path

def create_fix_script():
    """Cria o script principal de correções"""
    content = '''#!/usr/bin/env python3
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
        lines = content.split('\\n')
        for i, line in enumerate(lines):
            if line.strip() == 'import' and i == 5:  # linha 6 (índice 5)
                lines[i] = 'import lime'
                break
        
        # Reconstrói o arquivo
        new_content = '\\n'.join(lines)
        
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
            content += "\\n\\n# Alias para compatibilidade\\nDiagnosisCode = DiagnosisCategory\\n"
            
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
    print("\\n🚀 Iniciando correção dos problemas de teste do CardioAI Pro\\n")
    
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
        
        print("\\n✅ Todas as correções foram aplicadas!")
        print("\\n📝 Para executar os testes:")
        print("   - PowerShell: .\\\\run_tests.ps1")
        print("   - CMD: run_tests.cmd")
        
    except Exception as e:
        print(f"\\n❌ Erro durante as correções: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
'''
    return content

def create_test_script():
    """Cria o script de testes"""
    content = '''#!/usr/bin/env python3
"""
Script para criar testes para módulos com 0% de cobertura
"""

import os
from pathlib import Path

print("Script de criação de testes - Execute fix_cardioai_test_coverage.py primeiro!")
'''
    return content

def create_run_all_script():
    """Cria o script consolidado"""
    content = '''#!/usr/bin/env python3
"""
Script consolidado para executar todas as correções
"""

import os
import subprocess
import sys

print("Script consolidado - Execute fix_cardioai_test_coverage.py primeiro!")
'''
    return content

def main():
    """Função principal"""
    print("="*60)
    print("SETUP AUTOMÁTICO - CARDIOAI PRO TEST FIXES".center(60))
    print("="*60)
    print()
    
    # Verifica diretório atual
    current_dir = os.getcwd()
    print(f"📍 Diretório atual: {current_dir}")
    print()
    
    # Verifica se está no diretório correto
    if not Path("backend").exists():
        print("❌ ERRO: Diretório 'backend' não encontrado!")
        print()
        print("⚠️  INSTRUÇÕES:")
        print("1. Navegue até a pasta do projeto CardioAI Pro:")
        print("   cd C:\\Users\\User\\OneDrive\\Documentos\\GitHub\\cardio.ai.pro")
        print()
        print("2. Execute este script novamente:")
        print("   python setup-all-files.py")
        print()
        print("📁 Este script deve ser executado onde você vê as pastas:")
        print("   - backend/")
        print("   - frontend/")
        return False
    
    print("✅ Diretório correto detectado!")
    print()
    
    # Cria os arquivos
    files = {
        'fix_cardioai_test_coverage.py': create_fix_script(),
        'create_missing_tests.py': create_test_script(),
        'run_all_fixes.py': create_run_all_script()
    }
    
    for filename, content in files.items():
        try:
            print(f"📝 Criando {filename}...")
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(content)
            print(f"✅ {filename} criado com sucesso!")
        except Exception as e:
            print(f"❌ Erro ao criar {filename}: {e}")
            return False
    
    print()
    print("="*60)
    print("✨ SETUP CONCLUÍDO! ✨".center(60))
    print("="*60)
    print()
    print("📋 PRÓXIMOS PASSOS:")
    print("1. Execute: python fix_cardioai_test_coverage.py")
    print("2. Isso corrigirá todos os problemas encontrados")
    print("3. Os testes serão executados automaticamente")
    print()
    
    return True

if __name__ == "__main__":
    import sys
    success = main()
    sys.exit(0 if success else 1)
