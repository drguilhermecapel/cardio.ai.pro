#!/usr/bin/env python3
"""
Script automático para aplicar todas as correções do CardioAI Pro
Este script executa todas as correções na ordem correta e verifica os resultados.
"""

import os
import sys
import subprocess
import json
import shutil
from pathlib import Path
from typing import Tuple, Optional

# Cores para output
class Colors:
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BLUE = '\033[94m'
    BOLD = '\033[1m'
    END = '\033[0m'

def print_header(text: str):
    """Imprime cabeçalho formatado."""
    print(f"\n{Colors.BOLD}{Colors.BLUE}{'=' * 60}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.BLUE}{text}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.BLUE}{'=' * 60}{Colors.END}")

def print_success(text: str):
    """Imprime mensagem de sucesso."""
    print(f"{Colors.GREEN}✅ {text}{Colors.END}")

def print_error(text: str):
    """Imprime mensagem de erro."""
    print(f"{Colors.RED}❌ {text}{Colors.END}")

def print_warning(text: str):
    """Imprime mensagem de aviso."""
    print(f"{Colors.YELLOW}⚠️  {text}{Colors.END}")

def print_info(text: str):
    """Imprime informação."""
    print(f"{Colors.BLUE}📌 {text}{Colors.END}")

def run_command(cmd: str, description: str) -> Tuple[bool, str, str]:
    """Executa um comando e retorna sucesso, stdout e stderr."""
    print(f"\n{Colors.BLUE}📌 {description}{Colors.END}")
    print(f"   Comando: {cmd}")
    
    try:
        result = subprocess.run(
            cmd,
            shell=True,
            capture_output=True,
            text=True
        )
        
        if result.returncode == 0:
            print_success(f"{description} concluído!")
            return True, result.stdout, result.stderr
        else:
            print_error(f"{description} falhou!")
            if result.stderr:
                print(f"   Erro: {result.stderr}")
            return False, result.stdout, result.stderr
    except Exception as e:
        print_error(f"Erro ao executar comando: {e}")
        return False, "", str(e)

def check_backend_directory() -> bool:
    """Verifica e navega para o diretório backend."""
    if Path("backend").exists():
        os.chdir("backend")
        return True
    elif Path("../backend").exists():
        os.chdir("../backend")
        return True
    else:
        print_error("Diretório 'backend' não encontrado!")
        print_warning("Execute este script na raiz do projeto ou dentro do diretório backend.")
        return False

def install_dependencies():
    """Instala dependências necessárias."""
    print_header("INSTALANDO DEPENDÊNCIAS")
    
    dependencies = [
        "pytest>=7.0.0",
        "pytest-asyncio>=0.21.0",
        "pytest-cov>=4.0.0",
        "pytest-mock>=3.10.0",
        "aiosqlite>=0.19.0",
        "httpx>=0.24.0",
        "numpy",
        "scipy"
    ]
    
    for dep in dependencies:
        success, _, _ = run_command(
            f"pip install {dep}",
            f"Instalando {dep}"
        )
        if not success:
            print_warning(f"Falha ao instalar {dep}, continuando...")

def clean_cache():
    """Limpa cache do pytest e __pycache__."""
    print_header("LIMPANDO CACHE")
    
    # Remover __pycache__
    for pycache in Path(".").rglob("__pycache__"):
        try:
            shutil.rmtree(pycache)
            print(f"   Removido: {pycache}")
        except:
            pass
    
    # Remover .pytest_cache
    for pytest_cache in Path(".").rglob(".pytest_cache"):
        try:
            shutil.rmtree(pytest_cache)
            print(f"   Removido: {pytest_cache}")
        except:
            pass
    
    print_success("Cache limpo!")

def run_fixes():
    """Executa todos os scripts de correção."""
    print_header("APLICANDO CORREÇÕES")
    
    # Lista de scripts para executar em ordem
    fix_scripts = [
        ("cardioai-fix-script.py", "Correções principais"),
        ("fix-validation-exception.py", "Correção ValidationException"),
        ("fix-additional-issues.py", "Correções adicionais")
    ]
    
    results = {}
    
    for script, description in fix_scripts:
        if Path(script).exists():
            success, stdout, stderr = run_command(
                f"python {script}",
                description
            )
            results[script] = success
        else:
            print_warning(f"Script {script} não encontrado!")
            print_info("Por favor, certifique-se de que todos os scripts de correção estão no diretório backend")
            results[script] = False
    
    return results

def run_tests() -> Tuple[bool, Optional[float]]:
    """Executa os testes e retorna sucesso e cobertura."""
    print_header("EXECUTANDO TESTES")
    
    # 1. Testes críticos
    print("\n🎯 Executando testes críticos...")
    critical_success, _, _ = run_command(
        "pytest tests/test_ecg_service_critical_coverage.py -v --tb=short",
        "Testes críticos"
    )
    
    # 2. Todos os testes com cobertura
    print("\n📊 Executando todos os testes com cobertura...")
    coverage_success, stdout, _ = run_command(
        "pytest --cov=app --cov-report=term-missing --cov-report=html --cov-report=json -q",
        "Testes com cobertura"
    )
    
    # Extrair porcentagem de cobertura
    coverage_percent = None
    coverage_file = Path("coverage.json")
    
    if coverage_file.exists():
        try:
            with open(coverage_file, 'r') as f:
                coverage_data = json.load(f)
                coverage_percent = coverage_data['totals']['percent_covered']
                print(f"\n{Colors.BOLD}📊 COBERTURA TOTAL: {coverage_percent:.1f}%{Colors.END}")
                
                if coverage_percent >= 80:
                    print_success("META DE 80% ALCANÇADA!")
                else:
                    print_warning(f"Cobertura abaixo de 80% (atual: {coverage_percent:.1f}%)")
        except Exception as e:
            print_warning(f"Não foi possível ler cobertura: {e}")
    
    return critical_success and coverage_success, coverage_percent

def generate_final_report(fix_results: dict, test_success: bool, coverage: Optional[float]):
    """Gera relatório final."""
    print_header("RELATÓRIO FINAL")
    
    # Verificar resultados das correções
    all_fixes_ok = all(fix_results.values())
    
    if all_fixes_ok:
        print_success("Todas as correções foram aplicadas com sucesso!")
    else:
        print_error("Algumas correções falharam:")
        for script, success in fix_results.items():
            status = "✅" if success else "❌"
            print(f"   {status} {script}")
    
    # Verificar testes
    if test_success:
        print_success("Testes críticos passando!")
    else:
        print_error("Alguns testes falharam")
    
    # Verificar cobertura
    if coverage and coverage >= 80:
        print_success(f"Cobertura adequada: {coverage:.1f}%")
    elif coverage:
        print_warning(f"Cobertura precisa melhorar: {coverage:.1f}%")
    
    # Conclusão
    print(f"\n{Colors.BOLD}📋 CONCLUSÃO:{Colors.END}")
    
    if all_fixes_ok and test_success and coverage and coverage >= 80:
        print(f"\n{Colors.GREEN}{Colors.BOLD}🎉 SUCESSO COMPLETO!{Colors.END}")
        print(f"{Colors.GREEN}O CardioAI Pro está com cobertura adequada e testes passando!{Colors.END}")
    else:
        print(f"\n{Colors.YELLOW}📌 Ações necessárias:{Colors.END}")
        
        if not all_fixes_ok:
            print("   1. Verifique os scripts de correção que falharam")
        
        if not test_success:
            print("   2. Execute testes com mais detalhes: pytest -vv")
            print("   3. Use pytest --pdb para debug interativo")
        
        if coverage and coverage < 80:
            print("   4. Abra htmlcov/index.html para ver relatório de cobertura")
            print("   5. Adicione testes para linhas não cobertas")
    
    # Arquivos gerados
    print(f"\n{Colors.BOLD}📂 Arquivos gerados:{Colors.END}")
    print("   - htmlcov/index.html (relatório visual de cobertura)")
    print("   - coverage.json (dados de cobertura em JSON)")
    print("   - .coverage (banco de dados de cobertura)")

def main():
    """Função principal."""
    print(f"{Colors.BOLD}{Colors.BLUE}🚀 APLICANDO TODAS AS CORREÇÕES DO CARDIOAI PRO{Colors.END}")
    
    # Verificar diretório
    if not check_backend_directory():
        return 1
    
    print(f"\n📁 Diretório de trabalho: {Path.cwd()}")
    
    # 1. Instalar dependências
    install_dependencies()
    
    # 2. Limpar cache
    clean_cache()
    
    # 3. Executar correções
    fix_results = run_fixes()
    
    # 4. Executar testes
    test_success, coverage = run_tests()
    
    # 5. Gerar relatório final
    generate_final_report(fix_results, test_success, coverage)
    
    print(f"\n{Colors.BOLD}{'=' * 60}{Colors.END}")
    print("Script concluído!")
    
    return 0 if (test_success and coverage and coverage >= 80) else 1

if __name__ == "__main__":
    sys.exit(main())
