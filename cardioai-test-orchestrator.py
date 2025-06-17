#!/usr/bin/env python3
"""
Orquestrador principal para correção de testes e aumento de cobertura do CardioAI Pro.
Executa todas as correções necessárias e gera relatório de cobertura.
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path
from datetime import datetime
import json

class CardioAITestOrchestrator:
    def __init__(self):
        self.project_root = Path.cwd()
        self.backend_dir = self.project_root / "backend"
        self.start_time = datetime.now()
        self.results = {
            "fixes_applied": [],
            "tests_created": [],
            "coverage_before": None,
            "coverage_after": None,
            "errors": []
        }
        
    def run(self):
        """Executa todo o processo de correção e teste."""
        print("=" * 70)
        print("CARDIOAI PRO - ORQUESTRADOR DE CORREÇÕES E TESTES".center(70))
        print("=" * 70)
        print(f"\n📍 Diretório: {self.project_root}")
        print(f"🕐 Início: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        
        # Verificar ambiente
        if not self._check_environment():
            return False
            
        # 1. Aplicar correções do primeiro script
        print("\n📋 FASE 1: Aplicando correções básicas...")
        self._run_basic_fixes()
        
        # 2. Criar testes críticos
        print("\n📋 FASE 2: Criando testes críticos...")
        self._create_critical_tests()
        
        # 3. Instalar dependências de teste
        print("\n📋 FASE 3: Instalando dependências...")
        self._install_test_dependencies()
        
        # 4. Executar testes e gerar cobertura
        print("\n📋 FASE 4: Executando testes...")
        self._run_tests_with_coverage()
        
        # 5. Gerar relatório final
        print("\n📋 FASE 5: Gerando relatório...")
        self._generate_final_report()
        
        return True
        
    def _check_environment(self):
        """Verifica se o ambiente está correto."""
        if not self.backend_dir.exists():
            print("❌ Diretório 'backend' não encontrado!")
            print("   Execute este script na raiz do projeto CardioAI Pro")
            self.results["errors"].append("Backend directory not found")
            return False
            
        # Verificar Python
        python_version = sys.version_info
        if python_version.major < 3 or python_version.minor < 8:
            print(f"⚠️  Python {python_version.major}.{python_version.minor} detectado")
            print("   Recomendado Python 3.8 ou superior")
            
        return True
        
    def _run_basic_fixes(self):
        """Aplica correções básicas do primeiro script."""
        try:
            # Executar o script de correções
            fix_script_path = self.project_root / "fix_cardioai_test_coverage.py"
            if fix_script_path.exists():
                result = subprocess.run(
                    [sys.executable, str(fix_script_path)],
                    capture_output=True,
                    text=True
                )
                if result.returncode == 0:
                    print("✅ Correções básicas aplicadas")
                    self.results["fixes_applied"].append("basic_fixes")
                else:
                    print(f"⚠️  Algumas correções falharam: {result.stderr}")
            else:
                # Criar e executar correções inline
                self._apply_inline_fixes()
                
        except Exception as e:
            print(f"❌ Erro ao aplicar correções: {e}")
            self.results["errors"].append(f"Basic fixes error: {e}")
            
    def _apply_inline_fixes(self):
        """Aplica correções diretamente sem script externo."""
        # Adicionar MultiPathologyException
        exceptions_file = self.backend_dir / "app" / "core" / "exceptions.py"
        if exceptions_file.exists():
            with open(exceptions_file, 'a', encoding='utf-8') as f:
                f.write('''

# Exceções adicionadas para correção de testes
class MultiPathologyException(CardioAIException):
    """Exception for multi-pathology service errors."""
    
    def __init__(self, message: str, pathologies: list[str] | None = None) -> None:
        details = {"pathologies": pathologies} if pathologies else {}
        super().__init__(
            message=message,
            error_code="MULTI_PATHOLOGY_ERROR",
            status_code=500,
            details=details,
        )

class ECGReaderException(CardioAIException):
    """Exception for ECG file reading errors."""
    
    def __init__(self, message: str, file_format: str | None = None) -> None:
        details = {"file_format": file_format} if file_format else {}
        super().__init__(
            message=message,
            error_code="ECG_READER_ERROR",
            status_code=422,
            details=details,
        )
''')
            print("✅ Exceções faltantes adicionadas")
            self.results["fixes_applied"].append("missing_exceptions")
            
        # Criar conftest.py com mocks
        self._create_conftest()
        
    def _create_conftest(self):
        """Cria arquivo conftest.py com configurações de teste."""
        conftest_path = self.backend_dir / "tests" / "conftest.py"
        conftest_path.parent.mkdir(exist_ok=True)
        
        with open(conftest_path, 'w', encoding='utf-8') as f:
            f.write('''"""Configuração global para testes do CardioAI Pro."""

import sys
import os
from unittest.mock import MagicMock
import pytest

# Configurar ambiente de teste
os.environ["ENVIRONMENT"] = "test"
os.environ["DATABASE_URL"] = "sqlite+aiosqlite:///:memory:"
os.environ["SECRET_KEY"] = "test-secret-key"

# Mock pyedflib
sys.modules["pyedflib"] = MagicMock()

# Mock outros módulos problemáticos
sys.modules["redis"] = MagicMock()
sys.modules["celery"] = MagicMock()
sys.modules["minio"] = MagicMock()

@pytest.fixture
def mock_ecg_data():
    """Mock ECG data for tests."""
    import numpy as np
    return {
        "signal": np.random.randn(12, 5000).tolist(),
        "sampling_rate": 500,
        "leads": ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
    }
''')
        print("✅ Configuração de testes criada")
        self.results["fixes_applied"].append("conftest_created")
        
    def _create_critical_tests(self):
        """Cria os testes críticos."""
        try:
            # Executar script de criação de testes críticos
            critical_script_path = self.project_root / "create_critical_tests.py"
            if critical_script_path.exists():
                result = subprocess.run(
                    [sys.executable, str(critical_script_path)],
                    capture_output=True,
                    text=True
                )
                if result.returncode == 0:
                    print("✅ Testes críticos criados")
                    self.results["tests_created"].extend([
                        "test_security_complete.py",
                        "test_ecg_processing_complete.py",
                        "test_medical_validation_complete.py",
                        "test_core_services_complete.py",
                        "test_utils_complete.py"
                    ])
            else:
                print("⚠️  Script de testes críticos não encontrado")
                
        except Exception as e:
            print(f"❌ Erro ao criar testes: {e}")
            self.results["errors"].append(f"Critical tests error: {e}")
            
    def _install_test_dependencies(self):
        """Instala dependências necessárias para testes."""
        print("📦 Instalando pytest e plugins...")
        
        test_deps = [
            "pytest>=8.0.0",
            "pytest-asyncio>=0.23.0",
            "pytest-cov>=4.1.0",
            "pytest-mock>=3.12.0",
            "coverage>=7.3.0",
            "aiosqlite>=0.19.0"  # Para testes com SQLite async
        ]
        
        for dep in test_deps:
            try:
                subprocess.run(
                    [sys.executable, "-m", "pip", "install", dep],
                    capture_output=True,
                    check=True
                )
                print(f"  ✅ {dep.split('>=')[0]} instalado")
            except subprocess.CalledProcessError:
                print(f"  ⚠️  Falha ao instalar {dep}")
                
    def _run_tests_with_coverage(self):
        """Executa testes com análise de cobertura."""
        os.chdir(self.backend_dir)
        
        # Configurar variáveis de ambiente
        env = os.environ.copy()
        env["PYTHONPATH"] = str(self.backend_dir)
        env["ENVIRONMENT"] = "test"
        
        print("\n🧪 Executando testes com cobertura...\n")
        
        # Comandos de teste em ordem
        test_commands = [
            # 1. Testes básicos primeiro (sem dependências externas)
            ["pytest", "tests/test_exceptions_coverage.py", "-v", "--tb=short"],
            ["pytest", "tests/test_config_coverage.py", "-v", "--tb=short"],
            
            # 2. Testes de segurança (crítico)
            ["pytest", "tests/test_security_complete.py", "-v", "--tb=short"],
            
            # 3. Todos os testes com cobertura
            ["pytest", "--cov=app", "--cov-report=term-missing", 
             "--cov-report=html", "--cov-report=json", "-v", "--tb=short", 
             "--maxfail=10"]
        ]
        
        for cmd in test_commands:
            print(f"\n📍 Executando: {' '.join(cmd)}")
            print("-" * 50)
            
            try:
                result = subprocess.run(
                    cmd,
                    env=env,
                    capture_output=False,
                    text=True
                )
                
                if result.returncode != 0:
                    print(f"\n⚠️  Alguns testes falharam (código: {result.returncode})")
                    
            except Exception as e:
                print(f"\n❌ Erro ao executar comando: {e}")
                
        # Ler cobertura final
        self._read_coverage_report()
        
    def _read_coverage_report(self):
        """Lê o relatório de cobertura gerado."""
        coverage_json = self.backend_dir / "coverage.json"
        
        if coverage_json.exists():
            try:
                with open(coverage_json, 'r') as f:
                    coverage_data = json.load(f)
                    
                total_coverage = coverage_data.get("totals", {}).get("percent_covered", 0)
                self.results["coverage_after"] = round(total_coverage, 2)
                
                print(f"\n📊 Cobertura Total: {total_coverage:.2f}%")
                
                if total_coverage >= 80:
                    print("✅ Meta de 80% alcançada!")
                else:
                    print(f"⚠️  Abaixo da meta (faltam {80 - total_coverage:.2f}%)")
                    
            except Exception as e:
                print(f"❌ Erro ao ler relatório de cobertura: {e}")
                
    def _generate_final_report(self):
        """Gera relatório final da execução."""
        end_time = datetime.now()
        duration = (end_time - self.start_time).total_seconds()
        
        print("\n" + "=" * 70)
        print("RELATÓRIO FINAL".center(70))
        print("=" * 70)
        
        print(f"\n📊 RESUMO DA EXECUÇÃO:")
        print(f"   • Duração: {duration:.1f} segundos")
        print(f"   • Correções aplicadas: {len(self.results['fixes_applied'])}")
        print(f"   • Testes criados: {len(self.results['tests_created'])}")
        print(f"   • Erros encontrados: {len(self.results['errors'])}")
        
        if self.results["coverage_after"]:
            print(f"\n📈 COBERTURA DE TESTES:")
            print(f"   • Cobertura alcançada: {self.results['coverage_after']}%")
            
            if self.results["coverage_after"] >= 80:
                print("   • ✅ META ATINGIDA!")
            else:
                gap = 80 - self.results["coverage_after"]
                print(f"   • ⚠️  Faltam {gap:.1f}% para a meta")
                
        print(f"\n📁 ARQUIVOS GERADOS:")
        print(f"   • Relatório HTML: backend/htmlcov/index.html")
        print(f"   • Relatório JSON: backend/coverage.json")
        print(f"   • Logs de teste: backend/pytest.log")
        
        if self.results["errors"]:
            print(f"\n⚠️  ERROS ENCONTRADOS:")
            for error in self.results["errors"]:
                print(f"   • {error}")
                
        print("\n📋 PRÓXIMOS PASSOS:")
        print("   1. Revisar o relatório HTML de cobertura")
        print("   2. Adicionar testes para módulos com baixa cobertura")
        print("   3. Corrigir testes que falharam")
        print("   4. Executar novamente para validar")
        
        # Salvar relatório em arquivo
        report_path = self.backend_dir / "test_execution_report.json"
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=2, default=str)
            
        print(f"\n💾 Relatório salvo em: {report_path}")
        print("\n" + "=" * 70)


def main():
    """Função principal."""
    orchestrator = CardioAITestOrchestrator()
    
    try:
        success = orchestrator.run()
        return 0 if success else 1
    except KeyboardInterrupt:
        print("\n\n⚠️  Execução interrompida pelo usuário")
        return 1
    except Exception as e:
        print(f"\n\n❌ Erro fatal: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
