#!/usr/bin/env python3
"""
Test Basic Setup
Verifica se a configuração básica está funcionando
"""

import sys
import importlib
import subprocess
from pathlib import Path


class BasicSetupTester:
    """Test basic setup"""
    
    def __init__(self):
        self.backend_path = Path(__file__).parent
        self.passed = 0
        self.failed = 0
        
    def run(self):
        """Run all basic tests"""
        print("=" * 60)
        print("TESTE BÁSICO DE CONFIGURAÇÃO")
        print("=" * 60)
        
        # 1. Test Python version
        self.test_python_version()
        
        # 2. Test required packages
        self.test_required_packages()
        
        # 3. Test app imports
        self.test_app_imports()
        
        # 4. Test specific modules
        self.test_specific_modules()
        
        # 5. Run minimal pytest
        self.test_minimal_pytest()
        
        # Print summary
        self.print_summary()
        
    def test_python_version(self):
        """Test Python version"""
        print("\n1. Verificando versão do Python...")
        
        version = sys.version_info
        print(f"   Python {version.major}.{version.minor}.{version.micro}")
        
        if version.major == 3 and version.minor >= 8:
            print("   ✅ Versão compatível")
            self.passed += 1
        else:
            print("   ❌ Versão incompatível (precisa Python 3.8+)")
            self.failed += 1
    
    def test_required_packages(self):
        """Test if required packages are installed"""
        print("\n2. Verificando pacotes essenciais...")
        
        packages = [
            "fastapi",
            "pydantic",
            "sqlalchemy",
            "pytest",
            "numpy",
            "scipy",
            "pandas",
        ]
        
        for package in packages:
            try:
                importlib.import_module(package)
                print(f"   ✅ {package}")
                self.passed += 1
            except ImportError:
                print(f"   ❌ {package} não instalado")
                self.failed += 1
    
    def test_app_imports(self):
        """Test if app modules can be imported"""
        print("\n3. Verificando imports do app...")
        
        # Add backend to path
        sys.path.insert(0, str(self.backend_path))
        
        modules = [
            "app.core.config",
            "app.core.exceptions",
            "app.schemas.ecg_analysis",
            "app.services.ecg_service",
            "app.utils.memory_monitor",
        ]
        
        for module in modules:
            try:
                importlib.import_module(module)
                print(f"   ✅ {module}")
                self.passed += 1
            except Exception as e:
                print(f"   ❌ {module}: {str(e)[:50]}...")
                self.failed += 1
    
    def test_specific_modules(self):
        """Test specific problematic modules"""
        print("\n4. Testando módulos específicos...")
        
        # Test ECGAnalysisService
        try:
            from app.services.ecg_service import ECGAnalysisService
            service = ECGAnalysisService()
            print("   ✅ ECGAnalysisService instanciado")
            self.passed += 1
        except Exception as e:
            print(f"   ❌ ECGAnalysisService: {str(e)[:50]}...")
            self.failed += 1
        
        # Test MemoryMonitor
        try:
            from app.utils.memory_monitor import MemoryMonitor
            monitor = MemoryMonitor()
            stats = monitor.get_memory_stats()
            print(f"   ✅ MemoryMonitor funcionando (Memória: {stats.memory_percent:.1f}%)")
            self.passed += 1
        except Exception as e:
            print(f"   ❌ MemoryMonitor: {str(e)[:50]}...")
            self.failed += 1
        
        # Test schemas
        try:
            from app.schemas.ecg_analysis import ECGAnalysisCreate, ProcessingStatus, FileType
            print(f"   ✅ Schemas carregados (FileType tem {len(FileType)} tipos)")
            self.passed += 1
        except Exception as e:
            print(f"   ❌ Schemas: {str(e)[:50]}...")
            self.failed += 1
    
    def test_minimal_pytest(self):
        """Run a minimal pytest"""
        print("\n5. Executando teste pytest mínimo...")
        
        # Create a minimal test file
        test_file = self.backend_path / "test_minimal.py"
        test_content = '''
def test_basic():
    """Basic test"""
    assert 1 + 1 == 2

def test_import():
    """Test imports"""
    import app.core.config
    assert app.core.config.settings is not None
'''
        
        try:
            with open(test_file, 'w', encoding='utf-8') as f:
                f.write(test_content)
            
            # Run pytest
            result = subprocess.run(
                [sys.executable, "-m", "pytest", str(test_file), "-v"],
                capture_output=True,
                text=True
            )
            
            if result.returncode == 0:
                print("   ✅ pytest funcionando")
                self.passed += 1
            else:
                print("   ❌ pytest com erros")
                print(f"   Erro: {result.stderr[:200]}...")
                self.failed += 1
            
            # Clean up
            test_file.unlink(missing_ok=True)
            
        except Exception as e:
            print(f"   ❌ Erro ao executar pytest: {e}")
            self.failed += 1
    
    def print_summary(self):
        """Print test summary"""
        print("\n" + "=" * 60)
        print("RESUMO DOS TESTES")
        print("=" * 60)
        
        total = self.passed + self.failed
        percentage = (self.passed / total * 100) if total > 0 else 0
        
        print(f"\nTotal de testes: {total}")
        print(f"✅ Passaram: {self.passed}")
        print(f"❌ Falharam: {self.failed}")
        print(f"📊 Taxa de sucesso: {percentage:.1f}%")
        
        if self.failed == 0:
            print("\n🎉 SISTEMA PRONTO PARA TESTES COMPLETOS!")
            print("\nPróximo passo:")
            print("python run_coverage_test.py")
        else:
            print("\n⚠️  CORRIJA OS ERROS ANTES DE CONTINUAR")
            print("\nPróximo passo:")
            print("python fix_syntax_and_encoding.py")


def main():
    """Main entry point"""
    tester = BasicSetupTester()
    tester.run()


if __name__ == "__main__":
    main()
