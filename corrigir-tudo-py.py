import os
from pathlib import Path

print("="*60)
print("CORRIGINDO TESTES DO CARDIOAI PRO")
print("="*60)

# Verificar diretório
if not Path("backend").exists():
    print("\nERRO: Execute este arquivo na pasta do projeto!")
    print("Faca: cd C:\\Users\\User\\OneDrive\\Documentos\\GitHub\\cardio.ai.pro")
    input("\nPressione ENTER para sair...")
    exit()

# 1. Adicionar exceções faltantes
print("\n[1/3] Adicionando excecoes faltantes...")
exc_file = Path("backend/app/core/exceptions.py")
if exc_file.exists():
    with open(exc_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    if "MultiPathologyException" not in content:
        with open(exc_file, 'a', encoding='utf-8') as f:
            f.write('''

class MultiPathologyException(CardioAIException):
    """Exception for multi-pathology service errors."""
    def __init__(self, message: str, pathologies=None):
        details = {"pathologies": pathologies} if pathologies else {}
        super().__init__(message, "MULTI_PATHOLOGY_ERROR", 500, details)

class ECGReaderException(CardioAIException):
    """Exception for ECG file reading errors."""
    def __init__(self, message: str, file_format=None):
        details = {"file_format": file_format} if file_format else {}
        super().__init__(message, "ECG_READER_ERROR", 422, details)
''')
        print("   OK - Excecoes adicionadas!")
    else:
        print("   OK - Excecoes ja existem!")

# 2. Criar mock do pyedflib
print("\n[2/3] Criando mock do pyedflib...")
tests_dir = Path("backend/tests")
tests_dir.mkdir(exist_ok=True)
(tests_dir / "__init__.py").touch()

with open(tests_dir / "conftest.py", 'w', encoding='utf-8') as f:
    f.write('''import sys
import os
from unittest.mock import MagicMock

# Configurar ambiente de teste
os.environ["ENVIRONMENT"] = "test"
os.environ["DATABASE_URL"] = "sqlite+aiosqlite:///:memory:"
os.environ["SECRET_KEY"] = "test-secret-key"

# Mock do pyedflib
sys.modules["pyedflib"] = MagicMock()
''')
print("   OK - Mock criado!")

# 3. Criar script de teste
print("\n[3/3] Criando script de teste...")
with open("TESTAR_AGORA.bat", 'w', encoding='utf-8') as f:
    f.write('''@echo off
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
''')
print("   OK - Script criado!")

print("\n" + "="*60)
print("TUDO PRONTO!")
print("="*60)
print("\nAGORA EXECUTE:")
print("   TESTAR_AGORA.bat")
print("\nOu manualmente:")
print("   cd backend")
print("   pytest --cov=app -v")
