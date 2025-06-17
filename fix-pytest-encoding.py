import os
from pathlib import Path

print("CORRIGINDO ERRO DE ENCODING DO PYTEST")
print("=" * 50)

# Encontrar e remover/corrigir pytest.ini problemático
backend_dir = Path("backend")

# Procurar por pytest.ini em vários lugares
possible_locations = [
    backend_dir / "pytest.ini",
    Path("pytest.ini"),
    backend_dir / "setup.cfg",
    Path("setup.cfg"),
    backend_dir / "tox.ini",
    Path("tox.ini"),
    backend_dir / "pyproject.toml",
    Path("pyproject.toml")
]

print("Procurando arquivos de configuracao...")
for config_file in possible_locations:
    if config_file.exists():
        print(f"\nEncontrado: {config_file}")
        try:
            # Tentar ler o arquivo
            with open(config_file, 'r', encoding='utf-8') as f:
                content = f.read()
            print(f"  - Leitura OK")
        except UnicodeDecodeError:
            print(f"  - ERRO de encoding! Removendo...")
            os.remove(config_file)
            print(f"  - Arquivo removido")

# Criar um pytest.ini limpo e simples
print("\nCriando novo pytest.ini...")
pytest_ini = backend_dir / "pytest.ini"
with open(pytest_ini, 'w', encoding='utf-8') as f:
    f.write("""[tool:pytest]
testpaths = tests
python_files = test_*.py
python_classes = Test*
python_functions = test_*
addopts = -v --tb=short
""")
print("OK - pytest.ini criado!")

# Criar script de teste direto
print("\nCriando script de teste direto...")
with open("TESTAR_DIRETO.bat", 'w', encoding='utf-8') as f:
    f.write("""@echo off
cd backend
set PYTHONPATH=%CD%
set ENVIRONMENT=test

echo.
echo ====================================
echo EXECUTANDO TESTES DIRETAMENTE
echo ====================================

REM Executar pytest sem arquivo de config
python -m pytest tests -v --tb=short --cov=app --cov-report=term-missing

echo.
echo ====================================
echo Se ainda der erro, tente:
echo python -m pytest tests -v
echo ====================================
pause
""")

print("\n" + "=" * 50)
print("PRONTO! Execute agora:")
print("   TESTAR_DIRETO.bat")
print("\nOu manualmente:")
print("   cd backend")
print("   python -m pytest tests -v")
