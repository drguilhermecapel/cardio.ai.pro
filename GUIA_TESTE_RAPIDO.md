# Guia Rápido para Testar o CardioAI Pro

Este guia fornece instruções passo a passo para testar o CardioAI Pro no seu computador, incluindo a correção de erros comuns.

## Pré-requisitos

- Windows 10 ou superior
- Python 3.8 ou superior
- Node.js 16+ e npm
- Git

## Passo 1: Configurar o Ambiente Python

```powershell
# Navegue até a pasta raiz do projeto
cd C:\caminho\para\cardio.ai.pro

# Crie um ambiente virtual
python -m venv cardioai_env

# Ative o ambiente virtual
.\cardioai_env\Scripts\activate

# Verifique se o pip está instalado corretamente
python -m pip --version

# Se o pip não estiver instalado, instale-o
python -m ensurepip --upgrade
python -m pip install --upgrade pip
```

## Passo 2: Corrigir o Erro de 'metadata'

Se você encontrar o erro "Attribute name 'metadata' is reserved", siga estas etapas:

```powershell
# Navegue até a pasta do backend
cd backend

# Crie um arquivo de correção
notepad fix_metadata_field.py
```

Cole o seguinte código no arquivo `fix_metadata_field.py`:

```python
#!/usr/bin/env python3
"""
Script para corrigir o erro 'Attribute name 'metadata' is reserved' no modelo ECGAnalysis.
Este script modifica o arquivo do modelo para renomear o campo 'metadata' para 'ecg_metadata'.
"""

import os
import sys
from pathlib import Path

def fix_metadata_field():
    """Corrige o campo metadata no modelo ECGAnalysis."""
    try:
        # Caminho para o arquivo do modelo
        model_path = Path(__file__).parent / "app" / "models" / "ecg_analysis.py"
        
        if not model_path.exists():
            print(f"❌ Arquivo do modelo não encontrado: {model_path}")
            return False
        
        # Ler o conteúdo do arquivo
        content = model_path.read_text(encoding="utf-8")
        
        # Verificar se o campo 'metadata' existe
        if "metadata = Column(JSON, nullable=True)" in content:
            # Substituir 'metadata' por 'ecg_metadata'
            new_content = content.replace(
                "metadata = Column(JSON, nullable=True)",
                "ecg_metadata = Column(JSON, nullable=True)  # Renomeado de 'metadata' para evitar conflito com SQLAlchemy"
            )
            
            # Salvar o arquivo modificado
            model_path.write_text(new_content, encoding="utf-8")
            print(f"✅ Campo 'metadata' renomeado para 'ecg_metadata' em {model_path}")
            return True
        else:
            print("ℹ️ O campo 'metadata' já foi corrigido ou não existe no modelo.")
            return True
            
    except Exception as e:
        print(f"❌ Erro ao corrigir o campo metadata: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🔧 Corrigindo campo 'metadata' no modelo ECGAnalysis...")
    success = fix_metadata_field()
    sys.exit(0 if success else 1)
```

Execute o script de correção:

```powershell
python fix_metadata_field.py
```

## Passo 3: Instalar Dependências do Backend

```powershell
# Ainda na pasta do backend
python -m pip install -r requirements.txt
```

## Passo 4: Inicializar o Banco de Dados

```powershell
# Inicializar o banco de dados
python init_database.py
```

## Passo 5: Iniciar o Servidor Backend

```powershell
# Iniciar o servidor backend
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000
```

## Passo 6: Configurar e Executar o Frontend (em outro terminal)

```powershell
# Abra um novo terminal
# Navegue até a pasta raiz do projeto
cd C:\caminho\para\cardio.ai.pro

# Ative o ambiente virtual (se estiver usando)
.\cardioai_env\Scripts\activate

# Navegue até a pasta do frontend
cd frontend

# Instale as dependências Node.js
npm install

# Inicie o servidor de desenvolvimento
npm run dev
```

## Passo 7: Acessar o CardioAI Pro

- Abra seu navegador e acesse:
  - Frontend: http://localhost:5173 (porta padrão do Vite)
  - API: http://localhost:8000
  - Documentação da API: http://localhost:8000/docs

## Solução de Problemas Comuns

### Erro "No module named 'pip'"

```powershell
# Reinstale o pip no ambiente virtual
python -m ensurepip --upgrade
# OU
# Use o Python do sistema para instalar as dependências
python -m pip install -r requirements.txt
```

### Erro "Attribute name 'metadata' is reserved"

Execute o script de correção conforme descrito no Passo 2.

### Erro ao instalar pacotes com requisitos de compilação

```powershell
# Instale as ferramentas de compilação do Visual C++
pip install --upgrade setuptools wheel
# Instale pacotes pré-compilados quando disponíveis
pip install --only-binary=:all: -r requirements.txt
```

### Erro de permissão ao executar scripts

```powershell
# Execute o PowerShell como administrador e defina a política de execução
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### Erro de CORS no frontend

- Verifique se o backend está rodando na porta 8000
- Verifique se o arquivo vite.config.ts está configurado corretamente para proxy

## Notas Importantes

- O sistema usa um banco de dados SQLite por padrão, que é adequado para testes, mas não para produção com muitos usuários.
- Para ambientes de produção, considere configurar um banco de dados PostgreSQL conforme descrito no arquivo INSTALACAO.md.
- As credenciais padrão devem ser alteradas após o primeiro login por motivos de segurança.

## Recursos Adicionais

- Documentação completa: Disponível em `/docs` no repositório
- Guia de instalação detalhado: Consulte o arquivo INSTALACAO.md
- Guia de instalação standalone: Consulte o arquivo INSTALACAO-STANDALONE.md