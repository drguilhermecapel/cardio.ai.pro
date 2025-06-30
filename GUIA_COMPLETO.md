# Guia Completo para Instalação e Correção de Erros do CardioAI Pro

Este guia fornece instruções detalhadas para instalar, configurar e corrigir erros comuns no CardioAI Pro.

## Índice

1. [Instalação Básica](#1-instalação-básica)
2. [Correção de Erros Comuns](#2-correção-de-erros-comuns)
   - [Erro de Indentação no ECGService](#21-erro-de-indentação-no-ecgservice)
   - [Erro de Inicialização do Banco de Dados](#22-erro-de-inicialização-do-banco-de-dados)
   - [Erro de Dependência pdf2image](#23-erro-de-dependência-pdf2image)
   - [Erro nas Rotas de Autenticação](#24-erro-nas-rotas-de-autenticação)
3. [Execução do Sistema](#3-execução-do-sistema)
4. [Solução de Problemas Adicionais](#4-solução-de-problemas-adicionais)

## 1. Instalação Básica

### 1.1. Clonar o Repositório

```powershell
git clone https://github.com/drguilhermecapel/cardio.ai.pro.git
cd cardio.ai.pro
```

### 1.2. Configurar o Backend

```powershell
# Navegue até a pasta do backend
cd backend

# Crie um ambiente virtual
python -m venv cardioai_env

# Ative o ambiente virtual
# No Windows:
cardioai_env\Scripts\activate
# No Linux/Mac:
# source cardioai_env/bin/activate

# Instale as dependências
pip install -r requirements.txt

# Instale o validador de email (necessário para o Pydantic)
pip install email-validator
```

### 1.3. Configurar o Frontend

```powershell
# Navegue até a pasta do frontend
cd frontend

# Instale as dependências
npm install
```

## 2. Correção de Erros Comuns

### 2.1. Erro de Indentação no ECGService

Se você encontrar o seguinte erro:

```
IndentationError: unexpected indent
```

Siga estas instruções:

1. Abra o arquivo `backend/app/services/ecg_service.py` em um editor de texto
2. Substitua todo o conteúdo pelo código correto (disponível em [CORRECAO_MANUAL.md](CORRECAO_MANUAL.md))

O problema ocorre porque os métodos da classe `ECGService` estão com indentação incorreta.

### 2.2. Erro de Inicialização do Banco de Dados

Se você encontrar o seguinte erro ao inicializar o banco de dados:

```
Failed to create admin user: When initializing mapper Mapper[User(users)], expression 'ECGAnalysis.created_by' failed to locate a name ("name 'ECGAnalysis' is not defined").
```

Siga estas instruções:

1. Abra o arquivo `backend/app/models/ecg_analysis.py`
2. Adicione o campo `created_by` após a linha que define `validated_by`:
   ```python
   created_by = Column(Integer, ForeignKey("users.id"), nullable=True)
   ```
3. Adicione o relacionamento na seção de relacionamentos:
   ```python
   created_by_user = relationship("User", foreign_keys=[created_by], back_populates="analyses")
   ```

Para instruções detalhadas, consulte [CORRECAO_ERRO_DATABASE.md](CORRECAO_ERRO_DATABASE.md).

### 2.3. Erro de Dependência pdf2image

Se você encontrar o seguinte aviso:

```
WARNING:app.main:API v1 router não encontrado: No module named 'pdf2image'
```

Siga estas instruções:

1. Instale o pacote pdf2image:
   ```powershell
   pip install pdf2image
   ```

2. Instale o Poppler (necessário para o pdf2image):
   - Windows: Baixe em https://github.com/oschwartz10612/poppler-windows/releases/
   - Linux: `sudo apt-get install poppler-utils`
   - macOS: `brew install poppler`

Para instruções detalhadas, consulte [CORRECAO_ERRO_PDF2IMAGE.md](CORRECAO_ERRO_PDF2IMAGE.md).

### 2.4. Erro nas Rotas de Autenticação

Se você encontrar erro 404 ao tentar fazer login:

```
INFO: 127.0.0.1:59873 - "POST /api/v1/auth/login HTTP/1.1" 404 Not Found
```

Siga estas instruções:

1. Abra o arquivo `backend/app/api/v1/api.py`
2. Altere a linha 32 de:
   ```python
   api_router.include_router(auth.router, prefix="/auth", tags=["authentication"])
   ```
   Para:
   ```python
   api_router.include_router(auth.router, tags=["authentication"])
   ```

Para instruções detalhadas, consulte [CORRECAO_ERRO_ROTAS.md](CORRECAO_ERRO_ROTAS.md).

## 3. Execução do Sistema

### 3.1. Inicializar o Banco de Dados

```powershell
# Na pasta backend
python init_database.py
```

### 3.2. Iniciar o Backend

```powershell
# Na pasta backend
python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### 3.3. Iniciar o Frontend

```powershell
# Na pasta frontend
npm run dev
```

### 3.4. Acessar o Sistema

Abra seu navegador e acesse `http://localhost:5173`. Use as seguintes credenciais para fazer login:
- **Email/Usuário**: `admin@cardioai.com`
- **Senha**: `admin123`

## 4. Solução de Problemas Adicionais

### 4.1. Verificar Logs do Backend

Se você encontrar problemas, verifique os logs do backend para obter mais informações sobre os erros.

### 4.2. Verificar Configuração do Banco de Dados

O sistema usa SQLite por padrão. Verifique se o arquivo de banco de dados foi criado corretamente na pasta `backend`.

### 4.3. Verificar Configuração do Frontend

Se o frontend não conseguir se conectar ao backend, verifique o arquivo de configuração em `frontend/src/config.js` para garantir que a URL da API esteja correta.

### 4.4. Problemas com CORS

Se você encontrar erros de CORS, verifique se o backend está configurado para permitir solicitações do frontend. Isso é gerenciado nas configurações CORS em `backend/app/main.py`.

## Recursos Adicionais

- [Documentação do FastAPI](https://fastapi.tiangolo.com/)
- [Documentação do React](https://reactjs.org/docs/getting-started.html)
- [Documentação do Vite](https://vitejs.dev/guide/)

Para mais detalhes sobre correções específicas, consulte os arquivos de correção individuais no repositório.