# Guia de Instalação e Execução do CardioAI Pro

Este guia fornece instruções passo a passo para instalar e executar o CardioAI Pro em seu computador.

## Requisitos

- Python 3.9 ou superior
- Node.js 16 ou superior
- npm 8 ou superior
- Git

## 1. Clonar o Repositório

Primeiro, clone o repositório do GitHub:

```powershell
git clone https://github.com/drguilhermecapel/cardio.ai.pro.git
cd cardio.ai.pro
```

## 2. Configurar o Backend

### 2.1. Criar um Ambiente Virtual

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
```

### 2.2. Instalar Dependências

```powershell
# Instale as dependências do backend
pip install -r requirements.txt

# Instale o validador de email (necessário para o Pydantic)
pip install email-validator
```

### 2.3. Iniciar o Servidor Backend

```powershell
# Inicie o servidor backend
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

O backend estará disponível em `http://localhost:8000`.

## 3. Configurar o Frontend

Abra um novo terminal e siga estas etapas:

### 3.1. Instalar Dependências

```powershell
# Navegue até a pasta do frontend
cd frontend

# Instale as dependências
npm install
```

### 3.2. Iniciar o Servidor de Desenvolvimento

```powershell
# Inicie o servidor de desenvolvimento
npm run dev
```

O frontend estará disponível em `http://localhost:5173`.

## 4. Acessar o Sistema

Abra seu navegador e acesse `http://localhost:5173`. Você verá a tela de login do CardioAI Pro.

Use as seguintes credenciais para fazer login:
- **Email/Usuário**: `admin@cardioai.com`
- **Senha**: `admin123`

## 5. Solução de Problemas Comuns

### 5.1. Erro de Login (404 Not Found)

Se você encontrar um erro 404 ao tentar fazer login, verifique:

1. Se o backend está rodando corretamente
2. Se o arquivo `backend/app/api/v1/endpoints/auth.py` está configurado corretamente:
   ```python
   # Deve ser:
   router = APIRouter(tags=["authentication"])
   # E não:
   # router = APIRouter(prefix="/auth", tags=["authentication"])
   ```

Para mais detalhes, consulte o arquivo `CORRECAO_ERRO_LOGIN.md`.

### 5.2. Erro de Indentação no ECGService

Se você encontrar um erro de indentação ao iniciar o backend, verifique o arquivo `backend/app/services/ecg_service.py`. A classe `ECGService` deve estar corretamente definida.

Para mais detalhes, consulte o arquivo `CORRECAO_ERRO_INDENTACAO.md`.

### 5.3. Erro de Importação no Backend

Se você encontrar erros de importação no backend, verifique se todas as classes necessárias estão definidas, especialmente a classe `UnauthorizedException` em `backend/app/core/exceptions.py`.

Para mais detalhes, consulte o arquivo `CORRECAO_ERRO_BACKEND.md`.

## 6. Desenvolvimento

### 6.1. Estrutura do Projeto

- **Backend**: Implementado com FastAPI
  - `app/main.py`: Ponto de entrada da aplicação
  - `app/api/`: Endpoints da API
  - `app/models/`: Modelos de dados
  - `app/services/`: Serviços de negócio

- **Frontend**: Implementado com React e Vite
  - `src/main.tsx`: Ponto de entrada da aplicação
  - `src/contexts/`: Contextos React (incluindo autenticação)
  - `src/pages/`: Páginas da aplicação
  - `src/components/`: Componentes reutilizáveis

### 6.2. API Endpoints

- **Autenticação**: `/api/v1/auth/login`
- **Pacientes**: `/api/v1/patients`
- **Análises de ECG**: `/api/v1/ecg-analyses`

## 7. Suporte

Se você encontrar problemas adicionais, consulte a documentação ou entre em contato com a equipe de suporte.