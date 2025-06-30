# Guia de Instalação Completo do CardioAI Pro

Este guia fornece instruções detalhadas para instalar e configurar o CardioAI Pro em seu computador. Siga os passos abaixo para configurar o ambiente de desenvolvimento e executar o sistema.

## Requisitos do Sistema

- **Sistema Operacional**: Windows 10/11, macOS ou Linux
- **Python**: Versão 3.8 ou superior
- **Node.js**: Versão 14 ou superior
- **Git**: Para clonar o repositório

## Passo 1: Clonar o Repositório

1. Abra o terminal ou prompt de comando
2. Clone o repositório do GitHub:

```bash
git clone https://github.com/drguilhermecapel/cardio.ai.pro.git
cd cardio.ai.pro
```

## Passo 2: Configurar o Backend

### 2.1 Criar e Ativar o Ambiente Virtual

#### No Windows:

```powershell
cd backend
python -m venv cardioai_env
cardioai_env\Scripts\activate
```

#### No macOS/Linux:

```bash
cd backend
python3 -m venv cardioai_env
source cardioai_env/bin/activate
```

### 2.2 Instalar Dependências do Backend

```bash
pip install -r requirements.txt
```

### 2.3 Aplicar Correções Necessárias

Para corrigir erros conhecidos no backend, execute:

```bash
python fix_type_annotations.py
```

### 2.4 Inicializar o Banco de Dados

```bash
python init_database.py
```

### 2.5 Criar Favicon (se necessário)

```bash
python create_favicon.py
```

## Passo 3: Configurar o Frontend

### 3.1 Instalar Dependências do Frontend

```bash
cd ../frontend
npm install
```

### 3.2 Aplicar Correções Necessárias

Para corrigir erros conhecidos no frontend, execute:

```bash
node fix_auth_context_improved.js
node fix_api_routes.js
```

## Passo 4: Executar o Sistema

### 4.1 Iniciar o Backend

Em um terminal, com o ambiente virtual ativado:

#### No Windows:

```powershell
cd backend
cardioai_env\Scripts\activate  # Se ainda não estiver ativado
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000
```

#### No macOS/Linux:

```bash
cd backend
source cardioai_env/bin/activate  # Se ainda não estiver ativado
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000
```

### 4.2 Iniciar o Frontend

Em outro terminal:

```bash
cd frontend
npm run dev
```

### 4.3 Acessar o Sistema

Abra seu navegador e acesse:

- **Frontend**: http://localhost:5173
- **API do Backend**: http://localhost:8000/docs

## Passo 5: Fazer Login no Sistema

Use as credenciais padrão para acessar o sistema:

- **Email/Usuário**: `admin@cardioai.com`
- **Senha**: `admin123`

## Solução de Problemas Comuns

### Erro "Attribute name 'metadata' is reserved"

Este erro ocorre no modelo ECGAnalysis. Execute o script de correção:

```bash
cd backend
python fix_type_annotations.py
```

### Erro "name 'List' is not defined"

Este erro ocorre no arquivo exceptions.py. Execute o script de correção:

```bash
cd backend
python fix_type_annotations.py
```

### Erro "useAuth must be used within an AuthProvider"

Este erro ocorre no frontend. Execute o script de correção:

```bash
cd frontend
node fix_auth_context_improved.js
```

### Erro 404 para favicon.ico

Execute o script para criar o favicon:

```bash
cd backend
python create_favicon.py
```

### Erro 404 Not Found ao tentar fazer login

Se você receber um erro 404 ao tentar fazer login, há duas soluções possíveis:

1. **Solução no Frontend**: Execute o script `fix_api_routes.js` na pasta frontend:

   ```bash
   cd frontend
   node fix_api_routes.js
   npm run dev
   ```

2. **Solução no Backend**: Reinicie o backend após a atualização mais recente que adiciona rotas de compatibilidade.

Para instruções detalhadas, consulte o arquivo CORRECAO_ERRO_LOGIN.md.

### Erro "Manifest: Line: 1, column: 1, Syntax error"

Este erro ocorre no console do navegador. Execute o script para criar o arquivo manifest.json:

```bash
cd frontend
node create_manifest.js
npm run dev
```

Para instruções detalhadas, consulte o arquivo CORRECAO_ERRO_MANIFEST.md.

### Erro de CORS no frontend

- Verifique se o backend está rodando na porta 8000
- Execute o script para corrigir a configuração do proxy:

```bash
cd frontend
node fix_vite_config.js
npm run dev
```

## Estrutura do Projeto

```
cardio.ai.pro/
├── backend/                # Servidor FastAPI
│   ├── app/                # Código principal do backend
│   │   ├── api/            # Endpoints da API
│   │   ├── core/           # Configurações e utilitários
│   │   ├── db/             # Configuração do banco de dados
│   │   ├── models/         # Modelos SQLAlchemy
│   │   ├── schemas/        # Esquemas Pydantic
│   │   ├── services/       # Lógica de negócios
│   │   └── static/         # Arquivos estáticos
│   ├── tests/              # Testes do backend
│   └── requirements.txt    # Dependências do Python
├── frontend/               # Cliente React
│   ├── public/             # Arquivos públicos
│   ├── src/                # Código fonte do frontend
│   │   ├── components/     # Componentes React
│   │   ├── contexts/       # Contextos React
│   │   ├── pages/          # Páginas da aplicação
│   │   ├── store/          # Estado global (Redux)
│   │   └── utils/          # Utilitários
│   ├── package.json        # Dependências do Node.js
│   └── vite.config.ts      # Configuração do Vite
└── README.md               # Documentação principal
```

## Desenvolvimento

### Comandos Úteis

#### Backend

- **Executar testes**: `pytest`
- **Verificar tipos**: `mypy app`
- **Formatar código**: `black app`

#### Frontend

- **Executar testes**: `npm test`
- **Verificar tipos**: `npm run typecheck`
- **Formatar código**: `npm run format`

## Suporte

Se você encontrar problemas durante a instalação ou uso do CardioAI Pro, consulte a documentação adicional nos arquivos:

- CORRECAO_ERRO_METADATA.md
- CORRECAO_ERRO_FRONTEND.md
- CORRECAO_ERRO_FAVICON.md
- CORRECAO_ERRO_LOGIN.md
- CREDENCIAIS_ACESSO.md

## Notas Importantes

- O sistema usa um banco de dados SQLite por padrão, que é adequado para testes, mas não para produção com muitos usuários.
- As credenciais padrão devem ser alteradas antes de usar o sistema em produção.
- Para ambientes de produção, considere configurar um banco de dados PostgreSQL e um servidor web como Nginx ou Apache.