# Guia de Teste Passo a Passo do CardioAI Pro

Este guia fornece instruções detalhadas para testar o sistema CardioAI Pro em seu computador. Siga cada etapa na ordem apresentada para garantir uma configuração adequada.

## Pré-requisitos

Antes de começar, certifique-se de ter instalado:

- [Python 3.10+](https://www.python.org/downloads/)
- [Node.js 18+](https://nodejs.org/)
- [Git](https://git-scm.com/downloads)

## 1. Obter o Código-Fonte

### Clonar o Repositório

```powershell
# Crie uma pasta para o projeto
mkdir CardioAI
cd CardioAI

# Clone o repositório
git clone https://github.com/drguilhermecapel/cardio.ai.pro.git
cd cardio.ai.pro
```

## 2. Configurar o Backend

### Instalar Dependências do Python

```powershell
# Navegue até a pasta do backend
cd backend

# Crie um ambiente virtual
python -m venv venv

# Ative o ambiente virtual
.\venv\Scripts\activate  # No Windows
# source venv/bin/activate  # No Linux/Mac

# Instale as dependências
pip install -r requirements.txt
```

### Inicializar o Banco de Dados

```powershell
# Ainda na pasta backend
cd app
python init_db.py
```

### Iniciar o Servidor Backend

```powershell
# Volte para a pasta backend
cd ..
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Mantenha esta janela do terminal aberta com o servidor backend em execução.

## 3. Configurar o Frontend

Abra uma nova janela do terminal para configurar o frontend.

### Instalar Dependências do Node.js

```powershell
# Navegue até a pasta do frontend
cd C:\caminho\para\cardio.ai.pro\frontend

# Instale as dependências
npm install
```

### Corrigir Problemas Conhecidos

#### Corrigir Configuração do Proxy

```powershell
# Execute o script para corrigir a configuração do proxy
node fix_vite_config.js
```

#### Corrigir Erro de Manifest

```powershell
# Execute o script para criar o arquivo manifest.json
node create_manifest.js
```

### Iniciar o Servidor Frontend

```powershell
# Inicie o servidor de desenvolvimento
npm run dev
```

## 4. Acessar o Sistema

1. Abra seu navegador e acesse: http://localhost:5173
2. Você verá a tela de login do CardioAI Pro

## 5. Fazer Login

Use as credenciais padrão para fazer login:

- **Email**: admin@cardioai.com
- **Senha**: admin123

## 6. Explorar o Sistema

Após fazer login com sucesso, você pode explorar as diferentes funcionalidades do sistema:

1. **Dashboard**: Visão geral dos dados e estatísticas
2. **Pacientes**: Gerenciamento de pacientes
3. **ECGs**: Visualização e análise de eletrocardiogramas
4. **Relatórios**: Geração de relatórios de análise
5. **Configurações**: Personalização do sistema

## Solução de Problemas

### Erro 404 ao Fazer Login

Se você encontrar um erro 404 ao tentar fazer login, verifique:

1. Se o backend está rodando na porta 8000
2. Se você executou o script `fix_vite_config.js` para corrigir a configuração do proxy
3. Se você reiniciou o servidor frontend após fazer as alterações

### Erro "Manifest: Line: 1, column: 1, Syntax error"

Este erro no console do navegador pode ser resolvido executando:

```powershell
cd C:\caminho\para\cardio.ai.pro\frontend
node create_manifest.js
npm run dev
```

### Erro de CORS

Se você encontrar erros de CORS, verifique:

1. Se o backend está rodando com `--host 0.0.0.0`
2. Se você executou o script `fix_vite_config.js`

### Outros Erros

Para outros erros, consulte os arquivos de documentação específicos:

- `CORRECAO_ERRO_LOGIN.md`
- `CORRECAO_ERRO_MANIFEST.md`
- `CORRECAO_ERRO_FRONTEND_COMPLETA.md`

## Notas Importantes

- O sistema usa um banco de dados SQLite por padrão, adequado para testes
- Os dados são armazenados localmente e não são enviados para nenhum servidor externo
- Para um ambiente de produção, considere configurar um banco de dados PostgreSQL

## Próximos Passos

Após testar com sucesso o sistema, você pode:

1. Personalizar as configurações
2. Adicionar pacientes de teste
3. Fazer upload de ECGs para análise
4. Explorar os recursos de IA para análise automática

Para mais informações, consulte a documentação completa no arquivo `GUIA_INSTALACAO_COMPLETO.md`.