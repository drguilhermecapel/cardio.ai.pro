# Comandos para Testar o CardioAI Pro

Este documento fornece os comandos essenciais para testar o sistema CardioAI Pro em seu computador.

## Correções Recentes

Foram adicionadas correções para os seguintes erros:

1. **Erro de importação no backend**: Corrigido o erro `cannot import name 'UnauthorizedException'` - veja [CORRECAO_ERRO_BACKEND.md](CORRECAO_ERRO_BACKEND.md)
2. **Erro de manifest no frontend**: Corrigido o erro `Manifest: Line: 1, column: 1, Syntax error` - veja [CORRECAO_ERRO_MANIFEST.md](CORRECAO_ERRO_MANIFEST.md)
3. **Erro 404 nas rotas de login**: Corrigido o erro de acesso às rotas de autenticação - veja [GUIA_TESTE_PASSO_A_PASSO.md](GUIA_TESTE_PASSO_A_PASSO.md)

## Comandos Rápidos

### Backend

```powershell
# Navegue até a pasta do backend
cd C:\caminho\para\cardio.ai.pro\backend

# Crie e ative o ambiente virtual
python -m venv venv
.\venv\Scripts\activate

# Instale as dependências
pip install -r requirements.txt

# Inicialize o banco de dados
cd app
python init_db.py
cd ..

# Inicie o servidor backend
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### Frontend

```powershell
# Em uma nova janela de terminal, navegue até a pasta do frontend
cd C:\caminho\para\cardio.ai.pro\frontend

# Instale as dependências
npm install

# Corrija a configuração do proxy
node fix_vite_config.js

# Crie o arquivo manifest.json
node create_manifest.js

# Inicie o servidor frontend
npm run dev
```

## Credenciais de Acesso

Use as seguintes credenciais para fazer login:

- **Email**: admin@cardioai.com
- **Senha**: admin123

## Solução de Problemas Comuns

### Erro 404 ao Fazer Login

Se você encontrar um erro 404 ao tentar fazer login:

```powershell
# Verifique se o backend está rodando na porta 8000
# Corrija a configuração do proxy no frontend
cd C:\caminho\para\cardio.ai.pro\frontend
node fix_vite_config.js
npm run dev
```

### Erro de Manifest

Se você ver o erro "Manifest: Line: 1, column: 1, Syntax error" no console:

```powershell
cd C:\caminho\para\cardio.ai.pro\frontend
node create_manifest.js
npm run dev
```

### Avisos do React Router

Os avisos sobre "React Router Future Flag Warning" são apenas avisos de depreciação e não afetam o funcionamento do aplicativo. Você pode ignorá-los com segurança.

## Verificação de Funcionamento

Para verificar se o sistema está funcionando corretamente:

1. Acesse http://localhost:5173 no navegador
2. Faça login com as credenciais fornecidas
3. Navegue pelo dashboard e outras seções
4. Tente adicionar um paciente de teste
5. Tente fazer upload de um ECG de exemplo

## Próximos Passos

Após testar com sucesso o sistema básico, você pode:

1. Explorar as funcionalidades de análise de ECG
2. Testar a geração de relatórios
3. Configurar preferências do sistema

Para instruções mais detalhadas, consulte o arquivo `GUIA_TESTE_PASSO_A_PASSO.md`.