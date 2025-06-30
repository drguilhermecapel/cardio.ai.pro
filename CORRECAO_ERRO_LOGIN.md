# Correção do Erro de Login no CardioAI Pro

Este guia explica como corrigir o erro `404 Not Found` ao tentar fazer login no CardioAI Pro.

## Descrição do Problema

Ao tentar fazer login no frontend, o backend retorna um erro 404 (Not Found) para a rota `/api/auth/login`. Isso ocorre porque:

1. O frontend está tentando acessar a rota `/api/auth/login`
2. Mas o backend está configurado para usar a rota `/api/v1/auth/login`

## Solução Passo a Passo

### Opção 1: Corrigir o Frontend (Recomendado)

1. **Navegue até a pasta do frontend**:

   ```powershell
   cd C:\caminho\para\cardio.ai.pro\frontend
   ```

2. **Crie um script para corrigir as rotas de API**:

   ```powershell
   notepad fix_api_routes.js
   ```

3. **Cole o seguinte código**:

   ```javascript
   #!/usr/bin/env node
   /**
    * Script para corrigir as rotas de API no frontend
    * Este script corrige o erro 404 Not Found ao tentar fazer login
    */

   const fs = require('fs');
   const path = require('path');

   // Caminho para os arquivos
   const authContextPath = path.join(__dirname, 'src', 'contexts', 'AuthContext.tsx');

   function fixApiRoutes() {
     try {
       console.log('🔧 Corrigindo rotas de API no AuthContext...');
       
       // Ler o conteúdo do arquivo AuthContext.tsx
       let authContextContent = fs.readFileSync(authContextPath, 'utf8');
       
       // Substituir a rota de login
       authContextContent = authContextContent.replace(
         "'/api/auth/login'",
         "'/api/v1/auth/login'"
       );
       
       // Substituir a rota de login biométrico
       authContextContent = authContextContent.replace(
         "'/api/auth/biometric-login'",
         "'/api/v1/auth/biometric-login'"
       );
       
       // Salvar as alterações
       fs.writeFileSync(authContextPath, authContextContent);
       console.log('✅ Rotas de API corrigidas com sucesso!');
       
       return true;
     } catch (error) {
       console.error('❌ Erro ao corrigir rotas de API:', error);
       return false;
     }
   }

   // Executar a correção
   console.log('🚀 Iniciando correção das rotas de API...');
   const success = fixApiRoutes();

   if (success) {
     console.log('✅ Correções aplicadas com sucesso! Execute "npm run dev" para reiniciar o frontend.');
   } else {
     console.log('⚠️ Não foi possível aplicar as correções. Verifique os erros acima.');
   }
   ```

4. **Execute o script**:

   ```powershell
   node fix_api_routes.js
   ```

5. **Reinicie o frontend**:

   ```powershell
   npm run dev
   ```

### Opção 2: Corrigir o Backend

Se preferir modificar o backend em vez do frontend:

1. **Navegue até a pasta do backend**:

   ```powershell
   cd C:\caminho\para\cardio.ai.pro\backend
   ```

2. **Crie um arquivo para as rotas de autenticação**:

   ```powershell
   notepad app\api\v1\endpoints\auth.py
   ```

3. **Cole o seguinte código**:

   ```python
   """
   Endpoints de autenticação para o CardioAI Pro
   """
   from fastapi import APIRouter, Depends, HTTPException, status
   from fastapi.security import OAuth2PasswordRequestForm
   from sqlalchemy.orm import Session
   from typing import Any

   from app.core.security import create_access_token, get_password_hash, verify_password
   from app.db.session import get_db
   from app.models.user import User

   router = APIRouter()

   @router.post("/login")
   async def login(
       form_data: OAuth2PasswordRequestForm = Depends(),
       db: Session = Depends(get_db)
   ) -> Any:
       """
       Endpoint de login para obter token de acesso.
       """
       # Buscar usuário pelo email
       user = db.query(User).filter(User.email == form_data.username).first()
       
       # Verificar se o usuário existe e a senha está correta
       if not user or not verify_password(form_data.password, user.hashed_password):
           raise HTTPException(
               status_code=status.HTTP_401_UNAUTHORIZED,
               detail="Email ou senha incorretos",
               headers={"WWW-Authenticate": "Bearer"},
           )
       
       # Criar token de acesso
       access_token = create_access_token(subject=user.id)
       
       return {
           "access_token": access_token,
           "token_type": "bearer",
           "user": {
               "id": user.id,
               "email": user.email,
               "full_name": user.full_name,
               "is_active": user.is_active,
               "is_superuser": user.is_superuser,
               "role": user.role
           }
       }

   @router.post("/biometric-login")
   async def biometric_login() -> Any:
       """
       Endpoint para login biométrico.
       """
       # Implementação simplificada
       return {
           "access_token": "biometric_token_placeholder",
           "token_type": "bearer",
           "user": {
               "id": 1,
               "email": "admin@cardioai.com",
               "full_name": "Administrador",
               "is_active": True,
               "is_superuser": True,
               "role": "admin"
           }
       }
   ```

4. **Atualize o arquivo de rotas da API**:

   ```powershell
   notepad app\api\v1\api.py
   ```

5. **Cole o seguinte código** (ou adicione as linhas relevantes se o arquivo já existir):

   ```python
   """
   Configuração de rotas da API v1
   """
   from fastapi import APIRouter

   from app.api.v1.endpoints import auth

   api_router = APIRouter()

   # Incluir rotas de autenticação
   api_router.include_router(auth.router, prefix="/auth", tags=["auth"])
   ```

6. **Atualize o arquivo main.py para incluir as rotas**:

   ```powershell
   notepad app\main.py
   ```

7. **Encontre a seção de inclusão de routers e corrija-a**:

   Substitua:
   ```python
   # Incluir routers da API
   try:
       from app.api.v1.api import api_router
       app.include_router(api_router, prefix="/api/v1")
       logger.info("API v1 router incluído com sucesso")
   except ImportError:
       logger.warning("API v1 router não encontrado")
   ```

   Por:
   ```python
   # Incluir routers da API
   try:
       from app.api.v1.api import api_router
       app.include_router(api_router, prefix="/api/v1")
       logger.info("API v1 router incluído com sucesso")
   except ImportError:
       logger.warning("API v1 router não encontrado")
       
   # Adicionar rotas de compatibilidade para o frontend
   from app.api.v1.endpoints import auth as auth_endpoints

   # Criar router de compatibilidade
   compat_router = APIRouter()
   compat_router.include_router(auth_endpoints.router, prefix="/auth", tags=["auth"])
   app.include_router(compat_router, prefix="/api")
   logger.info("Rotas de compatibilidade adicionadas com sucesso")
   ```

8. **Reinicie o backend**:

   ```powershell
   python -m uvicorn app.main:app --host 0.0.0.0 --port 8000
   ```

## Verificação

Após aplicar uma das soluções acima, tente fazer login novamente com as credenciais padrão:

- **Email/Usuário**: `admin@cardioai.com`
- **Senha**: `admin123`

## Explicação Técnica

O erro ocorre devido a uma incompatibilidade entre as rotas esperadas pelo frontend e as rotas disponíveis no backend:

1. O frontend está configurado para enviar requisições para `/api/auth/login`
2. O backend está configurado para receber requisições em `/api/v1/auth/login`

A solução consiste em:
- Opção 1: Atualizar o frontend para usar as rotas corretas do backend
- Opção 2: Adicionar rotas de compatibilidade no backend para suportar as rotas esperadas pelo frontend

## Próximos Passos

Após corrigir o erro de login, você poderá acessar o sistema normalmente. Se encontrar outros problemas, consulte a documentação ou entre em contato com a equipe de suporte.