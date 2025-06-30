# Correção do Erro de Login no CardioAI Pro

Este guia explica como corrigir o erro `404 Not Found` ao tentar fazer login no CardioAI Pro.

## Descrição do Problema

Ao tentar fazer login no frontend, o backend retorna um erro 404 (Not Found) para a rota `/api/v1/auth/login`. Isso ocorre porque:

1. O router de autenticação no backend tem um prefixo duplicado
2. Também pode faltar o pacote `email-validator` necessário para validação

## Solução Passo a Passo

### 1. Instalar o validador de email

Primeiro, instale o pacote `email-validator` que está faltando:

```powershell
pip install email-validator
# ou
pip install pydantic[email]
```

### 2. Corrigir a configuração da rota de autenticação

O problema ocorre porque há um prefixo duplicado na rota de autenticação. No arquivo `backend/app/api/v1/endpoints/auth.py`, o router já está configurado com um prefixo `/auth`, mas esse mesmo prefixo é adicionado novamente quando o router é incluído no arquivo `api.py`.

Para corrigir, edite o arquivo `backend/app/api/v1/endpoints/auth.py` e remova o prefixo do router:

```python
# Antes:
router = APIRouter(prefix="/auth", tags=["authentication"])

# Depois:
router = APIRouter(tags=["authentication"])
```

### 3. Reiniciar o servidor backend

Após fazer essas alterações, reinicie o servidor backend:

```powershell
# Pressione Ctrl+C para parar o servidor atual
# Em seguida, inicie novamente
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

## Verificação

Após aplicar essas correções, você deve ver as seguintes mensagens no log do servidor:

```
INFO:app.main:API v1 router incluído com sucesso
INFO:app.main:Rotas de compatibilidade adicionadas com sucesso
```

E ao tentar fazer login, a rota `/api/v1/auth/login` deve estar disponível e funcionar corretamente.

## Explicação Técnica

O erro ocorre porque:

1. O arquivo `app/main.py` tenta importar e usar a classe `UnauthorizedException` do módulo `app.core.exceptions`, mas essa classe não está definida nesse módulo.

2. Além disso, o arquivo `app/api/v1/endpoints/auth.py` define o router com um prefixo `/auth`:
   ```python
   router = APIRouter(prefix="/auth", tags=["authentication"])
   ```

3. No arquivo `app/api/v1/api.py`, esse router é incluído com outro prefixo `/auth`:
   ```python
   api_router.include_router(auth.router, prefix="/auth", tags=["authentication"])
   ```

4. Isso resulta em uma rota final com prefixo duplicado: `/api/v1/auth/auth/login`, em vez de `/api/v1/auth/login`.

As correções acima resolvem esses problemas, permitindo que o backend inicialize corretamente e disponibilize as rotas de autenticação.

## Próximos Passos

Após aplicar essas correções:

1. Instale o pacote `email-validator`:
   ```powershell
   pip install email-validator
   ```

2. Reinicie o servidor backend:
   ```powershell
   uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
   ```

3. Verifique se as mensagens de log mostram:
   ```
   INFO:app.main:API v1 router incluído com sucesso
   INFO:app.main:Rotas de compatibilidade adicionadas com sucesso
   ```

4. Tente fazer login no frontend novamente.

Se encontrar outros problemas, consulte a documentação ou entre em contato com a equipe de suporte.