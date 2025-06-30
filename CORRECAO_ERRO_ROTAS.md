# Correção do Erro de Rotas no CardioAI Pro

Este documento explica como corrigir o erro 404 nas rotas de autenticação do CardioAI Pro.

## Descrição do Problema

Ao tentar acessar a rota de login (`/api/v1/auth/login`), você recebe um erro 404 (Not Found). Isso ocorre porque há uma configuração duplicada de prefixos nas rotas de autenticação.

## Análise do Problema

Existem dois arquivos principais envolvidos:

1. `backend/app/api/v1/api.py` - Define o roteador principal da API e inclui os sub-roteadores
2. `backend/app/api/v1/endpoints/auth.py` - Define as rotas de autenticação

O problema é que:

- Em `api.py`, a linha 32 adiciona o prefixo `/auth` ao roteador de autenticação:
  ```python
  api_router.include_router(auth.router, prefix="/auth", tags=["authentication"])
  ```

- Em `auth.py`, o roteador já foi corrigido para não ter um prefixo duplicado:
  ```python
  router = APIRouter(tags=["authentication"])
  ```

No entanto, quando o frontend tenta acessar `/api/v1/auth/login`, o backend está esperando a rota em `/api/v1/auth/auth/login` devido à duplicação de prefixos.

## Solução

Existem duas maneiras de resolver este problema:

### Opção 1: Modificar o arquivo api.py (Recomendada)

Edite o arquivo `backend/app/api/v1/api.py` e remova o prefixo `/auth` da inclusão do roteador de autenticação:

```powershell
notepad C:\Users\User\OneDrive\Documentos\GitHub\cardio.ai.pro2\backend\app\api\v1\api.py
```

Altere a linha 32 de:
```python
api_router.include_router(auth.router, prefix="/auth", tags=["authentication"])
```

Para:
```python
api_router.include_router(auth.router, tags=["authentication"])
```

### Opção 2: Modificar o arquivo auth.py

Alternativamente, você pode editar o arquivo `backend/app/api/v1/endpoints/auth.py` e adicionar o prefixo `/auth` ao roteador:

```powershell
notepad C:\Users\User\OneDrive\Documentos\GitHub\cardio.ai.pro2\backend\app\api\v1\endpoints\auth.py
```

Altere a linha 19 de:
```python
router = APIRouter(tags=["authentication"])
```

Para:
```python
router = APIRouter(prefix="/auth", tags=["authentication"])
```

## Reiniciar o Servidor

Após fazer as alterações, reinicie o servidor backend:

```powershell
python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

## Explicação Técnica

O erro ocorre devido a uma configuração incorreta dos prefixos de rota no FastAPI:

1. O FastAPI permite definir prefixos de rota em diferentes níveis:
   - No roteador principal (`api_router`)
   - Nos sub-roteadores (como `auth.router`)

2. Quando um sub-roteador é incluído com um prefixo, esse prefixo é adicionado a todas as rotas definidas no sub-roteador.

3. Se o sub-roteador já tiver um prefixo definido, os prefixos são combinados, resultando em uma duplicação.

A solução remove a duplicação de prefixos, garantindo que as rotas sejam acessíveis nos caminhos corretos.