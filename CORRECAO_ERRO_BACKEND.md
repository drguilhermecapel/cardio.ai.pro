# Correção de Erro no Backend do CardioAI Pro

Este documento explica como corrigir o erro de importação `UnauthorizedException` no backend do CardioAI Pro.

## Problema

Ao iniciar o backend, você pode encontrar o seguinte erro:

```
WARNING:app.main:API v1 router não encontrado: cannot import name 'UnauthorizedException' from 'app.core.exceptions' (C:\Users\User\OneDrive\Documentos\GitHub\cardio.ai.pro2\backend\app\core\exceptions.py)
```

E ao tentar fazer login, você pode receber um erro 404 Not Found para a rota `/api/v1/auth/login`.

## Solução

### 1. Adicionar a classe `UnauthorizedException` ao arquivo de exceções

Abra o arquivo `backend/app/core/exceptions.py` e adicione a seguinte classe no início do arquivo (após os imports):

```python
class UnauthorizedException(Exception):
    """Exceção para usuário não autorizado"""
    
    def __init__(self, message: str = "Não autorizado"):
        super().__init__(message)
        self.message = message
        self.status_code = 401
```

### 2. Corrigir a importação de `APIRouter` no arquivo principal

Abra o arquivo `backend/app/main.py` e adicione `APIRouter` à importação do FastAPI:

```python
from fastapi import FastAPI, APIRouter
```

### 3. Reinicie o servidor backend

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

2. Além disso, o arquivo `app/main.py` tenta usar a classe `APIRouter` sem importá-la explicitamente.

As correções acima resolvem esses dois problemas, permitindo que o backend inicialize corretamente e disponibilize as rotas de autenticação.