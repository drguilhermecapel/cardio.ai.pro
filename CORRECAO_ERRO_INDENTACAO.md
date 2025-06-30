# Correção do Erro de Indentação no ECGService

Este documento explica como corrigir o erro de indentação no arquivo `ecg_service.py` do CardioAI Pro.

## Descrição do Problema

Ao tentar iniciar o backend, ocorre um erro de indentação no arquivo `backend/app/services/ecg_service.py`:

```
IndentationError: unexpected indent
```

O problema ocorre porque os métodos da classe `ECGService` estão definidos fora da classe, com indentação incorreta.

## Solução

### 1. Corrigir a estrutura da classe ECGService

O arquivo `ecg_service.py` precisa ser corrigido para definir corretamente a classe `ECGService` e seus métodos:

```python
"""ECG Analysis Service - Fixed."""
from typing import Dict, Any, Optional, List
import numpy as np
from datetime import datetime

class ECGService:
    """Service for ECG data operations."""
    
    async def get_analyses_by_patient(self, patient_id: int, limit: int = 50, offset: int = 0):
        """Recupera análises de ECG por paciente."""
        if hasattr(self, 'repository') and self.repository:
            return await self.repository.get_analyses_by_patient(patient_id, limit, offset)
        # Implementação direta se não houver repository
        from sqlalchemy import select
        from app.models.ecg_analysis import ECGAnalysis
        
        query = select(ECGAnalysis).where(ECGAnalysis.patient_id == patient_id)
        query = query.limit(limit).offset(offset)
        
        if hasattr(self, 'db'):
            result = await self.db.execute(query)
            return result.scalars().all()
        return []
        
    # ... outros métodos da classe ...
```

### 2. Reiniciar o servidor backend

Após corrigir o arquivo, reinicie o servidor backend:

```powershell
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

## Explicação Técnica

O erro ocorre porque:

1. No arquivo original, os métodos da classe `ECGService` estão definidos fora da classe, mas com indentação, o que causa um erro de sintaxe em Python.

2. A solução consiste em:
   - Definir corretamente a classe `ECGService`
   - Colocar todos os métodos dentro da classe com a indentação apropriada
   - Manter a classe `ECGAnalysisService` separada como estava originalmente

Esta correção permite que o Python interprete corretamente a estrutura de classes e métodos no arquivo.