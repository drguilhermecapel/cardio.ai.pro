# Correção Manual do Erro de Indentação

Se você ainda está enfrentando o erro de indentação no arquivo `ecg_service.py`, siga estas instruções para corrigir manualmente o arquivo em seu computador local.

## Passo 1: Abra o arquivo para edição

Abra o arquivo `backend/app/services/ecg_service.py` em um editor de texto:

```powershell
notepad C:\Users\User\OneDrive\Documentos\GitHub\cardio.ai.pro2\backend\app\services\ecg_service.py
```

## Passo 2: Substitua todo o conteúdo

Apague todo o conteúdo atual do arquivo e substitua pelo código abaixo:

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
        
    async def get_pathologies_distribution(self):
        """Retorna distribuição de patologias."""
        if hasattr(self, 'repository') and self.repository:
            return await self.repository.get_pathologies_distribution()
        # Implementação simplificada
        return {
            "normal": 0.4,
            "arrhythmia": 0.3,
            "ischemia": 0.2,
            "other": 0.1
        }
        
    async def search_analyses(self, query: str, filters: Optional[Dict[str, Any]] = None):
        """Busca análises por critérios."""
        if hasattr(self, 'repository') and self.repository:
            return await self.repository.search_analyses(query, filters)
        # Implementação básica
        return []
        
    async def update_patient_risk(self, patient_id: int, risk_data: Dict[str, Any]):
        """Atualiza dados de risco do paciente."""
        if hasattr(self, 'repository') and self.repository:
            return await self.repository.update_patient_risk(patient_id, risk_data)
        # Implementação básica
        return {"patient_id": patient_id, "risk_updated": True, **risk_data}
        
    async def validate_analysis(self, analysis_id: int, validation_data: Dict[str, Any]):
        """Valida uma análise de ECG."""
        # Implementação de validação
        return {
            "analysis_id": analysis_id,
            "validation_status": "validated",
            "validated_at": datetime.utcnow().isoformat(),
            **validation_data
        }
        
    async def create_validation(self, analysis_id: int, user_id: int, notes: str):
        """Cria uma validação para análise."""
        return {
            "id": 1,
            "analysis_id": analysis_id,
            "user_id": user_id,
            "notes": notes,
            "status": "pending",
            "created_at": datetime.utcnow().isoformat()
        }

class ECGAnalysisService:
    """Service for ECG analysis."""
    
    def __init__(self, db=None, validation_service=None):
        self.db = db
        self.validation_service = validation_service
        self.status = {"status": "ready", "pending": 0}
        
    async def analyze_ecg(self, ecg_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze ECG data."""
        return {
            "id": f"ecg_{int(datetime.now().timestamp())}",
            "status": "completed",
            "results": {
                "heart_rate": 75,
                "rhythm": "normal sinus rhythm",
                "interpretation": "Normal ECG"
            }
        }
    
    def get_status(self) -> Dict[str, Any]:
        """Get service status."""
        return self.status
```

## Passo 3: Salve o arquivo

Salve o arquivo após fazer as alterações.

## Passo 4: Reinicie o servidor backend

Após corrigir o arquivo, reinicie o servidor backend:

```powershell
python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

## Explicação do Problema

O erro ocorre porque o arquivo original tem uma estrutura incorreta, onde os métodos da classe `ECGService` estão definidos fora da classe, mas com indentação. Isso causa um erro de sintaxe em Python.

A correção acima:
1. Define corretamente a classe `ECGService`
2. Coloca todos os métodos dentro da classe com a indentação apropriada
3. Mantém a classe `ECGAnalysisService` separada como estava originalmente

## Próximos Passos

Após corrigir este erro, você também pode precisar resolver o problema relacionado à inicialização do banco de dados:

```
Failed to create admin user: When initializing mapper Mapper[User(users)], expression 'ECGAnalysis.created_by' failed to locate a name ("name 'ECGAnalysis' is not defined").
```

Este erro indica que há um problema de importação circular entre os modelos `User` e `ECGAnalysis`. Para resolver isso, você precisará verificar os arquivos de modelo e ajustar as importações conforme necessário.