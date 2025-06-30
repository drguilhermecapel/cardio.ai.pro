# Correção do Erro de Inicialização do Banco de Dados

Este documento explica como corrigir o erro que ocorre ao inicializar o banco de dados do CardioAI Pro.

## Descrição do Problema

Ao executar o script `init_database.py`, você pode encontrar o seguinte erro:

```
Failed to create admin user: When initializing mapper Mapper[User(users)], expression 'ECGAnalysis.created_by' failed to locate a name ("name 'ECGAnalysis' is not defined").
```

Este erro ocorre porque há uma referência a um campo `created_by` no modelo `ECGAnalysis` que não está definido, mas é referenciado no modelo `User`.

## Solução

### 1. Corrigir o modelo ECGAnalysis

Abra o arquivo `backend/app/models/ecg_analysis.py` e adicione o campo `created_by` que está faltando:

```powershell
notepad C:\Users\User\OneDrive\Documentos\GitHub\cardio.ai.pro2\backend\app\models\ecg_analysis.py
```

Adicione a seguinte linha após a linha 31 (logo após `validated_by`):

```python
created_by = Column(Integer, ForeignKey("users.id"), nullable=True)
```

O trecho do código deve ficar assim:

```python
# Validação
validated = Column(Boolean, default=False)
validated_by = Column(Integer, ForeignKey("users.id"), nullable=True)
created_by = Column(Integer, ForeignKey("users.id"), nullable=True)
validated_at = Column(DateTime, nullable=True)
```

### 2. Adicionar o relacionamento no modelo ECGAnalysis

Adicione o relacionamento para o usuário que criou a análise. Procure a seção de relacionamentos (por volta da linha 43) e adicione:

```python
created_by_user = relationship("User", foreign_keys=[created_by], back_populates="analyses")
```

A seção de relacionamentos deve ficar assim:

```python
# Relacionamentos
patient = relationship("Patient", back_populates="ecg_analyses")
validator = relationship("User", foreign_keys=[validated_by])
created_by_user = relationship("User", foreign_keys=[created_by], back_populates="analyses")
validations = relationship("ECGValidation", back_populates="analysis")
```

### 3. Reinicializar o banco de dados

Após fazer essas alterações, execute novamente o script de inicialização do banco de dados:

```powershell
python init_database.py
```

## Explicação Técnica

O erro ocorre devido a uma referência circular entre os modelos `User` e `ECGAnalysis`:

1. No modelo `User`, há uma relação definida para `analyses` que aponta para `ECGAnalysis.created_by`
2. No entanto, o campo `created_by` não está definido no modelo `ECGAnalysis`

A solução adiciona o campo `created_by` e o relacionamento correspondente no modelo `ECGAnalysis`, resolvendo a referência circular.

## Próximos Passos

Após corrigir este erro, você deve ser capaz de inicializar o banco de dados com sucesso e iniciar o servidor backend:

```powershell
python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```