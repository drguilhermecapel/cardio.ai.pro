# Correção de Erros Comuns no CardioAI Pro

Este guia explica como corrigir erros comuns que podem ocorrer ao inicializar o banco de dados do CardioAI Pro.

## 1. Erro "Attribute name 'metadata' is reserved"

### Descrição do Problema

O erro ocorre porque o modelo `ECGAnalysis` usa um campo chamado `metadata`, que é um nome reservado na API Declarativa do SQLAlchemy. Isso causa um conflito quando o SQLAlchemy tenta criar as tabelas do banco de dados.

### Solução Passo a Passo

#### Opção 1: Usando o Script de Correção Automática

1. **Crie o script de correção**:

   Navegue até a pasta do backend e crie um arquivo chamado `fix_metadata_field.py`:

   ```powershell
   cd C:\caminho\para\cardio.ai.pro\backend
   notepad fix_metadata_field.py
   ```

2. **Cole o seguinte código**:

   ```python
   #!/usr/bin/env python3
   """
   Script para corrigir o erro 'Attribute name 'metadata' is reserved' no modelo ECGAnalysis.
   Este script modifica o arquivo do modelo para renomear o campo 'metadata' para 'ecg_metadata'.
   """

   import os
   import sys
   from pathlib import Path

   def fix_metadata_field():
       """Corrige o campo metadata no modelo ECGAnalysis."""
       try:
           # Caminho para o arquivo do modelo
           model_path = Path(__file__).parent / "app" / "models" / "ecg_analysis.py"
           
           if not model_path.exists():
               print(f"❌ Arquivo do modelo não encontrado: {model_path}")
               return False
           
           # Ler o conteúdo do arquivo
           content = model_path.read_text(encoding="utf-8")
           
           # Verificar se o campo 'metadata' existe
           if "metadata = Column(JSON, nullable=True)" in content:
               # Substituir 'metadata' por 'ecg_metadata'
               new_content = content.replace(
                   "metadata = Column(JSON, nullable=True)",
                   "ecg_metadata = Column(JSON, nullable=True)  # Renomeado de 'metadata' para evitar conflito com SQLAlchemy"
               )
               
               # Salvar o arquivo modificado
               model_path.write_text(new_content, encoding="utf-8")
               print(f"✅ Campo 'metadata' renomeado para 'ecg_metadata' em {model_path}")
               return True
           else:
               print("ℹ️ O campo 'metadata' já foi corrigido ou não existe no modelo.")
               return True
               
       except Exception as e:
           print(f"❌ Erro ao corrigir o campo metadata: {e}")
           import traceback
           traceback.print_exc()
           return False

   if __name__ == "__main__":
       print("🔧 Corrigindo campo 'metadata' no modelo ECGAnalysis...")
       success = fix_metadata_field()
       sys.exit(0 if success else 1)
   ```

3. **Execute o script de correção**:

   ```powershell
   python fix_metadata_field.py
   ```

4. **Inicialize o banco de dados novamente**:

   ```powershell
   python init_database.py
   ```

#### Opção 2: Correção Manual

Se preferir corrigir manualmente, siga estes passos:

1. **Abra o arquivo do modelo**:

   ```powershell
   notepad app\models\ecg_analysis.py
   ```

2. **Localize a linha com o campo 'metadata'**:

   Procure por esta linha (aproximadamente linha 36):

   ```python
   metadata = Column(JSON, nullable=True)
   ```

3. **Substitua por**:

   ```python
   ecg_metadata = Column(JSON, nullable=True)  # Renomeado de 'metadata' para evitar conflito com SQLAlchemy
   ```

4. **Salve o arquivo e inicialize o banco de dados novamente**:

   ```powershell
   python init_database.py
   ```

### Explicação Técnica

O SQLAlchemy usa o nome `metadata` internamente como parte da API Declarativa para armazenar informações sobre tabelas, colunas e outros objetos do banco de dados. Quando você tenta usar esse nome como um campo em um modelo, ocorre um conflito.

A solução é simplesmente renomear o campo para algo que não entre em conflito com os nomes reservados do SQLAlchemy, como `ecg_metadata`.

### Verificação

Após aplicar a correção, você deve poder inicializar o banco de dados sem erros:

```powershell
python init_database.py
```

Você deverá ver a mensagem:

```
Creating database tables...
Initializing database with default data...
✅ Database initialized successfully
```

## 2. Erro "name 'List' is not defined"

### Descrição do Problema

Este erro ocorre porque o tipo `List` é usado no arquivo `exceptions.py`, mas não está sendo importado do módulo `typing`. Isso causa um erro quando o Python tenta usar o tipo `List` sem tê-lo definido.

### Solução Passo a Passo

#### Opção 1: Usando o Script de Correção Automática

1. **Crie o script de correção**:

   Navegue até a pasta do backend e crie um arquivo chamado `fix_list_import.py`:

   ```powershell
   cd C:\caminho\para\cardio.ai.pro\backend
   notepad fix_list_import.py
   ```

2. **Cole o seguinte código**:

   ```python
   #!/usr/bin/env python3
   """
   Script para corrigir o erro 'name 'List' is not defined' no arquivo exceptions.py.
   Este script adiciona a importação do tipo List do módulo typing.
   """

   import os
   import sys
   from pathlib import Path

   def fix_list_import():
       """Corrige a importação do tipo List no arquivo exceptions.py."""
       try:
           # Caminho para o arquivo de exceções
           exceptions_path = Path(__file__).parent / "app" / "core" / "exceptions.py"
           
           if not exceptions_path.exists():
               print(f"❌ Arquivo de exceções não encontrado: {exceptions_path}")
               return False
           
           # Ler o conteúdo do arquivo
           content = exceptions_path.read_text(encoding="utf-8")
           
           # Verificar se a importação de List já existe
           if "from typing import" in content and "List" not in content.split("from typing import")[1].split("\n")[0]:
               # Adicionar List à importação de typing
               new_content = content.replace(
                   "from typing import Dict, Any, Optional, Union",
                   "from typing import Dict, Any, Optional, Union, List"
               )
               
               # Salvar o arquivo modificado
               exceptions_path.write_text(new_content, encoding="utf-8")
               print(f"✅ Importação de List adicionada em {exceptions_path}")
               return True
           else:
               print("ℹ️ A importação de List já existe ou o padrão de importação é diferente.")
               return True
               
       except Exception as e:
           print(f"❌ Erro ao corrigir a importação de List: {e}")
           import traceback
           traceback.print_exc()
           return False

   if __name__ == "__main__":
       print("🔧 Corrigindo importação de List no arquivo exceptions.py...")
       success = fix_list_import()
       sys.exit(0 if success else 1)
   ```

3. **Execute o script de correção**:

   ```powershell
   python fix_list_import.py
   ```

4. **Inicialize o banco de dados novamente**:

   ```powershell
   python init_database.py
   ```

#### Opção 2: Correção Manual

Se preferir corrigir manualmente, siga estes passos:

1. **Abra o arquivo de exceções**:

   ```powershell
   notepad app\core\exceptions.py
   ```

2. **Localize a linha de importação do typing**:

   Procure por esta linha (aproximadamente linha 4):

   ```python
   from typing import Dict, Any, Optional, Union
   ```

3. **Adicione o tipo List à importação**:

   ```python
   from typing import Dict, Any, Optional, Union, List
   ```

4. **Salve o arquivo e inicialize o banco de dados novamente**:

   ```powershell
   python init_database.py
   ```

### Explicação Técnica

O Python 3.9+ permite usar tipos genéricos como `list[str]` diretamente, mas para versões anteriores ou para manter compatibilidade, é necessário importar os tipos genéricos do módulo `typing`. Neste caso, o tipo `List` é usado para tipar uma lista de strings (`List[str]`), mas não foi importado.

### Verificação

Após aplicar a correção, você deve poder inicializar o banco de dados sem erros:

```powershell
python init_database.py
```

## Próximos Passos

Após corrigir este erro, você pode continuar com a configuração do CardioAI Pro seguindo o guia principal de instalação.

Se encontrar outros erros, consulte a documentação ou entre em contato com a equipe de suporte.