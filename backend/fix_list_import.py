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