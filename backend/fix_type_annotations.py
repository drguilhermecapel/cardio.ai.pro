#!/usr/bin/env python3
"""
Script para corrigir anotações de tipo incompatíveis com versões mais antigas do Python.
Este script substitui a sintaxe de união de tipos (Type | None) pela sintaxe compatível (Optional[Type]).
"""

import os
import sys
import re
from pathlib import Path

def fix_type_annotations():
    """Corrige anotações de tipo incompatíveis."""
    try:
        # Caminho para o arquivo init_db.py
        init_db_path = Path(__file__).parent / "app" / "db" / "init_db.py"
        
        if not init_db_path.exists():
            print(f"❌ Arquivo init_db.py não encontrado: {init_db_path}")
            return False
        
        # Ler o conteúdo do arquivo
        content = init_db_path.read_text(encoding="utf-8")
        
        # Verificar se a importação de Optional já existe
        if "from typing import" in content and "Optional" not in content.split("from typing import")[1].split("\n")[0]:
            # Adicionar Optional à importação de typing
            if "from typing import" in content:
                content = re.sub(
                    r"from typing import (.*)",
                    r"from typing import \1, Optional",
                    content
                )
            else:
                # Adicionar a importação se não existir
                content = "from typing import Optional\n" + content
        
        # Substituir a sintaxe de união de tipos pela sintaxe Optional
        content = re.sub(
            r"-> ([A-Za-z0-9_]+) \| None:",
            r"-> Optional[\1]:",
            content
        )
        
        # Salvar o arquivo modificado
        init_db_path.write_text(content, encoding="utf-8")
        print(f"✅ Anotações de tipo corrigidas em {init_db_path}")
        return True
            
    except Exception as e:
        print(f"❌ Erro ao corrigir anotações de tipo: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🔧 Corrigindo anotações de tipo incompatíveis...")
    success = fix_type_annotations()
    sys.exit(0 if success else 1)