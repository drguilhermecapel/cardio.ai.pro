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