#!/usr/bin/env python3
"""
Script para criar um favicon simples para o CardioAI Pro.
Este script gera um ícone com um coração vermelho em fundo branco.
"""

import os
import sys
from pathlib import Path

try:
    from PIL import Image, ImageDraw
except ImportError:
    print("Instalando dependências necessárias...")
    os.system("pip install pillow")
    from PIL import Image, ImageDraw

def create_favicon():
    """Cria um favicon simples com um coração."""
    try:
        # Caminho para o diretório static
        static_dir = Path(__file__).parent / "app" / "static"
        static_dir.mkdir(exist_ok=True, parents=True)
        
        favicon_path = static_dir / "favicon.ico"
        
        # Criar uma imagem 32x32 com fundo branco
        img = Image.new('RGBA', (32, 32), color=(255, 255, 255, 0))
        draw = ImageDraw.Draw(img)
        
        # Desenhar um coração simples
        heart_color = (220, 20, 60)  # Vermelho
        
        # Coordenadas para um coração simples
        heart = [
            (16, 8),   # Topo
            (20, 4),   # Superior direito
            (24, 8),   # Direito
            (16, 24),  # Base
            (8, 8),    # Esquerdo
            (12, 4),   # Superior esquerdo
            (16, 8)    # Volta ao topo
        ]
        
        # Desenhar o contorno e preencher
        draw.polygon(heart, fill=heart_color)
        
        # Salvar como .ico
        img.save(favicon_path, format='ICO')
        
        print(f"✅ Favicon criado com sucesso em {favicon_path}")
        return True
            
    except Exception as e:
        print(f"❌ Erro ao criar favicon: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🔧 Criando favicon para o CardioAI Pro...")
    success = create_favicon()
    sys.exit(0 if success else 1)