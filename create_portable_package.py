#!/usr/bin/env python3
"""
Create Portable CardioAI Pro Package for Windows
Simple packaging without cross-compilation requirements
"""

import os
import shutil
import subprocess
import sys
from pathlib import Path
import zipfile


def run_command(cmd: str, cwd: str = None) -> bool:
    """Run a shell command and return success status."""
    print(f"Running: {cmd}")
    try:
        result = subprocess.run(
            cmd,
            shell=True,
            cwd=cwd,
            check=True,
            capture_output=True,
            text=True
        )
        print(f"✓ Command completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Command failed: {e}")
        return False


def create_portable_package():
    """Create a portable Windows package."""
    print("=" * 60)
    print("CardioAI Pro - Portable Package Creator")
    print("Creating Windows-ready portable application")
    print("=" * 60)
    
    base_dir = Path(__file__).parent
    package_dir = base_dir / "CardioAI-Pro-Portable"
    
    if package_dir.exists():
        shutil.rmtree(package_dir)
    package_dir.mkdir()
    
    print(f"Creating package in: {package_dir}")
    
    backend_exe = base_dir / "backend" / "dist" / "cardioai-pro-backend"
    if backend_exe.exists():
        print("✓ Copying backend executable...")
        shutil.copy2(backend_exe, package_dir / "cardioai-pro-backend.exe")
    else:
        print("✗ Backend executable not found!")
        return False
    
    backend_launcher = base_dir / "backend" / "dist" / "start-cardioai-pro.bat"
    if backend_launcher.exists():
        shutil.copy2(backend_launcher, package_dir / "start-backend.bat")
    
    frontend_dir = base_dir / "frontend"
    print("Building React frontend...")
    if not run_command("npm run build", str(frontend_dir)):
        print("✗ Failed to build frontend")
        return False
    
    frontend_dist = frontend_dir / "dist"
    if frontend_dist.exists():
        print("✓ Copying frontend files...")
        frontend_target = package_dir / "frontend"
        shutil.copytree(frontend_dist, frontend_target)
    else:
        print("✗ Frontend build not found!")
        return False
    
    electron_dir = frontend_dir / "electron"
    if electron_dir.exists():
        print("✓ Copying Electron configuration...")
        electron_target = package_dir / "electron"
        shutil.copytree(electron_dir, electron_target)
    
    launcher_script = package_dir / "CardioAI-Pro.bat"
    launcher_content = '''@echo off
title CardioAI Pro - ECG Analysis System
echo.
echo ╔══════════════════════════════════════════════════════════════╗
echo ║                    CardioAI Pro v1.0.0                      ║
echo ║              Sistema de Análise de ECG com IA               ║
echo ╚══════════════════════════════════════════════════════════════╝
echo.
echo Iniciando CardioAI Pro...
echo.

REM Start backend server in background
echo [INFO] Iniciando servidor backend...
start /B cardioai-pro-backend.exe

REM Wait for backend to start
timeout /t 3 /nobreak >nul

REM Open frontend in default browser
echo [INFO] Abrindo interface web...
start http://localhost:8000

echo.
echo ╔══════════════════════════════════════════════════════════════╗
echo ║  CardioAI Pro está rodando!                                 ║
echo ║                                                              ║
echo ║  Interface web: http://localhost:8000                       ║
echo ║                                                              ║
echo ║  Para parar o sistema, feche esta janela                    ║
echo ╚══════════════════════════════════════════════════════════════╝
echo.
echo Pressione qualquer tecla para parar o sistema...
pause >nul

REM Kill backend process when user exits
taskkill /f /im cardioai-pro-backend.exe >nul 2>&1
echo Sistema parado.
'''
    
    with open(launcher_script, 'w', encoding='utf-8') as f:
        f.write(launcher_content)
    
    readme_file = package_dir / "LEIA-ME.txt"
    readme_content = '''CardioAI Pro - Sistema de Análise de ECG com IA
================================================

COMO USAR:
1. Extraia todos os arquivos para uma pasta
2. Clique duas vezes em "CardioAI-Pro.bat"
3. Aguarde o sistema inicializar
4. A interface web abrirá automaticamente no seu navegador

REQUISITOS:
- Windows 7 ou superior
- Navegador web (Chrome, Firefox, Edge)
- Pelo menos 2GB de RAM disponível

SOLUÇÃO DE PROBLEMAS:
- Se a interface não abrir automaticamente, acesse: http://localhost:8000
- Se houver erro de "porta em uso", feche outros programas e tente novamente
- Para parar o sistema, feche a janela do prompt de comando

SUPORTE:
Para suporte técnico, entre em contato através do GitHub:
https://github.com/drguilhermecapel/cardio.ai.pro

Versão: 1.0.0
Data: Junho 2025
'''
    
    with open(readme_file, 'w', encoding='utf-8') as f:
        f.write(readme_content)
    
    zip_file = base_dir / "CardioAI-Pro-v1.0.0-Portable.zip"
    print(f"Creating zip package: {zip_file}")
    
    with zipfile.ZipFile(zip_file, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for file_path in package_dir.rglob('*'):
            if file_path.is_file():
                arcname = file_path.relative_to(package_dir.parent)
                zipf.write(file_path, arcname)
    
    print("\n" + "=" * 60)
    print("Portable package created successfully!")
    print("=" * 60)
    print(f"Package directory: {package_dir}")
    print(f"Zip file: {zip_file}")
    print(f"Package size: {zip_file.stat().st_size / (1024*1024):.1f} MB")
    
    print("\nPackage contents:")
    for item in package_dir.rglob('*'):
        if item.is_file():
            print(f"  - {item.relative_to(package_dir)}")
    
    print("\nInstructions for user:")
    print("1. Download the zip file to Windows computer")
    print("2. Extract all files to a folder")
    print("3. Double-click 'CardioAI-Pro.bat' to start")
    print("4. The web interface will open automatically")
    
    return True


if __name__ == "__main__":
    success = create_portable_package()
    sys.exit(0 if success else 1)
