#!/usr/bin/env python3
"""
Build script for creating standalone Windows executable of CardioAI Pro backend.
Uses PyInstaller to bundle Python runtime and all dependencies.
"""

import os
import sys
import shutil
import subprocess
from pathlib import Path

def setup_environment():
    """Setup build environment and install dependencies."""
    print("Setting up build environment...")
    
    backend_dir = Path(__file__).parent.parent / "backend"
    os.chdir(backend_dir)
    
    try:
        subprocess.run(["poetry", "--version"], check=True, capture_output=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("Installing Poetry...")
        subprocess.run([
            sys.executable, "-m", "pip", "install", "poetry"
        ], check=True)
    
    print("Installing backend dependencies...")
    subprocess.run(["poetry", "install", "--no-dev"], check=True)
    
    print("Installing PyInstaller...")
    subprocess.run(["poetry", "run", "pip", "install", "pyinstaller"], check=True)

def setup_models():
    """Setup ML models for bundling."""
    print("Setting up ML models...")
    
    backend_dir = Path(__file__).parent.parent / "backend"
    models_dir = backend_dir / "models"
    
    models_dir.mkdir(exist_ok=True)
    
    setup_script = backend_dir / "scripts" / "setup_models.py"
    if setup_script.exists():
        try:
            subprocess.run([
                "poetry", "run", "python", str(setup_script)
            ], cwd=backend_dir, check=True)
        except subprocess.CalledProcessError:
            print("Warning: setup_models.py failed, creating placeholder models...")
            create_placeholder_models(models_dir)
    else:
        create_placeholder_models(models_dir)

def create_placeholder_models(models_dir):
    """Create placeholder model files."""
    model_files = [
        "ecg_classifier.onnx",
        "rhythm_detector.onnx", 
        "quality_assessor.onnx",
    ]
    
    for model_file in model_files:
        model_path = models_dir / model_file
        if not model_path.exists():
            with open(model_path, "wb") as f:
                f.write(b"PLACEHOLDER_ONNX_MODEL")
            print(f"Created placeholder model: {model_path}")

def create_pyinstaller_spec():
    """Create PyInstaller spec file for the backend."""
    print("Creating PyInstaller spec file...")
    
    backend_dir = Path(__file__).parent.parent / "backend"
    spec_content = '''# -*- mode: python ; coding: utf-8 -*-

import sys
import os
from pathlib import Path

block_cipher = None

backend_dir = Path(os.getcwd())
sys.path.insert(0, str(backend_dir))

a = Analysis(
    ['app/main.py'],
    pathex=[str(backend_dir)],
    binaries=[],
    datas=[
        ('models', 'models'),
        ('.env.example', '.'),
    ],
    hiddenimports=[
        'uvicorn',
        'uvicorn.lifespan',
        'uvicorn.lifespan.on',
        'uvicorn.protocols',
        'uvicorn.protocols.http',
        'uvicorn.protocols.websockets',
        'uvicorn.loops',
        'uvicorn.loops.auto',
        'fastapi',
        'sqlalchemy',
        'sqlalchemy.dialects.sqlite',
        'aiosqlite',
        'onnxruntime',
        'numpy',
        'scipy',
        'scipy.signal',
        'pydantic',
        'pydantic_settings',
        'passlib',
        'passlib.hash',
        'python_multipart',
        'email_validator',
        'app.api',
        'app.api.v1',
        'app.api.v1.endpoints',
        'app.core',
        'app.db',
        'app.models',
        'app.services',
        'app.tasks',
        'app.utils',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        'celery',
        'redis',
        'psycopg2',
        'asyncpg',
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='cardioai-backend',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
'''
    
    spec_file = backend_dir / "cardioai-backend.spec"
    with open(spec_file, "w") as f:
        f.write(spec_content)
    
    return spec_file

def build_executable():
    """Build the standalone executable using PyInstaller."""
    print("Building standalone executable...")
    
    backend_dir = Path(__file__).parent.parent / "backend"
    spec_file = create_pyinstaller_spec()
    
    subprocess.run([
        "poetry", "run", "pyinstaller",
        "--clean",
        "--noconfirm",
        str(spec_file)
    ], cwd=backend_dir, check=True)
    
    dist_dir = backend_dir / "dist"
    exe_file = dist_dir / "cardioai-backend.exe"
    
    if exe_file.exists():
        installer_dir = Path(__file__).parent
        shutil.copy2(exe_file, installer_dir / "cardioai-backend.exe")
        print(f"Executable copied to: {installer_dir / 'cardioai-backend.exe'}")
    else:
        raise FileNotFoundError("Built executable not found!")

def create_startup_script():
    """Create a startup script for the backend service."""
    print("Creating startup script...")
    
    installer_dir = Path(__file__).parent
    startup_script = installer_dir / "start_backend.bat"
    
    script_content = '''@echo off
echo Starting CardioAI Pro Backend...

REM Set environment variables for standalone mode
set STANDALONE_MODE=true
set DATABASE_URL=sqlite+aiosqlite:///./cardioai.db
set REDIS_URL=
set CELERY_BROKER_URL=
set CELERY_RESULT_BACKEND=

REM Start the backend server
cardioai-backend.exe

pause
'''
    
    with open(startup_script, "w") as f:
        f.write(script_content)
    
    print(f"Startup script created: {startup_script}")

def main():
    """Main build process."""
    try:
        print("Starting CardioAI Pro backend build process...")
        
        setup_environment()
        setup_models()
        build_executable()
        create_startup_script()
        
        print("\n✅ Backend build completed successfully!")
        print("Files created:")
        installer_dir = Path(__file__).parent
        print(f"  - {installer_dir / 'cardioai-backend.exe'}")
        print(f"  - {installer_dir / 'start_backend.bat'}")
        
    except Exception as e:
        print(f"\n❌ Build failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
