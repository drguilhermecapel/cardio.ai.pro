#!/usr/bin/env python3
"""
Build script for creating standalone Windows executable of CardioAI Pro backend.
Uses PyInstaller to bundle Python runtime and all dependencies.
"""

import os
import sys
import shutil
import subprocess
import time
import traceback
import logging
import argparse
from pathlib import Path
from contextlib import contextmanager

class BuildError(Exception):
    """Custom exception with detailed context."""
    def __init__(self, message, phase=None, details=None):
        self.phase = phase
        self.details = details or {}
        super().__init__(message)

@contextmanager
def build_phase(name, progress_callback=None):
    """Context manager for build phases with detailed error tracking."""
    start_time = time.time()
    try:
        logger.info(f"Starting phase: {name}")
        print(f"🔄 {name}...")
        if progress_callback:
            progress_callback(f"Phase: {name}", 0)
        yield
        duration = time.time() - start_time
        logger.info(f"Completed phase: {name} ({duration:.2f}s)")
        print(f"✅ {name} completed ({duration:.2f}s)")
    except Exception as e:
        duration = time.time() - start_time
        logger.error(f"Failed phase: {name} after {duration:.2f}s")
        logger.error(traceback.format_exc())
        
        diagnostics = {
            'phase': name,
            'duration': duration,
            'error_type': type(e).__name__,
            'error_message': str(e),
            'traceback': traceback.format_exc(),
            'working_directory': os.getcwd(),
            'python_version': sys.version,
            'environment_vars': dict(os.environ)
        }
        
        save_diagnostic_snapshot(diagnostics)
        
        raise BuildError(f"Build failed at phase: {name}", 
                        phase=name, 
                        details=diagnostics)

def save_diagnostic_snapshot(diagnostics):
    """Save diagnostic information to file for debugging."""
    try:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        snapshot_file = f"build_error_snapshot_{timestamp}.log"
        
        with open(snapshot_file, 'w') as f:
            f.write("=" * 60 + "\n")
            f.write(f"BUILD ERROR DIAGNOSTIC SNAPSHOT\n")
            f.write(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 60 + "\n\n")
            
            for key, value in diagnostics.items():
                f.write(f"{key.upper()}:\n")
                f.write(f"{value}\n\n")
        
        print(f"📋 Diagnostic snapshot saved: {snapshot_file}")
        logger.info(f"Diagnostic snapshot saved: {snapshot_file}")
    except Exception as e:
        logger.error(f"Failed to save diagnostic snapshot: {e}")

def setup_logging(debug_mode=False):
    """Configure comprehensive logging."""
    log_level = logging.DEBUG if debug_mode else logging.INFO
    
    logs_dir = Path("logs")
    logs_dir.mkdir(exist_ok=True)
    
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s',
        handlers=[
            logging.FileHandler(logs_dir / 'build_backend.log'),
            logging.FileHandler(logs_dir / 'build_errors.log', mode='a') if not debug_mode else logging.NullHandler(),
            logging.StreamHandler() if debug_mode else logging.NullHandler()
        ]
    )
    
    return logging.getLogger(__name__)

def check_environment():
    """Comprehensive environment validation."""
    checks = []
    
    if sys.version_info < (3, 8):
        checks.append(("Python Version", "FAIL", f"Python 3.8+ required, found {sys.version}"))
    else:
        checks.append(("Python Version", "PASS", f"Python {sys.version_info.major}.{sys.version_info.minor}"))
    
    if not Path("../backend").exists():
        checks.append(("Backend Directory", "FAIL", "Backend directory not found at ../backend"))
    else:
        checks.append(("Backend Directory", "PASS", "Backend directory found"))
    
    try:
        import shutil
        free_space = shutil.disk_usage('.').free / (1024**3)  # GB
        if free_space < 2:
            checks.append(("Disk Space", "FAIL", f"Insufficient disk space: {free_space:.1f}GB (2GB required)"))
        else:
            checks.append(("Disk Space", "PASS", f"Available: {free_space:.1f}GB"))
    except Exception as e:
        checks.append(("Disk Space", "WARNING", f"Could not check disk space: {e}"))
    
    try:
        test_file = Path("test_write_permissions.tmp")
        test_file.write_text("test")
        test_file.unlink()
        checks.append(("Write Permissions", "PASS", "Write permissions verified"))
    except Exception as e:
        checks.append(("Write Permissions", "FAIL", f"No write permissions: {e}"))
    
    print("\n" + "=" * 50)
    print("ENVIRONMENT VALIDATION REPORT")
    print("=" * 50)
    
    for name, status, message in checks:
        status_icon = "✅" if status == "PASS" else "❌" if status == "FAIL" else "⚠️"
        print(f"{status_icon} {name}: {message}")
    
    print("=" * 50 + "\n")
    
    critical_failures = [check for check in checks if check[1] == "FAIL"]
    if critical_failures:
        print("❌ CRITICAL VALIDATION FAILURES DETECTED:")
        for name, _, message in critical_failures:
            print(f"   • {name}: {message}")
        print("\nPlease resolve these issues before continuing.")
        return False
    
    return True

def install_poetry_with_retry():
    """Install Poetry with retry logic and detailed error handling."""
    max_retries = 3
    retry_delay = 5
    
    for attempt in range(max_retries):
        try:
            print(f"⚠️ Poetry not found, installing... (attempt {attempt + 1}/{max_retries})")
            
            result = subprocess.run([
                sys.executable, "-m", "pip", "install", "poetry"
            ], check=True, capture_output=True, text=True, timeout=300)
            
            print("✅ Poetry installed successfully")
            logger.info(f"Poetry installation successful on attempt {attempt + 1}")
            return True
            
        except subprocess.TimeoutExpired:
            error_msg = f"Poetry installation timed out after 5 minutes (attempt {attempt + 1})"
            print(f"❌ {error_msg}")
            logger.error(error_msg)
            
            if attempt < max_retries - 1:
                print(f"⏳ Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
                continue
            else:
                print("❌ ERROR: Poetry installation failed after all retry attempts")
                print("SOLUTIONS:")
                print("1. Check your internet connection")
                print("2. Install Poetry manually: https://python-poetry.org/docs/#installation")
                print("3. Use pip install --user poetry")
                return False
                
        except subprocess.CalledProcessError as e:
            error_msg = f"Poetry installation failed with exit code {e.returncode}"
            print(f"❌ {error_msg}")
            logger.error(f"{error_msg}\nSTDOUT: {e.stdout}\nSTDERR: {e.stderr}")
            
            if attempt < max_retries - 1:
                print(f"⏳ Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
                continue
            else:
                print("❌ ERROR: Poetry installation failed after all retry attempts")
                print("SOLUTIONS:")
                print("1. Install Poetry manually: https://python-poetry.org/docs/#installation")
                print("2. Check pip is working: python -m pip --version")
                print("3. Try: python -m pip install --user poetry")
                return False
        
        except Exception as e:
            error_msg = f"Unexpected error during Poetry installation: {e}"
            print(f"❌ {error_msg}")
            logger.error(f"{error_msg}\n{traceback.format_exc()}")
            return False
    
    return False

def install_dependencies_with_progress():
    """Install dependencies with progress tracking and timeout handling."""
    print("📦 Installing backend dependencies...")
    print("This may take 5-10 minutes depending on your internet connection...")
    
    try:
        process = subprocess.Popen(
            ["poetry", "install", "--without=dev"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True
        )
        
        start_time = time.time()
        timeout = 600  # 10 minutes
        
        while True:
            output = process.stdout.readline()
            if output == '' and process.poll() is not None:
                break
            if output:
                print(f"   {output.strip()}")
                logger.debug(f"Poetry install output: {output.strip()}")
            
            if time.time() - start_time > timeout:
                process.terminate()
                raise subprocess.TimeoutExpired("poetry install", timeout)
        
        return_code = process.poll()
        if return_code != 0:
            raise subprocess.CalledProcessError(return_code, "poetry install")
        
        print("✅ Dependencies installed successfully")
        return True
        
    except subprocess.TimeoutExpired:
        print("❌ ERROR: Dependency installation timed out after 10 minutes")
        print("SOLUTIONS:")
        print("1. Check your internet connection")
        print("2. Clear Poetry cache: poetry cache clear --all .")
        print("3. Try manual installation: poetry install --no-dev")
        return False
        
    except subprocess.CalledProcessError as e:
        print(f"❌ ERROR: Failed to install dependencies (exit code: {e.returncode})")
        print("SOLUTIONS:")
        print("1. Check pyproject.toml file exists and is valid")
        print("2. Clear Poetry cache: poetry cache clear --all .")
        print("3. Delete poetry.lock and try again")
        print("4. Check Poetry configuration: poetry config --list")
        return False
        
    except Exception as e:
        print(f"❌ ERROR: Unexpected error during dependency installation: {e}")
        logger.error(f"Dependency installation error: {e}\n{traceback.format_exc()}")
        return False

def setup_environment():
    """Setup build environment and install dependencies with comprehensive error handling."""
    with build_phase("Environment Setup"):
        backend_dir = Path(__file__).parent.parent / "backend"
        os.chdir(backend_dir)
        print(f"📁 Working in: {backend_dir}")
        
        try:
            result = subprocess.run(["poetry", "--version"], check=True, capture_output=True, text=True)
            print(f"✅ Poetry found: {result.stdout.strip()}")
            logger.info(f"Poetry version: {result.stdout.strip()}")
        except (subprocess.CalledProcessError, FileNotFoundError):
            if not install_poetry_with_retry():
                raise BuildError("Poetry installation failed", phase="poetry_install")
        
        if not install_dependencies_with_progress():
            raise BuildError("Dependency installation failed", phase="dependencies")
        
        print("📦 Installing PyInstaller...")
        try:
            subprocess.run(["poetry", "run", "pip", "install", "pyinstaller"], 
                         check=True, timeout=120, capture_output=True, text=True)
            print("✅ PyInstaller installed")
        except subprocess.TimeoutExpired:
            print("❌ ERROR: PyInstaller installation timed out")
            raise BuildError("PyInstaller installation timeout", phase="pyinstaller_install")
        except subprocess.CalledProcessError as e:
            print(f"❌ ERROR: Failed to install PyInstaller: {e}")
            raise BuildError("PyInstaller installation failed", phase="pyinstaller_install")

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
    """Build the standalone executable using PyInstaller with progress monitoring."""
    with build_phase("Executable Build"):
        backend_dir = Path(__file__).parent.parent / "backend"
        spec_file = create_pyinstaller_spec()
        
        print("🔨 Building executable (this may take several minutes)...")
        print("⏳ Please wait... Progress will be shown below:")
        
        try:
            process = subprocess.Popen([
                "poetry", "run", "pyinstaller",
                "--clean",
                "--noconfirm",
                str(spec_file)
            ], cwd=backend_dir, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            
            start_time = time.time()
            timeout = 900  # 15 minutes for build
            
            while True:
                output = process.stdout.readline()
                if output == '' and process.poll() is not None:
                    break
                if output:
                    print(f"   {output.strip()}")
                    logger.debug(f"PyInstaller output: {output.strip()}")
                
                if time.time() - start_time > timeout:
                    process.terminate()
                    raise subprocess.TimeoutExpired("pyinstaller", timeout)
            
            return_code = process.poll()
            if return_code != 0:
                raise subprocess.CalledProcessError(return_code, "pyinstaller")
            
            build_time = time.time() - start_time
            print(f"✅ Executable built successfully in {build_time:.1f} seconds")
            
        except subprocess.TimeoutExpired:
            print("❌ ERROR: Executable build timed out after 15 minutes")
            print("SOLUTIONS:")
            print("1. Check available disk space")
            print("2. Close other applications to free memory")
            print("3. Try building with --debug flag for more details")
            raise BuildError("Build timeout", phase="executable_build")
            
        except subprocess.CalledProcessError as e:
            print(f"❌ ERROR: PyInstaller build failed (exit code: {e.returncode})")
            print("SOLUTIONS:")
            print("1. Check the error messages above")
            print("2. Verify all dependencies are installed")
            print("3. Try: poetry run pyinstaller --debug")
            raise BuildError("PyInstaller build failed", phase="executable_build")
        
        dist_dir = backend_dir / "dist"
        exe_file = dist_dir / "cardioai-backend"
        exe_file_windows = dist_dir / "cardioai-backend.exe"
        
        installer_dir = Path(__file__).parent
        
        if exe_file.exists():
            shutil.copy2(exe_file, installer_dir / "cardioai-backend.exe")
            exe_size = (installer_dir / "cardioai-backend.exe").stat().st_size / (1024*1024)  # MB
            print(f"✅ Executable copied to: {installer_dir / 'cardioai-backend.exe'} ({exe_size:.1f} MB)")
        elif exe_file_windows.exists():
            shutil.copy2(exe_file_windows, installer_dir / "cardioai-backend.exe")
            exe_size = (installer_dir / "cardioai-backend.exe").stat().st_size / (1024*1024)  # MB
            print(f"✅ Executable copied to: {installer_dir / 'cardioai-backend.exe'} ({exe_size:.1f} MB)")
        else:
            print(f"❌ ERROR: Built executable not found!")
            print(f"Checked locations:")
            print(f"  - {exe_file}")
            print(f"  - {exe_file_windows}")
            print(f"Available files in dist directory:")
            if dist_dir.exists():
                for file in dist_dir.iterdir():
                    print(f"  - {file}")
            else:
                print("  - dist directory does not exist")
            raise BuildError("Executable not found after build", phase="executable_deployment")

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
    """Main build function with comprehensive error handling."""
    parser = argparse.ArgumentParser(description='Build CardioAI Pro Backend for Windows')
    parser.add_argument('--debug', '-d', action='store_true', 
                        help='Enable debug mode with verbose output')
    parser.add_argument('--log-level', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                        default='INFO', help='Set logging level')
    args = parser.parse_args()
    
    global logger
    logger = setup_logging(args.debug)
    
    print("🚀 Building CardioAI Pro Backend for Windows...")
    print("=" * 50)
    
    if args.debug:
        print("🐛 DEBUG MODE ENABLED - Verbose output active")
        print(f"🐛 Log level: {args.log_level}")
        print(f"🐛 Python version: {sys.version}")
        print(f"🐛 Working directory: {os.getcwd()}")
        print("=" * 50)
    
    try:
        with build_phase("Environment Validation"):
            if not check_environment():
                raise BuildError("Environment validation failed", phase="validation")
        
        setup_environment()
        
        with build_phase("Models Setup"):
            setup_models()
        
        build_executable()
        
        with build_phase("Startup Script Creation"):
            create_startup_script()
        
        print("\n🎉 Backend build completed successfully!")
        print("Files created:")
        installer_dir = Path(__file__).parent
        print(f"  - {installer_dir / 'cardioai-backend.exe'}")
        print(f"  - {installer_dir / 'start_backend.bat'}")
        print(f"📋 Build logs available in: logs/")
        
        if args.debug:
            print(f"🐛 Debug logs written to: logs/build_backend.log")
        
    except BuildError as e:
        print(f"\n❌ Build failed at phase: {e.phase}")
        print(f"❌ Error: {e}")
        if e.details:
            print(f"📋 Diagnostic information saved for debugging")
        logger.error(f"Build failed: {e}")
        input("Press Enter to exit...")
        sys.exit(1)
        
    except Exception as e:
        print(f"\n❌ Unexpected build failure: {e}")
        logger.error(f"Unexpected error: {e}\n{traceback.format_exc()}")
        
        emergency_diagnostics = {
            'error_type': type(e).__name__,
            'error_message': str(e),
            'traceback': traceback.format_exc(),
            'working_directory': os.getcwd(),
            'python_version': sys.version
        }
        save_diagnostic_snapshot(emergency_diagnostics)
        
        input("Press Enter to exit...")
        sys.exit(1)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n⚠️ Build interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        sys.exit(1)
