@echo off
REM CardioAI Pro Windows Installer Build Script
REM This script orchestrates the entire build process for creating a standalone Windows installer

echo ========================================
echo CardioAI Pro Windows Installer Builder
echo ========================================
echo.

REM Check if we're in the correct directory
if not exist "build_backend.py" (
    echo ERROR: Please run this script from the windows_installer directory
    echo Current directory should contain build_backend.py, build_frontend.py, etc.
    pause
    exit /b 1
)

REM Check for required tools
echo Checking for required tools...

REM Check Python
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.8+ from https://python.org
    pause
    exit /b 1
)

REM Check Node.js
node --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Node.js is not installed or not in PATH
    echo Please install Node.js from https://nodejs.org
    pause
    exit /b 1
)

REM Check NSIS (optional - will provide instructions if missing)
makensis /VERSION >nul 2>&1
if errorlevel 1 (
    echo WARNING: NSIS is not installed or not in PATH
    echo You can download NSIS from https://nsis.sourceforge.io/
    echo The installer script will be created but you'll need to compile it manually
    set NSIS_AVAILABLE=false
) else (
    echo NSIS found - installer will be compiled automatically
    set NSIS_AVAILABLE=true
)

echo.
echo All required tools are available. Starting build process...
echo.

REM Step 1: Build Backend
echo ========================================
echo Step 1: Building Backend Executable
echo ========================================
echo.

python build_backend.py
if errorlevel 1 (
    echo ERROR: Backend build failed
    pause
    exit /b 1
)

echo.
echo Backend build completed successfully!
echo.

REM Step 2: Build Frontend
echo ========================================
echo Step 2: Building Frontend Application
echo ========================================
echo.

python build_frontend.py
if errorlevel 1 (
    echo ERROR: Frontend build failed
    pause
    exit /b 1
)

echo.
echo Frontend build completed successfully!
echo.

REM Step 3: Create required files for installer
echo ========================================
echo Step 3: Preparing Installer Files
echo ========================================
echo.

REM Create a simple icon file (placeholder)
if not exist "cardioai.ico" (
    echo Creating placeholder icon file...
    copy nul cardioai.ico >nul
    echo Placeholder icon created. Replace cardioai.ico with actual icon file.
)

REM Create LICENSE.txt if it doesn't exist
if not exist "LICENSE.txt" (
    echo Creating LICENSE.txt...
    echo MIT License > LICENSE.txt
    echo. >> LICENSE.txt
    echo Copyright ^(c^) 2024 CardioAI Pro >> LICENSE.txt
    echo. >> LICENSE.txt
    echo Permission is hereby granted, free of charge, to any person obtaining a copy >> LICENSE.txt
    echo of this software and associated documentation files ^(the "Software"^), to deal >> LICENSE.txt
    echo in the Software without restriction, including without limitation the rights >> LICENSE.txt
    echo to use, copy, modify, merge, publish, distribute, sublicense, and/or sell >> LICENSE.txt
    echo copies of the Software, and to permit persons to whom the Software is >> LICENSE.txt
    echo furnished to do so, subject to the following conditions: >> LICENSE.txt
    echo. >> LICENSE.txt
    echo The above copyright notice and this permission notice shall be included in all >> LICENSE.txt
    echo copies or substantial portions of the Software. >> LICENSE.txt
    echo. >> LICENSE.txt
    echo THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR >> LICENSE.txt
    echo IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, >> LICENSE.txt
    echo FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE >> LICENSE.txt
    echo AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER >> LICENSE.txt
    echo LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, >> LICENSE.txt
    echo OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE >> LICENSE.txt
    echo SOFTWARE. >> LICENSE.txt
)

REM Create .env.example if it doesn't exist
if not exist ".env.example" (
    echo Creating .env.example...
    echo # CardioAI Pro Configuration > .env.example
    echo STANDALONE_MODE=true >> .env.example
    echo DATABASE_URL=sqlite+aiosqlite:///./cardioai.db >> .env.example
    echo SECRET_KEY=your-secret-key-here >> .env.example
    echo REDIS_URL= >> .env.example
    echo CELERY_BROKER_URL= >> .env.example
    echo CELERY_RESULT_BACKEND= >> .env.example
)

echo Installer files prepared successfully!
echo.

REM Step 4: Compile NSIS Installer
echo ========================================
echo Step 4: Compiling Windows Installer
echo ========================================
echo.

if "%NSIS_AVAILABLE%"=="true" (
    echo Compiling installer with NSIS...
    makensis cardioai_installer.nsi
    if errorlevel 1 (
        echo ERROR: NSIS compilation failed
        pause
        exit /b 1
    )
    echo.
    echo ✅ SUCCESS: Windows installer created successfully!
    echo.
    echo The installer file "CardioAI-Pro-Installer.exe" is ready for distribution.
) else (
    echo NSIS is not available. To create the installer:
    echo 1. Install NSIS from https://nsis.sourceforge.io/
    echo 2. Add NSIS to your PATH
    echo 3. Run: makensis cardioai_installer.nsi
    echo.
    echo All build files are ready in the windows_installer directory.
)

echo.
echo ========================================
echo Build Process Complete!
echo ========================================
echo.

if "%NSIS_AVAILABLE%"=="true" (
    echo Files created:
    echo   - CardioAI-Pro-Installer.exe ^(Windows installer^)
    echo   - cardioai-backend.exe ^(Backend executable^)
    echo   - frontend_build\ ^(Frontend files^)
    echo   - serve_frontend.py ^(Frontend server^)
    echo.
    echo The installer is ready for distribution to end users.
    echo Users can simply run CardioAI-Pro-Installer.exe to install the application.
) else (
    echo Files created:
    echo   - cardioai-backend.exe ^(Backend executable^)
    echo   - frontend_build\ ^(Frontend files^)
    echo   - serve_frontend.py ^(Frontend server^)
    echo   - cardioai_installer.nsi ^(NSIS script^)
    echo.
    echo To complete the build, install NSIS and run:
    echo   makensis cardioai_installer.nsi
)

echo.
pause
