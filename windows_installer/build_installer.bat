@echo off
REM CardioAI Pro Windows Installer Build Script
REM This script orchestrates the entire build process for creating a standalone Windows installer

echo ========================================
echo CardioAI Pro Windows Installer Builder
echo ========================================
echo.
echo NOTE: This script can be run from anywhere - it will automatically
echo navigate to the correct directory. You can double-click it or run
echo it from Command Prompt.
echo.

REM Get the directory where this script is located
set SCRIPT_DIR=%~dp0
REM Remove trailing backslash to prevent issues with paths containing parentheses
if "%SCRIPT_DIR:~-1%"=="\" set SCRIPT_DIR=%SCRIPT_DIR:~0,-1%


REM Navigate to the script's directory first
echo Navigating to installer directory: %SCRIPT_DIR%
pushd "%SCRIPT_DIR%"
if errorlevel 1 (
    echo ERROR: Failed to navigate to script directory: %SCRIPT_DIR%
    echo Current directory: %CD%
    pause
    exit /b 1
)
set PUSHED_DIR=1

REM Now verify we're in the correct directory (should have build_backend.py)
if not exist "build_backend.py" (
    echo ERROR: Script directory validation failed - build_backend.py not found
    echo Script directory: %SCRIPT_DIR%
    echo Current directory: %CD%
    echo.
    echo This script must be located in the windows_installer directory of the CardioAI Pro project.
    echo Expected files: build_backend.py, build_frontend.py, cardioai_installer.nsi
    echo.
    echo SOLUTION:
    echo 1. Make sure you downloaded the complete CardioAI Pro project
    echo 2. Ensure this script is in the windows_installer folder
    echo 3. Do not move this script to other locations
    pause
    exit /b 1
)

echo [OK] Successfully located in installer directory

REM Check for required tools
echo Checking for required tools...

REM Check Python with version verification
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.8+ from https://python.org
    pause
    exit /b 1
)

REM Verify Python version is 3.8+
for /f "tokens=2" %%i in ('python --version 2^>^&1') do set PYTHON_VERSION=%%i
echo Found Python version: %PYTHON_VERSION%
python -c "import sys; sys.exit(0 if sys.version_info >= (3, 8) else 1)" >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python version 3.8+ required, found %PYTHON_VERSION%
    echo Please upgrade Python from https://python.org
    pause
    exit /b 1
)

REM Check for Node.js - try portable version first, then system installation
set PORTABLE_NODE_DIR=portable_node
set PORTABLE_NODE=%PORTABLE_NODE_DIR%\node.exe
set PORTABLE_NPM=%PORTABLE_NODE_DIR%\npm.cmd

if exist "%PORTABLE_NODE%" (
    echo Found portable Node.js at: %PORTABLE_NODE%
    set NODE_CMD=%PORTABLE_NODE%
    set NPM_CMD=%PORTABLE_NPM%
    goto :check_node_version
)

REM Check system Node.js installation
node --version >nul 2>&1
if errorlevel 1 (
    echo.
    echo ========================================
    echo Node.js not found - downloading portable version...
    echo ========================================
    echo.
    call :download_portable_nodejs
    if errorlevel 1 (
        echo.
        echo ========================================
        echo ERROR: Failed to download portable Node.js
        echo ========================================
        echo.
        echo MANUAL SOLUTION:
        echo 1. Download Node.js LTS from: https://nodejs.org
        echo 2. During installation, check "Add to PATH"
        echo 3. Restart Command Prompt after installation
        echo 4. Run this script again
        echo.
        echo For detailed instructions, see: SOLUCAO_NODEJS.md
        echo.
        pause
        exit /b 1
    )
    set NODE_CMD=%PORTABLE_NODE%
    set NPM_CMD=%PORTABLE_NPM%
) else (
    echo Found system Node.js installation
    set NODE_CMD=node
    set NPM_CMD=npm
)

:check_node_version

REM Verify Node.js version is 16+
REM Use PowerShell to capture Node.js version to avoid batch parsing issues
for /f "tokens=*" %%i in ('"%NODE_CMD%" --version 2^>nul') do set NODE_VERSION=%%i
echo Found Node.js version: %NODE_VERSION%
REM Check if Node.js version is 16+ using cmd commands
if defined NODE_VERSION (
    if not "%NODE_VERSION%"=="unknown" (
        for /f "tokens=1 delims=." %%a in ("%NODE_VERSION:v=%") do (
            if %%a geq 16 (
                echo Node.js version check passed
            ) else (
                goto :nodejs_version_error
            )
        )
    )
)
goto :continue_after_version_check

:nodejs_version_error
if errorlevel 1 (
    echo.
    echo ========================================
    echo ERROR: Node.js version 16 or higher required, found %NODE_VERSION%
    echo ========================================
    echo.
    echo Attempting to download newer portable Node.js...
    call :download_portable_nodejs
    if errorlevel 1 (
        echo.
        echo MANUAL SOLUTION:
        echo 1. Download Node.js LTS (v18 or v20) from: https://nodejs.org
        echo 2. Uninstall old version first if needed
        echo 3. Install new version with "Add to PATH" checked
        echo 4. Run this script again
        echo.
        echo For detailed instructions, see: SOLUCAO_NODEJS.md
        echo.
        pause
        exit /b 1
    )
    set NODE_CMD=%PORTABLE_NODE%
    set NPM_CMD=%PORTABLE_NPM%
)

REM Check NSIS (optional - will provide instructions if missing)
makensis /VERSION >nul 2>&1
if errorlevel 1 (
    echo WARNING: NSIS is not installed or not in PATH
    echo You can download NSIS from https://nsis.sourceforge.io/
    echo The installer script will be created but you'll need to compile it manually
    set NSIS_AVAILABLE=false
) else (
    for /f "tokens=*" %%i in ('makensis /VERSION 2^>^&1') do set NSIS_VERSION=%%i
    echo NSIS found - version: %NSIS_VERSION%
    echo Installer will be compiled automatically
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

REM Verify backend directory exists
if not exist "..\backend" (
    echo ERROR: Backend directory not found
    echo Please ensure you're running this from the windows_installer directory
    echo and that the backend directory exists at ../backend
    pause
    exit /b 1
)

REM Check if Poetry is available before building
poetry --version >nul 2>&1
if errorlevel 1 (
    echo WARNING: Poetry not found, attempting to install...
    python -m pip install poetry
    if errorlevel 1 (
        echo ERROR: Failed to install Poetry
        echo Please install Poetry manually: https://python-poetry.org/docs/#installation
        pause
        exit /b 1
    )
)

python build_backend.py
if errorlevel 1 (
    echo ERROR: Backend build failed
    echo Check the error messages above for details
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

REM Verify frontend directory exists
if not exist "..\frontend" (
    echo ERROR: Frontend directory not found
    echo Please ensure you're running this from the windows_installer directory
    echo and that the frontend directory exists at ../frontend
    pause
    exit /b 1
)

REM Check if npm is available
"%NPM_CMD%" --version >nul 2>&1
if errorlevel 1 (
    echo.
    echo ========================================
    echo ERROR: npm not found but Node.js was detected earlier
    echo ========================================
    echo.
    echo This indicates a PATH issue or incomplete Node.js installation
    echo.
    echo SOLUTION:
    echo 1. Reinstall Node.js from https://nodejs.org
    echo 2. Make sure to check "Add to PATH" during installation
    echo 3. Restart your computer (not just Command Prompt)
    echo 4. Run this script again
    echo.
    echo For detailed instructions, see: SOLUCAO_NODEJS.md
    echo.
    pause
    exit /b 1
)

python build_frontend.py
if errorlevel 1 (
    echo ERROR: Frontend build failed
    echo Check the error messages above for details
    echo Common issues:
    echo - Missing dependencies: run 'npm install' in frontend directory
    echo - TypeScript errors: check frontend code for syntax issues
    echo - Memory issues: try 'npm run build' manually with --max-old-space-size=4096
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

REM Create a proper minimal ICO file instead of empty placeholder
if not exist "cardioai.ico" (
    echo Creating minimal valid icon file...
    REM Create a minimal 16x16 ICO file with proper header
    %SystemRoot%\System32\WindowsPowerShell\v1.0\powershell.exe -Command "Add-Type -AssemblyName System.Drawing; $bmp = New-Object System.Drawing.Bitmap(16,16); $bmp.Save('cardioai.ico', [System.Drawing.Imaging.ImageFormat]::Icon); $bmp.Dispose()"
    echo Valid icon file created. Replace with custom icon if desired.
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
    echo [SUCCESS] Windows installer created successfully!
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

REM Restore original directory if we navigated
if defined PUSHED_DIR popd

pause

REM Function to download portable Node.js
:download_portable_nodejs
echo Downloading portable Node.js v18.19.0...
set NODE_URL=https://nodejs.org/dist/v18.19.0/node-v18.19.0-win-x64.zip
set NODE_ZIP=%TEMP%\node-portable.zip

REM Create portable_node directory
if not exist "%PORTABLE_NODE_DIR%" mkdir "%PORTABLE_NODE_DIR%"

REM Download Node.js using PowerShell
%SystemRoot%\System32\WindowsPowerShell\v1.0\powershell.exe -Command "try { Invoke-WebRequest -Uri '%NODE_URL%' -OutFile '%NODE_ZIP%' -UseBasicParsing; Write-Host 'Download completed' } catch { Write-Host 'Download failed:' $_.Exception.Message; exit 1 }"
if errorlevel 1 (
    echo Failed to download Node.js
    exit /b 1
)

REM Extract Node.js using PowerShell
echo Extracting portable Node.js...
%SystemRoot%\System32\WindowsPowerShell\v1.0\powershell.exe -Command "try { Expand-Archive -Path '%NODE_ZIP%' -DestinationPath '%TEMP%\node-extract' -Force; Write-Host 'Extraction completed' } catch { Write-Host 'Extraction failed:' $_.Exception.Message; exit 1 }"
if errorlevel 1 (
    echo Failed to extract Node.js
    del "%NODE_ZIP%" >nul 2>&1
    exit /b 1
)

REM Move files to portable_node directory
echo Setting up portable Node.js...
xcopy "%TEMP%\node-extract\node-v18.19.0-win-x64\*" "%PORTABLE_NODE_DIR%\" /E /I /Y >nul
if errorlevel 1 (
    echo Failed to setup portable Node.js
    del "%NODE_ZIP%" >nul 2>&1
    rmdir /s /q "%TEMP%\node-extract" >nul 2>&1
    exit /b 1
)

REM Cleanup
del "%NODE_ZIP%" >nul 2>&1
rmdir /s /q "%TEMP%\node-extract" >nul 2>&1

echo [OK] Portable Node.js installed successfully!
echo Location: %PORTABLE_NODE_DIR%
exit /b 0
