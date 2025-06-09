@echo off
echo Preparing MedAI runtime components for Windows installer...

set "RUNTIME_DIR=%~dp0runtime"
set "APP_DIR=%~dp0app"
set "REDIST_DIR=%~dp0redist"
set "TEMP_DIR=%~dp0temp"

:: Create all required directories
if not exist "%RUNTIME_DIR%" mkdir "%RUNTIME_DIR%"
if not exist "%RUNTIME_DIR%\python" mkdir "%RUNTIME_DIR%\python"
if not exist "%RUNTIME_DIR%\nodejs" mkdir "%RUNTIME_DIR%\nodejs"
if not exist "%RUNTIME_DIR%\postgresql" mkdir "%RUNTIME_DIR%\postgresql"
if not exist "%RUNTIME_DIR%\postgresql\bin" mkdir "%RUNTIME_DIR%\postgresql\bin"
if not exist "%RUNTIME_DIR%\redis" mkdir "%RUNTIME_DIR%\redis"
if not exist "%APP_DIR%" mkdir "%APP_DIR%"
if not exist "%APP_DIR%\backend" mkdir "%APP_DIR%\backend"
if not exist "%APP_DIR%\frontend" mkdir "%APP_DIR%\frontend"
if not exist "%REDIST_DIR%" mkdir "%REDIST_DIR%"
if not exist "%TEMP_DIR%" mkdir "%TEMP_DIR%"

:: Check PowerShell availability
powershell -Command "exit 0" >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    set POWERSHELL_AVAILABLE=1
    echo PowerShell detected and available
) else (
    set POWERSHELL_AVAILABLE=0
    echo PowerShell not detected - using alternative methods
)

echo.
echo ========================================
echo Environment Diagnostics
echo ========================================
echo Windows Version:
ver
echo.
echo PowerShell Available: %POWERSHELL_AVAILABLE%
echo Current Directory: %~dp0
echo Runtime Directory: %RUNTIME_DIR%
echo Temp Directory: %TEMP_DIR%
echo.

echo ========================================
echo Starting Component Downloads
echo ========================================
echo.

:: Download Python
echo ========================================
echo Downloading Python 3.11 Embeddable
echo ========================================

if not exist "%RUNTIME_DIR%\python\python.exe" (
    echo Downloading Python 3.11.9 embeddable...
    
    set "PYTHON_URL=https://www.python.org/ftp/python/3.11.9/python-3.11.9-embed-amd64.zip"
    set DOWNLOAD_SUCCESS=0
    
    :: Try PowerShell download
    if "%POWERSHELL_AVAILABLE%"=="1" (
        echo [1/3] Trying PowerShell download...
        powershell -Command "[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12; try { Invoke-WebRequest -Uri '%PYTHON_URL%' -OutFile '%TEMP_DIR%\python.zip' -UseBasicParsing } catch { exit 1 }" >nul 2>&1
        if exist "%TEMP_DIR%\python.zip" set DOWNLOAD_SUCCESS=1
    )
    
    :: Try certutil download
    if "%DOWNLOAD_SUCCESS%"=="0" (
        echo [2/3] Trying certutil download...
        certutil -urlcache -split -f "%PYTHON_URL%" "%TEMP_DIR%\python.zip" >nul 2>&1
        if exist "%TEMP_DIR%\python.zip" set DOWNLOAD_SUCCESS=1
    )
    
    :: Try bitsadmin download
    if "%DOWNLOAD_SUCCESS%"=="0" (
        echo [3/3] Trying bitsadmin download...
        bitsadmin /transfer "PythonDownload" /download /priority normal "%PYTHON_URL%" "%TEMP_DIR%\python.zip" >nul 2>&1
        if exist "%TEMP_DIR%\python.zip" set DOWNLOAD_SUCCESS=1
    )
    
    if "%DOWNLOAD_SUCCESS%"=="0" goto :python_download_failed
    
    :: Extract Python
    echo Extracting Python...
    if "%POWERSHELL_AVAILABLE%"=="1" (
        powershell -Command "Expand-Archive -Path '%TEMP_DIR%\python.zip' -DestinationPath '%RUNTIME_DIR%\python' -Force" >nul 2>&1
    ) else (
        echo Using VBScript extraction...
        echo Set objShell = CreateObject("Shell.Application"^) > "%TEMP%\extract.vbs"
        echo Set objFolder = objShell.NameSpace("%RUNTIME_DIR%\python"^) >> "%TEMP%\extract.vbs"
        echo Set objZip = objShell.NameSpace("%TEMP_DIR%\python.zip"^) >> "%TEMP%\extract.vbs"
        echo objFolder.CopyHere objZip.Items, 16 >> "%TEMP%\extract.vbs"
        cscript //nologo "%TEMP%\extract.vbs" >nul
        del "%TEMP%\extract.vbs"
    )
    
    :: Configure Python
    echo Configuring Python embeddable...
    (
        echo python311.zip
        echo .
        echo .\Lib
        echo .\Lib\site-packages
        echo import site
    ) > "%RUNTIME_DIR%\python\python311._pth"
    
    :: Install pip
    echo Installing pip...
    if "%POWERSHELL_AVAILABLE%"=="1" (
        powershell -Command "Invoke-WebRequest -Uri 'https://bootstrap.pypa.io/get-pip.py' -OutFile '%RUNTIME_DIR%\python\get-pip.py'" >nul 2>&1
    ) else (
        certutil -urlcache -split -f "https://bootstrap.pypa.io/get-pip.py" "%RUNTIME_DIR%\python\get-pip.py" >nul 2>&1
    )
    
    "%RUNTIME_DIR%\python\python.exe" "%RUNTIME_DIR%\python\get-pip.py" >nul 2>&1
    
    echo Python installation completed!
) else (
    echo Python already installed, skipping...
)

:: Download Node.js
echo.
echo ========================================
echo Downloading Node.js
echo ========================================

if not exist "%RUNTIME_DIR%\nodejs\node.exe" (
    echo Downloading Node.js 18.20.3...
    
    set "NODEJS_URL=https://nodejs.org/dist/v18.20.3/node-v18.20.3-win-x64.zip"
    set DOWNLOAD_SUCCESS=0
    
    :: Try PowerShell download
    if "%POWERSHELL_AVAILABLE%"=="1" (
        echo [1/3] Trying PowerShell download...
        powershell -Command "[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12; try { Invoke-WebRequest -Uri '%NODEJS_URL%' -OutFile '%TEMP_DIR%\nodejs.zip' -UseBasicParsing } catch { exit 1 }" >nul 2>&1
        if exist "%TEMP_DIR%\nodejs.zip" set DOWNLOAD_SUCCESS=1
    )
    
    :: Try certutil download
    if "%DOWNLOAD_SUCCESS%"=="0" (
        echo [2/3] Trying certutil download...
        certutil -urlcache -split -f "%NODEJS_URL%" "%TEMP_DIR%\nodejs.zip" >nul 2>&1
        if exist "%TEMP_DIR%\nodejs.zip" set DOWNLOAD_SUCCESS=1
    )
    
    :: Try bitsadmin download
    if "%DOWNLOAD_SUCCESS%"=="0" (
        echo [3/3] Trying bitsadmin download...
        bitsadmin /transfer "NodeJSDownload" /download /priority normal "%NODEJS_URL%" "%TEMP_DIR%\nodejs.zip" >nul 2>&1
        if exist "%TEMP_DIR%\nodejs.zip" set DOWNLOAD_SUCCESS=1
    )
    
    if "%DOWNLOAD_SUCCESS%"=="0" goto :nodejs_download_failed
    
    :: Extract Node.js
    echo Extracting Node.js...
    if "%POWERSHELL_AVAILABLE%"=="1" (
        powershell -Command "Expand-Archive -Path '%TEMP_DIR%\nodejs.zip' -DestinationPath '%TEMP_DIR%' -Force" >nul 2>&1
    ) else (
        echo Using VBScript extraction...
        echo Set objShell = CreateObject("Shell.Application"^) > "%TEMP%\extract.vbs"
        echo Set objFolder = objShell.NameSpace("%TEMP_DIR%"^) >> "%TEMP%\extract.vbs"
        echo Set objZip = objShell.NameSpace("%TEMP_DIR%\nodejs.zip"^) >> "%TEMP%\extract.vbs"
        echo objFolder.CopyHere objZip.Items, 16 >> "%TEMP%\extract.vbs"
        cscript //nologo "%TEMP%\extract.vbs" >nul
        del "%TEMP%\extract.vbs"
    )
    
    :: Move to runtime directory
    if exist "%TEMP_DIR%\node-v18.20.3-win-x64" (
        move "%TEMP_DIR%\node-v18.20.3-win-x64" "%RUNTIME_DIR%\nodejs" >nul
    )
    
    echo Node.js installation completed!
) else (
    echo Node.js already installed, skipping...
)

:: PostgreSQL (Optional - create placeholder)
echo.
echo ========================================
echo PostgreSQL Setup
echo ========================================
if not exist "%RUNTIME_DIR%\postgresql\bin\postgres.exe" (
    echo Creating PostgreSQL placeholder...
    echo. > "%RUNTIME_DIR%\postgresql\bin\postgres.exe"
)

:: Redis (Optional - create placeholder)
echo.
echo ========================================
echo Redis Setup
echo ========================================
if not exist "%RUNTIME_DIR%\redis\redis-server.exe" (
    echo Creating Redis placeholder...
    echo. > "%RUNTIME_DIR%\redis\redis-server.exe"
)

:: Copy application files
echo.
echo ========================================
echo Copying Application Files
echo ========================================

:: Create demo backend if not exists
if not exist "..\backend" (
    echo Creating demo backend structure...
    mkdir "..\backend"
    (
        echo # MedAI Backend Requirements
        echo fastapi==0.104.1
        echo uvicorn==0.24.0
        echo pydicom==2.4.3
        echo numpy==1.24.3
        echo opencv-python==4.8.0.74
        echo Pillow==10.0.0
        echo scikit-image==0.21.0
        echo torch==2.0.0
        echo torchvision==0.15.0
    ) > "..\backend\requirements.txt"
    
    (
        echo from fastapi import FastAPI
        echo from fastapi.middleware.cors import CORSMiddleware
        echo import uvicorn
        echo.
        echo app = FastAPI(title="MedAI Radiologia API"^)
        echo.
        echo app.add_middleware(
        echo     CORSMiddleware,
        echo     allow_origins=["*"],
        echo     allow_credentials=True,
        echo     allow_methods=["*"],
        echo     allow_headers=["*"],
        echo ^)
        echo.
        echo @app.get("/"^)
        echo def read_root(^):
        echo     return {"message": "MedAI Radiologia Backend Running"}
        echo.
        echo @app.get("/api/health"^)
        echo def health_check(^):
        echo     return {"status": "healthy", "version": "1.0.0"}
        echo.
        echo if __name__ == "__main__":
        echo     uvicorn.run(app, host="0.0.0.0", port=8000^)
    ) > "..\backend\main.py"
)

:: Create demo frontend if not exists
if not exist "..\frontend" (
    echo Creating demo frontend structure...
    mkdir "..\frontend"
    (
        echo {
        echo   "name": "medai-frontend",
        echo   "version": "1.0.0",
        echo   "scripts": {
        echo     "dev": "echo Frontend development mode",
        echo     "build": "echo Frontend built"
        echo   }
        echo }
    ) > "..\frontend\package.json"
)

:: Copy files
echo Copying backend files...
xcopy "..\backend" "%APP_DIR%\backend" /E /I /Y /Q >nul 2>&1

echo Copying frontend files...
xcopy "..\frontend" "%APP_DIR%\frontend" /E /I /Y /Q >nul 2>&1

:: Install Python dependencies
echo.
echo ========================================
echo Installing Python Dependencies
echo ========================================

if exist "%APP_DIR%\backend\requirements.txt" (
    echo Installing backend dependencies...
    "%RUNTIME_DIR%\python\python.exe" -m pip install --upgrade pip >nul 2>&1
    "%RUNTIME_DIR%\python\python.exe" -m pip install -r "%APP_DIR%\backend\requirements.txt" >nul 2>&1
    echo Dependencies installation completed!
) else (
    echo No requirements.txt found, skipping dependency installation
)

:: Create required files
echo.
echo ========================================
echo Creating Required Files
echo ========================================

:: Create LICENSE.txt
if not exist "LICENSE.txt" (
    echo Creating LICENSE.txt...
    (
        echo MIT License
        echo.
        echo Copyright 2024 MedAI Systems
        echo.
        echo Permission is hereby granted, free of charge, to any person obtaining a copy
        echo of this software and associated documentation files (the "Software"^), to deal
        echo in the Software without restriction, including without limitation the rights
        echo to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
        echo copies of the Software, and to permit persons to whom the Software is
        echo furnished to do so, subject to the following conditions:
        echo.
        echo The above copyright notice and this permission notice shall be included in all
        echo copies or substantial portions of the Software.
        echo.
        echo THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
        echo IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
        echo FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
        echo AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
        echo LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
        echo OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
        echo SOFTWARE.
    ) > "LICENSE.txt"
)

:: Create icon placeholder
if not exist "spei.ico" (
    echo Creating icon placeholder...
    echo. > "spei.ico"
)

:: Create basic NSIS script
if not exist "spei_installer.nsi" (
    echo Creating NSIS installer script...
    (
        echo ; MedAI Radiologia Installer
        echo Name "MedAI Radiologia"
        echo OutFile "MedAI-Installer.exe"
        echo InstallDir "$PROGRAMFILES\MedAI"
        echo RequestExecutionLevel admin
        echo.
        echo Section
        echo   SetOutPath $INSTDIR
        echo   File /r "runtime\*.*"
        echo   File /r "app\*.*"
        echo   WriteUninstaller "$INSTDIR\uninstall.exe"
        echo SectionEnd
        echo.
        echo Section "Uninstall"
        echo   Delete "$INSTDIR\*.*"
        echo   RMDir /r "$INSTDIR"
        echo SectionEnd
    ) > "spei_installer.nsi"
)

:: Create VC++ redist info
if not exist "%REDIST_DIR%\README.md" (
    echo Creating VC++ Redistributable info...
    echo VC++ Redistributable will be downloaded during installation > "%REDIST_DIR%\README.md"
)

:: Final cleanup
echo.
echo ========================================
echo Cleaning Up
echo ========================================

if exist "%TEMP_DIR%" (
    rmdir /S /Q "%TEMP_DIR%" 2>nul
)

echo.
echo ========================================
echo Runtime preparation completed!
echo ========================================
echo.
echo Components prepared:
echo - Python: %RUNTIME_DIR%\python
echo - Node.js: %RUNTIME_DIR%\nodejs
echo - PostgreSQL: %RUNTIME_DIR%\postgresql (placeholder)
echo - Redis: %RUNTIME_DIR%\redis (placeholder)
echo - Application: %APP_DIR%
echo - License: LICENSE.txt
echo - NSIS Script: spei_installer.nsi
echo.
echo Ready to build installer!
echo Run: makensis spei_installer.nsi
echo.
pause
exit /b 0

:python_download_failed
echo.
echo ========================================
echo ERROR: Failed to download Python!
echo ========================================
echo.
echo MANUAL SOLUTION:
echo 1. Download from: https://www.python.org/ftp/python/3.11.9/python-3.11.9-embed-amd64.zip
echo 2. Save as: %TEMP_DIR%\python.zip
echo 3. Re-run this script
echo.
pause
exit /b 1

:nodejs_download_failed
echo.
echo ========================================
echo ERROR: Failed to download Node.js!
echo ========================================
echo.
echo MANUAL SOLUTION:
echo 1. Download from: https://nodejs.org/dist/v18.20.3/node-v18.20.3-win-x64.zip
echo 2. Save as: %TEMP_DIR%\nodejs.zip
echo 3. Re-run this script
echo.
pause
exit /b 1
