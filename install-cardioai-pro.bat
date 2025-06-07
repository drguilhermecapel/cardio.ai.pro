@echo off
setlocal enabledelayedexpansion

:: CardioAI Pro Windows Installer
:: Requires Windows 10+ with WSL2 support

title CardioAI Pro v1.0.0 - Windows Installer

echo.
echo ================================================================================
echo                          CardioAI Pro v1.0.0                              
echo                   Sistema de Analise de ECG com IA                       
echo                                                                              
echo   * Analise automatica de ECG com IA                                      
echo   * Compliance medico ANVISA/FDA                                          
echo   * Interface web responsiva                                              
echo   * API REST completa                                                     
echo   * Seguranca LGPD/HIPAA                                                  
echo ================================================================================
echo.

echo [INFO] Verificando privilegios de administrador...

:: Check if running as administrator
net session >nul 2>&1
if %errorLevel% neq 0 (
    echo [ERROR] Este instalador precisa ser executado como Administrador
    echo.
    echo Para executar como administrador:
    echo 1. Clique com o botão direito no arquivo install-cardioai-pro.bat
    echo 2. Selecione "Executar como administrador"
    echo.
    pause
    exit /b 1
)

echo [OK] Privilegios de administrador confirmados

echo.
echo [INFO] Iniciando instalacao avancada via PowerShell...
echo [INFO] O PowerShell ira gerenciar a instalacao completa do sistema

:: Check if PowerShell is available
where powershell >nul 2>&1
if %errorLevel% neq 0 (
    echo [ERROR] PowerShell nao esta disponivel neste sistema
    echo [INFO] PowerShell 5.1+ e necessario para a instalacao
    echo.
    echo Solucoes:
    echo 1. Instale o PowerShell 5.1+ do site da Microsoft
    echo 2. Ou use o Windows 10/11 que ja inclui PowerShell
    echo 3. Verifique se PowerShell esta no PATH do sistema
    echo.
    pause
    exit /b 1
)

:: Test PowerShell execution
powershell -Command "Write-Host 'PowerShell OK'" >nul 2>&1
if %errorLevel% neq 0 (
    echo [ERROR] PowerShell encontrado mas nao pode executar comandos
    echo [INFO] Verifique as politicas de execucao do PowerShell
    echo.
    echo Para corrigir, execute como Administrador:
    echo Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
    echo.
    pause
    exit /b 1
)

:: Execute the PowerShell installer script
powershell -ExecutionPolicy Bypass -File "%~dp0install-cardioai-pro.ps1"

:: Check PowerShell script exit code
if %errorLevel% neq 0 (
    echo.
    echo [ERROR] A instalacao encontrou problemas
    echo [INFO] Verifique as mensagens acima para mais detalhes
    pause
    exit /b 1
)

echo.
echo [SUCCESS] Instalacao concluida com sucesso!
echo [INFO] O CardioAI Pro esta pronto para uso
echo.
pause
