@echo off
setlocal enabledelayedexpansion

:: CardioAI Pro Windows Installer
:: Requires Windows 10+ with WSL2 support

title CardioAI Pro v1.0.0 - Windows Installer

echo.
echo ╔══════════════════════════════════════════════════════════════════════════════╗
echo ║                          CardioAI Pro v1.0.0                              ║
echo ║                   Sistema de Análise de ECG com IA                       ║
echo ║                                                                              ║
echo ║  ✓ Análise automática de ECG com IA                                      ║
echo ║  ✓ Compliance médico ANVISA/FDA                                          ║
echo ║  ✓ Interface web responsiva                                              ║
echo ║  ✓ API REST completa                                                     ║
echo ║  ✓ Segurança LGPD/HIPAA                                                  ║
echo ╚══════════════════════════════════════════════════════════════════════════════╝
echo.

echo [INFO] Verificando privilégios de administrador...

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

echo [OK] Privilégios de administrador confirmados

echo.
echo [INFO] Iniciando instalação avançada via PowerShell...
echo [INFO] O PowerShell irá gerenciar a instalação completa do sistema

:: Check if PowerShell is available
powershell -Command "Get-Host" >nul 2>&1
if %errorLevel% neq 0 (
    echo [ERROR] PowerShell não está disponível neste sistema
    echo [INFO] PowerShell 5.1+ é necessário para a instalação
    pause
    exit /b 1
)

:: Execute the PowerShell installer script
powershell -ExecutionPolicy Bypass -File "%~dp0install-cardioai-pro.ps1"

:: Check PowerShell script exit code
if %errorLevel% neq 0 (
    echo.
    echo [ERROR] A instalação encontrou problemas
    echo [INFO] Verifique as mensagens acima para mais detalhes
    pause
    exit /b 1
)

echo.
echo [SUCCESS] Instalação concluída com sucesso!
echo [INFO] O CardioAI Pro está pronto para uso
echo.
pause
