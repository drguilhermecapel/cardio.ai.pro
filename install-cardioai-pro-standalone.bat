@echo off
setlocal enabledelayedexpansion
title CardioAI Pro - Instalador Standalone

echo.
echo ================================================================================
echo                          CardioAI Pro v1.0.0                              
echo                   Sistema de Analise de ECG com IA                       
echo                     Instalador Standalone                   
echo ================================================================================
echo.

echo [INFO] Verificando privilegios de administrador...
net session >nul 2>&1
if %errorlevel% == 0 (
    echo [OK] Privilegios de administrador confirmados
) else (
    echo [ERRO] Este instalador requer privilegios de administrador
    echo.
    echo Por favor, execute este arquivo como administrador:
    echo 1. Clique com o botao direito neste arquivo
    echo 2. Selecione "Executar como administrador"
    echo.
    pause
    exit /b 1
)

echo.
echo [INFO] Verificando arquivo do instalador...

:: Check if installer exists in same directory
set "INSTALLER_FILE=%~dp0CardioAI-Pro-v1.0.0-Setup.exe"

if not exist "%INSTALLER_FILE%" (
    echo.
    echo [ERRO] Arquivo do instalador nao encontrado!
    echo.
    echo Certifique-se de que o arquivo "CardioAI-Pro-v1.0.0-Setup.exe"
    echo esta na mesma pasta que este instalador.
    echo.
    echo Pasta atual: %~dp0
    echo Arquivo procurado: %INSTALLER_FILE%
    echo.
    echo Arquivos na pasta atual:
    dir "%~dp0*.exe" /b 2>nul
    echo.
    pause
    exit /b 1
)

echo [OK] Arquivo do instalador encontrado: %INSTALLER_FILE%

echo.
echo [INFO] Iniciando instalacao...
echo [INFO] O instalador sera executado com privilegios de administrador
echo.

:: Execute the installer
"%INSTALLER_FILE%" /S

if %errorlevel% == 0 (
    echo.
    echo [SUCESSO] CardioAI Pro foi instalado com sucesso!
    echo.
    echo - Atalho criado na area de trabalho
    echo - Entrada adicionada ao menu Iniciar
    echo - Sistema pronto para uso
    echo.
    echo Para iniciar o CardioAI Pro:
    echo - Clique no atalho da area de trabalho, ou
    echo - Acesse: Menu Iniciar ^> CardioAI Pro
    echo.
    echo O sistema abrira automaticamente no seu navegador.
    echo Login inicial: admin / admin123
    echo.
) else (
    echo.
    echo [ERRO] Falha na instalacao (codigo: %errorlevel%)
    echo.
    echo Possiveis solucoes:
    echo 1. Verifique se tem privilegios de administrador
    echo 2. Desative temporariamente o antivirus
    echo 3. Execute o instalador manualmente: %INSTALLER_FILE%
    echo.
)

echo.
echo Pressione qualquer tecla para finalizar...
pause >nul
