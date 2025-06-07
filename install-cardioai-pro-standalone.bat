@echo off
chcp 65001 >nul
title CardioAI Pro - Instalador Standalone
color 0A

echo.
echo ╔══════════════════════════════════════════════════════════════╗
echo ║                    CardioAI Pro v1.0.0                      ║
echo ║              Sistema de Análise de ECG com IA               ║
echo ║                     Instalador Standalone                   ║
echo ╚══════════════════════════════════════════════════════════════╝
echo.

REM Check if installer exists
if not exist "CardioAI-Pro-v1.0.0-Setup.exe" (
    echo [ERRO] Arquivo do instalador não encontrado!
    echo.
    echo Certifique-se de que o arquivo "CardioAI-Pro-v1.0.0-Setup.exe"
    echo está na mesma pasta que este instalador.
    echo.
    echo Pressione qualquer tecla para sair...
    pause >nul
    exit /b 1
)

echo [INFO] Iniciando instalação do CardioAI Pro...
echo.
echo ┌──────────────────────────────────────────────────────────────┐
echo │  O instalador será executado com privilégios de administrador │
echo │  Clique em "Sim" quando solicitado pelo Windows               │
echo └──────────────────────────────────────────────────────────────┘
echo.

REM Run the installer with admin privileges
echo [INFO] Executando instalador...
powershell -Command "Start-Process 'CardioAI-Pro-v1.0.0-Setup.exe' -Verb RunAs -Wait"

REM Check if installation was successful
if %errorlevel% equ 0 (
    echo.
    echo ╔══════════════════════════════════════════════════════════════╗
    echo ║                 Instalação Concluída!                       ║
    echo ║                                                              ║
    echo ║  ✓ CardioAI Pro foi instalado com sucesso                   ║
    echo ║  ✓ Atalho criado na área de trabalho                        ║
    echo ║  ✓ Entrada criada no menu Iniciar                           ║
    echo ║                                                              ║
    echo ║  Para usar o sistema:                                       ║
    echo ║  • Clique duas vezes no atalho da área de trabalho          ║
    echo ║  • Ou acesse pelo menu Iniciar > CardioAI Pro               ║
    echo ╚══════════════════════════════════════════════════════════════╝
    echo.
    echo Pressione qualquer tecla para finalizar...
    pause >nul
) else (
    echo.
    echo ╔══════════════════════════════════════════════════════════════╗
    echo ║                    Erro na Instalação                       ║
    echo ║                                                              ║
    echo ║  A instalação não foi concluída com sucesso.                ║
    echo ║                                                              ║
    echo ║  Possíveis soluções:                                        ║
    echo ║  • Execute como administrador                                ║
    echo ║  • Verifique se há espaço suficiente em disco               ║
    echo ║  • Desative temporariamente o antivírus                     ║
    echo ║  • Feche outros programas e tente novamente                 ║
    echo ╚══════════════════════════════════════════════════════════════╝
    echo.
    echo Pressione qualquer tecla para sair...
    pause >nul
    exit /b 1
)
