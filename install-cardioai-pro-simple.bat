@echo off
setlocal enabledelayedexpansion

:: CardioAI Pro Windows Simple Installer (No PowerShell Required)
:: Alternative installer for systems without PowerShell

title CardioAI Pro v1.0.0 - Simple Windows Installer

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

echo [INFO] Instalador Simples - Sem PowerShell
echo [INFO] Este instalador fornece instrucoes manuais para instalacao
echo.

echo [INFO] Verificando privilegios de administrador...

:: Check if running as administrator
net session >nul 2>&1
if %errorLevel% neq 0 (
    echo [WARNING] Este instalador funciona melhor como Administrador
    echo [INFO] Mas pode continuar com usuario normal
    echo.
)

echo [OK] Continuando com a instalacao...
echo.

echo ================================================================================
echo                           INSTRUCOES DE INSTALACAO                           
echo ================================================================================
echo.
echo PASSO 1: Instalar WSL2 (Windows Subsystem for Linux)
echo ----------------------------------------
echo 1. Abra o Prompt de Comando como Administrador
echo 2. Execute: dism.exe /online /enable-feature /featurename:Microsoft-Windows-Subsystem-Linux /all /norestart
echo 3. Execute: dism.exe /online /enable-feature /featurename:VirtualMachinePlatform /all /norestart
echo 4. Reinicie o computador
echo 5. Baixe o kernel WSL2: https://aka.ms/wsl2kernel
echo 6. Execute: wsl --set-default-version 2
echo.
pause
echo.

echo PASSO 2: Instalar Ubuntu no WSL2
echo --------------------------------
echo 1. Abra a Microsoft Store
echo 2. Procure por "Ubuntu" 
echo 3. Instale "Ubuntu" (versao mais recente)
echo 4. Abra Ubuntu e configure usuario/senha
echo.
pause
echo.

echo PASSO 3: Instalar Docker Desktop
echo --------------------------------
echo 1. Baixe Docker Desktop: https://desktop.docker.com/win/main/amd64/Docker%%20Desktop%%20Installer.exe
echo 2. Execute o instalador
echo 3. Marque "Use WSL 2 instead of Hyper-V"
echo 4. Reinicie se solicitado
echo 5. Abra Docker Desktop e aguarde inicializacao
echo.
pause
echo.

echo PASSO 4: Instalar CardioAI Pro
echo ------------------------------
echo 1. Abra Ubuntu (WSL2)
echo 2. Execute os comandos:
echo    cd /tmp
echo    git clone https://github.com/drguilhermecapel/cardio.ai.pro.git
echo    cd cardio.ai.pro
echo    chmod +x install-cardioai-pro.sh
echo    ./install-cardioai-pro.sh
echo.
pause
echo.

echo ================================================================================
echo                              ACESSO AO SISTEMA                               
echo ================================================================================
echo.
echo Apos a instalacao completa:
echo.
echo URLs de Acesso:
echo - Frontend: http://localhost:3000
echo - API: http://localhost:8000
echo - Documentacao: http://localhost:8000/docs
echo.
echo Credenciais Padrao:
echo - Usuario: admin@cardioai.pro
echo - Senha: admin123
echo.
echo ================================================================================
echo                                 SUPORTE                                      
echo ================================================================================
echo.
echo Se encontrar problemas:
echo 1. Verifique se WSL2 esta funcionando: wsl --list --verbose
echo 2. Verifique se Docker esta rodando: docker ps
echo 3. Consulte INSTALACAO.md para troubleshooting detalhado
echo.
echo Para suporte: https://github.com/drguilhermecapel/cardio.ai.pro/issues
echo.

echo [SUCCESS] Instrucoes de instalacao exibidas!
echo [INFO] Siga os passos acima para instalar o CardioAI Pro
echo.
pause
