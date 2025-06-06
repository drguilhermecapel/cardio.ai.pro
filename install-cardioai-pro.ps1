# CardioAI Pro Windows PowerShell Installer
# Requires Windows 10+ with PowerShell 5.1+

param(
    [switch]$Quiet = $false
)

$ErrorActionPreference = "Stop"

function Write-Banner {
    Clear-Host
    Write-Host "╔══════════════════════════════════════════════════════════════════════════════╗" -ForegroundColor Cyan
    Write-Host "║                          CardioAI Pro v1.0.0                              ║" -ForegroundColor Cyan
    Write-Host "║                   Sistema de Análise de ECG com IA                       ║" -ForegroundColor Cyan
    Write-Host "║                                                                              ║" -ForegroundColor Cyan
    Write-Host "║  ✓ Análise automática de ECG com IA                                      ║" -ForegroundColor Cyan
    Write-Host "║  ✓ Compliance médico ANVISA/FDA                                          ║" -ForegroundColor Cyan
    Write-Host "║  ✓ Interface web responsiva                                              ║" -ForegroundColor Cyan
    Write-Host "║  ✓ API REST completa                                                     ║" -ForegroundColor Cyan
    Write-Host "║  ✓ Segurança LGPD/HIPAA                                                  ║" -ForegroundColor Cyan
    Write-Host "╚══════════════════════════════════════════════════════════════════════════════╝" -ForegroundColor Cyan
    Write-Host ""
}

function Test-Administrator {
    $currentUser = [Security.Principal.WindowsIdentity]::GetCurrent()
    $principal = New-Object Security.Principal.WindowsPrincipal($currentUser)
    return $principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
}

function Test-WindowsVersion {
    $osVersion = [System.Environment]::OSVersion.Version
    if ($osVersion.Major -lt 10) {
        Write-Host "[ERROR] Windows 10 ou superior é necessário" -ForegroundColor Red
        Write-Host "[INFO] Versão atual: Windows $($osVersion.Major).$($osVersion.Minor)" -ForegroundColor Yellow
        return $false
    }
    Write-Host "[OK] Windows $($osVersion.Major).$($osVersion.Minor) detectado" -ForegroundColor Green
    return $true
}

function Install-WSL2 {
    Write-Host "[INFO] Instalando WSL2..." -ForegroundColor Yellow
    
    try {
        # Enable WSL feature
        Write-Host "[INFO] Habilitando recurso WSL..." -ForegroundColor Blue
        dism.exe /online /enable-feature /featurename:Microsoft-Windows-Subsystem-Linux /all /norestart | Out-Null
        
        # Enable Virtual Machine Platform
        Write-Host "[INFO] Habilitando plataforma de máquina virtual..." -ForegroundColor Blue
        dism.exe /online /enable-feature /featurename:VirtualMachinePlatform /all /norestart | Out-Null
        
        # Download and install WSL2 kernel update
        Write-Host "[INFO] Baixando atualização do kernel WSL2..." -ForegroundColor Blue
        $wslUpdateUrl = "https://wslstorestorage.blob.core.windows.net/wslblob/wsl_update_x64.msi"
        $wslUpdatePath = "$env:TEMP\wsl_update_x64.msi"
        
        Invoke-WebRequest -Uri $wslUpdateUrl -OutFile $wslUpdatePath -UseBasicParsing
        
        Write-Host "[INFO] Instalando atualização do kernel WSL2..." -ForegroundColor Blue
        Start-Process msiexec.exe -ArgumentList "/i", $wslUpdatePath, "/quiet" -Wait
        Remove-Item $wslUpdatePath -ErrorAction SilentlyContinue
        
        # Set WSL2 as default
        wsl --set-default-version 2 | Out-Null
        
        Write-Host "[SUCCESS] WSL2 instalado com sucesso" -ForegroundColor Green
        return $true
    } catch {
        Write-Host "[ERROR] Falha ao instalar WSL2: $($_.Exception.Message)" -ForegroundColor Red
        return $false
    }
}

function Test-WSL2 {
    try {
        $wslStatus = wsl --status 2>&1
        if ($LASTEXITCODE -eq 0) {
            Write-Host "[OK] WSL2 está disponível" -ForegroundColor Green
            return $true
        } else {
            return $false
        }
    } catch {
        return $false
    }
}

function Install-DockerDesktop {
    Write-Host "[INFO] Baixando Docker Desktop..." -ForegroundColor Yellow
    
    try {
        $dockerUrl = "https://desktop.docker.com/win/main/amd64/Docker%20Desktop%20Installer.exe"
        $dockerPath = "$env:TEMP\DockerDesktopInstaller.exe"
        
        Invoke-WebRequest -Uri $dockerUrl -OutFile $dockerPath -UseBasicParsing
        
        Write-Host "[INFO] Instalando Docker Desktop..." -ForegroundColor Blue
        Write-Host "[INFO] Isso pode levar alguns minutos..." -ForegroundColor Yellow
        
        Start-Process $dockerPath -ArgumentList "install", "--quiet" -Wait
        
        Remove-Item $dockerPath -ErrorAction SilentlyContinue
        Write-Host "[SUCCESS] Docker Desktop instalado" -ForegroundColor Green
        return $true
    } catch {
        Write-Host "[ERROR] Falha ao instalar Docker Desktop: $($_.Exception.Message)" -ForegroundColor Red
        return $false
    }
}

function Test-DockerDesktop {
    try {
        docker --version | Out-Null
        if ($LASTEXITCODE -eq 0) {
            Write-Host "[OK] Docker Desktop está instalado" -ForegroundColor Green
            return $true
        } else {
            return $false
        }
    } catch {
        return $false
    }
}

function Start-DockerDesktop {
    $dockerPath = "${env:ProgramFiles}\Docker\Docker\Docker Desktop.exe"
    if (Test-Path $dockerPath) {
        Write-Host "[INFO] Iniciando Docker Desktop..." -ForegroundColor Yellow
        Start-Process $dockerPath -WindowStyle Hidden
        
        # Wait for Docker to be ready
        Write-Host "[INFO] Aguardando Docker Desktop iniciar..." -ForegroundColor Blue
        $timeout = 120
        $elapsed = 0
        $dockerReady = $false
        
        do {
            Start-Sleep 5
            $elapsed += 5
            try {
                docker ps | Out-Null
                if ($LASTEXITCODE -eq 0) {
                    $dockerReady = $true
                }
            } catch {
                $dockerReady = $false
            }
            
            if ($elapsed % 15 -eq 0) {
                Write-Host "[INFO] Aguardando... ($elapsed/$timeout segundos)" -ForegroundColor Blue
            }
        } while (-not $dockerReady -and $elapsed -lt $timeout)
        
        if ($dockerReady) {
            Write-Host "[SUCCESS] Docker Desktop está rodando" -ForegroundColor Green
            return $true
        } else {
            Write-Host "[WARNING] Docker Desktop demorou para iniciar" -ForegroundColor Yellow
            Write-Host "[INFO] Você pode precisar iniciar o Docker Desktop manualmente" -ForegroundColor Yellow
            return $false
        }
    } else {
        Write-Host "[ERROR] Docker Desktop não encontrado em: $dockerPath" -ForegroundColor Red
        return $false
    }
}

function Test-DockerRunning {
    try {
        docker ps | Out-Null
        if ($LASTEXITCODE -eq 0) {
            Write-Host "[OK] Docker Desktop está rodando" -ForegroundColor Green
            return $true
        } else {
            return $false
        }
    } catch {
        return $false
    }
}

function Install-Ubuntu {
    Write-Host "[INFO] Instalando Ubuntu no WSL2..." -ForegroundColor Yellow
    
    try {
        wsl --install -d Ubuntu --no-launch
        Write-Host "[SUCCESS] Ubuntu instalado no WSL2" -ForegroundColor Green
        Write-Host "[INFO] Configure o usuário Ubuntu quando solicitado" -ForegroundColor Yellow
        return $true
    } catch {
        Write-Host "[ERROR] Falha ao instalar Ubuntu: $($_.Exception.Message)" -ForegroundColor Red
        return $false
    }
}

function Test-Ubuntu {
    try {
        $ubuntuInstalled = wsl -l -v | Select-String "Ubuntu"
        if ($ubuntuInstalled) {
            Write-Host "[OK] Ubuntu WSL está instalado" -ForegroundColor Green
            return $true
        } else {
            return $false
        }
    } catch {
        return $false
    }
}

function Install-CardioAI {
    Write-Host "[INFO] Instalando CardioAI Pro..." -ForegroundColor Yellow
    
    try {
        $currentDir = Get-Location
        $windowsPath = $currentDir.Path
        $wslPath = $windowsPath -replace '^([A-Z]):', '/mnt/$1' -replace '\\', '/' | ForEach-Object { $_.ToLower() }
        
        Write-Host "[INFO] Clonando repositório no WSL2..." -ForegroundColor Blue
        $cloneCommand = "cd /tmp && git clone https://github.com/drguilhermecapel/cardio.ai.pro.git"
        wsl -d Ubuntu -e bash -c $cloneCommand
        
        Write-Host "[INFO] Executando instalador no WSL2..." -ForegroundColor Blue
        $installCommand = "cd /tmp/cardio.ai.pro && chmod +x install-cardioai-pro.sh && ./install-cardioai-pro.sh"
        wsl -d Ubuntu -e bash -c $installCommand
        
        Write-Host "[SUCCESS] CardioAI Pro instalado com sucesso!" -ForegroundColor Green
        return $true
    } catch {
        Write-Host "[ERROR] Falha ao instalar CardioAI Pro: $($_.Exception.Message)" -ForegroundColor Red
        return $false
    }
}

function Show-AccessInfo {
    Write-Host ""
    Write-Host "🎉 INSTALAÇÃO CONCLUÍDA COM SUCESSO! 🎉" -ForegroundColor Green
    Write-Host ""
    Write-Host "📋 Informações de Acesso:" -ForegroundColor White
    Write-Host "   • Frontend: http://localhost:3000" -ForegroundColor Cyan
    Write-Host "   • API: http://localhost:8000" -ForegroundColor Cyan
    Write-Host "   • Documentação: http://localhost:8000/docs" -ForegroundColor Cyan
    Write-Host "   • Admin: admin@cardioai.pro / admin123" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "🔧 Comandos Úteis (PowerShell):" -ForegroundColor White
    Write-Host "   • Ver logs: wsl -d Ubuntu -e docker-compose logs -f" -ForegroundColor Yellow
    Write-Host "   • Parar: wsl -d Ubuntu -e docker-compose down" -ForegroundColor Yellow
    Write-Host "   • Reiniciar: wsl -d Ubuntu -e docker-compose restart" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "💡 Dicas:" -ForegroundColor White
    Write-Host "   • O sistema roda no WSL2 com Docker" -ForegroundColor Gray
    Write-Host "   • Para acessar arquivos: \\wsl$\Ubuntu\tmp\cardio.ai.pro" -ForegroundColor Gray
    Write-Host "   • Para terminal Ubuntu: wsl -d Ubuntu" -ForegroundColor Gray
    Write-Host ""
}

function Show-RestartInfo {
    Write-Host ""
    Write-Host "⚠️  REINICIALIZAÇÃO NECESSÁRIA ⚠️" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "Alguns recursos foram instalados que requerem reinicialização:" -ForegroundColor White
    Write-Host "• WSL2 (Windows Subsystem for Linux)" -ForegroundColor Gray
    Write-Host "• Recursos de virtualização" -ForegroundColor Gray
    Write-Host ""
    Write-Host "Após reiniciar:" -ForegroundColor White
    Write-Host "1. Execute este instalador novamente" -ForegroundColor Cyan
    Write-Host "2. A instalação continuará automaticamente" -ForegroundColor Cyan
    Write-Host ""
}

# Main installation process
try {
    Write-Banner
    
    # Check if running as administrator
    if (-not (Test-Administrator)) {
        Write-Host "[ERROR] Execute como Administrador" -ForegroundColor Red
        Write-Host "Clique com botão direito e selecione 'Executar como administrador'" -ForegroundColor Yellow
        if (-not $Quiet) { Read-Host "Pressione Enter para sair" }
        exit 1
    }
    
    Write-Host "[INFO] Verificando pré-requisitos do Windows..." -ForegroundColor Blue
    
    # Check Windows version
    if (-not (Test-WindowsVersion)) {
        if (-not $Quiet) { Read-Host "Pressione Enter para sair" }
        exit 1
    }
    
    $needsRestart = $false
    
    # Check WSL2
    if (-not (Test-WSL2)) {
        if (Install-WSL2) {
            $needsRestart = $true
        } else {
            Write-Host "[ERROR] Falha ao instalar WSL2" -ForegroundColor Red
            if (-not $Quiet) { Read-Host "Pressione Enter para sair" }
            exit 1
        }
    }
    
    # Check Docker Desktop
    if (-not (Test-DockerDesktop)) {
        if (-not (Install-DockerDesktop)) {
            Write-Host "[ERROR] Falha ao instalar Docker Desktop" -ForegroundColor Red
            if (-not $Quiet) { Read-Host "Pressione Enter para sair" }
            exit 1
        }
        $needsRestart = $true
    }
    
    # If restart is needed, show restart info and exit
    if ($needsRestart) {
        Show-RestartInfo
        if (-not $Quiet) { Read-Host "Pressione Enter para sair" }
        exit 0
    }
    
    # Start Docker Desktop if not running
    if (-not (Test-DockerRunning)) {
        if (-not (Start-DockerDesktop)) {
            Write-Host "[WARNING] Docker Desktop pode não estar funcionando corretamente" -ForegroundColor Yellow
            Write-Host "[INFO] Tente iniciar o Docker Desktop manualmente" -ForegroundColor Yellow
        }
    }
    
    # Check Ubuntu WSL
    if (-not (Test-Ubuntu)) {
        if (-not (Install-Ubuntu)) {
            Write-Host "[ERROR] Falha ao instalar Ubuntu" -ForegroundColor Red
            if (-not $Quiet) { Read-Host "Pressione Enter para sair" }
            exit 1
        }
    }
    
    # Install CardioAI Pro
    if (Install-CardioAI) {
        Show-AccessInfo
    } else {
        Write-Host "[ERROR] Falha ao instalar CardioAI Pro" -ForegroundColor Red
        Write-Host "[INFO] Verifique se o Docker Desktop está rodando" -ForegroundColor Yellow
        Write-Host "[INFO] Tente executar o instalador novamente" -ForegroundColor Yellow
        if (-not $Quiet) { Read-Host "Pressione Enter para sair" }
        exit 1
    }
    
} catch {
    Write-Host "[ERROR] Erro durante a instalação: $($_.Exception.Message)" -ForegroundColor Red
    Write-Host "[INFO] Stack trace: $($_.ScriptStackTrace)" -ForegroundColor Gray
    if (-not $Quiet) { Read-Host "Pressione Enter para sair" }
    exit 1
}

if (-not $Quiet) {
    Read-Host "Pressione Enter para sair"
}
