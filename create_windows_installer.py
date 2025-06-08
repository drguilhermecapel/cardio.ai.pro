#!/usr/bin/env python3
"""
Create Windows Installer for CardioAI Pro
Uses NSIS to create a professional Windows installer package
"""

import os
import shutil
import subprocess
import sys
from pathlib import Path


def run_command(cmd: str, cwd: str = None) -> bool:
    """Run a shell command and return success status."""
    print(f"Running: {cmd}")
    try:
        result = subprocess.run(
            cmd,
            shell=True,
            cwd=cwd,
            check=True,
            capture_output=True,
            text=True
        )
        print(f"✓ Command completed successfully")
        if result.stdout:
            print(f"Output: {result.stdout}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Command failed with exit code {e.returncode}")
        if e.stdout:
            print(f"Stdout: {e.stdout}")
        if e.stderr:
            print(f"Stderr: {e.stderr}")
        return False


def install_nsis():
    """Install NSIS if not already available."""
    print("Checking for NSIS installation...")
    
    try:
        result = subprocess.run(['which', 'makensis'], capture_output=True, text=True)
        if result.returncode == 0:
            print("✓ NSIS is already installed")
            return True
    except:
        pass
    
    print("Installing NSIS...")
    return run_command("sudo apt-get update && sudo apt-get install -y nsis")


def create_nsis_script():
    """Create NSIS installer script."""
    script_content = '''
; CardioAI Pro Windows Installer Script
; Created with NSIS (Nullsoft Scriptable Install System)

!define APPNAME "CardioAI Pro"
!define COMPANYNAME "CardioAI"
!define DESCRIPTION "Sistema de Análise de ECG com IA"
!define VERSIONMAJOR 1
!define VERSIONMINOR 0
!define VERSIONBUILD 0
!define HELPURL "https://github.com/drguilhermecapel/cardio.ai.pro"
!define UPDATEURL "https://github.com/drguilhermecapel/cardio.ai.pro"
!define ABOUTURL "https://github.com/drguilhermecapel/cardio.ai.pro"
!define INSTALLSIZE 40960

RequestExecutionLevel admin
InstallDir "$PROGRAMFILES\\${APPNAME}"
LicenseData "LICENSE.txt"
Name "${APPNAME}"
outFile "CardioAI-Pro-v${VERSIONMAJOR}.${VERSIONMINOR}.${VERSIONBUILD}-Setup.exe"

!include LogicLib.nsh

page license
page directory
page instfiles

!macro VerifyUserIsAdmin
UserInfo::GetAccountType
pop $0
${If} $0 != "admin"
    messageBox mb_iconstop "Privilégios de administrador são necessários!"
    setErrorLevel 740
    quit
${EndIf}
!macroend

function .onInit
    setShellVarContext all
    !insertmacro VerifyUserIsAdmin
functionEnd

section "install"
    setOutPath $INSTDIR
    
    ; Copy all application files
    file /r "CardioAI-Pro-Portable\\*"
    
    ; Create desktop shortcut
    createShortCut "$DESKTOP\\${APPNAME}.lnk" "$INSTDIR\\CardioAI-Pro.bat"
    
    ; Create start menu shortcuts
    createDirectory "$SMPROGRAMS\\${APPNAME}"
    createShortCut "$SMPROGRAMS\\${APPNAME}\\${APPNAME}.lnk" "$INSTDIR\\CardioAI-Pro.bat"
    createShortCut "$SMPROGRAMS\\${APPNAME}\\Uninstall.lnk" "$INSTDIR\\uninstall.exe"
    
    ; Registry information for add/remove programs
    writeRegStr HKLM "Software\\Microsoft\\Windows\\CurrentVersion\\Uninstall\\${APPNAME}" "DisplayName" "${APPNAME} - ${DESCRIPTION}"
    writeRegStr HKLM "Software\\Microsoft\\Windows\\CurrentVersion\\Uninstall\\${APPNAME}" "UninstallString" "$\\"$INSTDIR\\uninstall.exe$\\""
    writeRegStr HKLM "Software\\Microsoft\\Windows\\CurrentVersion\\Uninstall\\${APPNAME}" "QuietUninstallString" "$\\"$INSTDIR\\uninstall.exe$\\" /S"
    writeRegStr HKLM "Software\\Microsoft\\Windows\\CurrentVersion\\Uninstall\\${APPNAME}" "InstallLocation" "$\\"$INSTDIR$\\""
    writeRegStr HKLM "Software\\Microsoft\\Windows\\CurrentVersion\\Uninstall\\${APPNAME}" "DisplayIcon" "$\\"$INSTDIR\\CardioAI-Pro.bat$\\""
    writeRegStr HKLM "Software\\Microsoft\\Windows\\CurrentVersion\\Uninstall\\${APPNAME}" "Publisher" "${COMPANYNAME}"
    writeRegStr HKLM "Software\\Microsoft\\Windows\\CurrentVersion\\Uninstall\\${APPNAME}" "HelpLink" "${HELPURL}"
    writeRegStr HKLM "Software\\Microsoft\\Windows\\CurrentVersion\\Uninstall\\${APPNAME}" "URLUpdateInfo" "${UPDATEURL}"
    writeRegStr HKLM "Software\\Microsoft\\Windows\\CurrentVersion\\Uninstall\\${APPNAME}" "URLInfoAbout" "${ABOUTURL}"
    writeRegStr HKLM "Software\\Microsoft\\Windows\\CurrentVersion\\Uninstall\\${APPNAME}" "DisplayVersion" "${VERSIONMAJOR}.${VERSIONMINOR}.${VERSIONBUILD}"
    writeRegDWORD HKLM "Software\\Microsoft\\Windows\\CurrentVersion\\Uninstall\\${APPNAME}" "VersionMajor" ${VERSIONMAJOR}
    writeRegDWORD HKLM "Software\\Microsoft\\Windows\\CurrentVersion\\Uninstall\\${APPNAME}" "VersionMinor" ${VERSIONMINOR}
    writeRegDWORD HKLM "Software\\Microsoft\\Windows\\CurrentVersion\\Uninstall\\${APPNAME}" "NoModify" 1
    writeRegDWORD HKLM "Software\\Microsoft\\Windows\\CurrentVersion\\Uninstall\\${APPNAME}" "NoRepair" 1
    writeRegDWORD HKLM "Software\\Microsoft\\Windows\\CurrentVersion\\Uninstall\\${APPNAME}" "EstimatedSize" ${INSTALLSIZE}
    
    ; Create uninstaller
    writeUninstaller "$INSTDIR\\uninstall.exe"
sectionEnd

section "uninstall"
    ; Remove Start Menu launcher
    delete "$SMPROGRAMS\\${APPNAME}\\${APPNAME}.lnk"
    delete "$SMPROGRAMS\\${APPNAME}\\Uninstall.lnk"
    rmDir "$SMPROGRAMS\\${APPNAME}"
    
    ; Remove desktop shortcut
    delete "$DESKTOP\\${APPNAME}.lnk"
    
    ; Try to remove the install directory
    rmDir /r "$INSTDIR"
    
    ; Remove uninstaller information from the registry
    deleteRegKey HKLM "Software\\Microsoft\\Windows\\CurrentVersion\\Uninstall\\${APPNAME}"
sectionEnd
'''
    
    return script_content


def create_installer():
    """Create the Windows installer using NSIS."""
    print("=" * 60)
    print("CardioAI Pro - Windows Installer Creator")
    print("Creating professional Windows installer with NSIS")
    print("=" * 60)
    
    base_dir = Path(__file__).parent
    installer_dir = base_dir / "installer"
    
    if installer_dir.exists():
        shutil.rmtree(installer_dir)
    installer_dir.mkdir()
    
    print(f"Creating installer in: {installer_dir}")
    
    if not install_nsis():
        print("✗ Failed to install NSIS")
        return False
    
    portable_dir = base_dir / "CardioAI-Pro-Portable"
    if not portable_dir.exists():
        print("✗ Portable package not found! Run create_portable_package.py first")
        return False
    
    print("✓ Copying portable package...")
    shutil.copytree(portable_dir, installer_dir / "CardioAI-Pro-Portable")
    
    license_file = installer_dir / "LICENSE.txt"
    license_content = '''CardioAI Pro - Sistema de Análise de ECG com IA
Licença de Uso

Copyright (c) 2025 CardioAI

Este software é fornecido "como está", sem garantias de qualquer tipo.
O uso deste software é por sua conta e risco.

Para mais informações, visite:
https://github.com/drguilhermecapel/cardio.ai.pro
'''
    
    with open(license_file, 'w', encoding='utf-8') as f:
        f.write(license_content)
    
    icon_file = installer_dir / "cardioai.ico"
    if not icon_file.exists():
        print("Icon file not found, creating placeholder cardioai.ico")
        icon_file.touch()

    nsis_script = installer_dir / "installer.nsi"
    with open(nsis_script, 'w', encoding='utf-8') as f:
        f.write(create_nsis_script())
    
    print("✓ NSIS script created")
    
    print("Building Windows installer...")
    if not run_command(f"makensis {nsis_script}", str(installer_dir)):
        print("✗ Failed to build installer")
        return False
    
    installer_exe = installer_dir / "CardioAI-Pro-v1.0.0-Setup.exe"
    if installer_exe.exists():
        print(f"✓ Installer created successfully!")
        print(f"Installer size: {installer_exe.stat().st_size / (1024*1024):.1f} MB")
        
        final_installer = base_dir / "CardioAI-Pro-v1.0.0-Setup.exe"
        shutil.move(installer_exe, final_installer)
        print(f"Final installer: {final_installer}")
        
    else:
        print("✗ Installer creation failed")
        return False
    
    print("\n" + "=" * 60)
    print("Windows installer created successfully!")
    print("=" * 60)
    print(f"Installer file: {final_installer}")
    print(f"Installer size: {final_installer.stat().st_size / (1024*1024):.1f} MB")
    
    print("\nInstaller features:")
    print("- Professional Windows installer (NSIS)")
    print("- Desktop shortcut creation")
    print("- Start Menu entries")
    print("- Add/Remove Programs integration")
    print("- Uninstaller included")
    print("- Admin privileges handling")
    
    print("\nInstructions for user:")
    print("1. Download the installer to Windows computer")
    print("2. Right-click and 'Run as Administrator'")
    print("3. Follow the installation wizard")
    print("4. Launch from desktop shortcut or Start Menu")
    
    return True


if __name__ == "__main__":
    success = create_installer()
    sys.exit(0 if success else 1)
