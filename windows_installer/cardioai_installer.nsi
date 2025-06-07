; CardioAI Pro Windows Installer
; NSIS Script for creating a standalone Windows installer
; This installer packages the backend executable, frontend files, and creates a desktop application

!define APPNAME "CardioAI Pro"
!define COMPANYNAME "CardioAI"
!define DESCRIPTION "Electronic Medical Record System with AI-powered ECG Analysis"
!define VERSIONMAJOR 1
!define VERSIONMINOR 0
!define VERSIONBUILD 0
!define HELPURL "https://github.com/drguilhermecapel/cardio.ai.pro"
!define UPDATEURL "https://github.com/drguilhermecapel/cardio.ai.pro/releases"
!define ABOUTURL "https://github.com/drguilhermecapel/cardio.ai.pro"
!define INSTALLSIZE 500000 ; Estimate in KB

RequestExecutionLevel admin ; Require admin rights on NT6+ (When UAC is turned on)

InstallDir "$PROGRAMFILES\${APPNAME}"

; rtf or txt file - remember if it is txt, it must be in the DOS text format (\r\n)
LicenseData "LICENSE.txt"
Name "${APPNAME}"
Icon "cardioai.ico"
outFile "CardioAI-Pro-Installer.exe"

!include LogicLib.nsh

; Just three pages - license agreement, install location, and installation
page license
page directory
Page instfiles

!macro VerifyUserIsAdmin
UserInfo::GetAccountType
pop $0
${If} $0 != "admin" ; Require admin rights on NT4+
    messageBox mb_iconstop "Administrator rights required!"
    setErrorLevel 740 ; ERROR_ELEVATION_REQUIRED
    quit
${EndIf}
!macroend

function .onInit
    setShellVarContext all
    !insertmacro VerifyUserIsAdmin
functionEnd

section "install"
    ; Files for the install directory - to build the installer, these should be in the same directory as the install script (this file)
    setOutPath $INSTDIR
    
    ; Backend executable and dependencies
    file "cardioai-backend.exe"
    file "start_backend.bat"
    
    ; Frontend files
    file /r "frontend_build"
    file "serve_frontend.py"
    
    ; Configuration files
    file /oname=".env" ".env.example"
    
    ; Create application data directory
    CreateDirectory "$APPDATA\${APPNAME}"
    CreateDirectory "$APPDATA\${APPNAME}\data"
    CreateDirectory "$APPDATA\${APPNAME}\logs"
    CreateDirectory "$APPDATA\${APPNAME}\uploads"
    
    ; Create startup script that launches both backend and frontend
    FileOpen $0 "$INSTDIR\start_cardioai.bat" w
    FileWrite $0 "@echo off$\r$\n"
    FileWrite $0 "echo Starting CardioAI Pro...$\r$\n"
    FileWrite $0 "echo.$\r$\n"
    FileWrite $0 "echo Starting backend server...$\r$\n"
    FileWrite $0 "cd /d $\"$INSTDIR$\"$\r$\n"
    FileWrite $0 "start $\"CardioAI Backend$\" /min cardioai-backend.exe$\r$\n"
    FileWrite $0 "echo Waiting for backend to start...$\r$\n"
    FileWrite $0 "timeout /t 5 /nobreak > nul$\r$\n"
    FileWrite $0 "echo Starting frontend...$\r$\n"
    FileWrite $0 "start $\"CardioAI Frontend$\" python serve_frontend.py$\r$\n"
    FileWrite $0 "echo.$\r$\n"
    FileWrite $0 "echo CardioAI Pro is starting...$\r$\n"
    FileWrite $0 "echo The application will open in your web browser shortly.$\r$\n"
    FileWrite $0 "echo.$\r$\n"
    FileWrite $0 "echo To stop CardioAI Pro, close this window.$\r$\n"
    FileWrite $0 "pause$\r$\n"
    FileClose $0
    
    ; Create stop script
    FileOpen $0 "$INSTDIR\stop_cardioai.bat" w
    FileWrite $0 "@echo off$\r$\n"
    FileWrite $0 "echo Stopping CardioAI Pro...$\r$\n"
    FileWrite $0 "taskkill /f /im cardioai-backend.exe 2>nul$\r$\n"
    FileWrite $0 "taskkill /f /im python.exe /fi $\"WINDOWTITLE eq CardioAI Frontend$\" 2>nul$\r$\n"
    FileWrite $0 "echo CardioAI Pro stopped.$\r$\n"
    FileWrite $0 "pause$\r$\n"
    FileClose $0
    
    ; Create desktop shortcut
    createShortCut "$DESKTOP\${APPNAME}.lnk" "$INSTDIR\start_cardioai.bat" "" "$INSTDIR\cardioai.ico"
    
    ; Create start menu shortcuts
    createDirectory "$SMPROGRAMS\${APPNAME}"
    createShortCut "$SMPROGRAMS\${APPNAME}\${APPNAME}.lnk" "$INSTDIR\start_cardioai.bat" "" "$INSTDIR\cardioai.ico"
    createShortCut "$SMPROGRAMS\${APPNAME}\Stop ${APPNAME}.lnk" "$INSTDIR\stop_cardioai.bat" "" "$INSTDIR\cardioai.ico"
    createShortCut "$SMPROGRAMS\${APPNAME}\Uninstall ${APPNAME}.lnk" "$INSTDIR\uninstall.exe" "" ""
    
    ; Registry information for add/remove programs
    WriteRegStr HKLM "Software\Microsoft\Windows\CurrentVersion\Uninstall\${APPNAME}" "DisplayName" "${APPNAME} - ${DESCRIPTION}"
    WriteRegStr HKLM "Software\Microsoft\Windows\CurrentVersion\Uninstall\${APPNAME}" "UninstallString" "$\"$INSTDIR\uninstall.exe$\""
    WriteRegStr HKLM "Software\Microsoft\Windows\CurrentVersion\Uninstall\${APPNAME}" "QuietUninstallString" "$\"$INSTDIR\uninstall.exe$\" /S"
    WriteRegStr HKLM "Software\Microsoft\Windows\CurrentVersion\Uninstall\${APPNAME}" "InstallLocation" "$\"$INSTDIR$\""
    WriteRegStr HKLM "Software\Microsoft\Windows\CurrentVersion\Uninstall\${APPNAME}" "DisplayIcon" "$\"$INSTDIR\cardioai.ico$\""
    WriteRegStr HKLM "Software\Microsoft\Windows\CurrentVersion\Uninstall\${APPNAME}" "Publisher" "${COMPANYNAME}"
    WriteRegStr HKLM "Software\Microsoft\Windows\CurrentVersion\Uninstall\${APPNAME}" "HelpLink" "${HELPURL}"
    WriteRegStr HKLM "Software\Microsoft\Windows\CurrentVersion\Uninstall\${APPNAME}" "URLUpdateInfo" "${UPDATEURL}"
    WriteRegStr HKLM "Software\Microsoft\Windows\CurrentVersion\Uninstall\${APPNAME}" "URLInfoAbout" "${ABOUTURL}"
    WriteRegStr HKLM "Software\Microsoft\Windows\CurrentVersion\Uninstall\${APPNAME}" "DisplayVersion" "${VERSIONMAJOR}.${VERSIONMINOR}.${VERSIONBUILD}"
    WriteRegDWORD HKLM "Software\Microsoft\Windows\CurrentVersion\Uninstall\${APPNAME}" "VersionMajor" ${VERSIONMAJOR}
    WriteRegDWORD HKLM "Software\Microsoft\Windows\CurrentVersion\Uninstall\${APPNAME}" "VersionMinor" ${VERSIONMINOR}
    WriteRegDWORD HKLM "Software\Microsoft\Windows\CurrentVersion\Uninstall\${APPNAME}" "NoModify" 1
    WriteRegDWORD HKLM "Software\Microsoft\Windows\CurrentVersion\Uninstall\${APPNAME}" "NoRepair" 1
    WriteRegDWORD HKLM "Software\Microsoft\Windows\CurrentVersion\Uninstall\${APPNAME}" "EstimatedSize" ${INSTALLSIZE}
    
    ; Create uninstaller
    writeUninstaller "$INSTDIR\uninstall.exe"
sectionEnd

; Uninstaller
function un.onInit
    SetShellVarContext all
    
    ; Verify the user is an admin
    UserInfo::GetAccountType
    pop $0
    ${If} $0 != "admin" ; Require admin rights on NT4+
        messageBox mb_iconstop "Administrator rights required!"
        setErrorLevel 740 ; ERROR_ELEVATION_REQUIRED
        quit
    ${EndIf}
functionEnd

section "uninstall"
    ; Stop any running CardioAI processes
    ExecWait "taskkill /f /im cardioai-backend.exe" $0
    ExecWait "taskkill /f /im python.exe" $0
    
    ; Remove Start Menu launcher
    delete "$SMPROGRAMS\${APPNAME}\${APPNAME}.lnk"
    delete "$SMPROGRAMS\${APPNAME}\Stop ${APPNAME}.lnk"
    delete "$SMPROGRAMS\${APPNAME}\Uninstall ${APPNAME}.lnk"
    rmDir "$SMPROGRAMS\${APPNAME}"
    
    ; Remove desktop shortcut
    delete "$DESKTOP\${APPNAME}.lnk"
    
    ; Remove files
    delete "$INSTDIR\cardioai-backend.exe"
    delete "$INSTDIR\start_backend.bat"
    delete "$INSTDIR\serve_frontend.py"
    delete "$INSTDIR\start_cardioai.bat"
    delete "$INSTDIR\stop_cardioai.bat"
    delete "$INSTDIR\.env"
    delete "$INSTDIR\cardioai.ico"
    
    ; Remove frontend files
    rmDir /r "$INSTDIR\frontend_build"
    
    ; Remove database file (optional - user might want to keep data)
    MessageBox MB_YESNO "Do you want to remove all CardioAI Pro data including patient records? This cannot be undone." IDYES removedata IDNO keepdata
    removedata:
        delete "$INSTDIR\cardioai.db"
        rmDir /r "$APPDATA\${APPNAME}"
    keepdata:
    
    ; Always remove uninstaller
    delete "$INSTDIR\uninstall.exe"
    
    ; Try to remove the install directory - this will only happen if it is empty
    rmDir "$INSTDIR"
    
    ; Remove uninstaller information from the registry
    DeleteRegKey HKLM "Software\Microsoft\Windows\CurrentVersion\Uninstall\${APPNAME}"
sectionEnd
