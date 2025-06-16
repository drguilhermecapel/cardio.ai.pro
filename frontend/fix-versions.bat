@echo off
REM Script para corrigir versões após npm audit fix --force

echo =====================================
echo  Corrigindo versoes dos pacotes...
echo =====================================
echo.

REM Remover node_modules e package-lock.json
echo Limpando instalacao anterior...
rmdir /s /q node_modules 2>nul
del package-lock.json 2>nul

REM Instalar versões específicas estáveis
echo.
echo Instalando versoes estaveis...
echo.

REM Downgrade para versões compatíveis
npm install --save-dev vite@5.0.11
npm install --save-dev vitest@1.2.1 @vitest/ui@1.2.1 @vitest/coverage-v8@1.2.1
npm install --save-dev eslint@8.56.0

REM Reinstalar todas as dependências
echo.
echo Reinstalando todas as dependencias...
npm install

echo.
echo =====================================
echo  Versoes corrigidas com sucesso!
echo =====================================
echo.
echo Proximos passos:
echo 1. npm run test:coverage
echo.
pause
