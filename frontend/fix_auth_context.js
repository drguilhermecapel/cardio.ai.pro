#!/usr/bin/env node
/**
 * Script para corrigir o problema de contexto de autenticação no frontend
 * Este script corrige o erro "useAuth must be used within an AuthProvider"
 */

const fs = require('fs');
const path = require('path');

// Caminho para os arquivos
const useAuthPath = path.join(__dirname, 'src', 'hooks', 'useAuth.ts');
const appTsxPath = path.join(__dirname, 'src', 'App.tsx');
const authContextPath = path.join(__dirname, 'src', 'contexts', 'AuthContext.tsx');

function fixUseAuthHook() {
  try {
    console.log('🔧 Corrigindo hook useAuth...');
    
    // Ler o conteúdo do arquivo useAuth.ts
    let useAuthContent = fs.readFileSync(useAuthPath, 'utf8');
    
    // Substituir a importação incorreta pela correta
    useAuthContent = useAuthContent.replace(
      "import { AuthContext } from '../contexts/AuthContextDefinition'",
      "import { AuthContext } from '../contexts/AuthContext'"
    );
    
    useAuthContent = useAuthContent.replace(
      "import type { AuthContextType } from '../contexts/AuthContextDefinition'",
      "import type { AuthContextType } from '../contexts/AuthContext'"
    );
    
    // Salvar as alterações
    fs.writeFileSync(useAuthPath, useAuthContent);
    console.log('✅ Hook useAuth corrigido com sucesso!');
    
    return true;
  } catch (error) {
    console.error('❌ Erro ao corrigir hook useAuth:', error);
    return false;
  }
}

function fixAppComponent() {
  try {
    console.log('🔧 Corrigindo componente App...');
    
    // Ler o conteúdo do arquivo App.tsx
    let appContent = fs.readFileSync(appTsxPath, 'utf8');
    
    // Verificar se o componente App está usando useAuth
    if (appContent.includes('const { isAuthenticated } = useAuth()')) {
      // Verificar se o useAuth está sendo importado
      if (!appContent.includes("import { useAuth }")) {
        // Adicionar a importação do useAuth
        appContent = appContent.replace(
          "import { useState, useEffect }",
          "import { useState, useEffect }\nimport { useAuth } from './hooks/useAuth'"
        );
      }
      
      // Salvar as alterações
      fs.writeFileSync(appTsxPath, appContent);
      console.log('✅ Componente App corrigido com sucesso!');
    } else {
      console.log('ℹ️ Componente App não precisa de correção.');
    }
    
    return true;
  } catch (error) {
    console.error('❌ Erro ao corrigir componente App:', error);
    return false;
  }
}

function fixAuthContext() {
  try {
    console.log('🔧 Corrigindo AuthContext...');
    
    // Ler o conteúdo do arquivo AuthContext.tsx
    let authContextContent = fs.readFileSync(authContextPath, 'utf8');
    
    // Verificar se o AuthContext já está sendo exportado
    if (!authContextContent.includes('export const AuthContext')) {
      // Substituir a declaração do AuthContext para exportá-lo
      authContextContent = authContextContent.replace(
        'const AuthContext = createContext<AuthContextType | undefined>(undefined)',
        'export const AuthContext = createContext<AuthContextType | undefined>(undefined)'
      );
      
      // Adicionar exportação do tipo AuthContextType
      authContextContent = authContextContent.replace(
        'interface AuthContextType extends AuthState {',
        'export interface AuthContextType extends AuthState {'
      );
      
      // Salvar as alterações
      fs.writeFileSync(authContextPath, authContextContent);
      console.log('✅ AuthContext corrigido com sucesso!');
    } else {
      console.log('ℹ️ AuthContext já está sendo exportado corretamente.');
    }
    
    return true;
  } catch (error) {
    console.error('❌ Erro ao corrigir AuthContext:', error);
    return false;
  }
}

// Executar as correções
console.log('🚀 Iniciando correção do contexto de autenticação...');
const authContextFixed = fixAuthContext();
const useAuthFixed = fixUseAuthHook();
const appFixed = fixAppComponent();

if (authContextFixed && useAuthFixed && appFixed) {
  console.log('✅ Correções aplicadas com sucesso! Execute "npm run dev" para iniciar o frontend.');
} else {
  console.log('⚠️ Algumas correções não puderam ser aplicadas. Verifique os erros acima.');
}