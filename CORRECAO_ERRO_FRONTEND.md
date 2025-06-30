# Correção de Erro no Frontend do CardioAI Pro

Este guia explica como corrigir o erro `useAuth must be used within an AuthProvider` que ocorre ao iniciar o frontend do CardioAI Pro.

## Descrição do Problema

Ao iniciar o frontend, o console do navegador mostra o seguinte erro:

```
Uncaught Error: useAuth must be used within an AuthProvider
    at useAuth (useAuth.ts:8:11)
    at App (App.tsx:757:31)
```

Este erro ocorre devido a uma incompatibilidade entre os arquivos de contexto de autenticação. O hook `useAuth` está importando o contexto de um arquivo, mas o componente `AuthProvider` está definido em outro arquivo.

## Solução Passo a Passo

### Opção 1: Usando o Script de Correção Automática (Recomendado)

1. **Navegue até a pasta do frontend**:

   ```powershell
   cd C:\caminho\para\cardio.ai.pro\frontend
   ```

2. **Crie o script de correção**:

   Crie um arquivo chamado `fix_auth_context.js` com o seguinte conteúdo:

   ```javascript
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
   ```

3. **Execute o script de correção**:

   ```powershell
   node fix_auth_context.js
   ```

4. **Inicie o frontend novamente**:

   ```powershell
   npm run dev
   ```

### Opção 2: Correção Manual

Se preferir corrigir manualmente, siga estes passos:

1. **Corrija o arquivo AuthContext.tsx**:

   Abra o arquivo `src/contexts/AuthContext.tsx` e faça as seguintes alterações:

   - Encontre a linha:
     ```typescript
     const AuthContext = createContext<AuthContextType | undefined>(undefined)
     ```
     
     E substitua por:
     ```typescript
     export const AuthContext = createContext<AuthContextType | undefined>(undefined)
     ```

   - Encontre a linha:
     ```typescript
     interface AuthContextType extends AuthState {
     ```
     
     E substitua por:
     ```typescript
     export interface AuthContextType extends AuthState {
     ```

2. **Corrija o arquivo useAuth.ts**:

   Abra o arquivo `src/hooks/useAuth.ts` e faça as seguintes alterações:

   - Substitua:
     ```typescript
     import { AuthContext } from '../contexts/AuthContextDefinition'
     import type { AuthContextType } from '../contexts/AuthContextDefinition'
     ```
     
     Por:
     ```typescript
     import { AuthContext } from '../contexts/AuthContext'
     import type { AuthContextType } from '../contexts/AuthContext'
     ```

3. **Verifique o arquivo App.tsx**:

   Abra o arquivo `src/App.tsx` e certifique-se de que o hook `useAuth` está sendo importado:

   ```typescript
   import { useAuth } from './hooks/useAuth'
   ```

4. **Inicie o frontend novamente**:

   ```powershell
   npm run dev
   ```

## Explicação Técnica

O erro ocorre porque:

1. O hook `useAuth` está importando o contexto de autenticação do arquivo `AuthContextDefinition.ts`
2. Mas o componente `AuthProvider` está definido no arquivo `AuthContext.tsx`
3. O contexto `AuthContext` não está sendo exportado corretamente do arquivo `AuthContext.tsx`

A correção consiste em:

1. Exportar o contexto `AuthContext` e o tipo `AuthContextType` do arquivo `AuthContext.tsx`
2. Atualizar as importações no hook `useAuth` para apontar para o arquivo correto
3. Garantir que o componente `App` esteja importando o hook `useAuth` corretamente

## Verificação

Após aplicar a correção, o frontend deve iniciar sem erros. Você pode verificar acessando:

```
http://localhost:5173
```

A página de login deve ser exibida corretamente, sem erros no console do navegador.

## Próximos Passos

Após corrigir este erro, você pode continuar com a configuração do CardioAI Pro seguindo o guia principal de instalação.

Se encontrar outros erros, consulte a documentação ou entre em contato com a equipe de suporte.