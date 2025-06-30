# Guia Completo para Correção de Erros no Frontend do CardioAI Pro

Este guia fornece instruções detalhadas para corrigir o erro `useAuth must be used within an AuthProvider` e outros problemas comuns no frontend do CardioAI Pro.

## Solução Aprimorada

Para garantir que todos os problemas sejam corrigidos de uma vez, criamos um script aprimorado que:

1. Verifica a existência de todos os arquivos necessários
2. Cria backups antes de fazer alterações
3. Corrige todos os problemas conhecidos de autenticação
4. Verifica e instala dependências necessárias
5. Permite restaurar backups se algo der errado

### Passo 1: Criar o Script de Correção Aprimorado

1. **Navegue até a pasta do frontend**:

   ```powershell
   cd C:\caminho\para\cardio.ai.pro\frontend
   ```

2. **Crie o arquivo de script**:

   ```powershell
   notepad fix_auth_context_improved.js
   ```

3. **Cole o seguinte código**:

   ```javascript
   #!/usr/bin/env node
   /**
    * Script aprimorado para corrigir o problema de contexto de autenticação no frontend
    * Este script corrige o erro "useAuth must be used within an AuthProvider"
    * e garante que todas as dependências estejam corretamente configuradas
    */

   const fs = require('fs');
   const path = require('path');
   const { execSync } = require('child_process');

   // Caminho para os arquivos
   const useAuthPath = path.join(__dirname, 'src', 'hooks', 'useAuth.ts');
   const appTsxPath = path.join(__dirname, 'src', 'App.tsx');
   const authContextPath = path.join(__dirname, 'src', 'contexts', 'AuthContext.tsx');
   const authContextDefPath = path.join(__dirname, 'src', 'contexts', 'AuthContextDefinition.ts');
   const mainTsxPath = path.join(__dirname, 'src', 'main.tsx');

   // Verifica se um arquivo existe
   function fileExists(filePath) {
     try {
       return fs.existsSync(filePath);
     } catch (error) {
       return false;
     }
   }

   // Cria um backup de um arquivo
   function createBackup(filePath) {
     try {
       if (fileExists(filePath)) {
         const backupPath = `${filePath}.backup`;
         fs.copyFileSync(filePath, backupPath);
         console.log(`📦 Backup criado: ${backupPath}`);
         return true;
       }
       return false;
     } catch (error) {
       console.error(`❌ Erro ao criar backup de ${filePath}:`, error);
       return false;
     }
   }

   // Restaura um backup
   function restoreBackup(filePath) {
     try {
       const backupPath = `${filePath}.backup`;
       if (fileExists(backupPath)) {
         fs.copyFileSync(backupPath, filePath);
         console.log(`🔄 Arquivo restaurado do backup: ${filePath}`);
         return true;
       }
       return false;
     } catch (error) {
       console.error(`❌ Erro ao restaurar backup de ${filePath}:`, error);
       return false;
     }
   }

   // Cria backups de todos os arquivos
   function createBackups() {
     createBackup(useAuthPath);
     createBackup(appTsxPath);
     createBackup(authContextPath);
     createBackup(authContextDefPath);
     createBackup(mainTsxPath);
   }

   function fixUseAuthHook() {
     try {
       console.log('🔧 Corrigindo hook useAuth...');
       
       if (!fileExists(useAuthPath)) {
         console.error(`❌ Arquivo não encontrado: ${useAuthPath}`);
         return false;
       }
       
       // Ler o conteúdo do arquivo useAuth.ts
       let useAuthContent = fs.readFileSync(useAuthPath, 'utf8');
       
       // Verificar qual importação está sendo usada
       const usesAuthContextDef = useAuthContent.includes("from '../contexts/AuthContextDefinition'");
       const usesAuthContext = useAuthContent.includes("from '../contexts/AuthContext'");
       
       if (usesAuthContextDef) {
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
       } else if (usesAuthContext) {
         console.log('ℹ️ Hook useAuth já está usando a importação correta.');
       } else {
         // Caso o padrão de importação seja diferente
         console.log('⚠️ Padrão de importação não reconhecido em useAuth.ts. Tentando corrigir...');
         
         // Substituir qualquer importação de contexto
         useAuthContent = useAuthContent.replace(
           /import\s+\{\s*AuthContext\s*\}\s+from\s+['"](.*)['"];?/,
           "import { AuthContext } from '../contexts/AuthContext';"
         );
         
         useAuthContent = useAuthContent.replace(
           /import\s+type\s+\{\s*AuthContextType\s*\}\s+from\s+['"](.*)['"];?/,
           "import type { AuthContextType } from '../contexts/AuthContext';"
         );
         
         // Salvar as alterações
         fs.writeFileSync(useAuthPath, useAuthContent);
         console.log('✅ Hook useAuth corrigido com sucesso (padrão não reconhecido)!');
       }
       
       return true;
     } catch (error) {
       console.error('❌ Erro ao corrigir hook useAuth:', error);
       return false;
     }
   }

   function fixAppComponent() {
     try {
       console.log('🔧 Corrigindo componente App...');
       
       if (!fileExists(appTsxPath)) {
         console.error(`❌ Arquivo não encontrado: ${appTsxPath}`);
         return false;
       }
       
       // Ler o conteúdo do arquivo App.tsx
       let appContent = fs.readFileSync(appTsxPath, 'utf8');
       
       // Verificar se o componente App está usando useAuth
       if (appContent.includes('const { isAuthenticated } = useAuth()') || 
           appContent.includes('useAuth()')) {
         
         // Verificar se o useAuth está sendo importado
         if (!appContent.includes("import { useAuth }")) {
           // Adicionar a importação do useAuth
           if (appContent.includes("import { useState, useEffect }")) {
             appContent = appContent.replace(
               "import { useState, useEffect }",
               "import { useState, useEffect }\nimport { useAuth } from './hooks/useAuth'"
             );
           } else if (appContent.includes("import React")) {
             appContent = appContent.replace(
               "import React",
               "import React\nimport { useAuth } from './hooks/useAuth'"
             );
           } else {
             // Adicionar no início do arquivo
             appContent = "import { useAuth } from './hooks/useAuth';\n" + appContent;
           }
           
           // Salvar as alterações
           fs.writeFileSync(appTsxPath, appContent);
           console.log('✅ Componente App corrigido com sucesso!');
         } else {
           console.log('ℹ️ Componente App já tem a importação de useAuth.');
         }
       } else {
         console.log('ℹ️ Componente App não parece usar useAuth diretamente.');
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
       
       if (!fileExists(authContextPath)) {
         console.error(`❌ Arquivo não encontrado: ${authContextPath}`);
         return false;
       }
       
       // Ler o conteúdo do arquivo AuthContext.tsx
       let authContextContent = fs.readFileSync(authContextPath, 'utf8');
       
       let modified = false;
       
       // Verificar se o AuthContext já está sendo exportado
       if (!authContextContent.includes('export const AuthContext')) {
         // Substituir a declaração do AuthContext para exportá-lo
         authContextContent = authContextContent.replace(
           /const\s+AuthContext\s*=\s*createContext<AuthContextType\s*\|\s*undefined>\(undefined\)/,
           'export const AuthContext = createContext<AuthContextType | undefined>(undefined)'
         );
         modified = true;
       }
       
       // Verificar se o tipo AuthContextType está sendo exportado
       if (!authContextContent.includes('export interface AuthContextType')) {
         // Adicionar exportação do tipo AuthContextType
         authContextContent = authContextContent.replace(
           /interface\s+AuthContextType\s+extends\s+AuthState\s*\{/,
           'export interface AuthContextType extends AuthState {'
         );
         modified = true;
       }
       
       if (modified) {
         // Salvar as alterações
         fs.writeFileSync(authContextPath, authContextContent);
         console.log('✅ AuthContext corrigido com sucesso!');
       } else {
         console.log('ℹ️ AuthContext já está configurado corretamente.');
       }
       
       return true;
     } catch (error) {
       console.error('❌ Erro ao corrigir AuthContext:', error);
       return false;
     }
   }

   function fixMainTsx() {
     try {
       console.log('🔧 Verificando main.tsx...');
       
       if (!fileExists(mainTsxPath)) {
         console.error(`❌ Arquivo não encontrado: ${mainTsxPath}`);
         return false;
       }
       
       // Ler o conteúdo do arquivo main.tsx
       let mainContent = fs.readFileSync(mainTsxPath, 'utf8');
       
       // Verificar se o AppWithAuth está sendo usado
       if (mainContent.includes('<AppWithAuth />')) {
         console.log('ℹ️ main.tsx já está usando AppWithAuth corretamente.');
         return true;
       }
       
       // Se estiver usando App diretamente, substituir por AppWithAuth
       if (mainContent.includes('<App />')) {
         mainContent = mainContent.replace(
           '<App />',
           '<AppWithAuth />'
         );
         
         // Atualizar a importação
         mainContent = mainContent.replace(
           /import\s+App\s+from\s+['"]\.\/App['"];?/,
           "import AppWithAuth from './App';"
         );
         
         // Salvar as alterações
         fs.writeFileSync(mainTsxPath, mainContent);
         console.log('✅ main.tsx corrigido para usar AppWithAuth!');
       } else {
         console.log('⚠️ Não foi possível identificar o padrão em main.tsx. Verifique manualmente.');
       }
       
       return true;
     } catch (error) {
       console.error('❌ Erro ao verificar main.tsx:', error);
       return false;
     }
   }

   function copyAuthContextDefinitionToAuthContext() {
     try {
       console.log('🔧 Verificando se é necessário copiar definições de AuthContextDefinition...');
       
       // Verificar se ambos os arquivos existem
       if (fileExists(authContextDefPath) && fileExists(authContextPath)) {
         // Ler o conteúdo dos arquivos
         const defContent = fs.readFileSync(authContextDefPath, 'utf8');
         let authContent = fs.readFileSync(authContextPath, 'utf8');
         
         // Verificar se há tipos no AuthContextDefinition que precisam ser copiados
         if (defContent.includes('export interface AuthContextType')) {
           console.log('⚠️ AuthContextDefinition.ts contém definições de tipos que podem ser necessárias.');
           console.log('ℹ️ Considere verificar manualmente se todos os tipos necessários estão em AuthContext.tsx');
         } else {
           console.log('ℹ️ Não parece haver tipos exclusivos em AuthContextDefinition.ts');
         }
         
         return true;
       } else if (!fileExists(authContextDefPath)) {
         console.log('ℹ️ Arquivo AuthContextDefinition.ts não encontrado, não é necessário copiar definições.');
         return true;
       } else {
         console.error('❌ Arquivo AuthContext.tsx não encontrado!');
         return false;
       }
     } catch (error) {
       console.error('❌ Erro ao verificar definições de contexto:', error);
       return false;
     }
   }

   function verifyAllFiles() {
     console.log('🔍 Verificando a existência de todos os arquivos necessários...');
     
     const files = [
       { path: useAuthPath, name: 'useAuth.ts' },
       { path: appTsxPath, name: 'App.tsx' },
       { path: authContextPath, name: 'AuthContext.tsx' },
       { path: mainTsxPath, name: 'main.tsx' }
     ];
     
     let allFilesExist = true;
     
     files.forEach(file => {
       if (fileExists(file.path)) {
         console.log(`✅ ${file.name} encontrado.`);
       } else {
         console.error(`❌ ${file.name} não encontrado em ${file.path}`);
         allFilesExist = false;
       }
     });
     
     return allFilesExist;
   }

   function runNpmInstall() {
     try {
       console.log('🔧 Verificando dependências do projeto...');
       
       // Verificar se o package.json existe
       const packageJsonPath = path.join(__dirname, 'package.json');
       if (!fileExists(packageJsonPath)) {
         console.error('❌ package.json não encontrado!');
         return false;
       }
       
       console.log('📦 Instalando dependências do projeto...');
       execSync('npm install', { stdio: 'inherit' });
       console.log('✅ Dependências instaladas com sucesso!');
       
       return true;
     } catch (error) {
       console.error('❌ Erro ao instalar dependências:', error);
       return false;
     }
   }

   // Função principal
   async function main() {
     console.log('🚀 Iniciando correção aprimorada do contexto de autenticação...');
     
     // Verificar arquivos
     if (!verifyAllFiles()) {
       console.error('❌ Alguns arquivos necessários não foram encontrados. Abortando...');
       return false;
     }
     
     // Criar backups
     console.log('📦 Criando backups dos arquivos...');
     createBackups();
     
     try {
       // Executar todas as correções
       const authContextFixed = fixAuthContext();
       const useAuthFixed = fixUseAuthHook();
       const appFixed = fixAppComponent();
       const mainFixed = fixMainTsx();
       const definitionsChecked = copyAuthContextDefinitionToAuthContext();
       
       // Verificar se todas as correções foram bem-sucedidas
       if (authContextFixed && useAuthFixed && appFixed && mainFixed && definitionsChecked) {
         console.log('✅ Todas as correções foram aplicadas com sucesso!');
         
         // Instalar dependências
         const depsInstalled = runNpmInstall();
         
         if (depsInstalled) {
           console.log('\n🎉 Configuração concluída! Execute "npm run dev" para iniciar o frontend.');
           console.log('📝 Se ainda encontrar problemas, verifique os arquivos manualmente ou restaure os backups.');
           return true;
         } else {
           console.warn('⚠️ Correções aplicadas, mas houve problemas ao instalar dependências.');
           return false;
         }
       } else {
         console.error('❌ Algumas correções não puderam ser aplicadas.');
         console.log('⚠️ Você pode tentar restaurar os backups usando a opção --restore.');
         return false;
       }
     } catch (error) {
       console.error('❌ Erro durante o processo de correção:', error);
       console.log('⚠️ Restaurando backups...');
       
       // Restaurar backups em caso de erro
       restoreBackup(useAuthPath);
       restoreBackup(appTsxPath);
       restoreBackup(authContextPath);
       restoreBackup(authContextDefPath);
       restoreBackup(mainTsxPath);
       
       return false;
     }
   }

   // Verificar se o usuário quer restaurar backups
   if (process.argv.includes('--restore')) {
     console.log('🔄 Restaurando backups...');
     restoreBackup(useAuthPath);
     restoreBackup(appTsxPath);
     restoreBackup(authContextPath);
     restoreBackup(authContextDefPath);
     restoreBackup(mainTsxPath);
     console.log('✅ Processo de restauração concluído.');
   } else {
     // Executar o script principal
     main().then(success => {
       if (success) {
         console.log('✅ Script concluído com sucesso!');
       } else {
         console.error('❌ Script concluído com erros.');
         console.log('💡 Você pode restaurar os backups executando: node fix_auth_context_improved.js --restore');
       }
     });
   }
   ```

4. **Salve o arquivo**.

### Passo 2: Execute o Script

1. **Execute o script**:

   ```powershell
   node fix_auth_context_improved.js
   ```

2. **Observe a saída do script**:
   - O script verificará todos os arquivos necessários
   - Criará backups antes de fazer alterações
   - Aplicará todas as correções necessárias
   - Instalará as dependências do projeto

3. **Inicie o frontend**:

   ```powershell
   npm run dev
   ```

### Passo 3: Verificação

Após executar o script e iniciar o frontend, acesse:

```
http://localhost:5173
```

A página de login deve ser exibida corretamente, sem erros no console do navegador.

## Solução de Problemas

### Se o Script Falhar

Se o script falhar por algum motivo, você pode restaurar os backups:

```powershell
node fix_auth_context_improved.js --restore
```

### Correção Manual

Se preferir corrigir manualmente ou se o script não funcionar, siga estas etapas:

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

4. **Verifique o arquivo main.tsx**:

   Abra o arquivo `src/main.tsx` e certifique-se de que está usando `AppWithAuth` em vez de `App`:

   ```typescript
   import AppWithAuth from './App'
   
   ReactDOM.createRoot(document.getElementById('root')!).render(
     <React.StrictMode>
       <AppWithAuth />
     </React.StrictMode>
   )
   ```

## Explicação Técnica

O erro "useAuth must be used within an AuthProvider" ocorre porque:

1. O hook `useAuth` está importando o contexto de autenticação do arquivo `AuthContextDefinition.ts`
2. Mas o componente `AuthProvider` está definido no arquivo `AuthContext.tsx`
3. O contexto `AuthContext` não está sendo exportado corretamente do arquivo `AuthContext.tsx`

A solução consiste em:

1. Exportar o contexto `AuthContext` e o tipo `AuthContextType` do arquivo `AuthContext.tsx`
2. Atualizar as importações no hook `useAuth` para apontar para o arquivo correto
3. Garantir que o componente `App` esteja importando o hook `useAuth` corretamente
4. Verificar se o arquivo `main.tsx` está usando o componente `AppWithAuth` que inclui o `AuthProvider`

## Próximos Passos

Após corrigir este erro, você pode continuar com a configuração do CardioAI Pro seguindo o guia principal de instalação.

Se encontrar outros erros, consulte a documentação ou entre em contato com a equipe de suporte.