#!/usr/bin/env node
/**
 * Script para corrigir as rotas de API no frontend
 * Este script corrige o erro 404 Not Found ao tentar fazer login
 */

const fs = require('fs');
const path = require('path');

// Caminho para os arquivos
const authContextPath = path.join(__dirname, 'src', 'contexts', 'AuthContext.tsx');
const authSlicePath = path.join(__dirname, 'src', 'store', 'slices', 'authSlice.ts');

function fixAuthContext() {
  try {
    console.log('🔧 Corrigindo rotas de API no AuthContext...');
    
    if (!fs.existsSync(authContextPath)) {
      console.error(`❌ Arquivo não encontrado: ${authContextPath}`);
      return false;
    }
    
    // Ler o conteúdo do arquivo AuthContext.tsx
    let authContextContent = fs.readFileSync(authContextPath, 'utf8');
    
    // Verificar se as rotas precisam ser corrigidas
    const needsCorrection = authContextContent.includes("'/api/auth/login'") || 
                           authContextContent.includes("'/api/auth/biometric-login'");
    
    if (needsCorrection) {
      // Substituir a rota de login
      authContextContent = authContextContent.replace(
        /['"]\/api\/auth\/login['"]/g,
        "'/api/v1/auth/login'"
      );
      
      // Substituir a rota de login biométrico
      authContextContent = authContextContent.replace(
        /['"]\/api\/auth\/biometric-login['"]/g,
        "'/api/v1/auth/biometric-login'"
      );
      
      // Salvar as alterações
      fs.writeFileSync(authContextPath, authContextContent);
      console.log('✅ Rotas de API no AuthContext corrigidas com sucesso!');
    } else {
      console.log('ℹ️ Rotas no AuthContext já estão corretas ou usam um formato diferente.');
    }
    
    return true;
  } catch (error) {
    console.error('❌ Erro ao corrigir rotas de API no AuthContext:', error);
    return false;
  }
}

function fixAuthSlice() {
  try {
    console.log('🔧 Corrigindo rotas de API no AuthSlice...');
    
    if (!fs.existsSync(authSlicePath)) {
      console.error(`❌ Arquivo não encontrado: ${authSlicePath}`);
      return false;
    }
    
    // Ler o conteúdo do arquivo authSlice.ts
    let authSliceContent = fs.readFileSync(authSlicePath, 'utf8');
    
    // Verificar se as rotas precisam ser corrigidas
    const needsCorrection = authSliceContent.includes("'/api/auth/login'") || 
                           authSliceContent.includes("'/api/auth/logout'");
    
    if (needsCorrection) {
      // Substituir a rota de login
      authSliceContent = authSliceContent.replace(
        /['"]\/api\/auth\/login['"]/g,
        "'/api/v1/auth/login'"
      );
      
      // Substituir a rota de logout
      authSliceContent = authSliceContent.replace(
        /['"]\/api\/auth\/logout['"]/g,
        "'/api/v1/auth/logout'"
      );
      
      // Salvar as alterações
      fs.writeFileSync(authSlicePath, authSliceContent);
      console.log('✅ Rotas de API no AuthSlice corrigidas com sucesso!');
    } else {
      console.log('ℹ️ Rotas no AuthSlice já estão corretas ou usam um formato diferente.');
    }
    
    return true;
  } catch (error) {
    console.error('❌ Erro ao corrigir rotas de API no AuthSlice:', error);
    return false;
  }
}

// Função para procurar e corrigir todas as referências a rotas de API no projeto
function findAndFixAllApiRoutes() {
  try {
    console.log('🔍 Procurando por todas as referências a rotas de API no projeto...');
    
    const srcDir = path.join(__dirname, 'src');
    
    // Função recursiva para percorrer diretórios
    function processDirectory(dir) {
      const files = fs.readdirSync(dir);
      
      for (const file of files) {
        const filePath = path.join(dir, file);
        const stat = fs.statSync(filePath);
        
        if (stat.isDirectory()) {
          // Recursivamente processar subdiretórios
          processDirectory(filePath);
        } else if (
          stat.isFile() && 
          (file.endsWith('.ts') || file.endsWith('.tsx') || file.endsWith('.js') || file.endsWith('.jsx'))
        ) {
          // Processar apenas arquivos TypeScript/JavaScript
          let content = fs.readFileSync(filePath, 'utf8');
          let modified = false;
          
          // Verificar e corrigir rotas de API
          if (content.includes('/api/auth/')) {
            console.log(`🔧 Corrigindo rotas em: ${filePath}`);
            
            // Substituir todas as ocorrências de /api/auth/ por /api/v1/auth/
            const newContent = content.replace(/['"]\/api\/auth\//g, "'/api/v1/auth/");
            
            if (newContent !== content) {
              fs.writeFileSync(filePath, newContent);
              modified = true;
            }
          }
          
          if (modified) {
            console.log(`✅ Arquivo corrigido: ${filePath}`);
          }
        }
      }
    }
    
    // Iniciar o processamento a partir do diretório src
    processDirectory(srcDir);
    console.log('✅ Busca e correção de rotas de API concluída!');
    
    return true;
  } catch (error) {
    console.error('❌ Erro ao procurar e corrigir rotas de API:', error);
    return false;
  }
}

// Executar as correções
console.log('🚀 Iniciando correção das rotas de API...');
const authContextFixed = fixAuthContext();
const authSliceFixed = fixAuthSlice();
const allRoutesFixed = findAndFixAllApiRoutes();

if (authContextFixed && authSliceFixed && allRoutesFixed) {
  console.log('\n✅ Todas as correções foram aplicadas com sucesso!');
  console.log('🔄 Execute "npm run dev" para reiniciar o frontend.');
} else {
  console.log('\n⚠️ Algumas correções não puderam ser aplicadas. Verifique os erros acima.');
}