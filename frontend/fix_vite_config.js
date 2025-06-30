#!/usr/bin/env node
/**
 * Script para corrigir a configuração do Vite
 * Este script adiciona um rewrite para redirecionar /api/auth/* para /api/v1/auth/*
 */

const fs = require('fs');
const path = require('path');

// Caminho para o arquivo vite.config.ts
const viteConfigPath = path.join(__dirname, 'vite.config.ts');

function fixViteConfig() {
  try {
    console.log('🔧 Corrigindo configuração do Vite...');
    
    if (!fs.existsSync(viteConfigPath)) {
      console.error(`❌ Arquivo não encontrado: ${viteConfigPath}`);
      return false;
    }
    
    // Ler o conteúdo do arquivo vite.config.ts
    let viteConfigContent = fs.readFileSync(viteConfigPath, 'utf8');
    
    // Verificar se a configuração já inclui o rewrite
    if (viteConfigContent.includes('rewrite')) {
      console.log('ℹ️ Configuração de rewrite já existe no arquivo vite.config.ts');
      return true;
    }
    
    // Substituir a configuração do proxy
    const oldProxyConfig = `  server: {
    proxy: {
      '/api': {
        target: 'http://localhost:8000',
        changeOrigin: true,
        secure: false,
      },
    },
  },`;
    
    const newProxyConfig = `  server: {
    proxy: {
      '/api': {
        target: 'http://localhost:8000',
        changeOrigin: true,
        secure: false,
        rewrite: (path) => {
          // Redirecionar /api/auth/* para /api/v1/auth/*
          if (path.startsWith('/api/auth/')) {
            return path.replace('/api/auth/', '/api/v1/auth/');
          }
          return path;
        },
      },
    },
    host: true,
    port: 5173,
    strictPort: true,
    cors: true,
  },`;
    
    // Substituir a configuração
    viteConfigContent = viteConfigContent.replace(oldProxyConfig, newProxyConfig);
    
    // Salvar as alterações
    fs.writeFileSync(viteConfigPath, viteConfigContent);
    console.log('✅ Configuração do Vite corrigida com sucesso!');
    
    return true;
  } catch (error) {
    console.error('❌ Erro ao corrigir configuração do Vite:', error);
    return false;
  }
}

// Executar a correção
console.log('🚀 Iniciando correção da configuração do Vite...');
const success = fixViteConfig();

if (success) {
  console.log('\n✅ Correção aplicada com sucesso!');
  console.log('🔄 Execute "npm run dev" para reiniciar o frontend com a nova configuração.');
} else {
  console.log('\n⚠️ Não foi possível aplicar a correção. Verifique os erros acima.');
}