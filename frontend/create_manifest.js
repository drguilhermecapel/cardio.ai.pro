#!/usr/bin/env node
/**
 * Script para criar o arquivo manifest.json
 * Este script corrige o erro "Manifest: Line: 1, column: 1, Syntax error"
 */

const fs = require('fs');
const path = require('path');

// Caminho para a pasta public
const publicDir = path.join(__dirname, 'public');

// Conteúdo do manifest.json
const manifestContent = {
  "name": "CardioAI Pro",
  "short_name": "CardioAI",
  "description": "AI-powered ECG analysis system",
  "theme_color": "#1976d2",
  "background_color": "#ffffff",
  "display": "standalone",
  "start_url": "/",
  "icons": [
    {
      "src": "pwa-192x192.png",
      "sizes": "192x192",
      "type": "image/png"
    },
    {
      "src": "pwa-512x512.png",
      "sizes": "512x512",
      "type": "image/png"
    }
  ]
};

function createManifest() {
  try {
    console.log('🔧 Criando arquivo manifest.json...');
    
    // Verificar se a pasta public existe
    if (!fs.existsSync(publicDir)) {
      console.log('📁 Criando pasta public...');
      fs.mkdirSync(publicDir, { recursive: true });
    }
    
    // Caminho para o arquivo manifest.json
    const manifestPath = path.join(publicDir, 'manifest.json');
    
    // Verificar se o arquivo já existe
    if (fs.existsSync(manifestPath)) {
      console.log('ℹ️ O arquivo manifest.json já existe. Substituindo...');
    }
    
    // Escrever o conteúdo do manifest.json
    fs.writeFileSync(
      manifestPath,
      JSON.stringify(manifestContent, null, 2)
    );
    
    console.log('✅ Arquivo manifest.json criado com sucesso!');
    
    // Criar ícones placeholder se não existirem
    createPlaceholderIcons();
    
    return true;
  } catch (error) {
    console.error('❌ Erro ao criar arquivo manifest.json:', error);
    return false;
  }
}

function createPlaceholderIcons() {
  try {
    console.log('🔧 Verificando ícones PWA...');
    
    // Caminhos para os ícones
    const icon192Path = path.join(publicDir, 'pwa-192x192.png');
    const icon512Path = path.join(publicDir, 'pwa-512x512.png');
    
    // Verificar se os ícones já existem
    if (!fs.existsSync(icon192Path)) {
      console.log('📝 Criando ícone placeholder de 192x192...');
      createPlaceholderIcon(icon192Path, 192);
    }
    
    if (!fs.existsSync(icon512Path)) {
      console.log('📝 Criando ícone placeholder de 512x512...');
      createPlaceholderIcon(icon512Path, 512);
    }
    
    console.log('✅ Ícones PWA verificados/criados com sucesso.');
    
    return true;
  } catch (error) {
    console.error('❌ Erro ao verificar/criar ícones PWA:', error);
    return false;
  }
}

// Função para criar um ícone placeholder simples usando Node.js puro
// Esta é uma solução temporária - idealmente você usaria uma biblioteca de manipulação de imagens
function createPlaceholderIcon(filePath, size) {
  try {
    // Criar um arquivo de texto com instruções
    fs.writeFileSync(
      filePath,
      `Este é um arquivo placeholder para o ícone PWA de ${size}x${size}.
Por favor, substitua este arquivo por uma imagem PNG real de ${size}x${size} pixels.
Você pode usar ferramentas online como https://realfavicongenerator.net/ para criar ícones.`
    );
    
    console.log(`✅ Arquivo placeholder criado em ${filePath}`);
    console.log(`⚠️ IMPORTANTE: Substitua este arquivo por uma imagem PNG real de ${size}x${size} pixels.`);
    
    return true;
  } catch (error) {
    console.error(`❌ Erro ao criar ícone placeholder ${size}x${size}:`, error);
    return false;
  }
}

// Executar a criação do manifest
console.log('🚀 Iniciando criação do manifest.json...');
const success = createManifest();

if (success) {
  console.log('\n✅ Manifest.json e ícones placeholder criados com sucesso!');
  console.log('⚠️ IMPORTANTE: Substitua os arquivos de ícone placeholder por imagens PNG reais.');
  console.log('🔄 Execute "npm run dev" para reiniciar o frontend.');
} else {
  console.log('\n⚠️ Não foi possível criar o manifest.json. Verifique os erros acima.');
}