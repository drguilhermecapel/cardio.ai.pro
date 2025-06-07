# 🔧 Solução para Erro do Node.js - CardioAI Pro

## ❌ Problema Identificado
```
ERROR: Node.js is not installed or not in PATH
Please install Node.js from https://nodejs.org
```

## 🚀 Solução Automática (NOVA!)

**O instalador agora baixa automaticamente o Node.js portátil!**

Quando você executar o `build_installer.bat`, ele irá:
1. ✅ Verificar se Node.js está instalado no sistema
2. ✅ Se não encontrar, baixar automaticamente Node.js v18.19.0 portátil
3. ✅ Extrair e configurar na pasta `portable_node/`
4. ✅ Usar a versão portátil para build do frontend
5. ✅ Não requer instalação manual ou modificação do PATH

## ✅ Solução Manual (se automática falhar)

### Passo 1: Instalar Node.js
1. **Acesse**: https://nodejs.org
2. **Baixe**: Versão LTS (Long Term Support) - Recomendada
3. **Execute**: o instalador baixado
4. **IMPORTANTE**: Durante a instalação, marque a opção "Add to PATH"

### Passo 2: Verificar Instalação
Abra o **Prompt de Comando** (cmd) e digite:
```cmd
node --version
npm --version
```

Deve aparecer algo como:
```
v18.17.0
9.6.7
```

### Passo 3: Executar o Instalador
Agora execute novamente o `build_installer.bat`

## 📁 Estrutura do Node.js Portátil

Após o download automático, a estrutura será:
```
windows_installer/
├── portable_node/
│   ├── node.exe          # Node.js executável
│   ├── npm.cmd           # npm command
│   ├── node_modules/     # Módulos Node.js
│   └── ...
├── build_installer.bat   # Script principal
└── SOLUCAO_NODEJS.md    # Este arquivo
```

## 🚀 Alternativa: Instalador Pré-Compilado

Se preferir **não usar download automático**, você pode usar o instalador já compilado:

### Opção A: Baixar Executável Pronto
- **Arquivo**: `CardioAI-Pro-1.0.0-portable.exe` (200MB)
- **Localização**: `windows_installer/` (se disponível)
- **Uso**: Duplo clique para executar

### Opção B: Solicitar Versão Compilada
Solicite ao desenvolvedor uma versão já compilada do instalador.

## 📋 Requisitos do Sistema

### Para Usar o CardioAI Pro:
- ✅ Windows 10/11
- ✅ 4GB RAM mínimo
- ✅ 500MB espaço em disco

### Para Compilar o Instalador:
- ✅ Node.js 16+ (LTS recomendado)
- ✅ Python 3.8+
- ✅ Poetry (gerenciador Python)

## 🆘 Suporte

Se continuar com problemas:
1. Verifique se Node.js está no PATH do sistema
2. Reinicie o computador após instalar Node.js
3. Execute como Administrador se necessário

## 📞 Contato
Para suporte técnico, entre em contato com o desenvolvedor.
