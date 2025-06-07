# CardioAI Pro - Versão Standalone

## Visão Geral

Esta é a versão standalone do CardioAI Pro, projetada para instalação simples em sistemas Windows sem necessidade de Docker, configurações complexas ou conhecimentos técnicos.

## Características da Versão Standalone

### ✅ Incluído
- ✓ **Executável único**: Backend empacotado com PyInstaller
- ✓ **Banco de dados local**: SQLite integrado
- ✓ **Interface web**: React frontend responsivo
- ✓ **Processamento local**: Análise de ECG sem internet
- ✓ **Modelos de IA**: Pré-treinados e inclusos
- ✓ **Instalação simples**: Clique duplo para instalar
- ✓ **Sem dependências**: Não requer Docker ou Python

### ❌ Removido da Versão Original
- ❌ Docker e containers
- ❌ Redis para cache
- ❌ Celery para tarefas assíncronas
- ❌ PostgreSQL (substituído por SQLite)
- ❌ Configurações complexas de ambiente

## Arquivos de Instalação

### Para Usuários Finais

1. **`CardioAI-Pro-v1.0.0-Setup.exe`** (38.6 MB)
   - Instalador profissional NSIS
   - Cria atalhos na área de trabalho e menu Iniciar
   - Inclui desinstalador
   - Requer privilégios de administrador

2. **`install-cardioai-pro-standalone.bat`** (4.3 KB)
   - Script de instalação automática
   - Executa o instalador NSIS com privilégios
   - Interface amigável em português
   - Tratamento de erros

3. **`CardioAI-Pro-v1.0.0-Portable.zip`** (40.4 MB)
   - Versão portátil (não requer instalação)
   - Extrair e executar `CardioAI-Pro.bat`
   - Ideal para testes ou uso temporário

### Para Desenvolvedores

4. **`create_portable_package.py`**
   - Script para criar pacote portátil
   - Combina backend e frontend
   - Gera arquivo ZIP para distribuição

5. **`create_windows_installer.py`**
   - Script para criar instalador NSIS
   - Gera instalador profissional
   - Inclui configurações de registro do Windows

## Instruções de Instalação

### Método 1: Instalação Automática (Recomendado)

```bash
# 1. Baixar arquivos
- install-cardioai-pro-standalone.bat
- CardioAI-Pro-v1.0.0-Setup.exe

# 2. Colocar na mesma pasta

# 3. Executar
Clique duplo em: install-cardioai-pro-standalone.bat
```

### Método 2: Instalação Manual

```bash
# 1. Baixar instalador
- CardioAI-Pro-v1.0.0-Setup.exe

# 2. Executar como administrador
Botão direito > "Executar como administrador"
```

### Método 3: Versão Portátil

```bash
# 1. Baixar e extrair
- CardioAI-Pro-v1.0.0-Portable.zip

# 2. Executar
Clique duplo em: CardioAI-Pro.bat
```

## Como Usar

### Inicialização

1. **Após instalação**: Clique no atalho da área de trabalho
2. **Ou pelo menu**: Iniciar > CardioAI Pro
3. **Aguarde**: Sistema inicializa em 3-5 segundos
4. **Interface**: Abre automaticamente no navegador

### Acesso

- **URL local**: http://localhost:8000
- **Login inicial**: admin / admin123
- **Alterar senha**: Recomendado na primeira utilização

### Parar o Sistema

- **Versão instalada**: Feche a janela do prompt
- **Versão portátil**: Pressione qualquer tecla na janela

## Arquitetura Técnica

### Backend Standalone
```
cardioai-pro-backend.exe
├── FastAPI application
├── SQLite database
├── ML models (embedded)
├── ECG processing
└── API endpoints
```

### Frontend
```
frontend/
├── React application (built)
├── Material-UI components
├── PWA capabilities
└── Responsive design
```

### Estrutura de Arquivos
```
CardioAI-Pro-Portable/
├── cardioai-pro-backend.exe    # Backend executável
├── CardioAI-Pro.bat           # Launcher principal
├── start-backend.bat          # Script auxiliar
├── LEIA-ME.txt               # Instruções
├── frontend/                 # Interface web
│   ├── index.html
│   ├── assets/
│   └── ...
└── electron/                 # Configuração Electron
    ├── main.js
    └── preload.js
```

## Desenvolvimento

### Construir Backend

```bash
cd backend
python build_backend.py
```

### Construir Frontend

```bash
cd frontend
python build_frontend.py
```

### Criar Pacote Portátil

```bash
python create_portable_package.py
```

### Criar Instalador Windows

```bash
python create_windows_installer.py
```

## Modificações da Versão Original

### Backend Changes

1. **`app/main.py`**
   - Removido lifespan manager complexo
   - Simplificado para execução standalone
   - Removido middleware de segurança avançado

2. **`app/tasks/ecg_tasks.py`**
   - Convertido de Celery para processamento síncrono
   - Removido Redis dependency
   - Mantida funcionalidade de análise

3. **`app/db/session.py`**
   - Configurado para SQLite
   - Criação automática de diretórios
   - Inicialização simplificada

4. **`pyproject.toml`**
   - Removido neurokit2 (problemas de compilação)
   - Adicionado scipy para processamento de sinais
   - Dependências mínimas

### Frontend Changes

1. **`package.json`**
   - Adicionado Electron dependencies
   - Scripts de build para desktop
   - Configuração electron-builder

2. **`electron/main.js`**
   - Gerenciamento do processo backend
   - Janela desktop nativa
   - IPC communication

3. **`electron/preload.js`**
   - Bridge segura entre frontend e backend
   - API exposure controlada

## Solução de Problemas

### Problemas Comuns

**Backend não inicia**
```bash
# Verificar se porta 8000 está livre
netstat -an | findstr :8000

# Matar processos conflitantes
taskkill /f /im cardioai-pro-backend.exe
```

**Interface não abre**
```bash
# Abrir manualmente
start http://localhost:8000

# Verificar firewall
# Windows Defender > Permitir aplicativo
```

**Erro de permissões**
```bash
# Executar como administrador
# Botão direito > "Executar como administrador"
```

### Logs e Diagnóstico

**Backend logs**
- Console output durante execução
- Erros aparecem na janela do prompt

**Frontend logs**
- F12 no navegador > Console
- Network tab para problemas de API

## Requisitos do Sistema

### Mínimos
- Windows 7 SP1 (64-bit)
- 2GB RAM
- 500MB espaço em disco
- Navegador moderno

### Recomendados
- Windows 10/11 (64-bit)
- 4GB RAM
- 1GB espaço em disco
- Chrome/Edge/Firefox (versões recentes)

## Segurança e Privacidade

- ✓ **Processamento local**: Dados não saem do computador
- ✓ **Sem telemetria**: Nenhum dado enviado para servidores
- ✓ **LGPD compliant**: Para dados processados localmente
- ✓ **Banco local**: SQLite criptografado (opcional)

## Limitações

### Funcionalidades Não Incluídas
- Sincronização em nuvem
- Backup automático remoto
- Atualizações automáticas
- Integração com sistemas hospitalares
- Processamento distribuído
- Cache Redis

### Performance
- Processamento sequencial (não paralelo)
- Limitado pela capacidade local
- Sem balanceamento de carga

## Suporte

### Documentação
- `INSTALACAO-STANDALONE.md`: Guia detalhado de instalação
- `LEIA-ME.txt`: Instruções básicas (incluído no pacote)

### Contato
- **GitHub**: https://github.com/drguilhermecapel/cardio.ai.pro
- **Issues**: Para reportar problemas
- **Discussions**: Para dúvidas gerais

## Versionamento

- **Versão atual**: 1.0.0 Standalone
- **Base**: CardioAI Pro v1.0.0
- **Tipo**: Desktop standalone application
- **Plataforma**: Windows (64-bit)

---

**Nota**: Esta versão standalone é otimizada para simplicidade e facilidade de uso. Para funcionalidades empresariais avançadas, considere a versão completa com Docker.
