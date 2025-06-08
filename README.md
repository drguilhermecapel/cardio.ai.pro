# CardioAI Pro

Sistema de análise de ECG com IA para diagnóstico médico avançado.

## 🚀 Versão Standalone Disponível!

**Nova versão simplificada para Windows - Instalação com clique duplo!**

### Downloads Standalone
- **[CardioAI-Pro-1.0.0-portable.exe](frontend/dist-electron/CardioAI-Pro-1.0.0-portable.exe)** (209 MB) - Executável portátil
- **[build_installer.bat](windows_installer/build_installer.bat)** - Script de compilação do instalador

### Instalação Simples
```bash
# Opção 1: Usar executável portátil (Recomendado)
# 1. Baixe: CardioAI-Pro-1.0.0-portable.exe
# 2. Clique duas vezes no arquivo
# 3. Pronto! Sistema inicia automaticamente

# Opção 2: Compilar do código fonte
# 1. Clone o repositório
# 2. Navegue para windows_installer/
# 3. Clique duas vezes em: build_installer.bat
# 4. Use o executável gerado
```

**📖 [Guia Completo da Versão Standalone](README-STANDALONE.md)**

---

## Características

- 🔬 Análise automática de ECG com IA
- 📊 Interface web responsiva
- 🏥 Compliance médico (ANVISA/FDA)
- 🔒 Segurança LGPD/HIPAA
- 📱 API REST completa
- 🚀 Deploy com Docker
- 💻 **NOVO**: Versão standalone para Windows

## Versões Disponíveis

### 🖥️ Standalone (Windows)
- ✅ Instalação simples (clique duplo)
- ✅ Sem Docker ou dependências
- ✅ Processamento 100% local
- ✅ Ideal para usuários finais
- ❌ Apenas Windows
- ❌ Funcionalidades limitadas

### 🐳 Docker (Multiplataforma)
- ✅ Funcionalidades completas
- ✅ Multiplataforma (Linux/Mac/Windows)
- ✅ Escalabilidade
- ✅ Ideal para desenvolvimento
- ❌ Requer conhecimento técnico
- ❌ Configuração complexa

## Instalação

### 🎯 Versão Standalone (Recomendada para Usuários)

**Requisitos**: Apenas Windows 7+ (64-bit)

```bash
# Método 1: Executável Portátil (Recomendado)
1. Baixe: CardioAI-Pro-1.0.0-portable.exe
2. Clique duas vezes no arquivo
3. Sistema inicia automaticamente
4. Acesse via http://localhost:8000

# Método 2: Compilar do Código Fonte
1. Clone o repositório
2. Navegue para windows_installer/
3. Clique duas vezes em: build_installer.bat
4. Use o executável gerado em frontend/dist-electron/
```

### 🐳 Versão Docker (Para Desenvolvedores)

**Requisitos**: Docker, Docker Compose, Git, 8GB RAM

```bash
# Clone o repositório
git clone https://github.com/drguilhermecapel/cardio.ai.pro.git
cd cardio.ai.pro

# Inicie os serviços
docker-compose up -d

# Acesse a aplicação
open http://localhost:3000
```

### 🔧 Compilação do Instalador (Para Desenvolvedores)

Se você deseja compilar o instalador a partir do código fonte:

1. **Clone o repositório**:
   ```bash
   git clone https://github.com/drguilhermecapel/cardio.ai.pro.git
   cd cardio.ai.pro
   ```

2. **Navegue para o diretório do instalador**:
   ```bash
   cd windows_installer
   ```

3. **Execute o script de compilação**:
   ```bash
   build_installer.bat
   ```

Para instruções detalhadas de desenvolvimento, consulte `windows_installer/README.md`.

## Uso

### Interface Web

**Versão Standalone**
- **URL**: http://localhost:8000 (abre automaticamente)
- **Login**: admin / (senha gerada automaticamente no primeiro uso - veja logs do sistema)

**Versão Docker**
- **URL**: http://localhost:3000
- **Admin**: admin@cardioai.pro / (senha gerada automaticamente no primeiro uso - veja logs do sistema)
- **Docs API**: http://localhost:8000/docs

## Suporte

- 📧 Email: suporte@cardioai.pro
- 💬 Discord: [CardioAI Community](https://discord.gg/cardioai)
- 📖 Docs: [docs.cardioai.pro](https://docs.cardioai.pro)
- 🐛 Issues: [GitHub Issues](https://github.com/drguilhermecapel/cardio.ai.pro/issues)

### Documentação Adicional
- **[DISTRIBUTION_GUIDE.md](DISTRIBUTION_GUIDE.md)** - Guia completo de distribuição
- **[WINDOWS_INSTALLER_TEST_REPORT.md](WINDOWS_INSTALLER_TEST_REPORT.md)** - Relatório de testes do instalador
- **[windows_installer/COMO_USAR_INSTALADOR.md](windows_installer/COMO_USAR_INSTALADOR.md)** - Instruções detalhadas de compilação

---

**CardioAI Pro** - Revolucionando o diagnóstico de ECG com Inteligência Artificial 🚀
