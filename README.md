# CardioAI Pro

Sistema de análise de ECG com IA para diagnóstico médico avançado.

## 🚀 Versão Standalone Disponível!

**Nova versão simplificada para Windows - Instalação com clique duplo!**

### Downloads Standalone
- **[CardioAI-Pro-1.0.0-installer.exe](frontend/dist-electron/CardioAI-Pro-1.0.0-installer.exe)** (229 MB) - Instalador unificado

### Instalação Simples
```bash
# Opção 1: Usar instalador unificado (Recomendado)
# 1. Baixe: CardioAI-Pro-1.0.0-installer.exe
# 2. Execute como administrador
# 3. Siga o assistente de instalação
# 4. Lance pelo atalho da área de trabalho

# Opção 2: Compilar do código fonte
# 1. Clone o repositório
# 2. Navegue para windows_installer/
# 3. Execute os scripts de build:
#    python build_backend.py
#    python build_frontend.py
# 4. Compile o instalador com o NSIS:
#    makensis cardioai_installer.nsi
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
- ✅ Instalação profissional (assistente NSIS)
- ✅ Sem Docker ou dependências
- ✅ Processamento 100% local
- ✅ Ideal para usuários finais
- ✅ Backend e frontend unificados
- ❌ Apenas Windows

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
# Método 1: Instalador Unificado (Recomendado)
1. Baixe: CardioAI-Pro-1.0.0-installer.exe
2. Execute como administrador
3. Siga o assistente de instalação
4. Lance pelo atalho da área de trabalho
5. Acesse via http://localhost:8000

# Método 2: Compilar do Código Fonte
1. Clone o repositório
2. Navegue para windows_installer/
3. Execute:
   ```bash
   python build_backend.py
   python build_frontend.py
   makensis cardioai_installer.nsi
   ```
4. O instalador será gerado em `frontend/dist-electron/`
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

3. **Execute os scripts de build**:
   ```bash
   python build_backend.py
   python build_frontend.py
   makensis cardioai_installer.nsi
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
