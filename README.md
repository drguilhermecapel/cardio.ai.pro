# CardioAI Pro

Sistema de análise de ECG com IA para diagnóstico médico avançado.

## 🚀 Versão Standalone Disponível!

**Nova versão simplificada para Windows - Instalação com clique duplo!**

### Downloads Standalone
- **[CardioAI-Pro-v1.0.0-Setup.exe](CardioAI-Pro-v1.0.0-Setup.exe)** (38.6 MB) - Instalador profissional
- **[install-cardioai-pro-standalone.bat](install-cardioai-pro-standalone.bat)** - Instalação automática
- **[CardioAI-Pro-v1.0.0-Portable.zip](CardioAI-Pro-v1.0.0-Portable.zip)** (40.4 MB) - Versão portátil

### Instalação Simples
```bash
# 1. Baixe os arquivos (instalador + script)
# 2. Coloque na mesma pasta
# 3. Clique duas vezes em: install-cardioai-pro-standalone.bat
# 4. Pronto! Sistema instalado e funcionando
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
# Método 1: Instalação Automática
1. Baixe: install-cardioai-pro-standalone.bat + CardioAI-Pro-v1.0.0-Setup.exe
2. Coloque na mesma pasta
3. Clique duas vezes no arquivo .bat
4. Aguarde a instalação
5. Use o atalho da área de trabalho

# Método 2: Versão Portátil
1. Baixe: CardioAI-Pro-v1.0.0-Portable.zip
2. Extraia para uma pasta
3. Clique duas vezes em: CardioAI-Pro.bat
4. Sistema inicia automaticamente
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
- **Login**: admin / admin123

**Versão Docker**
- **URL**: http://localhost:3000
- **Admin**: admin@cardioai.pro / admin123
- **Docs API**: http://localhost:8000/docs

## Suporte

- 📧 Email: suporte@cardioai.pro
- 💬 Discord: [CardioAI Community](https://discord.gg/cardioai)
- 📖 Docs: [docs.cardioai.pro](https://docs.cardioai.pro)
- 🐛 Issues: [GitHub Issues](https://github.com/drguilhermecapel/cardio.ai.pro/issues)

### Documentação Adicional
- **[README-STANDALONE.md](README-STANDALONE.md)** - Guia completo da versão standalone
- **[INSTALACAO-STANDALONE.md](INSTALACAO-STANDALONE.md)** - Instruções detalhadas de instalação

---

**CardioAI Pro** - Revolucionando o diagnóstico de ECG com Inteligência Artificial 🚀
