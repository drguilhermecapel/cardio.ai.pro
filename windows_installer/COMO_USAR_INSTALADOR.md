# Como Usar o Instalador Windows do CardioAI Pro

## 📋 Pré-requisitos
- Windows 10 ou 11 (64-bit)
- Conexão com internet (para download automático do Node.js se necessário)

## 🚀 Instruções de Uso

### Passo 1: Baixar o Projeto
1. Baixe ou clone o projeto CardioAI Pro
2. Extraia os arquivos se necessário

### Passo 2: Navegar para o Diretório Correto
1. Abra o **Prompt de Comando** (cmd) ou **PowerShell**
2. Navegue até a pasta do projeto:
   ```cmd
   cd caminho\para\cardio.ai.pro
   ```
3. Entre na pasta do instalador:
   ```cmd
   cd windows_installer
   ```

### Passo 3: Executar o Instalador
```cmd
build_installer.bat
```

## ✅ O que o Instalador Faz Automaticamente

### 1. **Verificação de Dependências**
- ✅ Verifica se Python 3.8+ está instalado
- ✅ Verifica se Node.js está disponível
- ✅ **Baixa Node.js automaticamente** se não encontrado
- ✅ Verifica se NSIS está disponível (opcional)

### 2. **Build do Backend**
- ✅ Instala Poetry automaticamente se necessário
- ✅ Instala dependências Python
- ✅ Cria executável standalone (`cardioai-backend.exe`)
- ✅ Gera script de inicialização (`start_backend.bat`)

### 3. **Build do Frontend**
- ✅ Usa Node.js portable ou do sistema
- ✅ Instala dependências npm automaticamente
- ✅ Compila aplicação React/Vite
- ✅ Cria servidor frontend (`serve_frontend.py`)

### 4. **Criação do Instalador**
- ✅ Prepara todos os arquivos necessários
- ✅ Cria ícone válido automaticamente
- ✅ Gera arquivo de licença
- ✅ Compila instalador NSIS (se disponível)

## 🎯 Resultado Final

Após a execução bem-sucedida, você terá:

### Se NSIS estiver disponível:
- `CardioAI-Pro-Installer.exe` - **Instalador final para distribuição**

### Se NSIS não estiver disponível:
- `cardioai-backend.exe` - Executável do backend
- `frontend_build/` - Arquivos do frontend
- `serve_frontend.py` - Servidor do frontend
- `cardioai_installer.nsi` - Script NSIS (para compilação manual)

## 🔧 Solução de Problemas

### Erro: "Please run this script from the windows_installer directory"
**Solução**: Certifique-se de estar no diretório correto:
```cmd
cd cardio.ai.pro\windows_installer
build_installer.bat
```

### Erro: "Python is not installed"
**Solução**: 
1. Baixe Python 3.8+ de https://python.org
2. Durante a instalação, marque "Add Python to PATH"
3. Reinicie o Prompt de Comando

### Erro: "Node.js is not installed"
**Solução**: O instalador baixará automaticamente! Se falhar:
1. Baixe Node.js LTS de https://nodejs.org
2. Durante a instalação, marque "Add to PATH"
3. Reinicie o computador

### Erro: "NSIS is not installed"
**Solução**: NSIS é opcional. Para instalá-lo:
1. Baixe NSIS de https://nsis.sourceforge.io/
2. Instale e adicione ao PATH
3. Execute o script novamente

## 📞 Suporte

Se encontrar problemas:
1. Verifique se está no diretório `windows_installer`
2. Certifique-se de ter permissões de administrador
3. Verifique sua conexão com internet
4. Consulte os arquivos de log gerados durante o build

## 🎉 Distribuição

O arquivo `CardioAI-Pro-Installer.exe` pode ser distribuído para usuários finais. Eles só precisam:
1. Executar o instalador
2. Seguir as instruções na tela
3. Usar o CardioAI Pro instalado

**Não há necessidade de instalação manual de dependências para usuários finais!**
