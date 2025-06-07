# CardioAI Pro

Sistema de prontuário eletrônico com análise de ECG baseada em inteligência artificial.

## Instalação para Windows

### Instalação Simples (Recomendada)

1. **Baixe o instalador**: `CardioAI-Pro-Installer.exe`
2. **Execute como administrador**: Clique com o botão direito no arquivo e selecione "Executar como administrador"
3. **Siga o assistente de instalação**: Aceite a licença e escolha o diretório de instalação
4. **Inicie o aplicativo**: Use o atalho criado na área de trabalho ou no menu Iniciar

O CardioAI Pro será instalado como um aplicativo desktop completo, sem necessidade de Docker, linha de comando ou configurações técnicas.

### Características da Instalação

- ✅ **Instalação com um clique**: Sem configurações complexas
- ✅ **Sem dependências externas**: Não requer Docker, PostgreSQL ou Redis
- ✅ **Banco de dados local**: Usa SQLite para armazenamento seguro
- ✅ **Interface web integrada**: Abre automaticamente no navegador
- ✅ **Atalhos convenientes**: Ícones na área de trabalho e menu Iniciar
- ✅ **Desinstalação limpa**: Remove completamente o aplicativo quando necessário

### Requisitos do Sistema

- **Sistema Operacional**: Windows 10 ou superior
- **Memória RAM**: Mínimo 4GB, recomendado 8GB
- **Espaço em disco**: 2GB livres
- **Navegador**: Chrome, Firefox, Edge ou Safari atualizado

### Para Desenvolvedores

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

### Instalação Manual (Avançada)

Para instalação manual usando Docker, consulte o arquivo `INSTALACAO.md`.
