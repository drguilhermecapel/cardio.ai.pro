# Correção do Erro de Dependência pdf2image

Este documento explica como corrigir o erro de dependência `pdf2image` no CardioAI Pro.

## Descrição do Problema

Ao iniciar o servidor backend, você pode ver o seguinte aviso:

```
WARNING:app.main:API v1 router não encontrado: No module named 'pdf2image'
```

Este erro ocorre porque o sistema tenta importar o módulo `pdf2image`, que é usado para converter arquivos PDF em imagens, mas ele não está instalado.

## Solução

### 1. Instalar a dependência pdf2image

Execute o seguinte comando para instalar o pacote `pdf2image`:

```powershell
pip install pdf2image
```

### 2. Instalar o Poppler (necessário para o pdf2image)

O pacote `pdf2image` depende do Poppler, que é uma biblioteca externa para processamento de PDFs.

#### No Windows:

1. Baixe o Poppler para Windows em: https://github.com/oschwartz10612/poppler-windows/releases/
   - Escolha a versão mais recente (por exemplo, `Release-24.02.0-0`)
   - Baixe o arquivo ZIP (por exemplo, `poppler-24.02.0-0-windows-x86_64.zip`)

2. Extraia o conteúdo para uma pasta em seu computador, por exemplo:
   ```
   C:\Program Files\poppler
   ```

3. Adicione o caminho para a pasta `bin` do Poppler ao PATH do sistema:
   - Abra o Painel de Controle > Sistema > Configurações avançadas do sistema
   - Clique em "Variáveis de Ambiente"
   - Na seção "Variáveis do Sistema", encontre a variável "Path" e clique em "Editar"
   - Clique em "Novo" e adicione o caminho para a pasta bin, por exemplo:
     ```
     C:\Program Files\poppler\bin
     ```
   - Clique em "OK" para fechar todas as janelas

#### No Linux:

```bash
# Ubuntu/Debian
sudo apt-get install poppler-utils

# Fedora/CentOS
sudo dnf install poppler-utils
```

#### No macOS:

```bash
brew install poppler
```

### 3. Reiniciar o servidor backend

Após instalar as dependências, reinicie o servidor backend:

```powershell
python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

## Explicação Técnica

O erro ocorre porque:

1. O CardioAI Pro usa o módulo `pdf2image` para processar arquivos PDF, provavelmente para extrair imagens de ECGs de documentos PDF.

2. Este módulo não está incluído nas dependências padrão listadas em `requirements.txt`.

3. O módulo `pdf2image` depende de uma biblioteca externa chamada Poppler, que precisa ser instalada separadamente do sistema de pacotes Python.

A solução instala tanto o pacote Python quanto a dependência externa necessária para seu funcionamento.