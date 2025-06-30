# Correção do Erro de Dependência pytesseract

Este documento explica como corrigir o erro de dependência `pytesseract` no CardioAI Pro.

## Descrição do Problema

Ao iniciar o servidor backend, você pode ver o seguinte aviso:

```
WARNING:app.main:API v1 router não encontrado: No module named 'pytesseract'
```

Este erro ocorre porque o sistema tenta importar o módulo `pytesseract`, que é usado para reconhecimento óptico de caracteres (OCR) em imagens, mas ele não está instalado.

## Solução

### 1. Instalar a dependência pytesseract

Execute o seguinte comando para instalar o pacote `pytesseract`:

```powershell
pip install pytesseract
```

### 2. Instalar o Tesseract OCR (necessário para o pytesseract)

O pacote `pytesseract` é apenas um wrapper Python para o Tesseract OCR, que é uma biblioteca externa que precisa ser instalada separadamente.

#### No Windows:

1. Baixe o instalador do Tesseract OCR para Windows em: https://github.com/UB-Mannheim/tesseract/wiki
   - Escolha a versão mais recente (por exemplo, `tesseract-ocr-w64-setup-5.3.3.20231005.exe` para 64-bit)

2. Execute o instalador e siga as instruções. Recomendamos instalar em:
   ```
   C:\Program Files\Tesseract-OCR
   ```

3. Adicione o caminho para a pasta do Tesseract ao PATH do sistema:
   - Abra o Painel de Controle > Sistema > Configurações avançadas do sistema
   - Clique em "Variáveis de Ambiente"
   - Na seção "Variáveis do Sistema", encontre a variável "Path" e clique em "Editar"
   - Clique em "Novo" e adicione o caminho para a pasta do Tesseract, por exemplo:
     ```
     C:\Program Files\Tesseract-OCR
     ```
   - Clique em "OK" para fechar todas as janelas

#### No Linux:

```bash
# Ubuntu/Debian
sudo apt-get install tesseract-ocr

# Fedora/CentOS
sudo dnf install tesseract
```

#### No macOS:

```bash
brew install tesseract
```

### 3. Reiniciar o servidor backend

Após instalar as dependências, reinicie o servidor backend:

```powershell
python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

## Explicação Técnica

O erro ocorre porque:

1. O CardioAI Pro usa o módulo `pytesseract` para realizar OCR em imagens, provavelmente para extrair texto de imagens de ECGs ou documentos médicos.

2. Este módulo não está incluído nas dependências padrão listadas em `requirements.txt`.

3. O módulo `pytesseract` depende de uma biblioteca externa chamada Tesseract OCR, que precisa ser instalada separadamente do sistema de pacotes Python.

A solução instala tanto o pacote Python quanto a dependência externa necessária para seu funcionamento.