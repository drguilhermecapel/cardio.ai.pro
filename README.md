# CardioAI Pro

Sistema de análise de ECG com IA para diagnóstico médico avançado.

## 🚀 Versão Standalone Disponível!

**Nova versão simplificada para Windows - Instalação com clique duplo!**

### Downloads Standalone
- **[CardioAI-Pro-1.0.0-installer.exe](frontend/dist-electron/CardioAI-Pro-1.0.0-installer.exe)** (229 MB) - Instalador unificado
- **[build_installer.bat](windows_installer/build_installer.bat)** - Script de compilação do instalador

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
# 3. Clique duas vezes em: build_installer.bat
# 4. Use o instalador gerado
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
- 🎭 **NOVO**: Geração de avatares hiper-realistas com IA

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
3. Clique duas vezes em: build_installer.bat
4. Use o instalador gerado em frontend/dist-electron/
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

## 🎭 Avatar Generation - Geração de Avatares com IA

### Quickstart

O CardioAI Pro agora inclui um sistema avançado de geração de avatares hiper-realistas usando Stable Diffusion. Esta funcionalidade permite criar retratos fotorealistas de uma mulher caucasiana de 50 anos com óculos, ideal para interfaces terapêuticas e apresentações médicas.

#### Instalação das Dependências

```bash
# Navegue para o diretório backend
cd backend

# Instale as dependências com Poetry
poetry install

# Ou com pip (se não usar Poetry)
pip install diffusers torch accelerate safetensors pillow
```

#### Uso via API

```bash
# Exemplo básico - gera avatar com prompt padrão
curl -X POST http://localhost:8000/api/v1/avatar \
  -H "Content-Type: application/json" \
  -d '{"prompt": null}'

# Exemplo com prompt personalizado
curl -X POST http://localhost:8000/api/v1/avatar \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "smiling woman with red glasses",
    "seed": 123,
    "steps": 30,
    "height": 768,
    "width": 512
  }'
```

#### Uso via CLI

```bash
# Geração básica
python -m app.cli.avatar_cli

# Com parâmetros personalizados
python -m app.cli.avatar_cli \
  --prompt "smiling woman with red glasses" \
  --seed 123 \
  --steps 30 \
  --height 768 \
  --width 512 \
  --output-dir media/avatars
```

### GPU vs CPU - Considerações de Performance

#### 🚀 Modo GPU (Recomendado)
- **Requisitos**: NVIDIA GPU com CUDA, 4GB+ VRAM
- **Performance**: ~5-8 segundos para 768x512px
- **Otimizações automáticas**:
  - Precision fp16 para economia de VRAM
  - Attention slicing para reduzir uso de memória
  - Memory efficient attention

#### 🐌 Modo CPU (Fallback)
- **Requisitos**: CPU moderno, 8GB+ RAM
- **Performance**: ~60-120 segundos para 512x384px
- **Limitações**:
  - Precision fp32 (maior uso de memória)
  - Sem otimizações de VRAM
  - Recomendado apenas para desenvolvimento/teste

#### Configuração de Ambiente

```bash
# Para GPU (CUDA)
export CUDA_VISIBLE_DEVICES=0

# Para forçar CPU (desenvolvimento)
export CUDA_VISIBLE_DEVICES=""
```

### Personalização de Prompts

#### Prompt Padrão
```
ultra-photorealistic portrait of a 50-year-old caucasian woman,
short grayish-blonde hair, wearing modern rectangular eyeglasses,
natural soft studio lighting, 85mm lens, f/1.8,
high-resolution skin texture, subtle makeup, neutral background
```

#### Customização por Características

**Idade:**
```json
{"prompt": "ultra-photorealistic portrait of a 30-year-old caucasian woman..."}
{"prompt": "ultra-photorealistic portrait of a 65-year-old caucasian woman..."}
```

**Etnia:**
```json
{"prompt": "ultra-photorealistic portrait of a 50-year-old african woman..."}
{"prompt": "ultra-photorealistic portrait of a 50-year-old asian woman..."}
{"prompt": "ultra-photorealistic portrait of a 50-year-old latina woman..."}
```

**Acessórios:**
```json
{"prompt": "...wearing round vintage eyeglasses..."}
{"prompt": "...wearing contact lenses, no glasses..."}
{"prompt": "...wearing a medical stethoscope around neck..."}
```

**Expressões:**
```json
{"prompt": "...gentle smile, warm expression..."}
{"prompt": "...serious professional expression..."}
{"prompt": "...concerned empathetic expression..."}
```

#### Prompt Negativo (Automático)
O sistema automaticamente aplica um prompt negativo para evitar:
- Imagens borradas ou de baixa qualidade
- Estilo cartoon/anime
- Múltiplas pessoas
- Marcas d'água ou texto
- Anatomia incorreta

### Exemplos de Uso Avançado

#### Geração em Lote
```bash
# Gerar múltiplos avatares com seeds diferentes
for seed in {1..5}; do
  python -m app.cli.avatar_cli --seed $seed --output-dir media/avatars/batch_$seed
done
```

#### Integração com Docker
```dockerfile
# Adicione ao Dockerfile para suporte GPU
FROM nvidia/cuda:11.8-runtime-ubuntu20.04
# ... resto da configuração
```

#### Monitoramento de Performance
```python
import time
from app.services.avatar_generator import generate_avatar

start_time = time.time()
avatar_path = generate_avatar(steps=20, height=512, width=512)
generation_time = time.time() - start_time

print(f"Avatar gerado em {generation_time:.2f}s: {avatar_path}")
```

### Estrutura de Arquivos

```
media/
├── avatars/
│   ├── .gitkeep
│   ├── 20250609_130500.png
│   └── 20250609_130530.png
└── .gitkeep

backend/app/
├── services/
│   └── avatar_generator.py
├── api/v1/endpoints/
│   └── avatar.py
└── cli/
    └── avatar_cli.py
```

### Troubleshooting

#### Erro: "CUDA out of memory"
```bash
# Reduza a resolução
python -m app.cli.avatar_cli --height 512 --width 384

# Ou force CPU
CUDA_VISIBLE_DEVICES="" python -m app.cli.avatar_cli
```

#### Erro: "Model not found"
```bash
# Verifique conexão com internet para download do modelo
# Primeira execução baixa ~4GB do Hugging Face
```

#### Performance lenta
```bash
# Reduza steps para desenvolvimento
python -m app.cli.avatar_cli --steps 15

# Use resolução menor
python -m app.cli.avatar_cli --height 384 --width 256
```

---

**CardioAI Pro** - Revolucionando o diagnóstico de ECG com Inteligência Artificial 🚀
