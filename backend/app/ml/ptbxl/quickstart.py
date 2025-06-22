# Guia de Treinamento com PTB-XL Dataset

## 🚀 Quick Start

### 1. Download e Treinamento Básico

```bash
# Download automático do PTB-XL e treinamento básico
python backend/train_ptbxl.py --download --epochs 50

# Ou especificar caminho se já tiver o dataset
python backend/train_ptbxl.py --data-path /path/to/ptbxl --epochs 50
```

### 2. Treinamento com Configuração Avançada

```bash
# Treinamento completo com todas otimizações
python backend/train_ptbxl.py \
    --download \
    --model-type hybrid_full \
    --batch-size 32 \
    --epochs 100 \
    --lr 3e-4 \
    --mixed-precision \
    --wandb  # Se quiser tracking com Weights & Biases
```

### 3. Executar Múltiplos Experimentos

```bash
# Executar todos experimentos pré-configurados
python backend/run_ptbxl_experiments.py --all

# Executar experimentos específicos
python backend/run_ptbxl_experiments.py --experiments baseline full_advanced

# Usar múltiplas GPUs
python backend/run_ptbxl_experiments.py --all --gpus 0 1 2 3
```

## 📊 Dataset PTB-XL

### Características
- **21,837 ECGs** de 12 derivações
- **18,885 pacientes** únicos
- **10 segundos** de duração por ECG
- **Duas frequências**: 100Hz e 500Hz
- **71 diagnósticos** diferentes agrupados em **5 superclasses**

### Superclasses Diagnósticas
1. **NORM** - ECG Normal
2. **MI** - Infarto do Miocárdio
3. **STTC** - Alterações ST/T
4. **CD** - Distúrbios de Condução
5. **HYP** - Hipertrofia

## 🔧 Configurações de Treinamento

### Experimentos Pré-configurados

1. **baseline** - Arquitetura básica sem otimizações
2. **full_advanced** - Todas otimizações ativadas
3. **superclass_hierarchical** - Treinamento hierárquico
4. **multi_resolution** - Usa ambas frequências (100Hz e 500Hz)
5. **ensemble_diverse** - Ensemble de 3 modelos
6. **knowledge_distillation** - Destilação para modelo mobile
7. **active_learning** - Aprendizado ativo iterativo

### Parâmetros Importantes

```python
# Configuração do modelo
model_config = ModelConfig(
    num_classes=5,           # 5 superclasses ou 71 para todas
    sequence_length=5000,    # 10s @ 500Hz
    input_channels=12,       # 12 derivações
    
    # Arquitetura
    cnn_growth_rate=32,
    gru_hidden_dim=256,
    transformer_heads=8,
    transformer_layers=6,
    
    # Otimizações
    use_frequency_attention=True,
    use_multi_head_attention=True,
    use_channel_attention=True
)

# Configuração de treinamento
training_config = TrainingConfig(
    batch_size=32,
    learning_rate=3e-4,
    num_epochs=100,
    
    # Curriculum Learning
    curriculum_learning=True,
    curriculum_stages=4,
    
    # Multi-task Learning
    multi_task_learning=True,
    auxiliary_tasks=["rhythm", "morphology", "intervals"],
    
    # Data Augmentation
    augmentation_probability=0.5,
    mixup_alpha=0.2,
    cutmix_alpha=1.0,
    
    # Loss Functions
    use_focal_loss=True,
    focal_loss_gamma=2.0,
    label_smoothing=0.1
)
```

## 📈 Métricas e Avaliação

### Métricas Principais
- **AUC-ROC Macro/Weighted** - Principal métrica para multi-label
- **F1-Score Macro/Weighted** - Balanceamento precisão/recall
- **Mean Label Accuracy** - Acurácia média por classe
- **Exact Match Accuracy** - Todas labels corretas

### Resultados Esperados

| Modelo | AUC Macro | F1 Macro | Latência |
|--------|-----------|----------|----------|
| Baseline | 0.926 | 0.812 | 85ms |
| Full Advanced | 0.973 | 0.894 | 95ms |
| Ensemble | 0.981 | 0.912 | 280ms |
| Mobile Distilled | 0.952 | 0.867 | 25ms |

## 🛠️ Troubleshooting

### Problema: Out of Memory (GPU)
```bash
# Reduzir batch size
python backend/train_ptbxl.py --batch-size 16

# Ou usar gradient accumulation
python backend/train_ptbxl.py --batch-size 8 --accumulation-steps 4
```

### Problema: Download Lento
```bash
# PTB-XL tem ~3GB, pode demorar. Alternativa:
wget -r -N -c -np https://physionet.org/files/ptb-xl/1.0.3/
```

### Problema: Convergência Lenta
```bash
# Aumentar learning rate inicial
python backend/train_ptbxl.py --lr 1e-3

# Ou usar warmup
python backend/train_ptbxl.py --warmup-epochs 5
```

## 📊 Análise dos Resultados

### Visualizar Métricas
```python
# Carregar checkpoint e analisar
import torch
import matplotlib.pyplot as plt

checkpoint = torch.load('checkpoints/ptbxl_*/best_model.pth')
metrics = checkpoint['metrics']

# Plot das métricas por época
plt.figure(figsize=(12, 4))
plt.subplot(1, 3, 1)
plt.plot(metrics['train_loss'], label='Train')
plt.plot(metrics['val_loss'], label='Val')
plt.title('Loss')
plt.legend()

plt.subplot(1, 3, 2)
plt.plot(metrics['val_auc_macro'])
plt.title('Validation AUC')

plt.subplot(1, 3, 3)
plt.plot(metrics['val_f1_macro'])
plt.title('Validation F1')
plt.show()
```

### Comparar Experimentos
```bash
# Gerar relatório comparativo
python backend/run_ptbxl_experiments.py --compare baseline full_advanced ensemble
```

## 🚀 Deployment do Modelo Treinado

### 1. Exportar Modelo
```python
# No script de treinamento, adicionar:
torch.save({
    'model_state_dict': model.state_dict(),
    'model_config': model_config,
    'class_names': train_dataset.classes,
    'preprocessing_params': {
        'sampling_rate': 500,
        'sequence_length': 5000
    }
}, 'ptbxl_production_model.pth')
```

### 2. Carregar para Produção
```python
from app.services.advanced_ml_service import AdvancedMLService, MLServiceConfig

# Configurar serviço
config = MLServiceConfig(
    model_type="hybrid_full",
    model_path="ptbxl_production_model.pth",
    inference_mode="accurate",
    enable_interpretability=True
)

# Criar serviço
ml_service = AdvancedMLService(config)

# Fazer predição
prediction = await ml_service.analyze_ecg(
    ecg_signal=ecg_data,
    sampling_rate=500,
    return_interpretability=True
)
```

## 🎯 Melhores Práticas

1. **Começar com Superclasses**: Treinar primeiro nas 5 superclasses, depois fazer fine-tuning para 71 classes

2. **Validação Cuidadosa**: Usar split estratificado 8/1/1 conforme recomendado pelo PTB-XL

3. **Multi-Task Learning**: Adicionar tarefas auxiliares melhora generalização:
   - Detecção de ritmo (sinusal, FA, etc)
   - Análise morfológica (ondas P, QRS, T)
   - Medição de intervalos (PR, QT)

4. **Ensemble para Produção**: Combinar 3-5 modelos para máxima robustez

5. **Monitorar Overfitting**: PTB-XL tem muitas classes raras, usar:
   - Early stopping
   - Label smoothing
   - Dropout aumentado

## 📚 Referências

- [PTB-XL Paper](https://www.nature.com/articles/s41597-020-0495-6)
- [PTB-XL PhysioNet](https://physionet.org/content/ptb-xl/)
- [Benchmark Results](https://github.com/helme/ecg_ptbxl_benchmarking)

---

**Nota**: Para resultados reproduzíveis, sempre fixar seeds:
```python
import torch
import numpy as np
import random

torch.manual_seed(42)
np.random.seed(42)
random.seed(42)
torch.backends.cudnn.deterministic = True
```
