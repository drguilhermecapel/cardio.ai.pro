"""
Arquivo de configuração principal atualizado para incluir a plataforma de treinamento
"""

# Adicionar ao final do arquivo de configuração existente

# =============================================================================
# CONFIGURAÇÕES DA PLATAFORMA DE TREINAMENTO DE IA
# =============================================================================

# Habilitar/desabilitar a plataforma de treinamento
TRAINING_ENABLED = True

# Configurações de API
TRAINING_API_PORT = 8001
TRAINING_API_HOST = "0.0.0.0"

# Caminhos de dados e modelos
TRAINING_DATA_ROOT = "./data/training"
TRAINING_MODELS_ROOT = "./models/training"
TRAINING_CHECKPOINTS_ROOT = "./checkpoints/training"
TRAINING_LOGS_ROOT = "./logs/training"
TRAINING_EXPORTS_ROOT = "./exports/training"

# Configurações de recursos
TRAINING_MAX_CONCURRENT_JOBS = 2
TRAINING_GPU_ENABLED = "auto"  # "auto", "true", "false"
TRAINING_MIXED_PRECISION = True
TRAINING_NUM_WORKERS = 4

# Configurações padrão de treinamento
TRAINING_DEFAULT_BATCH_SIZE = 32
TRAINING_DEFAULT_LEARNING_RATE = 1e-4
TRAINING_DEFAULT_EPOCHS = 100
TRAINING_DEFAULT_EARLY_STOPPING_PATIENCE = 10

# Configurações de validação
TRAINING_DEFAULT_VAL_SPLIT = 0.2
TRAINING_DEFAULT_TEST_SPLIT = 0.1
TRAINING_CROSS_VALIDATION_FOLDS = 5

# Configurações de dados
TRAINING_DEFAULT_SAMPLING_RATE = 500
TRAINING_DEFAULT_SIGNAL_LENGTH = 5000
TRAINING_DEFAULT_NUM_LEADS = 12

# Configurações de augmentation
TRAINING_AUGMENTATION_PROB = 0.5
TRAINING_NOISE_LEVEL = 0.01
TRAINING_BASELINE_WANDER = True

# Configurações de logging
TRAINING_LOG_LEVEL = "INFO"
TRAINING_LOG_EVERY_N_STEPS = 50
TRAINING_SAVE_TOP_K = 3
TRAINING_TENSORBOARD_ENABLED = True
TRAINING_WANDB_PROJECT = None  # Opcional

# Configurações de segurança
TRAINING_ALLOWED_ROLES = ["admin", "researcher"]
TRAINING_REQUIRE_APPROVAL = False
TRAINING_MAX_DATASET_SIZE_GB = 100
TRAINING_MAX_MODEL_SIZE_MB = 500

# Configurações de notificação
TRAINING_NOTIFY_ON_COMPLETION = True
TRAINING_NOTIFY_ON_FAILURE = True
TRAINING_EMAIL_NOTIFICATIONS = False

# Configurações de backup
TRAINING_AUTO_BACKUP_CHECKPOINTS = True
TRAINING_BACKUP_RETENTION_DAYS = 30
TRAINING_BACKUP_LOCATION = "./backups/training"

# Configurações de monitoramento
TRAINING_MONITOR_GPU_USAGE = True
TRAINING_MONITOR_MEMORY_USAGE = True
TRAINING_ALERT_ON_HIGH_USAGE = True

# Configurações de exportação
TRAINING_DEFAULT_EXPORT_FORMAT = "pytorch"  # "pytorch", "onnx", "torchscript"
TRAINING_EXPORT_OPTIMIZATION = True
TRAINING_EXPORT_QUANTIZATION = False

# Configurações de datasets
TRAINING_AUTO_DOWNLOAD_DATASETS = False
TRAINING_DATASET_CACHE_ENABLED = True
TRAINING_DATASET_PREPROCESSING_CACHE = True

# Configurações de modelos pré-treinados
TRAINING_PRETRAINED_MODELS_PATH = "./pretrained/training"
TRAINING_AUTO_DOWNLOAD_PRETRAINED = False
TRAINING_PRETRAINED_CACHE_ENABLED = True

# Configurações de experimentação
TRAINING_EXPERIMENT_TRACKING = True
TRAINING_HYPERPARAMETER_TUNING = False
TRAINING_AUTO_MODEL_SELECTION = False

# Configurações de produção
TRAINING_PRODUCTION_READY_EXPORT = True
TRAINING_MODEL_VALIDATION_REQUIRED = True
TRAINING_PERFORMANCE_BENCHMARKING = True

# =============================================================================
# FIM DAS CONFIGURAÇÕES DE TREINAMENTO
# =============================================================================

