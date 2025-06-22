#!/bin/bash
# Automated PTB-XL Training Pipeline for CardioAI Pro
# This script automates the entire process from setup to final model evaluation

set -e  # Exit on error

# Configuration
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXPERIMENTS_DIR="${PROJECT_DIR}/experiments/ptbxl"
VENV_DIR="${PROJECT_DIR}/venv"
PYTHON="${VENV_DIR}/bin/python"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="${EXPERIMENTS_DIR}/pipeline_${TIMESTAMP}.log"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Functions
log() {
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')]${NC} $1" | tee -a "${LOG_FILE}"
}

error() {
    echo -e "${RED}[ERROR]${NC} $1" | tee -a "${LOG_FILE}"
    exit 1
}

warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1" | tee -a "${LOG_FILE}"
}

check_requirements() {
    log "Checking system requirements..."
    
    # Check Python
    if ! command -v python3 &> /dev/null; then
        error "Python 3 is not installed"
    fi
    
    # Check CUDA (optional)
    if command -v nvidia-smi &> /dev/null; then
        log "CUDA detected:"
        nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
    else
        warning "CUDA not detected. Training will use CPU (much slower)"
    fi
    
    # Check disk space
    REQUIRED_SPACE_GB=10
    AVAILABLE_SPACE_GB=$(df -BG . | awk 'NR==2 {print $4}' | sed 's/G//')
    if [ "$AVAILABLE_SPACE_GB" -lt "$REQUIRED_SPACE_GB" ]; then
        error "Insufficient disk space. Required: ${REQUIRED_SPACE_GB}GB, Available: ${AVAILABLE_SPACE_GB}GB"
    fi
}

setup_environment() {
    log "Setting up Python environment..."
    
    # Create virtual environment if it doesn't exist
    if [ ! -d "${VENV_DIR}" ]; then
        python3 -m venv "${VENV_DIR}"
        log "Created virtual environment"
    fi
    
    # Activate virtual environment
    source "${VENV_DIR}/bin/activate"
    
    # Upgrade pip
    pip install --upgrade pip
    
    # Install requirements
    log "Installing dependencies..."
    pip install -r "${PROJECT_DIR}/backend/requirements.txt"
    pip install -r "${PROJECT_DIR}/backend/requirements-ml.txt"
}

download_ptbxl() {
    log "Checking PTB-XL dataset..."
    
    DATA_DIR="${EXPERIMENTS_DIR}/data/ptbxl"
    
    if [ -d "${DATA_DIR}" ] && [ -f "${DATA_DIR}/ptbxl_database.csv" ]; then
        log "PTB-XL dataset already exists"
    else
        log "Downloading PTB-XL dataset (~3GB)..."
        mkdir -p "${DATA_DIR}"
        
        # Download using wget with progress bar
        cd "${DATA_DIR}/.."
        wget -c -r -np -nH --cut-dirs=3 \
            --reject "index.html*" \
            --show-progress \
            https://physionet.org/files/ptb-xl/1.0.3/
        
        # Rename directory
        if [ -d "ptb-xl" ]; then
            mv ptb-xl/* ptbxl/
            rmdir ptb-xl
        fi
        
        cd "${PROJECT_DIR}"
        log "PTB-XL dataset downloaded successfully"
    fi
    
    echo "${DATA_DIR}"
}

run_baseline_training() {
    local data_path=$1
    log "Running baseline training..."
    
    ${PYTHON} backend/train_ptbxl.py \
        --data-path "${data_path}" \
        --model-type hybrid_full \
        --batch-size 32 \
        --epochs 50 \
        --lr 3e-4 \
        --mixed-precision \
        2>&1 | tee -a "${LOG_FILE}"
}

run_advanced_training() {
    local data_path=$1
    log "Running advanced training with all optimizations..."
    
    # Create custom config
    cat > "${EXPERIMENTS_DIR}/advanced_config.json" << EOF
{
    "model_type": "hybrid_full",
    "batch_size": 32,
    "epochs": 100,
    "lr": 3e-4,
    "use_curriculum": true,
    "use_multi_task": true,
    "use_augmentation": true,
    "mixup_alpha": 0.2,
    "cutmix_alpha": 1.0,
    "focal_loss": true,
    "label_smoothing": 0.1
}
EOF
    
    ${PYTHON} backend/train_ptbxl.py \
        --data-path "${data_path}" \
        --config "${EXPERIMENTS_DIR}/advanced_config.json" \
        --mixed-precision \
        2>&1 | tee -a "${LOG_FILE}"
}

run_experiments() {
    local data_path=$1
    local mode=$2
    
    case $mode in
        "quick")
            log "Running quick experiment (baseline only)..."
            ${PYTHON} backend/run_ptbxl_experiments.py \
                --experiments baseline \
                --base-dir "${EXPERIMENTS_DIR}" \
                2>&1 | tee -a "${LOG_FILE}"
            ;;
        "standard")
            log "Running standard experiments..."
            ${PYTHON} backend/run_ptbxl_experiments.py \
                --experiments baseline full_advanced superclass_hierarchical \
                --base-dir "${EXPERIMENTS_DIR}" \
                2>&1 | tee -a "${LOG_FILE}"
            ;;
        "full")
            log "Running all experiments..."
            ${PYTHON} backend/run_ptbxl_experiments.py \
                --all \
                --base-dir "${EXPERIMENTS_DIR}" \
                2>&1 | tee -a "${LOG_FILE}"
            ;;
        *)
            error "Unknown mode: $mode"
            ;;
    esac
}

evaluate_models() {
    log "Evaluating trained models..."
    
    # Find best model
    BEST_MODEL=$(find "${EXPERIMENTS_DIR}" -name "best_model.pth" -type f | \
                 xargs ls -t | head -n 1)
    
    if [ -z "$BEST_MODEL" ]; then
        error "No trained models found"
    fi
    
    log "Best model found: ${BEST_MODEL}"
    
    # Run evaluation script
    ${PYTHON} - << EOF
import torch
import json
from pathlib import Path

# Load model
checkpoint = torch.load("${BEST_MODEL}", map_location='cpu')
print("\nModel Information:")
print(f"Epoch: {checkpoint.get('epoch', 'Unknown')}")
print(f"Best validation metrics:")
for key, value in checkpoint.get('metrics', {}).items():
    if isinstance(value, float):
        print(f"  {key}: {value:.4f}")

# Save summary
summary = {
    "model_path": "${BEST_MODEL}",
    "metrics": checkpoint.get('metrics', {}),
    "config": checkpoint.get('config', {})
}

summary_path = Path("${EXPERIMENTS_DIR}") / "best_model_summary.json"
with open(summary_path, 'w') as f:
    json.dump(summary, f, indent=2)

print(f"\nSummary saved to: {summary_path}")
EOF
}

generate_report() {
    log "Generating final report..."
    
    REPORT_FILE="${EXPERIMENTS_DIR}/final_report_${TIMESTAMP}.md"
    
    cat > "${REPORT_FILE}" << EOF
# PTB-XL Training Pipeline Report

Generated: $(date)

## System Information
- Python: $(python3 --version)
- PyTorch: $(${PYTHON} -c "import torch; print(f'PyTorch {torch.__version__}')")
- CUDA Available: $(${PYTHON} -c "import torch; print(torch.cuda.is_available())")
- GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo "N/A")

## Training Summary

### Experiments Run
$(ls -la "${EXPERIMENTS_DIR}"/*/summary.json 2>/dev/null | wc -l) experiments completed

### Best Model Performance
$(${PYTHON} -c "
import json
from pathlib import Path
summary_file = Path('${EXPERIMENTS_DIR}') / 'best_model_summary.json'
if summary_file.exists():
    with open(summary_file) as f:
        data = json.load(f)
        metrics = data.get('metrics', {})
        print(f'- Validation AUC: {metrics.get(\"val_auc_macro\", 0):.4f}')
        print(f'- Validation F1: {metrics.get(\"val_f1_macro\", 0):.4f}')
else:
    print('No summary found')
")

### Training Logs
See: ${LOG_FILE}

### Model Checkpoints
$(find "${EXPERIMENTS_DIR}" -name "*.pth" -type f | wc -l) model files saved

## Next Steps

1. **Deploy Best Model**:
   \`\`\`bash
   cp ${BEST_MODEL:-"best_model.pth"} /path/to/production/
   \`\`\`

2. **Run Inference**:
   \`\`\`python
   from app.services.advanced_ml_service import AdvancedMLService
   service = AdvancedMLService(model_path="${BEST_MODEL:-"best_model.pth"}")
   \`\`\`

3. **Fine-tune on Custom Data**:
   Use the trained model as starting point for your specific ECG data.

EOF
    
    log "Report generated: ${REPORT_FILE}"
    cat "${REPORT_FILE}"
}

# Main execution
main() {
    # Parse arguments
    MODE=${1:-"standard"}  # quick, standard, or full
    
    # Create directories
    mkdir -p "${EXPERIMENTS_DIR}"
    
    # Start logging
    log "Starting PTB-XL automated pipeline (mode: ${MODE})"
    
    # Run pipeline steps
    check_requirements
    setup_environment
    
    # Download dataset
    DATA_PATH=$(download_ptbxl)
    
    # Run training based on mode
    case $MODE in
        "baseline")
            run_baseline_training "${DATA_PATH}"
            ;;
        "advanced")
            run_advanced_training "${DATA_PATH}"
            ;;
        *)
            run_experiments "${DATA_PATH}" "${MODE}"
            ;;
    esac
    
    # Evaluate and report
    evaluate_models
    generate_report
    
    log "Pipeline completed successfully!"
    log "Results available in: ${EXPERIMENTS_DIR}"
}

# Show usage if needed
if [ "$1" == "-h" ] || [ "$1" == "--help" ]; then
    echo "Usage: $0 [mode]"
    echo ""
    echo "Modes:"
    echo "  quick     - Run baseline experiment only (~1 hour)"
    echo "  standard  - Run baseline, advanced, and hierarchical (~4 hours)"
    echo "  full      - Run all experiments (~8 hours)"
    echo "  baseline  - Train single baseline model"
    echo "  advanced  - Train single advanced model"
    echo ""
    echo "Example:"
    echo "  $0 standard"
    exit 0
fi

# Run main function
main "$@"
