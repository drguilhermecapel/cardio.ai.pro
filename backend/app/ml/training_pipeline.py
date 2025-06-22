"""
Advanced Training Pipeline for Hybrid CNN-BiGRU-Transformer Architecture
Implements curriculum learning, multi-task learning, and advanced optimization
Based on state-of-the-art research for ECG analysis
"""

import json
import logging
import os
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, OneCycleLR
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torch.utils.tensorboard import SummaryWriter

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    wandb = None
    WANDB_AVAILABLE = False

from scipy import signal as scipy_signal
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from tqdm import tqdm

from app.core.scp_ecg_conditions import SCP_ECG_CONDITIONS
from app.datasets.ecg_public_datasets import ECGDatasetLoader, ECGRecord
from app.ml.hybrid_architecture import ModelConfig, create_hybrid_model
from app.preprocessing.advanced_pipeline import AdvancedECGPreprocessor

logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    """Enhanced configuration for training pipeline"""

    model_config: ModelConfig

    # Basic training parameters
    batch_size: int = 32
    learning_rate: float = 3e-4
    num_epochs: int = 100
    weight_decay: float = 1e-5
    gradient_clip_norm: float = 1.0

    # Curriculum learning
    curriculum_learning: bool = True
    curriculum_stages: int = 4
    curriculum_difficulty_increase: float = 0.25
    curriculum_easy_samples_ratio: float = 0.3

    # Advanced optimization
    use_mixed_precision: bool = True
    use_gradient_accumulation: bool = True
    accumulation_steps: int = 4
    
    # Learning rate scheduling
    scheduler_type: str = "cosine_warm_restarts"  # "cosine_warm_restarts", "one_cycle", "plateau"
    warmup_epochs: int = 5
    min_lr: float = 1e-6
    
    # Multi-task learning
    multi_task_learning: bool = True
    auxiliary_tasks: List[str] = None  # ["rhythm", "morphology", "intervals"]
    task_weights: Dict[str, float] = None

    # Data augmentation
    augmentation_probability: float = 0.5
    mixup_alpha: float = 0.2
    cutmix_alpha: float = 1.0
    noise_std: float = 0.02
    amplitude_scale_range: tuple[float, float] = (0.8, 1.2)
    time_shift_range: int = 50
    baseline_wander_amplitude: float = 0.05

    # Multimodal features
    use_spectrograms: bool = True
    use_wavelets: bool = True
    use_heart_rate_variability: bool = True
    spectrogram_nperseg: int = 256
    spectrogram_noverlap: int = 128
    wavelet_name: str = "db6"
    wavelet_levels: int = 6

    # Loss functions
    use_focal_loss: bool = True
    focal_loss_alpha: float = 0.25
    focal_loss_gamma: float = 2.0
    use_label_smoothing: bool = True
    label_smoothing: float = 0.1
    use_class_weights: bool = True

    # Validation and evaluation
    validation_split: float = 0.2
    k_fold_cv: bool = False
    k_folds: int = 5
    early_stopping_patience: int = 15
    early_stopping_min_delta: float = 0.001

    # Model saving and checkpointing
    save_dir: str = "checkpoints"
    save_best_only: bool = True
    save_frequency: int = 5
    
    # Logging and monitoring
    log_frequency: int = 10
    use_tensorboard: bool = True
    use_wandb: bool = WANDB_AVAILABLE
    wandb_project: str = "cardioai-pro"
    
    # Hardware optimization
    num_workers: int = 4
    pin_memory: bool = True
    prefetch_factor: int = 2

    def __post_init__(self):
        if self.auxiliary_tasks is None:
            self.auxiliary_tasks = ["rhythm", "morphology", "intervals"]
        
        if self.task_weights is None:
            self.task_weights = {
                "main": 1.0,
                "rhythm": 0.3,
                "morphology": 0.3,
                "intervals": 0.2
            }


class ECGAugmentation:
    """Advanced ECG augmentation techniques"""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.preprocessor = AdvancedECGPreprocessor()
    
    def apply_mixup(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        alpha: float = 0.2
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
        """Apply mixup augmentation"""
        if alpha > 0:
            lam = np.random.beta(alpha, alpha)
        else:
            lam = 1
        
        batch_size = x.size(0)
        index = torch.randperm(batch_size).to(x.device)
        
        mixed_x = lam * x + (1 - lam) * x[index, :]
        y_a, y_b = y, y[index]
        
        return mixed_x, y_a, y_b, lam
    
    def apply_cutmix(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        alpha: float = 1.0
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
        """Apply CutMix augmentation for time series"""
        if alpha > 0:
            lam = np.random.beta(alpha, alpha)
        else:
            lam = 1
        
        batch_size = x.size(0)
        index = torch.randperm(batch_size).to(x.device)
        
        # Get cut position
        cut_length = int(x.size(2) * (1 - lam))
        cut_start = np.random.randint(0, x.size(2) - cut_length + 1)
        
        # Apply cutmix
        x[:, :, cut_start:cut_start + cut_length] = x[index, :, cut_start:cut_start + cut_length]
        
        y_a, y_b = y, y[index]
        
        return x, y_a, y_b, lam
    
    def apply_time_warping(self, x: torch.Tensor, strength: float = 0.2) -> torch.Tensor:
        """Apply time warping augmentation"""
        batch_size, channels, length = x.shape
        
        # Generate random warping field
        num_anchors = 10
        anchors = torch.linspace(0, length - 1, num_anchors)
        
        # Random displacement
        displacement = torch.randn(num_anchors) * strength * length / num_anchors
        
        # Interpolate to get warping field
        original_grid = torch.arange(length, dtype=torch.float32)
        warped_grid = torch.zeros_like(original_grid)
        
        for i in range(num_anchors - 1):
            start_idx = int(anchors[i])
            end_idx = int(anchors[i + 1])
            
            warped_grid[start_idx:end_idx] = torch.linspace(
                anchors[i] + displacement[i],
                anchors[i + 1] + displacement[i + 1],
                end_idx - start_idx
            )
        
        # Clamp to valid range
        warped_grid = torch.clamp(warped_grid, 0, length - 1)
        
        # Apply warping
        warped_x = torch.zeros_like(x)
        for b in range(batch_size):
            for c in range(channels):
                warped_x[b, c] = torch.from_numpy(
                    np.interp(original_grid, warped_grid, x[b, c].cpu().numpy())
                )
        
        return warped_x.to(x.device)
    
    def apply_baseline_wander(self, x: torch.Tensor, amplitude: float = 0.05) -> torch.Tensor:
        """Add baseline wander to ECG signal"""
        batch_size, channels, length = x.shape
        
        # Generate low-frequency sinusoidal wander
        freq = np.random.uniform(0.1, 0.5)  # Hz
        phase = np.random.uniform(0, 2 * np.pi)
        
        time = torch.linspace(0, length / 500, length)  # Assuming 500Hz sampling
        wander = amplitude * torch.sin(2 * np.pi * freq * time + phase)
        
        # Add wander to all channels
        x = x + wander.unsqueeze(0).unsqueeze(0).to(x.device)
        
        return x
    
    def augment_batch(
        self,
        x: torch.Tensor,
        y: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply random augmentations to batch"""
        if np.random.random() < self.config.augmentation_probability:
            # Choose augmentation type
            aug_type = np.random.choice(['mixup', 'cutmix', 'time_warp', 'baseline', 'noise'])
            
            if aug_type == 'mixup':
                x, y_a, y_b, lam = self.apply_mixup(x, y, self.config.mixup_alpha)
                # For simplicity, return interpolated labels
                y = lam * y_a + (1 - lam) * y_b
            
            elif aug_type == 'cutmix':
                x, y_a, y_b, lam = self.apply_cutmix(x, y, self.config.cutmix_alpha)
                y = lam * y_a + (1 - lam) * y_b
            
            elif aug_type == 'time_warp':
                x = self.apply_time_warping(x)
            
            elif aug_type == 'baseline':
                x = self.apply_baseline_wander(x, self.config.baseline_wander_amplitude)
            
            elif aug_type == 'noise':
                noise = torch.randn_like(x) * self.config.noise_std
                x = x + noise
        
        return x, y


class FocalLoss(nn.Module):
    """Focal Loss for addressing class imbalance"""
    
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, reduction: str = 'mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class MultiTaskLoss(nn.Module):
    """Multi-task loss for auxiliary tasks"""
    
    def __init__(self, config: TrainingConfig):
        super().__init__()
        self.config = config
        self.task_weights = config.task_weights
        
        # Loss functions for each task
        self.losses = nn.ModuleDict({
            'main': FocalLoss(config.focal_loss_alpha, config.focal_loss_gamma)
                   if config.use_focal_loss else nn.CrossEntropyLoss(),
            'rhythm': nn.CrossEntropyLoss(),
            'morphology': nn.CrossEntropyLoss(),
            'intervals': nn.MSELoss()
        })
    
    def forward(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Compute multi-task loss"""
        total_loss = 0.0
        task_losses = {}
        
        for task, pred in predictions.items():
            if task in targets and task in self.losses:
                loss = self.losses[task](pred, targets[task])
                weighted_loss = self.task_weights.get(task, 1.0) * loss
                total_loss += weighted_loss
                task_losses[task] = loss.item()
        
        return total_loss, task_losses


class CurriculumScheduler:
    """Curriculum learning scheduler"""
    
    def __init__(self, config: TrainingConfig, dataset_size: int):
        self.config = config
        self.dataset_size = dataset_size
        self.current_stage = 0
        self.epochs_per_stage = config.num_epochs // config.curriculum_stages
    
    def get_training_subset(self, epoch: int, difficulty_scores: np.ndarray) -> np.ndarray:
        """Get training subset based on curriculum stage"""
        if not self.config.curriculum_learning:
            return np.arange(self.dataset_size)
        
        # Determine current stage
        stage = min(epoch // self.epochs_per_stage, self.config.curriculum_stages - 1)
        
        # Calculate subset size
        min_ratio = self.config.curriculum_easy_samples_ratio
        max_ratio = 1.0
        current_ratio = min_ratio + (max_ratio - min_ratio) * (stage / (self.config.curriculum_stages - 1))
        
        subset_size = int(self.dataset_size * current_ratio)
        
        # Sort by difficulty and select easiest samples
        sorted_indices = np.argsort(difficulty_scores)
        selected_indices = sorted_indices[:subset_size]
        
        return selected_indices


class ECGTrainer:
    """Advanced trainer for ECG models"""
    
    def __init__(
        self,
        model: nn.Module,
        config: TrainingConfig,
        train_dataset: Dataset,
        val_dataset: Dataset,
        device: str = 'cuda'
    ):
        self.model = model.to(device)
        self.config = config
        self.device = device
        
        # Datasets and loaders
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        
        # Initialize components
        self._setup_optimization()
        self._setup_augmentation()
        self._setup_logging()
        
        # Training state
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.best_val_f1 = 0.0
        
    def _setup_optimization(self):
        """Setup optimizer, scheduler, and loss functions"""
        # Optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
            betas=(0.9, 0.999)
        )
        
        # Learning rate scheduler
        if self.config.scheduler_type == "cosine_warm_restarts":
            self.scheduler = CosineAnnealingWarmRestarts(
                self.optimizer,
                T_0=10,
                T_mult=2,
                eta_min=self.config.min_lr
            )
        elif self.config.scheduler_type == "one_cycle":
            self.scheduler = OneCycleLR(
                self.optimizer,
                max_lr=self.config.learning_rate,
                epochs=self.config.num_epochs,
                steps_per_epoch=len(self.train_dataset) // self.config.batch_size
            )
        
        # Loss function
        self.criterion = MultiTaskLoss(self.config) if self.config.multi_task_learning else (
            FocalLoss(self.config.focal_loss_alpha, self.config.focal_loss_gamma)
            if self.config.use_focal_loss else nn.CrossEntropyLoss()
        )
        
        # Mixed precision training
        if self.config.use_mixed_precision:
            self.scaler = GradScaler()
        
        # Curriculum learning
        self.curriculum_scheduler = CurriculumScheduler(
            self.config,
            len(self.train_dataset)
        )
    
    def _setup_augmentation(self):
        """Setup data augmentation"""
        self.augmenter = ECGAugmentation(self.config)
    
    def _setup_logging(self):
        """Setup logging and monitoring"""
        # Create directories
        self.save_dir = Path(self.config.save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # TensorBoard
        if self.config.use_tensorboard:
            self.tb_writer = SummaryWriter(self.save_dir / 'tensorboard')
        
        # Weights & Biases
        if self.config.use_wandb and WANDB_AVAILABLE:
            wandb.init(
                project=self.config.wandb_project,
                config=asdict(self.config),
                name=f"ecg_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            )
    
    def _create_data_loaders(self, epoch: int = 0) -> Tuple[DataLoader, DataLoader]:
        """Create data loaders with optional curriculum learning"""
        # Get training subset for curriculum learning
        if hasattr(self.train_dataset, 'difficulty_scores'):
            train_indices = self.curriculum_scheduler.get_training_subset(
                epoch,
                self.train_dataset.difficulty_scores
            )
        else:
            train_indices = np.arange(len(self.train_dataset))
        
        # Create subset dataset
        train_subset = torch.utils.data.Subset(self.train_dataset, train_indices)
        
        # Compute class weights for balanced sampling
        if self.config.use_class_weights and hasattr(self.train_dataset, 'labels'):
            labels = np.array([self.train_dataset.labels[i] for i in train_indices])
            class_counts = np.bincount(labels)
            class_weights = 1.0 / (class_counts + 1e-6)
            sample_weights = class_weights[labels]
            sampler = WeightedRandomSampler(sample_weights, len(sample_weights))
            shuffle = False
        else:
            sampler = None
            shuffle = True
        
        # Create loaders
        train_loader = DataLoader(
            train_subset,
            batch_size=self.config.batch_size,
            shuffle=shuffle,
            sampler=sampler,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory,
            prefetch_factor=self.config.prefetch_factor
        )
        
        val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory
        )
        
        return train_loader, val_loader
    
    def train_epoch(self, train_loader: DataLoader, epoch: int) -> Dict[str, float]:
        """Train for one epoch"""
        self.model.train()
        
        total_loss = 0.0
        task_losses = {task: 0.0 for task in self.config.auxiliary_tasks}
        correct = 0
        total = 0
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch}/{self.config.num_epochs}')
        
        for batch_idx, (data, target) in enumerate(pbar):
            data, target = data.to(self.device), target.to(self.device)
            
            # Apply augmentation
            data, target = self.augmenter.augment_batch(data, target)
            
            # Mixed precision training
            if self.config.use_mixed_precision:
                with autocast():
                    output = self.model(data)
                    
                    if self.config.multi_task_learning:
                        loss, batch_task_losses = self.criterion(output, target)
                        predictions = output['main']
                    else:
                        loss = self.criterion(output['logits'], target)
                        predictions = output['predictions']
                
                # Gradient accumulation
                loss = loss / self.config.accumulation_steps
                self.scaler.scale(loss).backward()
                
                if (batch_idx + 1) % self.config.accumulation_steps == 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.gradient_clip_norm
                    )
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad()
            else:
                # Standard training
                output = self.model(data)
                
                if self.config.multi_task_learning:
                    loss, batch_task_losses = self.criterion(output, target)
                    predictions = output['main']
                else:
                    loss = self.criterion(output['logits'], target)
                    predictions = output['predictions']
                
                loss = loss / self.config.accumulation_steps
                loss.backward()
                
                if (batch_idx + 1) % self.config.accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.gradient_clip_norm
                    )
                    self.optimizer.step()
                    self.optimizer.zero_grad()
            
            # Update metrics
            total_loss += loss.item() * self.config.accumulation_steps
            _, predicted = predictions.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
            
            # Update task losses
            if self.config.multi_task_learning:
                for task, task_loss in batch_task_losses.items():
                    task_losses[task] += task_loss
            
            # Update progress bar
            pbar.set_postfix({
                'loss': total_loss / (batch_idx + 1),
                'acc': 100. * correct / total
            })
            
            # Learning rate scheduling
            if self.config.scheduler_type == "one_cycle":
                self.scheduler.step()
        
        # Compute epoch metrics
        epoch_loss = total_loss / len(train_loader)
        epoch_acc = 100. * correct / total
        
        metrics = {
            'train_loss': epoch_loss,
            'train_acc': epoch_acc
        }
        
        if self.config.multi_task_learning:
            for task, task_loss in task_losses.items():
                metrics[f'train_{task}_loss'] = task_loss / len(train_loader)
        
        return metrics
    
    def validate(self, val_loader: DataLoader) -> Dict[str, float]:
        """Validate the model"""
        self.model.eval()
        
        total_loss = 0.0
        all_predictions = []
        all_targets = []
        all_probabilities = []
        
        with torch.no_grad():
            for data, target in tqdm(val_loader, desc='Validation'):
                data, target = data.to(self.device), target.to(self.device)
                
                # Forward pass
                output = self.model(data)
                
                if self.config.multi_task_learning:
                    loss, _ = self.criterion(output, target)
                    predictions = output['main']
                else:
                    loss = self.criterion(output['logits'], target)
                    predictions = output['predictions']
                
                total_loss += loss.item()
                
                # Collect predictions
                all_predictions.extend(predictions.argmax(1).cpu().numpy())
                all_targets.extend(target.cpu().numpy())
                all_probabilities.extend(predictions.cpu().numpy())
        
        # Compute metrics
        all_predictions = np.array(all_predictions)
        all_targets = np.array(all_targets)
        all_probabilities = np.array(all_probabilities)
        
        # Basic metrics
        val_loss = total_loss / len(val_loader)
        val_acc = (all_predictions == all_targets).mean() * 100
        
        # Advanced metrics
        val_f1 = f1_score(all_targets, all_predictions, average='weighted')
        
        # Compute AUC for multi-class
        try:
            val_auc = roc_auc_score(
                all_targets,
                all_probabilities,
                multi_class='ovr',
                average='weighted'
            )
        except:
            val_auc = 0.0
        
        # Classification report
        report = classification_report(
            all_targets,
            all_predictions,
            output_dict=True
        )
        
        metrics = {
            'val_loss': val_loss,
            'val_acc': val_acc,
            'val_f1': val_f1,
            'val_auc': val_auc,
            'val_report': report
        }
        
        return metrics
    
    def train(self):
        """Main training loop"""
        logger.info(f"Starting training for {self.config.num_epochs} epochs")
        
        for epoch in range(self.config.num_epochs):
            self.current_epoch = epoch
            
            # Create data loaders (may change with curriculum learning)
            train_loader, val_loader = self._create_data_loaders(epoch)
            
            # Train epoch
            train_metrics = self.train_epoch(train_loader, epoch)
            
            # Validate
            val_metrics = self.validate(val_loader)
            
            # Learning rate scheduling
            if self.config.scheduler_type == "cosine_warm_restarts":
                self.scheduler.step()
            
            # Logging
            self._log_metrics(train_metrics, val_metrics, epoch)
            
            # Save checkpoint
            if val_metrics['val_f1'] > self.best_val_f1:
                self.best_val_f1 = val_metrics['val_f1']
                self._save_checkpoint(epoch, val_metrics, is_best=True)
            elif epoch % self.config.save_frequency == 0:
                self._save_checkpoint(epoch, val_metrics, is_best=False)
            
            # Early stopping
            if self._check_early_stopping(val_metrics):
                logger.info(f"Early stopping triggered at epoch {epoch}")
                break
        
        logger.info("Training completed!")
        self._cleanup()
    
    def _log_metrics(self, train_metrics: Dict, val_metrics: Dict, epoch: int):
        """Log metrics to various backends"""
        # Console logging
        logger.info(
            f"Epoch {epoch}: "
            f"Train Loss: {train_metrics['train_loss']:.4f}, "
            f"Train Acc: {train_metrics['train_acc']:.2f}%, "
            f"Val Loss: {val_metrics['val_loss']:.4f}, "
            f"Val Acc: {val_metrics['val_acc']:.2f}%, "
            f"Val F1: {val_metrics['val_f1']:.4f}"
        )
        
        # TensorBoard
        if self.config.use_tensorboard:
            for key, value in train_metrics.items():
                self.tb_writer.add_scalar(f'Train/{key}', value, epoch)
            
            for key, value in val_metrics.items():
                if key != 'val_report':
                    self.tb_writer.add_scalar(f'Val/{key}', value, epoch)
        
        # Weights & Biases
        if self.config.use_wandb and WANDB_AVAILABLE:
            wandb.log({
                **{f'train/{k}': v for k, v in train_metrics.items()},
                **{f'val/{k}': v for k, v in val_metrics.items() if k != 'val_report'},
                'epoch': epoch
            })
    
    def _save_checkpoint(self, epoch: int, metrics: Dict, is_best: bool = False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'metrics': metrics,
            'config': asdict(self.config)
        }
        
        # Save regular checkpoint
        checkpoint_path = self.save_dir / f'checkpoint_epoch_{epoch}.pth'
        torch.save(checkpoint, checkpoint_path)
        
        # Save best model
        if is_best:
            best_path = self.save_dir / 'best_model.pth'
            torch.save(checkpoint, best_path)
            logger.info(f"Saved best model with F1: {metrics['val_f1']:.4f}")
    
    def _check_early_stopping(self, val_metrics: Dict) -> bool:
        """Check if early stopping should be triggered"""
        # Implementation of early stopping logic
        # This is a simplified version
        return False
    
    def _cleanup(self):
        """Cleanup resources"""
        if self.config.use_tensorboard:
            self.tb_writer.close()
        
        if self.config.use_wandb and WANDB_AVAILABLE:
            wandb.finish()


def create_trainer(
    model: nn.Module,
    train_dataset: Dataset,
    val_dataset: Dataset,
    config: Optional[TrainingConfig] = None
) -> ECGTrainer:
    """Factory function to create trainer"""
    if config is None:
        config = TrainingConfig(model_config=ModelConfig())
    
    return ECGTrainer(model, config, train_dataset, val_dataset)
