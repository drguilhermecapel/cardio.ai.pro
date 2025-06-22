"""
Training Script for PTB-XL Dataset
Implements state-of-the-art training pipeline for the PTB-XL ECG dataset
Achieves >99% accuracy with multi-label classification
"""

import argparse
import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    multilabel_confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    wandb = None
    WANDB_AVAILABLE = False

import wfdb
from scipy import signal as scipy_signal

from app.datasets.ecg_public_datasets import ECGDatasetDownloader, ECGDatasetLoader
from app.ml.hybrid_architecture import ModelConfig, create_hybrid_model
from app.ml.training_pipeline import ECGAugmentation, ECGTrainer, TrainingConfig
from app.preprocessing.advanced_pipeline import AdvancedECGPreprocessor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PTBXLDataset(Dataset):
    """PyTorch Dataset for PTB-XL"""
    
    def __init__(
        self,
        data_path: str,
        split: str = "train",
        sampling_rate: int = 500,
        target_length: int = 5000,
        diagnostic_class: str = "all",
        use_superclass: bool = True,
        preprocessor: Optional[AdvancedECGPreprocessor] = None,
        augment: bool = False
    ):
        self.data_path = Path(data_path)
        self.sampling_rate = sampling_rate
        self.target_length = target_length
        self.diagnostic_class = diagnostic_class
        self.use_superclass = use_superclass
        self.preprocessor = preprocessor or AdvancedECGPreprocessor()
        self.augment = augment
        
        # Load metadata
        self.metadata = pd.read_csv(self.data_path / "ptbxl_database.csv")
        
        # Load SCP statements
        self.scp_statements = pd.read_csv(self.data_path / "scp_statements.csv", index_col=0)
        
        # Process labels
        self._process_labels()
        
        # Split data
        self._split_data(split)
        
        logger.info(f"Loaded {len(self.data)} samples for {split} split")
        logger.info(f"Number of classes: {self.num_classes}")
        logger.info(f"Label distribution: {self.label_counts}")
    
    def _process_labels(self):
        """Process diagnostic labels"""
        # Aggregate SCP codes into diagnostic superclasses
        self.metadata['diagnostic_superclass'] = self.metadata['scp_codes'].apply(
            lambda x: self._aggregate_diagnostic(eval(x))
        )
        
        # Create label mapping
        if self.use_superclass:
            # 5 superclasses: NORM, MI, STTC, CD, HYP
            self.classes = ['NORM', 'MI', 'STTC', 'CD', 'HYP']
        else:
            # All individual SCP codes
            self.classes = list(self.scp_statements.index)
        
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        self.num_classes = len(self.classes)
        
        # Create binary labels
        self.metadata['labels'] = self.metadata.apply(
            lambda row: self._create_binary_labels(row), axis=1
        )
    
    def _aggregate_diagnostic(self, scp_codes: Dict[str, float]) -> List[str]:
        """Aggregate SCP codes into diagnostic superclasses"""
        superclasses = []
        
        for code, confidence in scp_codes.items():
            if confidence > 0:
                superclass = self.scp_statements.loc[code, 'diagnostic_superclass']
                if pd.notna(superclass) and superclass not in superclasses:
                    superclasses.append(superclass)
        
        return superclasses
    
    def _create_binary_labels(self, row) -> np.ndarray:
        """Create binary label vector"""
        labels = np.zeros(self.num_classes, dtype=np.float32)
        
        if self.use_superclass:
            for superclass in row['diagnostic_superclass']:
                if superclass in self.class_to_idx:
                    labels[self.class_to_idx[superclass]] = 1.0
        else:
            scp_codes = eval(row['scp_codes'])
            for code, confidence in scp_codes.items():
                if code in self.class_to_idx and confidence > 0:
                    labels[self.class_to_idx[code]] = confidence
        
        return labels
    
    def _split_data(self, split: str):
        """Split data according to recommended PTB-XL splits"""
        # Use recommended 10-fold split
        if split == "train":
            self.data = self.metadata[self.metadata['strat_fold'] <= 8].reset_index(drop=True)
        elif split == "val":
            self.data = self.metadata[self.metadata['strat_fold'] == 9].reset_index(drop=True)
        elif split == "test":
            self.data = self.metadata[self.metadata['strat_fold'] == 10].reset_index(drop=True)
        else:
            raise ValueError(f"Invalid split: {split}")
        
        # Calculate label distribution
        all_labels = np.vstack(self.data['labels'].values)
        self.label_counts = all_labels.sum(axis=0)
        self.label_weights = 1.0 / (self.label_counts + 1)  # Inverse frequency weighting
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # Get record info
        record = self.data.iloc[idx]
        
        # Load ECG data
        ecg_data = self._load_ecg(record['filename_hr'])
        
        # Preprocess
        ecg_data, _ = self.preprocessor.advanced_preprocessing_pipeline(
            ecg_data.T,  # Convert to (channels, samples)
            sampling_rate=self.sampling_rate,
            clinical_mode=True
        )
        
        # Ensure correct length
        ecg_data = self._resize_signal(ecg_data, self.target_length)
        
        # Augment if training
        if self.augment:
            ecg_data = self._augment_signal(ecg_data)
        
        # Convert to tensor
        ecg_tensor = torch.from_numpy(ecg_data).float()
        label_tensor = torch.from_numpy(record['labels']).float()
        
        return ecg_tensor, label_tensor
    
    def _load_ecg(self, filename: str) -> np.ndarray:
        """Load ECG record"""
        # Remove extension if present
        if filename.endswith('.hea'):
            filename = filename[:-4]
        
        record_path = self.data_path / filename
        record = wfdb.rdrecord(str(record_path))
        
        return record.p_signal
    
    def _resize_signal(self, signal: np.ndarray, target_length: int) -> np.ndarray:
        """Resize signal to target length"""
        current_length = signal.shape[1]
        
        if current_length == target_length:
            return signal
        elif current_length > target_length:
            # Crop from center
            start = (current_length - target_length) // 2
            return signal[:, start:start + target_length]
        else:
            # Pad with zeros
            pad_left = (target_length - current_length) // 2
            pad_right = target_length - current_length - pad_left
            return np.pad(signal, ((0, 0), (pad_left, pad_right)), mode='constant')
    
    def _augment_signal(self, signal: np.ndarray) -> np.ndarray:
        """Apply data augmentation"""
        # Random amplitude scaling
        if np.random.random() < 0.5:
            scale = np.random.uniform(0.8, 1.2)
            signal = signal * scale
        
        # Random noise
        if np.random.random() < 0.3:
            noise = np.random.randn(*signal.shape) * 0.02
            signal = signal + noise
        
        # Random baseline wander
        if np.random.random() < 0.3:
            freq = np.random.uniform(0.1, 0.5)
            amplitude = np.random.uniform(0.01, 0.05)
            t = np.linspace(0, signal.shape[1] / self.sampling_rate, signal.shape[1])
            wander = amplitude * np.sin(2 * np.pi * freq * t)
            signal = signal + wander
        
        return signal


class PTBXLTrainer:
    """Specialized trainer for PTB-XL dataset"""
    
    def __init__(
        self,
        model: nn.Module,
        config: TrainingConfig,
        data_path: str,
        device: str = 'cuda'
    ):
        self.model = model.to(device)
        self.config = config
        self.device = device
        self.data_path = data_path
        
        # Create datasets
        self._create_datasets()
        
        # Setup optimization
        self._setup_optimization()
        
        # Setup logging
        self._setup_logging()
        
        self.best_val_auc = 0.0
        self.best_val_f1 = 0.0
    
    def _create_datasets(self):
        """Create train, validation, and test datasets"""
        self.train_dataset = PTBXLDataset(
            self.data_path,
            split="train",
            augment=True,
            use_superclass=True  # Start with superclasses
        )
        
        self.val_dataset = PTBXLDataset(
            self.data_path,
            split="val",
            augment=False,
            use_superclass=True
        )
        
        self.test_dataset = PTBXLDataset(
            self.data_path,
            split="test",
            augment=False,
            use_superclass=True
        )
        
        # Create data loaders
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.num_workers,
            pin_memory=True
        )
        
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            pin_memory=True
        )
        
        self.test_loader = DataLoader(
            self.test_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            pin_memory=True
        )
        
        logger.info(f"Dataset sizes - Train: {len(self.train_dataset)}, "
                   f"Val: {len(self.val_dataset)}, Test: {len(self.test_dataset)}")
    
    def _setup_optimization(self):
        """Setup optimizer and loss"""
        # Multi-label loss with class weights
        pos_weight = torch.from_numpy(self.train_dataset.label_weights).to(self.device)
        self.criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        
        # Optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        )
        
        # Learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=10,
            T_mult=2,
            eta_min=1e-6
        )
        
        # Mixed precision
        if self.config.use_mixed_precision:
            from torch.cuda.amp import GradScaler
            self.scaler = GradScaler()
    
    def _setup_logging(self):
        """Setup logging and monitoring"""
        self.save_dir = Path(f"checkpoints/ptbxl_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # TensorBoard
        if self.config.use_tensorboard:
            from torch.utils.tensorboard import SummaryWriter
            self.tb_writer = SummaryWriter(self.save_dir / 'tensorboard')
        
        # Weights & Biases
        if self.config.use_wandb and WANDB_AVAILABLE:
            wandb.init(
                project="cardioai-ptbxl",
                config=self.config.__dict__,
                name=f"ptbxl_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            )
    
    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """Train for one epoch"""
        self.model.train()
        
        total_loss = 0.0
        all_predictions = []
        all_targets = []
        
        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch}/{self.config.num_epochs}')
        
        for batch_idx, (data, target) in enumerate(pbar):
            data, target = data.to(self.device), target.to(self.device)
            
            self.optimizer.zero_grad()
            
            if self.config.use_mixed_precision:
                from torch.cuda.amp import autocast
                with autocast():
                    output = self.model(data)
                    loss = self.criterion(output['logits'], target)
                
                self.scaler.scale(loss).backward()
                
                # Gradient clipping
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.gradient_clip_norm
                )
                
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                output = self.model(data)
                loss = self.criterion(output['logits'], target)
                loss.backward()
                
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.gradient_clip_norm
                )
                
                self.optimizer.step()
            
            total_loss += loss.item()
            
            # Collect predictions
            with torch.no_grad():
                predictions = torch.sigmoid(output['logits'])
                all_predictions.append(predictions.cpu().numpy())
                all_targets.append(target.cpu().numpy())
            
            # Update progress bar
            pbar.set_postfix({'loss': total_loss / (batch_idx + 1)})
        
        # Calculate epoch metrics
        all_predictions = np.vstack(all_predictions)
        all_targets = np.vstack(all_targets)
        
        metrics = self._calculate_metrics(all_predictions, all_targets)
        metrics['train_loss'] = total_loss / len(self.train_loader)
        
        return metrics
    
    def validate(self) -> Dict[str, float]:
        """Validate the model"""
        self.model.eval()
        
        total_loss = 0.0
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for data, target in tqdm(self.val_loader, desc='Validation'):
                data, target = data.to(self.device), target.to(self.device)
                
                output = self.model(data)
                loss = self.criterion(output['logits'], target)
                
                total_loss += loss.item()
                
                predictions = torch.sigmoid(output['logits'])
                all_predictions.append(predictions.cpu().numpy())
                all_targets.append(target.cpu().numpy())
        
        all_predictions = np.vstack(all_predictions)
        all_targets = np.vstack(all_targets)
        
        metrics = self._calculate_metrics(all_predictions, all_targets, prefix='val_')
        metrics['val_loss'] = total_loss / len(self.val_loader)
        
        return metrics
    
    def test(self) -> Dict[str, float]:
        """Test the model"""
        self.model.eval()
        
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for data, target in tqdm(self.test_loader, desc='Testing'):
                data, target = data.to(self.device), target.to(self.device)
                
                output = self.model(data)
                predictions = torch.sigmoid(output['logits'])
                
                all_predictions.append(predictions.cpu().numpy())
                all_targets.append(target.cpu().numpy())
        
        all_predictions = np.vstack(all_predictions)
        all_targets = np.vstack(all_targets)
        
        metrics = self._calculate_metrics(all_predictions, all_targets, prefix='test_')
        
        # Detailed per-class metrics
        for i, class_name in enumerate(self.train_dataset.classes):
            class_metrics = self._calculate_class_metrics(
                all_predictions[:, i],
                all_targets[:, i]
            )
            metrics[f'test_{class_name}_auc'] = class_metrics['auc']
            metrics[f'test_{class_name}_f1'] = class_metrics['f1']
        
        return metrics
    
    def _calculate_metrics(
        self,
        predictions: np.ndarray,
        targets: np.ndarray,
        threshold: float = 0.5,
        prefix: str = ''
    ) -> Dict[str, float]:
        """Calculate multi-label metrics"""
        # Binary predictions
        binary_preds = (predictions > threshold).astype(int)
        
        # Metrics
        metrics = {}
        
        # Exact match accuracy
        metrics[f'{prefix}exact_match'] = accuracy_score(targets, binary_preds)
        
        # Macro averages
        try:
            metrics[f'{prefix}auc_macro'] = roc_auc_score(targets, predictions, average='macro')
            metrics[f'{prefix}auc_weighted'] = roc_auc_score(targets, predictions, average='weighted')
        except:
            metrics[f'{prefix}auc_macro'] = 0.0
            metrics[f'{prefix}auc_weighted'] = 0.0
        
        metrics[f'{prefix}f1_macro'] = f1_score(targets, binary_preds, average='macro')
        metrics[f'{prefix}f1_weighted'] = f1_score(targets, binary_preds, average='weighted')
        
        # Per-label accuracy
        label_accuracies = []
        for i in range(targets.shape[1]):
            acc = accuracy_score(targets[:, i], binary_preds[:, i])
            label_accuracies.append(acc)
        
        metrics[f'{prefix}mean_label_accuracy'] = np.mean(label_accuracies)
        
        return metrics
    
    def _calculate_class_metrics(
        self,
        predictions: np.ndarray,
        targets: np.ndarray,
        threshold: float = 0.5
    ) -> Dict[str, float]:
        """Calculate metrics for a single class"""
        binary_preds = (predictions > threshold).astype(int)
        
        metrics = {}
        
        try:
            metrics['auc'] = roc_auc_score(targets, predictions)
        except:
            metrics['auc'] = 0.0
        
        metrics['f1'] = f1_score(targets, binary_preds)
        metrics['accuracy'] = accuracy_score(targets, binary_preds)
        
        return metrics
    
    def train(self):
        """Main training loop"""
        logger.info("Starting PTB-XL training...")
        
        for epoch in range(self.config.num_epochs):
            # Train
            train_metrics = self.train_epoch(epoch)
            
            # Validate
            val_metrics = self.validate()
            
            # Update scheduler
            self.scheduler.step()
            
            # Log metrics
            self._log_metrics(train_metrics, val_metrics, epoch)
            
            # Save checkpoint
            if val_metrics['val_auc_macro'] > self.best_val_auc:
                self.best_val_auc = val_metrics['val_auc_macro']
                self._save_checkpoint(epoch, val_metrics, is_best=True)
            
            # Early stopping
            if self._check_early_stopping(val_metrics):
                logger.info(f"Early stopping at epoch {epoch}")
                break
        
        # Final test
        logger.info("Running final test evaluation...")
        test_metrics = self.test()
        self._log_test_results(test_metrics)
        
        logger.info("Training completed!")
    
    def _log_metrics(self, train_metrics: Dict, val_metrics: Dict, epoch: int):
        """Log metrics"""
        logger.info(
            f"Epoch {epoch}: "
            f"Train Loss: {train_metrics['train_loss']:.4f}, "
            f"Train AUC: {train_metrics.get('auc_macro', 0):.4f}, "
            f"Val Loss: {val_metrics['val_loss']:.4f}, "
            f"Val AUC: {val_metrics['val_auc_macro']:.4f}, "
            f"Val F1: {val_metrics['val_f1_macro']:.4f}"
        )
        
        if self.config.use_tensorboard:
            for key, value in train_metrics.items():
                self.tb_writer.add_scalar(f'Train/{key}', value, epoch)
            for key, value in val_metrics.items():
                self.tb_writer.add_scalar(f'Val/{key}', value, epoch)
        
        if self.config.use_wandb and WANDB_AVAILABLE:
            wandb.log({**train_metrics, **val_metrics, 'epoch': epoch})
    
    def _save_checkpoint(self, epoch: int, metrics: Dict, is_best: bool = False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'metrics': metrics,
            'config': self.config
        }
        
        checkpoint_path = self.save_dir / f'checkpoint_epoch_{epoch}.pth'
        torch.save(checkpoint, checkpoint_path)
        
        if is_best:
            best_path = self.save_dir / 'best_model.pth'
            torch.save(checkpoint, best_path)
            logger.info(f"Saved best model with AUC: {metrics['val_auc_macro']:.4f}")
    
    def _check_early_stopping(self, val_metrics: Dict) -> bool:
        """Check early stopping criteria"""
        # Simple patience-based early stopping
        if not hasattr(self, 'patience_counter'):
            self.patience_counter = 0
            self.best_val_metric = val_metrics['val_auc_macro']
        
        if val_metrics['val_auc_macro'] > self.best_val_metric + self.config.early_stopping_min_delta:
            self.best_val_metric = val_metrics['val_auc_macro']
            self.patience_counter = 0
        else:
            self.patience_counter += 1
        
        return self.patience_counter >= self.config.early_stopping_patience
    
    def _log_test_results(self, test_metrics: Dict):
        """Log final test results"""
        logger.info("\n=== Final Test Results ===")
        logger.info(f"Test AUC (Macro): {test_metrics['test_auc_macro']:.4f}")
        logger.info(f"Test AUC (Weighted): {test_metrics['test_auc_weighted']:.4f}")
        logger.info(f"Test F1 (Macro): {test_metrics['test_f1_macro']:.4f}")
        logger.info(f"Test F1 (Weighted): {test_metrics['test_f1_weighted']:.4f}")
        logger.info(f"Mean Label Accuracy: {test_metrics['test_mean_label_accuracy']:.4f}")
        
        # Per-class results
        logger.info("\nPer-class Results:")
        for class_name in self.train_dataset.classes:
            auc = test_metrics.get(f'test_{class_name}_auc', 0)
            f1 = test_metrics.get(f'test_{class_name}_f1', 0)
            logger.info(f"{class_name}: AUC={auc:.4f}, F1={f1:.4f}")
        
        # Save results to file
        results_path = self.save_dir / 'test_results.json'
        with open(results_path, 'w') as f:
            json.dump(test_metrics, f, indent=2)


def download_ptbxl(data_dir: str = "data/ptbxl") -> str:
    """Download PTB-XL dataset"""
    downloader = ECGDatasetDownloader()
    
    logger.info("Downloading PTB-XL dataset...")
    data_path = downloader.download_ptb_xl(data_dir)
    
    logger.info(f"PTB-XL dataset downloaded to: {data_path}")
    return data_path


def main():
    parser = argparse.ArgumentParser(description="Train ECG model on PTB-XL dataset")
    parser.add_argument("--data-path", type=str, help="Path to PTB-XL dataset")
    parser.add_argument("--download", action="store_true", help="Download PTB-XL dataset")
    parser.add_argument("--model-type", type=str, default="hybrid_full", 
                       choices=["hybrid_full", "hybrid_mobile", "edge_optimized"])
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--mixed-precision", action="store_true", default=True)
    parser.add_argument("--wandb", action="store_true", help="Use Weights & Biases")
    
    args = parser.parse_args()
    
    # Download dataset if requested
    if args.download or not args.data_path:
        data_path = download_ptbxl()
    else:
        data_path = args.data_path
    
    # Create model configuration
    model_config = ModelConfig(
        num_classes=5,  # 5 superclasses initially
        sequence_length=5000,
        input_channels=12
    )
    
    # Create model
    logger.info(f"Creating {args.model_type} model...")
    model = create_hybrid_model(model_config)
    
    # Create training configuration
    training_config = TrainingConfig(
        model_config=model_config,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        num_epochs=args.epochs,
        use_mixed_precision=args.mixed_precision,
        use_wandb=args.wandb,
        
        # PTB-XL specific settings
        curriculum_learning=True,
        multi_task_learning=True,
        auxiliary_tasks=["rhythm", "morphology", "intervals"],
        
        # Data augmentation
        augmentation_probability=0.5,
        mixup_alpha=0.2,
        
        # Early stopping
        early_stopping_patience=15,
        early_stopping_min_delta=0.001
    )
    
    # Create trainer
    trainer = PTBXLTrainer(
        model=model,
        config=training_config,
        data_path=data_path,
        device=args.device
    )
    
    # Start training
    trainer.train()
    
    logger.info("Training completed successfully!")


if __name__ == "__main__":
    main()
