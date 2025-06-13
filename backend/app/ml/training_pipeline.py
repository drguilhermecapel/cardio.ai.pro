"""
Training Pipeline for Hybrid CNN-BiLSTM-Transformer Architecture
Implements curriculum learning and multimodal processing for ECG analysis
Based on scientific recommendations for CardioAI Pro
"""

import json
import logging
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pywt
import torch
import torch.nn as nn

try:
    import torch.nn.functional as F
except ImportError:
    F = None
import torch.optim as optim

try:
    import wandb
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    wandb = None
from scipy import signal as scipy_signal

try:
    from sklearn.metrics import classification_report, roc_auc_score
    from sklearn.model_selection import train_test_split
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    classification_report = None
    roc_auc_score = None
    train_test_split = None
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from tqdm import tqdm

from app.core.scp_ecg_conditions import SCP_ECG_CONDITIONS, get_condition_by_code
from app.datasets.ecg_public_datasets import ECGDatasetLoader
from app.ml.hybrid_architecture import ModelConfig, create_hybrid_model
from app.preprocessing.advanced_pipeline import AdvancedECGPreprocessor

logger = logging.getLogger(__name__)

@dataclass
class TrainingConfig:
    """Configuration for training pipeline"""
    model_config: ModelConfig

    batch_size: int = 32
    learning_rate: float = 1e-4
    num_epochs: int = 100
    weight_decay: float = 1e-5
    gradient_clip_norm: float = 1.0

    curriculum_learning: bool = True
    curriculum_stages: int = 3
    curriculum_epochs_per_stage: int = 30

    augmentation_probability: float = 0.5
    noise_std: float = 0.02
    amplitude_scale_range: tuple[float, float] = (0.8, 1.2)
    time_shift_range: int = 50

    use_spectrograms: bool = True
    use_wavelets: bool = True
    spectrogram_nperseg: int = 256
    spectrogram_noverlap: int = 128
    wavelet_name: str = 'db6'
    wavelet_levels: int = 6

    use_class_weights: bool = True
    focal_loss_alpha: float = 0.25
    focal_loss_gamma: float = 2.0
    label_smoothing: float = 0.1

    validation_split: float = 0.2
    k_fold_cv: bool = False
    k_folds: int = 5

    save_checkpoints: bool = True
    checkpoint_dir: str = "checkpoints"
    save_best_only: bool = True
    early_stopping_patience: int = 15

    use_wandb: bool = False
    wandb_project: str = "cardioai-pro"
    log_interval: int = 100

    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    num_workers: int = 4
    pin_memory: bool = True

class ECGMultimodalDataset(Dataset):
    """
    Multimodal ECG dataset supporting 1D signals, 2D spectrograms, and wavelet representations
    """

    def __init__(
        self,
        signals: list[np.ndarray],
        labels: list[int],
        condition_codes: list[str],
        config: TrainingConfig,
        is_training: bool = True,
        preprocessor: AdvancedECGPreprocessor | None = None
    ):
        self.signals = signals
        self.labels = labels
        self.condition_codes = condition_codes
        self.config = config
        self.is_training = is_training
        self.preprocessor = preprocessor or AdvancedECGPreprocessor()

        self.spectrograms = []
        self.wavelets = []

        if config.use_spectrograms or config.use_wavelets:
            self._precompute_multimodal_representations()

    def _precompute_multimodal_representations(self) -> None:
        """Precompute spectrograms and wavelet representations"""
        logger.info("Precomputing multimodal representations...")

        for _i, signal in enumerate(tqdm(self.signals, desc="Computing multimodal features")):
            if self.config.use_spectrograms:
                spectrogram = self._compute_spectrogram(signal)
                self.spectrograms.append(spectrogram)

            if self.config.use_wavelets:
                wavelet_features = self._compute_wavelet_features(signal)
                self.wavelets.append(wavelet_features)

    def _compute_spectrogram(self, signal: np.ndarray) -> np.ndarray:
        """Compute spectrogram for ECG signal"""
        try:
            lead_ii = signal[:, 1] if signal.shape[1] > 1 else signal[:, 0]

            f, t, Sxx = scipy_signal.spectrogram(
                lead_ii,
                fs=500,  # Assuming 500 Hz sampling rate
                nperseg=self.config.spectrogram_nperseg,
                noverlap=self.config.spectrogram_noverlap,
                window='hann'
            )

            Sxx_db = 10 * np.log10(Sxx + 1e-10)

            Sxx_normalized = (Sxx_db - np.mean(Sxx_db)) / (np.std(Sxx_db) + 1e-8)

            target_size = (128, 128)
            spectrogram_resized = cv2.resize(Sxx_normalized, target_size)

            return spectrogram_resized.astype(np.float32)

        except Exception as e:
            logger.warning(f"Error computing spectrogram: {e}")
            return np.zeros((128, 128), dtype=np.float32)

    def _compute_wavelet_features(self, signal: np.ndarray) -> np.ndarray:
        """Compute wavelet features for ECG signal"""
        try:
            lead_ii = signal[:, 1] if signal.shape[1] > 1 else signal[:, 0]

            coeffs = pywt.wavedec(
                lead_ii,
                self.config.wavelet_name,
                level=self.config.wavelet_levels
            )

            wavelet_features = []

            for coeff in coeffs:
                wavelet_features.extend([
                    np.mean(coeff),
                    np.std(coeff),
                    np.var(coeff),
                    np.max(coeff),
                    np.min(coeff),
                    np.median(coeff),
                    np.percentile(coeff, 25),
                    np.percentile(coeff, 75)
                ])

            target_size = 256
            if len(wavelet_features) > target_size:
                wavelet_features = wavelet_features[:target_size]
            else:
                wavelet_features.extend([0.0] * (target_size - len(wavelet_features)))

            return np.array(wavelet_features, dtype=np.float32)

        except Exception as e:
            logger.warning(f"Error computing wavelet features: {e}")
            return np.zeros(256, dtype=np.float32)

    def _augment_signal(self, signal: np.ndarray) -> np.ndarray:
        """Apply data augmentation to ECG signal"""
        if not self.is_training or np.random.random() > self.config.augmentation_probability:
            return signal

        augmented_signal = signal.copy()

        if np.random.random() < 0.5:
            noise = np.random.normal(0, self.config.noise_std, signal.shape)
            augmented_signal += noise

        if np.random.random() < 0.5:
            scale = np.random.uniform(*self.config.amplitude_scale_range)
            augmented_signal *= scale

        if np.random.random() < 0.5:
            shift = np.random.randint(-self.config.time_shift_range, self.config.time_shift_range)
            if shift > 0:
                augmented_signal = np.concatenate([
                    augmented_signal[shift:],
                    np.zeros((shift, signal.shape[1]))
                ])
            elif shift < 0:
                augmented_signal = np.concatenate([
                    np.zeros((-shift, signal.shape[1])),
                    augmented_signal[:shift]
                ])

        return augmented_signal

    def __len__(self) -> int:
        return len(self.signals)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        signal = self.signals[idx]
        signal = self._augment_signal(signal)

        signal_tensor = torch.from_numpy(signal.T).float()  # (channels, time)

        label = torch.tensor(self.labels[idx], dtype=torch.long)

        item = {
            'signal': signal_tensor,
            'label': label,
            'condition_code': self.condition_codes[idx]
        }

        if self.config.use_spectrograms and self.spectrograms:
            spectrogram = torch.from_numpy(self.spectrograms[idx]).float()
            item['spectrogram'] = spectrogram.unsqueeze(0)  # Add channel dimension

        if self.config.use_wavelets and self.wavelets:
            wavelet = torch.from_numpy(self.wavelets[idx]).float()
            item['wavelet'] = wavelet

        return item

class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance"""

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

class CurriculumScheduler:
    """Curriculum learning scheduler"""

    def __init__(self, config: TrainingConfig, dataset_difficulty_scores: list[float]):
        self.config = config
        self.difficulty_scores = np.array(dataset_difficulty_scores)
        self.current_stage = 0
        self.epochs_in_stage = 0

    def get_current_subset_indices(self) -> list[int]:
        """Get indices for current curriculum stage"""
        if not self.config.curriculum_learning:
            return list(range(len(self.difficulty_scores)))

        if self.current_stage == 0:
            threshold = np.percentile(self.difficulty_scores, 33)
            indices = np.where(self.difficulty_scores <= threshold)[0]
        elif self.current_stage == 1:
            threshold_low = np.percentile(self.difficulty_scores, 33)
            threshold_high = np.percentile(self.difficulty_scores, 66)
            indices = np.where(
                (self.difficulty_scores > threshold_low) &
                (self.difficulty_scores <= threshold_high)
            )[0]
        else:
            indices = np.arange(len(self.difficulty_scores))

        return indices.tolist()

    def step_epoch(self):
        """Step to next epoch and potentially next curriculum stage"""
        self.epochs_in_stage += 1

        if (self.epochs_in_stage >= self.config.curriculum_epochs_per_stage and
            self.current_stage < self.config.curriculum_stages - 1):
            self.current_stage += 1
            self.epochs_in_stage = 0
            logger.info(f"Advanced to curriculum stage {self.current_stage}")

class ECGTrainingPipeline:
    """
    Complete training pipeline for hybrid ECG model
    Supports curriculum learning, multimodal processing, and advanced training strategies
    """

    def __init__(self, config: TrainingConfig):
        self.config = config
        self.device = torch.device(config.device)

        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.criterion = None
        self.dataset_manager = ECGDatasetLoader()
        self.preprocessor = AdvancedECGPreprocessor()

        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.best_val_accuracy = 0.0
        self.training_history = {
            'train_loss': [],
            'val_loss': [],
            'train_accuracy': [],
            'val_accuracy': [],
            'learning_rate': []
        }

        if config.use_wandb:
            if wandb is None:
                raise ImportError("Weights & Biases is required when use_wandb=True")
            wandb.init(
                project=config.wandb_project,
                config=asdict(config),
                name=f"hybrid_ecg_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            )

    def setup_model(self) -> None:
        """Initialize model, optimizer, and loss function"""
        self.model = create_hybrid_model(
            num_classes=self.config.model_config.num_classes,
            input_channels=self.config.model_config.input_channels,
            sequence_length=self.config.model_config.sequence_length
        ).to(self.device)

        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        )

        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=10,
            verbose=True
        )

        if self.config.focal_loss_alpha > 0:
            self.criterion = FocalLoss(
                alpha=self.config.focal_loss_alpha,
                gamma=self.config.focal_loss_gamma
            )
        else:
            self.criterion = nn.CrossEntropyLoss(
                label_smoothing=self.config.label_smoothing
            )

        logger.info(f"Model initialized with {self.model.count_parameters():,} parameters")

    def load_and_prepare_data(self) -> tuple[ECGMultimodalDataset, ECGMultimodalDataset]:
        """Load and prepare training and validation datasets"""
        logger.info("Loading ECG datasets...")

        datasets = {}

        try:
            mitbih_data = self.dataset_manager.load_mitbih_arrhythmia()
            datasets['mitbih'] = mitbih_data
            logger.info(f"Loaded MIT-BIH: {len(mitbih_data['signals'])} samples")
        except Exception as e:
            logger.warning(f"Failed to load MIT-BIH: {e}")

        try:
            ptbxl_data = self.dataset_manager.load_ptbxl()
            datasets['ptbxl'] = ptbxl_data
            logger.info(f"Loaded PTB-XL: {len(ptbxl_data['signals'])} samples")
        except Exception as e:
            logger.warning(f"Failed to load PTB-XL: {e}")

        try:
            cpsc_data = self.dataset_manager.load_cpsc2018()
            datasets['cpsc'] = cpsc_data
            logger.info(f"Loaded CPSC-2018: {len(cpsc_data['signals'])} samples")
        except Exception as e:
            logger.warning(f"Failed to load CPSC-2018: {e}")

        if not datasets:
            raise ValueError("No datasets could be loaded")

        all_signals = []
        all_labels = []
        all_condition_codes = []
        all_difficulty_scores = []

        for dataset_name, data in datasets.items():
            signals = data['signals']
            labels = data['labels']
            condition_codes = data.get('condition_codes', ['NORM'] * len(signals))

            processed_signals = []
            for signal in tqdm(signals, desc=f"Preprocessing {dataset_name}"):
                try:
                    result = self.preprocessor.process(signal)
                    if result.quality_metrics.overall_score > 0.7:  # Quality threshold
                        processed_signals.append(result.clean_signal)
                    else:
                        logger.debug(f"Rejected signal with quality {result.quality_metrics.overall_score}")
                except Exception as e:
                    logger.warning(f"Failed to preprocess signal: {e}")
                    continue

            mapped_labels = []
            mapped_codes = []
            difficulty_scores = []

            for i, (_, code) in enumerate(zip(processed_signals, condition_codes, strict=False)):
                condition = get_condition_by_code(code)
                if condition:
                    label = list(SCP_ECG_CONDITIONS.keys()).index(code)
                    mapped_labels.append(label)
                    mapped_codes.append(code)

                    rarity_score = 1.0 / (labels.count(labels[i]) + 1)  # Rarer = more difficult
                    urgency_score = {'low': 0.2, 'medium': 0.5, 'high': 0.8, 'critical': 1.0}.get(
                        condition.clinical_urgency, 0.5
                    )
                    difficulty_scores.append(rarity_score * 0.7 + urgency_score * 0.3)
                else:
                    mapped_labels.append(0)  # Assuming NORM is first
                    mapped_codes.append('NORM')
                    difficulty_scores.append(0.1)  # Easy case

            all_signals.extend(processed_signals)
            all_labels.extend(mapped_labels)
            all_condition_codes.extend(mapped_codes)
            all_difficulty_scores.extend(difficulty_scores)

        logger.info(f"Total processed samples: {len(all_signals)}")

        if train_test_split is None:
            raise ImportError(
                "scikit-learn is required for data splitting"
            )
        train_signals, val_signals, train_labels, val_labels, train_codes, val_codes, train_difficulty, val_difficulty = train_test_split(
            all_signals,
            all_labels,
            all_condition_codes,
            all_difficulty_scores,
            test_size=self.config.validation_split,
            stratify=all_labels,
            random_state=42,
        )

        train_dataset = ECGMultimodalDataset(
            signals=train_signals,
            labels=train_labels,
            condition_codes=train_codes,
            config=self.config,
            is_training=True,
            preprocessor=self.preprocessor
        )

        val_dataset = ECGMultimodalDataset(
            signals=val_signals,
            labels=val_labels,
            condition_codes=val_codes,
            config=self.config,
            is_training=False,
            preprocessor=self.preprocessor
        )

        self.curriculum_scheduler = CurriculumScheduler(self.config, train_difficulty)

        logger.info(f"Training samples: {len(train_dataset)}")
        logger.info(f"Validation samples: {len(val_dataset)}")

        return train_dataset, val_dataset

    def create_data_loaders(
        self,
        train_dataset: ECGMultimodalDataset,
        val_dataset: ECGMultimodalDataset
    ) -> tuple[DataLoader, DataLoader]:
        """Create data loaders with appropriate sampling strategies"""

        if self.config.use_class_weights:
            class_counts = np.bincount(train_dataset.labels)
            class_weights = 1.0 / (class_counts + 1e-6)
            sample_weights = [class_weights[label] for label in train_dataset.labels]
            sampler = WeightedRandomSampler(
                weights=sample_weights,
                num_samples=len(train_dataset),
                replacement=True
            )
        else:
            sampler = None

        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            sampler=sampler,
            shuffle=(sampler is None),
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory,
            drop_last=True
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory
        )

        return train_loader, val_loader

    def train_epoch(self, train_loader: DataLoader, epoch: int) -> dict[str, float]:
        """Train for one epoch"""
        self.model.train()

        total_loss = 0.0
        correct_predictions = 0
        total_samples = 0

        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch}")

        for batch_idx, batch in enumerate(progress_bar):
            signals = batch['signal'].to(self.device)
            labels = batch['label'].to(self.device)

            self.optimizer.zero_grad()

            if self.config.use_spectrograms or self.config.use_wavelets:
                outputs = self.model(signals, return_features=True)
                logits = outputs['logits']
            else:
                logits = self.model(signals)

            loss = self.criterion(logits, labels)

            loss.backward()

            if self.config.gradient_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.gradient_clip_norm
                )

            self.optimizer.step()

            total_loss += loss.item()
            _, predicted = torch.max(logits.data, 1)
            correct_predictions += (predicted == labels).sum().item()
            total_samples += labels.size(0)

            if batch_idx % self.config.log_interval == 0:
                current_lr = self.optimizer.param_groups[0]['lr']
                progress_bar.set_postfix({
                    'Loss': f'{loss.item():.4f}',
                    'Acc': f'{100. * correct_predictions / total_samples:.2f}%',
                    'LR': f'{current_lr:.2e}'
                })

                if self.config.use_wandb:
                    wandb.log({
                        'train_loss_step': loss.item(),
                        'train_accuracy_step': 100. * correct_predictions / total_samples,
                        'learning_rate': current_lr,
                        'epoch': epoch,
                        'step': epoch * len(train_loader) + batch_idx
                    })

        avg_loss = total_loss / len(train_loader)
        accuracy = 100. * correct_predictions / total_samples

        return {
            'loss': avg_loss,
            'accuracy': accuracy
        }

    def validate_epoch(self, val_loader: DataLoader, epoch: int) -> dict[str, float]:
        """Validate for one epoch"""
        self.model.eval()

        total_loss = 0.0
        correct_predictions = 0
        total_samples = 0
        all_predictions = []
        all_labels = []

        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
                signals = batch['signal'].to(self.device)
                labels = batch['label'].to(self.device)

                if self.config.use_spectrograms or self.config.use_wavelets:
                    outputs = self.model(signals, return_features=True)
                    logits = outputs['logits']
                else:
                    logits = self.model(signals)

                loss = self.criterion(logits, labels)

                total_loss += loss.item()
                _, predicted = torch.max(logits.data, 1)
                correct_predictions += (predicted == labels).sum().item()
                total_samples += labels.size(0)

                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        avg_loss = total_loss / len(val_loader)
        accuracy = 100. * correct_predictions / total_samples

        try:
            if classification_report is None or roc_auc_score is None:
                raise ImportError("scikit-learn is required for metrics")
            report = classification_report(
                all_labels,
                all_predictions,
                output_dict=True,
                zero_division=0,
            )

            try:
                auc_score = roc_auc_score(
                    all_labels,
                    all_predictions,
                    multi_class="ovr",
                    average="weighted",
                )
            except Exception:
                auc_score = 0.0

        except Exception as e:
            logger.warning(f"Error computing detailed metrics: {e}")
            report = {}
            auc_score = 0.0

        return {
            'loss': avg_loss,
            'accuracy': accuracy,
            'auc': auc_score,
            'classification_report': report
        }

    def save_checkpoint(self, epoch: int, val_metrics: dict[str, float], is_best: bool = False):
        """Save model checkpoint"""
        if not self.config.save_checkpoints:
            return

        checkpoint_dir = Path(self.config.checkpoint_dir)
        checkpoint_dir.mkdir(exist_ok=True)

        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'config': self.config,
            'training_history': self.training_history,
            'val_metrics': val_metrics
        }

        if not self.config.save_best_only:
            checkpoint_path = checkpoint_dir / f"checkpoint_epoch_{epoch}.pt"
            torch.save(checkpoint, checkpoint_path)

        if is_best:
            best_path = checkpoint_dir / "best_model.pt"
            torch.save(checkpoint, best_path)
            logger.info(f"Saved best model with validation accuracy: {val_metrics['accuracy']:.2f}%")

    def train(self):
        """Main training loop"""
        logger.info("Starting training pipeline...")

        self.setup_model()

        train_dataset, val_dataset = self.load_and_prepare_data()

        patience_counter = 0

        for epoch in range(self.config.num_epochs):
            self.current_epoch = epoch

            if self.config.curriculum_learning:
                self.curriculum_scheduler.step_epoch()
                current_indices = self.curriculum_scheduler.get_current_subset_indices()

                current_train_dataset = ECGMultimodalDataset(
                    signals=[train_dataset.signals[i] for i in current_indices],
                    labels=[train_dataset.labels[i] for i in current_indices],
                    condition_codes=[train_dataset.condition_codes[i] for i in current_indices],
                    config=self.config,
                    is_training=True,
                    preprocessor=self.preprocessor
                )
            else:
                current_train_dataset = train_dataset

            train_loader, val_loader = self.create_data_loaders(current_train_dataset, val_dataset)

            train_metrics = self.train_epoch(train_loader, epoch)

            val_metrics = self.validate_epoch(val_loader, epoch)

            if self.scheduler:
                self.scheduler.step(val_metrics['loss'])

            self.training_history['train_loss'].append(train_metrics['loss'])
            self.training_history['val_loss'].append(val_metrics['loss'])
            self.training_history['train_accuracy'].append(train_metrics['accuracy'])
            self.training_history['val_accuracy'].append(val_metrics['accuracy'])
            self.training_history['learning_rate'].append(self.optimizer.param_groups[0]['lr'])

            is_best = val_metrics['accuracy'] > self.best_val_accuracy
            if is_best:
                self.best_val_accuracy = val_metrics['accuracy']
                self.best_val_loss = val_metrics['loss']
                patience_counter = 0
            else:
                patience_counter += 1

            self.save_checkpoint(epoch, val_metrics, is_best)

            logger.info(
                f"Epoch {epoch}: "
                f"Train Loss: {train_metrics['loss']:.4f}, "
                f"Train Acc: {train_metrics['accuracy']:.2f}%, "
                f"Val Loss: {val_metrics['loss']:.4f}, "
                f"Val Acc: {val_metrics['accuracy']:.2f}%"
            )

            if self.config.use_wandb:
                wandb.log({
                    'epoch': epoch,
                    'train_loss': train_metrics['loss'],
                    'train_accuracy': train_metrics['accuracy'],
                    'val_loss': val_metrics['loss'],
                    'val_accuracy': val_metrics['accuracy'],
                    'val_auc': val_metrics['auc'],
                    'best_val_accuracy': self.best_val_accuracy,
                    'curriculum_stage': self.curriculum_scheduler.current_stage if self.config.curriculum_learning else 0
                })

            if patience_counter >= self.config.early_stopping_patience:
                logger.info(f"Early stopping triggered after {epoch + 1} epochs")
                break

        logger.info("Training completed!")
        logger.info(f"Best validation accuracy: {self.best_val_accuracy:.2f}%")

        self.evaluate_final_model(val_dataset)

    def evaluate_final_model(self, val_dataset: ECGMultimodalDataset):
        """Evaluate the final trained model"""
        logger.info("Evaluating final model...")

        if self.config.save_checkpoints:
            best_model_path = Path(self.config.checkpoint_dir) / "best_model.pt"
            if best_model_path.exists():
                checkpoint = torch.load(best_model_path, map_location=self.device)
                self.model.load_state_dict(checkpoint['model_state_dict'])
                logger.info("Loaded best model for final evaluation")

        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers
        )

        val_metrics = self.validate_epoch(val_loader, -1)

        self._generate_evaluation_report(val_metrics)

    def _generate_evaluation_report(self, metrics: dict[str, Any]):
        """Generate detailed evaluation report"""
        report_path = Path(self.config.checkpoint_dir) / "evaluation_report.json"

        report = {
            'final_metrics': {
                'accuracy': metrics['accuracy'],
                'loss': metrics['loss'],
                'auc': metrics['auc']
            },
            'training_history': self.training_history,
            'model_config': asdict(self.config.model_config),
            'training_config': asdict(self.config),
            'best_validation_accuracy': self.best_val_accuracy,
            'total_parameters': self.model.count_parameters()
        }

        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)

        logger.info(f"Evaluation report saved to {report_path}")

def create_training_config(**kwargs) -> TrainingConfig:
    """Factory function to create training configuration"""

    model_config = ModelConfig(
        input_channels=12,
        sequence_length=5000,
        num_classes=71,
        cnn_growth_rate=32,
        lstm_hidden_dim=256,
        transformer_heads=8,
        transformer_layers=4,
        dropout_rate=0.2
    )

    if 'model_config' in kwargs:
        model_config = kwargs.pop('model_config')

    return TrainingConfig(
        model_config=model_config,
        **kwargs
    )

if __name__ == "__main__":
    config = create_training_config(
        batch_size=16,  # Smaller batch size for memory efficiency
        learning_rate=1e-4,
        num_epochs=50,
        curriculum_learning=True,
        use_spectrograms=True,
        use_wavelets=True,
        use_wandb=False,  # Set to True if you want to use Weights & Biases
        device="cuda" if torch.cuda.is_available() else "cpu"
    )

    pipeline = ECGTrainingPipeline(config)
    pipeline.train()
