"""
TimeGAN Implementation for ECG Synthesis
Generates synthetic ECG signals for rare conditions and data augmentation
Based on scientific recommendations for CardioAI Pro
"""

import logging
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from app.preprocessing.advanced_pipeline import AdvancedECGPreprocessor

warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

@dataclass
class TimeGANConfig:
    """Configuration for TimeGAN model"""
    sequence_length: int = 5000
    num_features: int = 12  # 12-lead ECG

    hidden_dim: int = 128
    num_layers: int = 3

    batch_size: int = 32
    learning_rate: float = 1e-4
    num_epochs: int = 1000

    gamma: float = 1.0  # Supervised loss weight
    eta: float = 1.0    # Generator loss weight

    realism_threshold: float = 0.7  # Target >70% realism

    device: str = "cuda" if torch.cuda.is_available() else "cpu"

class ECGTimeSeriesDataset(Dataset):
    """Dataset for ECG time series data"""

    def __init__(
        self,
        signals: list[np.ndarray],
        condition_codes: list[str],
        sequence_length: int = 5000,
        preprocessor: AdvancedECGPreprocessor | None = None
    ):
        self.signals = signals
        self.condition_codes = condition_codes
        self.sequence_length = sequence_length
        self.preprocessor = preprocessor or AdvancedECGPreprocessor()

        self.processed_signals = []
        self.scalers = []

        self._preprocess_signals()

    def _preprocess_signals(self):
        """Preprocess and normalize ECG signals"""
        logger.info("Preprocessing ECG signals for TimeGAN...")

        for signal in tqdm(self.signals, desc="Processing signals"):
            try:
                result = self.preprocessor.process(signal)

                if result.quality_metrics.overall_score > 0.7:
                    processed_signal = result.clean_signal

                    if len(processed_signal) > self.sequence_length:
                        processed_signal = processed_signal[:self.sequence_length]
                    elif len(processed_signal) < self.sequence_length:
                        padding = np.zeros((self.sequence_length - len(processed_signal), processed_signal.shape[1]))
                        processed_signal = np.vstack([processed_signal, padding])

                    scaler = MinMaxScaler(feature_range=(-1, 1))
                    normalized_signal = scaler.fit_transform(processed_signal)

                    self.processed_signals.append(normalized_signal)
                    self.scalers.append(scaler)
                else:
                    logger.debug(f"Rejected signal with quality {result.quality_metrics.overall_score}")

            except Exception as e:
                logger.warning(f"Failed to preprocess signal: {e}")
                continue

        logger.info(f"Successfully processed {len(self.processed_signals)} signals")

    def __len__(self) -> int:
        return len(self.processed_signals)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        signal = torch.from_numpy(self.processed_signals[idx]).float()

        return {
            'signal': signal,
            'condition_code': self.condition_codes[idx] if idx < len(self.condition_codes) else 'NORM'
        }

class LSTMGenerator(nn.Module):
    """LSTM-based generator for TimeGAN"""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, num_layers: int = 3):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2 if num_layers > 1 else 0
        )

        self.output_projection = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, output_dim),
            nn.Tanh()  # Output in [-1, 1] range
        )

    def forward(self, x, hidden=None):
        lstm_out, hidden = self.lstm(x, hidden)

        output = self.output_projection(lstm_out)

        return output, hidden

class LSTMDiscriminator(nn.Module):
    """LSTM-based discriminator for TimeGAN"""

    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int = 3):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2 if num_layers > 1 else 0
        )

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        lstm_out, _ = self.lstm(x)

        last_output = lstm_out[:, -1, :]  # (batch, hidden_dim)

        output = self.classifier(last_output)

        return output

class LSTMEmbedder(nn.Module):
    """LSTM-based embedder for TimeGAN (maps real data to latent space)"""

    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int = 3):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2 if num_layers > 1 else 0
        )

        self.output_projection = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Sigmoid()  # Latent representation in [0, 1]
        )

    def forward(self, x):
        lstm_out, _ = self.lstm(x)

        latent = self.output_projection(lstm_out)

        return latent

class LSTMRecovery(nn.Module):
    """LSTM-based recovery network for TimeGAN (maps latent space back to data space)"""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, num_layers: int = 3):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2 if num_layers > 1 else 0
        )

        self.output_projection = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, output_dim),
            nn.Tanh()  # Output in [-1, 1] range
        )

    def forward(self, x):
        lstm_out, _ = self.lstm(x)

        output = self.output_projection(lstm_out)

        return output

class TimeGAN(nn.Module):
    """
    TimeGAN implementation for ECG synthesis
    Based on "Time-series Generative Adversarial Networks" (Yoon et al., 2019)
    """

    def __init__(self, config: TimeGANConfig):
        super().__init__()

        self.config = config

        self.embedder = LSTMEmbedder(
            input_dim=config.num_features,
            hidden_dim=config.hidden_dim,
            num_layers=config.num_layers
        )

        self.recovery = LSTMRecovery(
            input_dim=config.hidden_dim,
            hidden_dim=config.hidden_dim,
            output_dim=config.num_features,
            num_layers=config.num_layers
        )

        self.generator = LSTMGenerator(
            input_dim=config.hidden_dim,  # Takes noise in latent space
            hidden_dim=config.hidden_dim,
            output_dim=config.hidden_dim,
            num_layers=config.num_layers
        )

        self.discriminator = LSTMDiscriminator(
            input_dim=config.hidden_dim,
            hidden_dim=config.hidden_dim,
            num_layers=config.num_layers
        )

        self.mse_loss = nn.MSELoss()
        self.bce_loss = nn.BCELoss()

    def forward(self, x, z=None):
        """Forward pass for training"""
        batch_size, seq_len, num_features = x.shape

        H = self.embedder(x)

        X_tilde = self.recovery(H)

        if z is None:
            z = torch.randn(batch_size, seq_len, self.config.hidden_dim, device=x.device)
        E, _ = self.generator(z)

        X_hat = self.recovery(E)

        Y_real = self.discriminator(H)
        Y_fake = self.discriminator(E)

        return {
            'H': H,
            'X_tilde': X_tilde,
            'E': E,
            'X_hat': X_hat,
            'Y_real': Y_real,
            'Y_fake': Y_fake
        }

    def generate(self, num_samples: int, condition_code: str | None = None) -> np.ndarray:
        """Generate synthetic ECG signals"""
        self.eval()

        with torch.no_grad():
            z = torch.randn(
                num_samples,
                self.config.sequence_length,
                self.config.hidden_dim,
                device=next(self.parameters()).device
            )

            E, _ = self.generator(z)

            X_hat = self.recovery(E)

            synthetic_signals = X_hat.cpu().numpy()

        self.train()
        return synthetic_signals

class ECGQualityValidator:
    """Validates the quality and realism of synthetic ECG signals"""

    def __init__(self, preprocessor: AdvancedECGPreprocessor | None = None):
        self.preprocessor = preprocessor or AdvancedECGPreprocessor()

    def validate_signal_quality(self, signal: np.ndarray) -> dict[str, float]:
        """Validate basic signal quality metrics"""
        try:
            result = self.preprocessor.process(signal)

            quality_metrics = {
                'overall_score': result.quality_metrics.overall_score,
                'snr': result.quality_metrics.snr,
                'baseline_wander': result.quality_metrics.baseline_wander,
                'powerline_interference': result.quality_metrics.powerline_interference,
                'muscle_artifacts': result.quality_metrics.muscle_artifacts,
                'electrode_artifacts': result.quality_metrics.electrode_artifacts
            }

            return quality_metrics

        except Exception as e:
            logger.warning(f"Quality validation failed: {e}")
            return {'overall_score': 0.0}

    def validate_morphological_features(self, signal: np.ndarray) -> dict[str, float]:
        """Validate ECG morphological features"""
        try:
            lead_ii = signal[:, 1] if signal.shape[1] > 1 else signal[:, 0]

            features = {}

            peaks = self._detect_r_peaks(lead_ii)
            if len(peaks) > 1:
                rr_intervals = np.diff(peaks) / 500.0  # Assuming 500 Hz
                heart_rate = 60.0 / np.mean(rr_intervals)
                features['heart_rate'] = heart_rate
                features['heart_rate_valid'] = 1.0 if 40 <= heart_rate <= 200 else 0.0
            else:
                features['heart_rate'] = 0.0
                features['heart_rate_valid'] = 0.0

            amplitude_range = np.max(lead_ii) - np.min(lead_ii)
            features['amplitude_range'] = amplitude_range
            features['amplitude_valid'] = 1.0 if 0.5 <= amplitude_range <= 5.0 else 0.0

            baseline_std = np.std(lead_ii[:500])  # First 1 second
            features['baseline_stability'] = 1.0 / (1.0 + baseline_std)

            return features

        except Exception as e:
            logger.warning(f"Morphological validation failed: {e}")
            return {'heart_rate_valid': 0.0, 'amplitude_valid': 0.0, 'baseline_stability': 0.0}

    def _detect_r_peaks(self, signal: np.ndarray, fs: int = 500) -> np.ndarray:
        """Simple R-peak detection"""
        try:
            diff_signal = np.diff(signal)
            threshold = np.std(diff_signal) * 2

            peaks = []
            for i in range(1, len(diff_signal) - 1):
                if (diff_signal[i] > threshold and
                    diff_signal[i] > diff_signal[i-1] and
                    diff_signal[i] > diff_signal[i+1]):
                    peaks.append(i)

            return np.array(peaks)

        except Exception:
            return np.array([])

    def compute_realism_score(self, synthetic_signals: list[np.ndarray]) -> float:
        """Compute overall realism score for synthetic signals"""
        total_score = 0.0
        valid_signals = 0

        for signal in synthetic_signals:
            try:
                quality_metrics = self.validate_signal_quality(signal)
                quality_score = quality_metrics.get('overall_score', 0.0)

                morph_metrics = self.validate_morphological_features(signal)
                morph_score = np.mean([
                    morph_metrics.get('heart_rate_valid', 0.0),
                    morph_metrics.get('amplitude_valid', 0.0),
                    morph_metrics.get('baseline_stability', 0.0)
                ])

                combined_score = (quality_score * 0.6 + morph_score * 0.4)
                total_score += combined_score
                valid_signals += 1

            except Exception as e:
                logger.warning(f"Failed to compute realism for signal: {e}")
                continue

        if valid_signals == 0:
            return 0.0

        average_realism = total_score / valid_signals
        return average_realism

class TimeGANTrainer:
    """Training pipeline for TimeGAN"""

    def __init__(self, config: TimeGANConfig):
        self.config = config
        self.device = torch.device(config.device)

        self.model = TimeGAN(config).to(self.device)

        self.optimizer_embedder = optim.Adam(
            self.model.embedder.parameters(),
            lr=config.learning_rate
        )
        self.optimizer_recovery = optim.Adam(
            self.model.recovery.parameters(),
            lr=config.learning_rate
        )
        self.optimizer_generator = optim.Adam(
            self.model.generator.parameters(),
            lr=config.learning_rate
        )
        self.optimizer_discriminator = optim.Adam(
            self.model.discriminator.parameters(),
            lr=config.learning_rate
        )

        self.quality_validator = ECGQualityValidator()

        self.training_history = {
            'embedder_loss': [],
            'generator_loss': [],
            'discriminator_loss': [],
            'realism_score': []
        }

    def train_embedder_recovery(self, data_loader: DataLoader, num_epochs: int = 100):
        """Phase 1: Train embedder and recovery networks"""
        logger.info("Phase 1: Training embedder and recovery networks...")

        for epoch in range(num_epochs):
            total_loss = 0.0
            num_batches = 0

            for batch in tqdm(data_loader, desc=f"Embedder Epoch {epoch+1}"):
                signals = batch['signal'].to(self.device)

                self.optimizer_embedder.zero_grad()
                self.optimizer_recovery.zero_grad()

                H = self.model.embedder(signals)
                X_tilde = self.model.recovery(H)

                loss = self.model.mse_loss(X_tilde, signals)

                loss.backward()
                self.optimizer_embedder.step()
                self.optimizer_recovery.step()

                total_loss += loss.item()
                num_batches += 1

            avg_loss = total_loss / num_batches
            self.training_history['embedder_loss'].append(avg_loss)

            if (epoch + 1) % 10 == 0:
                logger.info(f"Embedder Epoch {epoch+1}: Loss = {avg_loss:.6f}")

    def train_generator_discriminator(self, data_loader: DataLoader, num_epochs: int = 500):
        """Phase 2: Train generator and discriminator"""
        logger.info("Phase 2: Training generator and discriminator...")

        for epoch in range(num_epochs):
            total_gen_loss = 0.0
            total_disc_loss = 0.0
            num_batches = 0

            for batch in tqdm(data_loader, desc=f"GAN Epoch {epoch+1}"):
                signals = batch['signal'].to(self.device)
                batch_size, seq_len, num_features = signals.shape

                z = torch.randn(batch_size, seq_len, self.config.hidden_dim, device=self.device)

                with torch.no_grad():
                    H = self.model.embedder(signals)

                self.optimizer_discriminator.zero_grad()

                E, _ = self.model.generator(z)
                Y_real = self.model.discriminator(H.detach())
                Y_fake = self.model.discriminator(E.detach())

                real_labels = torch.ones_like(Y_real)
                fake_labels = torch.zeros_like(Y_fake)

                disc_loss_real = self.model.bce_loss(Y_real, real_labels)
                disc_loss_fake = self.model.bce_loss(Y_fake, fake_labels)
                disc_loss = disc_loss_real + disc_loss_fake

                disc_loss.backward()
                self.optimizer_discriminator.step()

                self.optimizer_generator.zero_grad()

                E, _ = self.model.generator(z)
                Y_fake = self.model.discriminator(E)

                gen_loss_adv = self.model.bce_loss(Y_fake, real_labels)

                X_hat = self.model.recovery(E)
                gen_loss_supervised = self.model.mse_loss(
                    torch.mean(X_hat, dim=1),
                    torch.mean(signals, dim=1)
                ) + self.model.mse_loss(
                    torch.var(X_hat, dim=1),
                    torch.var(signals, dim=1)
                )

                gen_loss = gen_loss_adv + self.config.gamma * gen_loss_supervised

                gen_loss.backward()
                self.optimizer_generator.step()

                total_gen_loss += gen_loss.item()
                total_disc_loss += disc_loss.item()
                num_batches += 1

            avg_gen_loss = total_gen_loss / num_batches
            avg_disc_loss = total_disc_loss / num_batches

            self.training_history['generator_loss'].append(avg_gen_loss)
            self.training_history['discriminator_loss'].append(avg_disc_loss)

            if (epoch + 1) % 50 == 0:
                realism_score = self._validate_generation_quality()
                self.training_history['realism_score'].append(realism_score)

                logger.info(
                    f"GAN Epoch {epoch+1}: "
                    f"Gen Loss = {avg_gen_loss:.6f}, "
                    f"Disc Loss = {avg_disc_loss:.6f}, "
                    f"Realism = {realism_score:.3f}"
                )

                if realism_score >= self.config.realism_threshold:
                    logger.info(f"Target realism {self.config.realism_threshold} achieved!")
                    break

    def _validate_generation_quality(self, num_samples: int = 100) -> float:
        """Validate the quality of generated samples"""
        try:
            synthetic_signals = self.model.generate(num_samples)

            signal_list = [synthetic_signals[i] for i in range(num_samples)]

            realism_score = self.quality_validator.compute_realism_score(signal_list)

            return realism_score

        except Exception as e:
            logger.warning(f"Quality validation failed: {e}")
            return 0.0

    def train(self, data_loader: DataLoader):
        """Complete training pipeline"""
        logger.info("Starting TimeGAN training...")

        self.train_embedder_recovery(data_loader, num_epochs=100)

        self.train_generator_discriminator(data_loader, num_epochs=500)

        final_realism = self._validate_generation_quality(num_samples=200)
        logger.info(f"Final realism score: {final_realism:.3f}")

        if final_realism >= self.config.realism_threshold:
            logger.info("✅ TimeGAN training successful - target realism achieved!")
        else:
            logger.warning(f"⚠️ TimeGAN training completed but realism {final_realism:.3f} < target {self.config.realism_threshold}")

        return final_realism

    def save_model(self, path: str):
        """Save trained model"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'config': self.config,
            'training_history': self.training_history
        }, path)
        logger.info(f"Model saved to {path}")

    def load_model(self, path: str):
        """Load trained model"""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.training_history = checkpoint.get('training_history', {})
        logger.info(f"Model loaded from {path}")

class ECGSynthesizer:
    """High-level interface for ECG synthesis using TimeGAN"""

    def __init__(self, model_path: str | None = None, config: TimeGANConfig | None = None):
        self.config = config or TimeGANConfig()
        self.trainer = TimeGANTrainer(self.config)

        if model_path and Path(model_path).exists():
            self.trainer.load_model(model_path)
            self.is_trained = True
        else:
            self.is_trained = False

    def train_on_data(
        self,
        signals: list[np.ndarray],
        condition_codes: list[str],
        save_path: str | None = None
    ) -> float:
        """Train TimeGAN on provided ECG data"""
        logger.info(f"Training TimeGAN on {len(signals)} ECG signals...")

        dataset = ECGTimeSeriesDataset(
            signals=signals,
            condition_codes=condition_codes,
            sequence_length=self.config.sequence_length
        )

        data_loader = DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            drop_last=True
        )

        final_realism = self.trainer.train(data_loader)

        if save_path:
            self.trainer.save_model(save_path)

        self.is_trained = True
        return final_realism

    def generate_synthetic_ecgs(
        self,
        num_samples: int,
        condition_code: str | None = None
    ) -> list[np.ndarray]:
        """Generate synthetic ECG signals"""
        if not self.is_trained:
            raise ValueError("Model must be trained before generating samples")

        logger.info(f"Generating {num_samples} synthetic ECG signals...")

        synthetic_signals = self.trainer.model.generate(num_samples, condition_code)

        signal_list = [synthetic_signals[i] for i in range(num_samples)]

        realism_score = self.trainer.quality_validator.compute_realism_score(signal_list)
        logger.info(f"Generated signals realism score: {realism_score:.3f}")

        return signal_list

    def balance_rare_conditions(
        self,
        original_signals: list[np.ndarray],
        original_conditions: list[str],
        target_balance: dict[str, int]
    ) -> tuple[list[np.ndarray], list[str]]:
        """Balance rare conditions by generating synthetic samples"""
        logger.info("Balancing rare conditions with synthetic data...")

        balanced_signals = original_signals.copy()
        balanced_conditions = original_conditions.copy()

        condition_counts = {}
        for condition in original_conditions:
            condition_counts[condition] = condition_counts.get(condition, 0) + 1

        for condition_code, target_count in target_balance.items():
            current_count = condition_counts.get(condition_code, 0)

            if current_count < target_count:
                needed_samples = target_count - current_count
                logger.info(f"Generating {needed_samples} synthetic samples for {condition_code}")

                synthetic_samples = self.generate_synthetic_ecgs(
                    num_samples=needed_samples,
                    condition_code=condition_code
                )

                balanced_signals.extend(synthetic_samples)
                balanced_conditions.extend([condition_code] * needed_samples)

        logger.info(f"Balanced dataset: {len(balanced_signals)} total samples")
        return balanced_signals, balanced_conditions

def create_timegan_config(**kwargs) -> TimeGANConfig:
    """Factory function to create TimeGAN configuration"""
    return TimeGANConfig(**kwargs)

def train_ecg_timegan(
    signals: list[np.ndarray],
    condition_codes: list[str],
    config: TimeGANConfig | None = None,
    save_path: str | None = None
) -> ECGSynthesizer:
    """Train TimeGAN on ECG data"""

    if config is None:
        config = create_timegan_config()

    synthesizer = ECGSynthesizer(config=config)
    final_realism = synthesizer.train_on_data(signals, condition_codes, save_path)

    logger.info(f"TimeGAN training completed with realism score: {final_realism:.3f}")

    return synthesizer

def balance_stemi_condition(
    original_signals: list[np.ndarray],
    original_conditions: list[str],
    synthesizer: ECGSynthesizer
) -> tuple[list[np.ndarray], list[str]]:
    """
    Example: Balance STEMI condition from 0.4% to 5% as specified in requirements
    """

    total_samples = len(original_signals)
    current_stemi_count = original_conditions.count('STEMI')
    current_stemi_percentage = (current_stemi_count / total_samples) * 100

    logger.info(f"Current STEMI percentage: {current_stemi_percentage:.2f}%")

    target_stemi_count = int(total_samples * 0.05)

    target_balance = {
        'STEMI': target_stemi_count
    }

    balanced_signals, balanced_conditions = synthesizer.balance_rare_conditions(
        original_signals, original_conditions, target_balance
    )

    new_stemi_count = balanced_conditions.count('STEMI')
    new_stemi_percentage = (new_stemi_count / len(balanced_signals)) * 100

    logger.info(f"Balanced STEMI percentage: {new_stemi_percentage:.2f}%")

    return balanced_signals, balanced_conditions

if __name__ == "__main__":
    config = create_timegan_config(
        sequence_length=5000,
        num_features=12,
        hidden_dim=128,
        batch_size=16,  # Smaller batch for memory efficiency
        learning_rate=1e-4,
        realism_threshold=0.7,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )

    logger.info("TimeGAN ECG Synthesizer initialized")
    logger.info(f"Configuration: {config}")
    logger.info("Ready for training on ECG datasets")
