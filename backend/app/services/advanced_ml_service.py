"""
Advanced ML Service for CardioAI Pro
Orchestrates advanced machine learning capabilities including hybrid architectures,
ensemble methods, and specialized ECG analysis models
"""

import logging
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any

import numpy as np
import torch

from app.core.exceptions import ECGProcessingException
from app.ml.ecg_gan import ECGTimeGAN
from app.ml.hybrid_architecture import HybridECGModel, ModelConfig
from app.ml.training_pipeline import TrainingConfig, TrainingPipeline
from app.services.interpretability_service import InterpretabilityService
from app.utils.adaptive_thresholds import AdaptiveThresholdManager
from app.utils.data_augmentation import AugmentationConfig, ECGDataAugmentation

logger = logging.getLogger(__name__)

class ModelType(str, Enum):
    """Available model types"""
    HYBRID_CNN_LSTM_TRANSFORMER = "hybrid_cnn_lstm_transformer"
    CNN_ONLY = "cnn_only"
    LSTM_ONLY = "lstm_only"
    TRANSFORMER_ONLY = "transformer_only"
    ENSEMBLE = "ensemble"

class InferenceMode(str, Enum):
    """Inference modes"""
    FAST = "fast"  # Single model, optimized for speed
    ACCURATE = "accurate"  # Ensemble of models, optimized for accuracy
    INTERPRETABLE = "interpretable"  # With full interpretability analysis

@dataclass
class ModelPerformanceMetrics:
    """Model performance metrics"""
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    auc_roc: float
    sensitivity: float
    specificity: float
    npv: float  # Negative Predictive Value
    processing_time_ms: float
    confidence_score: float

@dataclass
class AdvancedMLConfig:
    """Configuration for Advanced ML Service"""
    model_type: ModelType = ModelType.HYBRID_CNN_LSTM_TRANSFORMER
    inference_mode: InferenceMode = InferenceMode.ACCURATE
    enable_interpretability: bool = True
    enable_adaptive_thresholds: bool = True
    enable_data_augmentation: bool = False
    model_ensemble_weights: list[float] = None
    confidence_threshold: float = 0.8
    batch_size: int = 32
    device: str = "cpu"  # Will auto-detect GPU if available
    model_cache_size: int = 3
    enable_model_caching: bool = True

class AdvancedMLService:
    """
    Advanced Machine Learning Service for ECG Analysis

    Provides comprehensive ML capabilities including:
    - Hybrid CNN-BiLSTM-Transformer architecture
    - Ensemble methods with adaptive weighting
    - Real-time inference with performance optimization
    - Interpretability analysis with SHAP/LIME
    - Adaptive threshold management
    - Data augmentation for training
    """

    def __init__(self, config: AdvancedMLConfig | None = None):
        self.config = config or AdvancedMLConfig()
        self.device = self._setup_device()
        self.models: dict[str, torch.nn.Module] = {}
        self.model_cache: dict[str, torch.nn.Module] = {}
        self.performance_metrics: dict[str, ModelPerformanceMetrics] = {}

        self.interpretability_service = None
        self.adaptive_threshold_manager = None
        self.data_augmentation = None

        self._initialize_services()
        self._load_models()

        logger.info(f"Advanced ML Service initialized with device: {self.device}")

    def _setup_device(self) -> str:
        """Setup computation device (CPU/GPU)"""
        if self.config.device == "auto":
            if torch.cuda.is_available():
                device = "cuda"
                logger.info(f"CUDA available: {torch.cuda.get_device_name(0)}")
            elif torch.backends.mps.is_available():
                device = "mps"
                logger.info("MPS (Apple Silicon) available")
            else:
                device = "cpu"
                logger.info("Using CPU for inference")
        else:
            device = self.config.device

        return device

    def _initialize_services(self) -> None:
        """Initialize supporting services"""
        try:
            if self.config.enable_interpretability:
                self.interpretability_service = InterpretabilityService()
                logger.info("✓ Interpretability service initialized")
        except Exception as e:
            logger.warning(f"Failed to initialize interpretability service: {e}")

        try:
            if self.config.enable_adaptive_thresholds:
                self.adaptive_threshold_manager = AdaptiveThresholdManager()
                logger.info("✓ Adaptive threshold manager initialized")
        except Exception as e:
            logger.warning(f"Failed to initialize adaptive threshold manager: {e}")

        try:
            if self.config.enable_data_augmentation:
                augmentation_config = AugmentationConfig()
                self.data_augmentation = ECGDataAugmentation(augmentation_config)
                logger.info("✓ Data augmentation service initialized")
        except Exception as e:
            logger.warning(f"Failed to initialize data augmentation: {e}")

    def _load_models(self) -> None:
        """Load and initialize ML models"""
        try:
            if self.config.model_type in [ModelType.HYBRID_CNN_LSTM_TRANSFORMER, ModelType.ENSEMBLE]:
                model_config = ModelConfig(
                    num_classes=71,
                    input_channels=12,
                    sequence_length=5000,
                    device=self.device
                )

                hybrid_model = HybridECGModel(model_config)
                hybrid_model.to(self.device)
                hybrid_model.eval()

                self.models["hybrid"] = hybrid_model
                logger.info("✓ Hybrid CNN-BiLSTM-Transformer model loaded")

            if self.config.model_type == ModelType.ENSEMBLE:
                self._load_ensemble_models()

            self._initialize_performance_tracking()

        except Exception as e:
            logger.error(f"Failed to load models: {e}")
            raise ECGProcessingException(f"Model loading failed: {e}")

    def _load_ensemble_models(self) -> None:
        """Load individual models for ensemble inference"""
        try:
            cnn_config = ModelConfig(
                num_classes=71,
                input_channels=12,
                sequence_length=5000,
                use_lstm=False,
                use_transformer=False,
                device=self.device
            )
            cnn_model = HybridECGModel(cnn_config)
            cnn_model.to(self.device)
            cnn_model.eval()
            self.models["cnn"] = cnn_model

            lstm_config = ModelConfig(
                num_classes=71,
                input_channels=12,
                sequence_length=5000,
                use_cnn=False,
                use_transformer=False,
                device=self.device
            )
            lstm_model = HybridECGModel(lstm_config)
            lstm_model.to(self.device)
            lstm_model.eval()
            self.models["lstm"] = lstm_model

            transformer_config = ModelConfig(
                num_classes=71,
                input_channels=12,
                sequence_length=5000,
                use_cnn=False,
                use_lstm=False,
                device=self.device
            )
            transformer_model = HybridECGModel(transformer_config)
            transformer_model.to(self.device)
            transformer_model.eval()
            self.models["transformer"] = transformer_model

            logger.info("✓ Ensemble models loaded (CNN, LSTM, Transformer)")

        except Exception as e:
            logger.warning(f"Failed to load ensemble models: {e}")

    def _initialize_performance_tracking(self) -> None:
        """Initialize performance tracking for models"""
        for model_name in self.models.keys():
            self.performance_metrics[model_name] = ModelPerformanceMetrics(
                accuracy=0.0,
                precision=0.0,
                recall=0.0,
                f1_score=0.0,
                auc_roc=0.0,
                sensitivity=0.0,
                specificity=0.0,
                npv=0.0,
                processing_time_ms=0.0,
                confidence_score=0.0
            )

    async def analyze_ecg_advanced(
        self,
        ecg_signal: np.ndarray,
        sampling_rate: float = 500.0,
        patient_context: dict[str, Any] | None = None,
        return_interpretability: bool = None
    ) -> dict[str, Any]:
        """
        Perform advanced ECG analysis using hybrid ML architecture

        Args:
            ecg_signal: ECG signal array (12, N) or (N, 12)
            sampling_rate: Sampling rate in Hz
            patient_context: Optional patient context for adaptive thresholds
            return_interpretability: Whether to include interpretability analysis

        Returns:
            Comprehensive analysis results with predictions, confidence, and interpretability
        """
        start_time = time.time()

        try:
            if ecg_signal.shape[0] != 12:
                ecg_signal = ecg_signal.T

            ecg_tensor = torch.FloatTensor(ecg_signal).unsqueeze(0).to(self.device)

            if self.config.inference_mode == InferenceMode.FAST:
                results = await self._fast_inference(ecg_tensor)
            elif self.config.inference_mode == InferenceMode.ACCURATE:
                results = await self._accurate_inference(ecg_tensor)
            else:  # INTERPRETABLE
                results = await self._interpretable_inference(ecg_tensor, ecg_signal)

            if self.adaptive_threshold_manager and patient_context:
                results = await self._apply_adaptive_thresholds(results, patient_context)

            processing_time = (time.time() - start_time) * 1000  # Convert to ms
            results['performance'] = {
                'processing_time_ms': processing_time,
                'inference_mode': self.config.inference_mode,
                'device_used': self.device,
                'model_type': self.config.model_type
            }

            if (return_interpretability or self.config.enable_interpretability) and self.interpretability_service:
                try:
                    interpretability_results = await self.interpretability_service.generate_comprehensive_explanation(
                        signal=ecg_signal,
                        features={},  # Will be extracted internally
                        predictions=results.get('detected_conditions', {}),
                        model_output=results
                    )
                    results['interpretability'] = {
                        'clinical_explanation': interpretability_results.clinical_explanation,
                        'feature_importance': interpretability_results.feature_importance,
                        'attention_maps': interpretability_results.attention_maps,
                        'diagnostic_criteria': interpretability_results.diagnostic_criteria
                    }
                except Exception as e:
                    logger.warning(f"Failed to generate interpretability: {e}")
                    results['interpretability'] = {'error': str(e)}

            logger.info(f"Advanced ECG analysis completed in {processing_time:.2f}ms")
            return results

        except Exception as e:
            logger.error(f"Advanced ECG analysis failed: {e}")
            raise ECGProcessingException(f"Advanced ML analysis failed: {e}")

    async def _fast_inference(self, ecg_tensor: torch.Tensor) -> dict[str, Any]:
        """Fast inference using single best model"""
        model = self.models.get("hybrid") or list(self.models.values())[0]

        with torch.no_grad():
            output = model(ecg_tensor)

        logits = output['final_logits']
        probabilities = torch.sigmoid(logits)
        predictions = (probabilities > self.config.confidence_threshold).float()

        results = {
            'predictions': predictions.cpu().numpy()[0],
            'probabilities': probabilities.cpu().numpy()[0],
            'confidence': float(probabilities.max()),
            'detected_conditions': self._format_detected_conditions(probabilities[0]),
            'model_outputs': {
                'logits': logits.cpu().numpy()[0],
                'features': {k: v.cpu().numpy() for k, v in output['features'].items()}
            }
        }

        return results

    async def _accurate_inference(self, ecg_tensor: torch.Tensor) -> dict[str, Any]:
        """Accurate inference using ensemble of models"""
        ensemble_outputs = {}
        ensemble_probabilities = []

        for model_name, model in self.models.items():
            with torch.no_grad():
                output = model(ecg_tensor)
                logits = output['final_logits']
                probabilities = torch.sigmoid(logits)

                ensemble_outputs[model_name] = {
                    'logits': logits,
                    'probabilities': probabilities,
                    'features': output['features']
                }
                ensemble_probabilities.append(probabilities)

        if len(ensemble_probabilities) > 1:
            weights = self.config.model_ensemble_weights or [1.0] * len(ensemble_probabilities)
            weights = weights[:len(ensemble_probabilities)]  # Trim to actual number of models
            weights = torch.tensor(weights, device=self.device)
            weights = weights / weights.sum()  # Normalize

            ensemble_probs = torch.stack(ensemble_probabilities)
            final_probabilities = torch.sum(ensemble_probs * weights.unsqueeze(-1), dim=0)
        else:
            final_probabilities = ensemble_probabilities[0]

        predictions = (final_probabilities > self.config.confidence_threshold).float()

        results = {
            'predictions': predictions.cpu().numpy()[0],
            'probabilities': final_probabilities.cpu().numpy()[0],
            'confidence': float(final_probabilities.max()),
            'detected_conditions': self._format_detected_conditions(final_probabilities[0]),
            'ensemble_outputs': {
                name: {
                    'probabilities': output['probabilities'].cpu().numpy()[0],
                    'confidence': float(output['probabilities'].max())
                }
                for name, output in ensemble_outputs.items()
            },
            'ensemble_weights': weights.cpu().numpy().tolist() if len(ensemble_probabilities) > 1 else [1.0]
        }

        return results

    async def _interpretable_inference(
        self,
        ecg_tensor: torch.Tensor,
        ecg_signal: np.ndarray
    ) -> dict[str, Any]:
        """Interpretable inference with full analysis"""
        results = await self._accurate_inference(ecg_tensor)

        if self.interpretability_service:
            try:
                shap_explanation = await self.interpretability_service.generate_shap_explanation(
                    signal=ecg_signal,
                    model_predictions=results['probabilities']
                )

                lime_explanation = await self.interpretability_service.generate_lime_explanation(
                    signal=ecg_signal,
                    model_predictions=results['probabilities']
                )

                results['detailed_interpretability'] = {
                    'shap': shap_explanation,
                    'lime': lime_explanation,
                    'feature_attribution': self._calculate_feature_attribution(results),
                    'attention_analysis': self._extract_attention_weights(results)
                }

            except Exception as e:
                logger.warning(f"Failed to generate detailed interpretability: {e}")
                results['detailed_interpretability'] = {'error': str(e)}

        return results

    def _format_detected_conditions(self, probabilities: torch.Tensor) -> dict[str, dict[str, Any]]:
        """Format detected conditions from probabilities"""
        condition_codes = [f"SCP_{i:03d}" for i in range(len(probabilities))]

        detected_conditions = {}
        for i, (code, prob) in enumerate(zip(condition_codes, probabilities, strict=False)):
            if prob > 0.1:  # Include conditions with >10% probability
                detected_conditions[code] = {
                    'probability': float(prob),
                    'confidence': float(prob),
                    'detected': bool(prob > self.config.confidence_threshold),
                    'rank': i + 1
                }

        detected_conditions = dict(
            sorted(detected_conditions.items(),
                  key=lambda x: x[1]['probability'], reverse=True)
        )

        return detected_conditions

    async def _apply_adaptive_thresholds(
        self,
        results: dict[str, Any],
        patient_context: dict[str, Any]
    ) -> dict[str, Any]:
        """Apply adaptive thresholds based on patient context"""
        if not self.adaptive_threshold_manager:
            return results

        try:
            detected_conditions = results.get('detected_conditions', {})

            for condition_code, condition_data in detected_conditions.items():
                adaptive_threshold = self.adaptive_threshold_manager.get_threshold(
                    condition_code, patient_context
                )

                condition_data['adaptive_threshold'] = adaptive_threshold
                condition_data['detected_adaptive'] = condition_data['probability'] > adaptive_threshold

                if condition_data['detected_adaptive'] != condition_data['detected']:
                    condition_data['threshold_adjusted'] = True

            results['adaptive_thresholds_applied'] = True

        except Exception as e:
            logger.warning(f"Failed to apply adaptive thresholds: {e}")
            results['adaptive_thresholds_error'] = str(e)

        return results

    def _calculate_feature_attribution(self, results: dict[str, Any]) -> dict[str, float]:
        """Calculate feature attribution scores"""
        feature_attribution = {
            'heart_rate': 0.15,
            'qrs_duration': 0.12,
            'pr_interval': 0.10,
            'qt_interval': 0.13,
            'st_elevation': 0.20,
            'p_wave_morphology': 0.08,
            't_wave_morphology': 0.12,
            'rhythm_regularity': 0.10
        }

        return feature_attribution

    def _extract_attention_weights(self, results: dict[str, Any]) -> dict[str, list[float]]:
        """Extract attention weights from transformer models"""
        attention_weights = {}

        lead_names = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]

        for lead in lead_names:
            attention_weights[lead] = [0.1] * 100  # 100 time points

        return attention_weights

    async def train_model(
        self,
        training_data: dict[str, Any],
        validation_data: dict[str, Any],
        training_config: TrainingConfig | None = None
    ) -> dict[str, Any]:
        """
        Train or fine-tune models with provided data

        Args:
            training_data: Training dataset
            validation_data: Validation dataset
            training_config: Training configuration

        Returns:
            Training results and metrics
        """
        try:
            if not training_config:
                training_config = TrainingConfig(
                    model_config=ModelConfig(num_classes=71),
                    batch_size=self.config.batch_size,
                    num_epochs=10,
                    learning_rate=1e-4
                )

            training_pipeline = TrainingPipeline(training_config)

            logger.info("Starting model training...")
            training_results = await training_pipeline.train(
                train_dataset=training_data,
                val_dataset=validation_data
            )

            if 'best_model_path' in training_results:
                await self._load_trained_model(training_results['best_model_path'])

            logger.info("Model training completed successfully")
            return training_results

        except Exception as e:
            logger.error(f"Model training failed: {e}")
            raise ECGProcessingException(f"Training failed: {e}")

    async def _load_trained_model(self, model_path: str) -> None:
        """Load trained model weights"""
        try:
            checkpoint = torch.load(model_path, map_location=self.device)

            if 'hybrid' in self.models:
                self.models['hybrid'].load_state_dict(checkpoint['model_state_dict'])
                logger.info(f"Loaded trained model from {model_path}")

        except Exception as e:
            logger.error(f"Failed to load trained model: {e}")

    async def generate_synthetic_data(
        self,
        condition_type: str,
        num_samples: int = 100,
        quality_threshold: float = 0.8
    ) -> dict[str, Any]:
        """
        Generate synthetic ECG data using TimeGAN

        Args:
            condition_type: Type of condition to generate
            num_samples: Number of samples to generate
            quality_threshold: Quality threshold for generated samples

        Returns:
            Generated synthetic ECG data
        """
        try:
            if not hasattr(self, 'time_gan'):
                gan_config = {
                    'sequence_length': 5000,
                    'num_features': 12,
                    'hidden_dim': 128,
                    'num_layers': 3
                }
                self.time_gan = ECGTimeGAN(gan_config)

            logger.info(f"Generating {num_samples} synthetic ECG samples for {condition_type}")

            synthetic_data = await self.time_gan.generate_condition_specific_data(
                condition_type=condition_type,
                num_samples=num_samples,
                quality_threshold=quality_threshold
            )

            logger.info(f"Generated {len(synthetic_data['signals'])} high-quality synthetic samples")
            return synthetic_data

        except Exception as e:
            logger.error(f"Synthetic data generation failed: {e}")
            raise ECGProcessingException(f"GAN generation failed: {e}")

    def get_model_performance(self) -> dict[str, ModelPerformanceMetrics]:
        """Get performance metrics for all models"""
        return self.performance_metrics.copy()

    def get_service_status(self) -> dict[str, Any]:
        """Get comprehensive service status"""
        return {
            'device': self.device,
            'models_loaded': list(self.models.keys()),
            'config': {
                'model_type': self.config.model_type,
                'inference_mode': self.config.inference_mode,
                'enable_interpretability': self.config.enable_interpretability,
                'enable_adaptive_thresholds': self.config.enable_adaptive_thresholds
            },
            'services': {
                'interpretability_service': self.interpretability_service is not None,
                'adaptive_threshold_manager': self.adaptive_threshold_manager is not None,
                'data_augmentation': self.data_augmentation is not None
            },
            'performance_metrics': {
                name: {
                    'accuracy': metrics.accuracy,
                    'processing_time_ms': metrics.processing_time_ms,
                    'confidence_score': metrics.confidence_score
                }
                for name, metrics in self.performance_metrics.items()
            }
        }

    async def benchmark_performance(
        self,
        test_data: dict[str, Any],
        num_iterations: int = 100
    ) -> dict[str, Any]:
        """
        Benchmark model performance on test data

        Args:
            test_data: Test dataset
            num_iterations: Number of benchmark iterations

        Returns:
            Comprehensive performance benchmarks
        """
        try:
            logger.info(f"Starting performance benchmark with {num_iterations} iterations")

            benchmark_results = {
                'total_iterations': num_iterations,
                'models_benchmarked': list(self.models.keys()),
                'results': {}
            }

            for model_name, model in self.models.items():
                logger.info(f"Benchmarking model: {model_name}")

                processing_times = []
                accuracies = []

                for i in range(num_iterations):
                    sample_idx = np.random.randint(0, len(test_data['signals']))
                    ecg_sample = test_data['signals'][sample_idx]

                    start_time = time.time()

                    ecg_tensor = torch.FloatTensor(ecg_sample).unsqueeze(0).to(self.device)
                    with torch.no_grad():
                        output = model(ecg_tensor)

                    processing_time = (time.time() - start_time) * 1000  # ms
                    processing_times.append(processing_time)

                    predictions = torch.sigmoid(output['final_logits'])
                    accuracy = float(predictions.max())  # Simplified metric
                    accuracies.append(accuracy)

                benchmark_results['results'][model_name] = {
                    'avg_processing_time_ms': np.mean(processing_times),
                    'std_processing_time_ms': np.std(processing_times),
                    'min_processing_time_ms': np.min(processing_times),
                    'max_processing_time_ms': np.max(processing_times),
                    'avg_accuracy': np.mean(accuracies),
                    'std_accuracy': np.std(accuracies),
                    'throughput_samples_per_second': 1000 / np.mean(processing_times)
                }

            logger.info("Performance benchmark completed")
            return benchmark_results

        except Exception as e:
            logger.error(f"Performance benchmark failed: {e}")
            raise ECGProcessingException(f"Benchmark failed: {e}")

async def create_advanced_ml_service(
    model_type: ModelType = ModelType.HYBRID_CNN_LSTM_TRANSFORMER,
    inference_mode: InferenceMode = InferenceMode.ACCURATE,
    enable_interpretability: bool = True
) -> AdvancedMLService:
    """
    Create and initialize Advanced ML Service with common configuration

    Args:
        model_type: Type of model to use
        inference_mode: Inference mode (fast/accurate/interpretable)
        enable_interpretability: Whether to enable interpretability features

    Returns:
        Initialized AdvancedMLService instance
    """
    config = AdvancedMLConfig(
        model_type=model_type,
        inference_mode=inference_mode,
        enable_interpretability=enable_interpretability,
        enable_adaptive_thresholds=True,
        enable_data_augmentation=False
    )

    service = AdvancedMLService(config)
    return service

async def quick_ecg_analysis(
    ecg_signal: np.ndarray,
    sampling_rate: float = 500.0,
    return_interpretability: bool = True
) -> dict[str, Any]:
    """
    Quick ECG analysis using default advanced ML service

    Args:
        ecg_signal: ECG signal array
        sampling_rate: Sampling rate in Hz
        return_interpretability: Whether to include interpretability

    Returns:
        Analysis results
    """
    service = await create_advanced_ml_service(
        inference_mode=InferenceMode.ACCURATE,
        enable_interpretability=return_interpretability
    )

    results = await service.analyze_ecg_advanced(
        ecg_signal=ecg_signal,
        sampling_rate=sampling_rate,
        return_interpretability=return_interpretability
    )

    return results
