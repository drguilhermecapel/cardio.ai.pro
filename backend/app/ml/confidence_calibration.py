"""
Confidence Calibration System for ECG AI Predictions
Implements advanced calibration techniques to ensure reliable confidence scores
Based on Phase 3 optimization specifications for CardioAI Pro
"""

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

logger = logging.getLogger(__name__)


@dataclass
class CalibrationMetrics:
    """Metrics for evaluating calibration quality"""
    expected_calibration_error: float
    maximum_calibration_error: float
    average_calibration_error: float
    brier_score: float
    reliability_diagram_data: dict[str, list[float]]
    confidence_histogram: dict[str, list[float]]


@dataclass
class CalibrationConfig:
    """Configuration for confidence calibration"""
    method: str = "platt"  # "platt", "isotonic", "temperature", "ensemble"
    n_bins: int = 10
    temperature_init: float = 1.0
    ensemble_weights: list[float] | None = None
    min_samples_per_bin: int = 10
    calibration_data_ratio: float = 0.2


class TemperatureScaling:
    """
    Temperature scaling calibration method
    Simple but effective post-hoc calibration technique
    """

    def __init__(self, temperature_init: float = 1.0):
        self.temperature = temperature_init
        self.is_fitted = False

    def fit(self, logits: npt.NDArray[np.float64], labels: npt.NDArray[np.int32]) -> None:
        """
        Fit temperature scaling parameter

        Args:
            logits: Raw model outputs (before softmax)
            labels: True binary labels
        """
        from scipy.optimize import minimize_scalar

        def temperature_loss(temp: float) -> float:
            """Negative log-likelihood loss for temperature scaling"""
            scaled_logits = logits / temp
            probs = 1 / (1 + np.exp(-scaled_logits))  # Sigmoid for binary
            probs = np.clip(probs, 1e-7, 1 - 1e-7)
            loss = -np.mean(labels * np.log(probs) + (1 - labels) * np.log(1 - probs))
            return float(loss)

        result = minimize_scalar(temperature_loss, bounds=(0.1, 10.0), args=(), method='bounded')
        self.temperature = float(result.x)
        self.is_fitted = True

        logger.info(f"Temperature scaling fitted with temperature: {self.temperature:.3f}")

    def predict_proba(self, logits: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """
        Apply temperature scaling to get calibrated probabilities

        Args:
            logits: Raw model outputs

        Returns:
            Calibrated probabilities
        """
        if not self.is_fitted:
            raise ValueError("Temperature scaling must be fitted before prediction")

        scaled_logits = logits / self.temperature
        calibrated_probs = 1 / (1 + np.exp(-scaled_logits))

        return np.asarray(calibrated_probs, dtype=np.float64)


class PlattScaling:
    """
    Platt scaling calibration using logistic regression
    Maps classifier outputs to calibrated probabilities
    """

    def __init__(self) -> None:
        self.calibrator = LogisticRegression()
        self.is_fitted = False

    def fit(self, scores: npt.NDArray[np.float64], labels: npt.NDArray[np.int32]) -> None:
        """
        Fit Platt scaling calibrator

        Args:
            scores: Raw classifier scores/probabilities
            labels: True binary labels
        """
        scores_reshaped = scores.reshape(-1, 1)

        self.calibrator.fit(scores_reshaped, labels)
        self.is_fitted = True

        logger.info("Platt scaling calibrator fitted successfully")

    def predict_proba(self, scores: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """
        Apply Platt scaling to get calibrated probabilities

        Args:
            scores: Raw classifier scores

        Returns:
            Calibrated probabilities
        """
        if not self.is_fitted:
            raise ValueError("Platt scaling must be fitted before prediction")

        scores_reshaped = scores.reshape(-1, 1)
        calibrated_probs = self.calibrator.predict_proba(scores_reshaped)[:, 1]

        return np.asarray(calibrated_probs, dtype=np.float64)


class IsotonicCalibration:
    """
    Isotonic regression calibration
    Non-parametric method that preserves ranking
    """

    def __init__(self) -> None:
        self.calibrator = IsotonicRegression(out_of_bounds='clip')
        self.is_fitted = False

    def fit(self, scores: npt.NDArray[np.float64], labels: npt.NDArray[np.int32]) -> None:
        """
        Fit isotonic regression calibrator

        Args:
            scores: Raw classifier scores/probabilities
            labels: True binary labels
        """
        self.calibrator.fit(scores, labels)
        self.is_fitted = True

        logger.info("Isotonic regression calibrator fitted successfully")

    def predict_proba(self, scores: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """
        Apply isotonic regression to get calibrated probabilities

        Args:
            scores: Raw classifier scores

        Returns:
            Calibrated probabilities
        """
        if not self.is_fitted:
            raise ValueError("Isotonic calibration must be fitted before prediction")

        calibrated_probs = self.calibrator.predict(scores)

        return np.asarray(calibrated_probs, dtype=np.float64)


class CalibratedPrediction:
    """
    Calibrated prediction system as specified in the optimization guide
    """

    def __init__(self, model: Any):
        self.model = model
        self.calibrator = IsotonicRegression()
        self.is_calibrated = False

    def calibrate(self, val_data: dict[str, Any]) -> None:
        """Calibra probabilidades usando validação"""
        raw_probs = self.model.predict_proba(val_data['X'])
        self.calibrator.fit(raw_probs, val_data['y'])
        self.is_calibrated = True

        logger.info("Model calibration completed using validation data")

    def predict_with_confidence(self, ecg_signal: npt.NDArray[np.float64]) -> dict[str, Any]:
        """
        Predict with calibrated confidence as specified in the guide

        Args:
            ecg_signal: Input ECG signal

        Returns:
            Dictionary with prediction, probability, uncertainty, and confidence level
        """
        if not self.is_calibrated:
            logger.warning("Model not calibrated, using raw probabilities")

        raw_prob = self.model.predict_proba(ecg_signal)

        if self.is_calibrated:
            calibrated_prob = self.calibrator.transform(raw_prob)
        else:
            calibrated_prob = raw_prob

        uncertainty = self._calculate_uncertainty(raw_prob)

        confidence_level = self._get_confidence_level(calibrated_prob, uncertainty)

        return {
            'prediction': int(np.argmax(calibrated_prob)),
            'probability': calibrated_prob.tolist(),
            'uncertainty': float(uncertainty),
            'confidence_level': confidence_level,
            'requires_review': confidence_level != 'HIGH'
        }

    def _calculate_uncertainty(self, probabilities: npt.NDArray[np.float64]) -> float:
        """Calculate prediction uncertainty using entropy"""
        probs = np.clip(probabilities, 1e-7, 1 - 1e-7)
        entropy = -np.sum(probs * np.log(probs))

        max_entropy = np.log(len(probs))
        normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0.0

        return float(normalized_entropy)

    def _get_confidence_level(self, probabilities: npt.NDArray[np.float64],
                            uncertainty: float) -> str:
        """Determine confidence level based on probability and uncertainty"""
        max_prob = np.max(probabilities)

        if max_prob > 0.9 and uncertainty < 0.1:
            return 'HIGH'
        elif max_prob > 0.7 and uncertainty < 0.3:
            return 'MEDIUM'
        else:
            return 'LOW'


class ConfidenceCalibrationSystem:
    """
    Comprehensive confidence calibration system for ECG AI predictions
    """

    def __init__(self, config: CalibrationConfig | None = None):
        self.config = config or CalibrationConfig()
        self.calibrators: dict[str, Any] = {}
        self.calibration_metrics: dict[str, CalibrationMetrics] = {}
        self.is_fitted = False

    def fit_calibration(self, predictions: dict[str, npt.NDArray[np.float64]],
                       labels: dict[str, npt.NDArray[np.int32]]) -> None:
        """
        Fit calibration models for multiple conditions

        Args:
            predictions: Dictionary mapping condition names to prediction scores
            labels: Dictionary mapping condition names to true labels
        """
        logger.info(f"Fitting calibration models using {self.config.method} method")

        for condition, scores in predictions.items():
            if condition not in labels:
                logger.warning(f"No labels found for condition: {condition}")
                continue

            condition_labels = labels[condition]

            if len(scores) != len(condition_labels):
                logger.warning(f"Mismatch in scores and labels length for {condition}")
                continue

            if self.config.method == "platt":
                calibrator: Any = PlattScaling()
            elif self.config.method == "isotonic":
                calibrator = IsotonicCalibration()
            elif self.config.method == "temperature":
                calibrator = TemperatureScaling(self.config.temperature_init)
            else:
                raise ValueError(f"Unknown calibration method: {self.config.method}")

            try:
                calibrator.fit(scores, condition_labels)
                self.calibrators[condition] = calibrator
                logger.info(f"Calibration fitted for condition: {condition}")
            except Exception as e:
                logger.error(f"Failed to fit calibration for {condition}: {e}")

        self.is_fitted = True

    def calibrate_predictions(self, predictions: dict[str, float]) -> dict[str, float]:
        """
        Apply calibration to new predictions

        Args:
            predictions: Dictionary mapping condition names to raw prediction scores

        Returns:
            Dictionary with calibrated confidence scores
        """
        if not self.is_fitted:
            logger.warning("Calibration system not fitted, returning original predictions")
            return predictions

        calibrated_predictions = {}

        for condition, score in predictions.items():
            if condition in self.calibrators:
                try:
                    score_array = np.array([score])
                    calibrated_score = self.calibrators[condition].predict_proba(score_array)[0]
                    calibrated_predictions[condition] = float(calibrated_score)

                    logger.debug(f"Calibrated {condition}: {score:.3f} -> {calibrated_score:.3f}")
                except Exception as e:
                    logger.warning(f"Failed to calibrate {condition}: {e}, using original score")
                    calibrated_predictions[condition] = score
            else:
                logger.debug(f"No calibrator found for {condition}, using original score")
                calibrated_predictions[condition] = score

        return calibrated_predictions

    def evaluate_calibration(self, predictions: dict[str, npt.NDArray[np.float64]],
                           labels: dict[str, npt.NDArray[np.int32]]) -> dict[str, CalibrationMetrics]:
        """
        Evaluate calibration quality using multiple metrics

        Args:
            predictions: Dictionary mapping condition names to prediction scores
            labels: Dictionary mapping condition names to true labels

        Returns:
            Dictionary with calibration metrics for each condition
        """
        metrics = {}

        for condition, scores in predictions.items():
            if condition not in labels:
                continue

            condition_labels = labels[condition]

            if len(scores) != len(condition_labels):
                continue

            ece = self._expected_calibration_error(scores, condition_labels)
            mce = self._maximum_calibration_error(scores, condition_labels)
            ace = self._average_calibration_error(scores, condition_labels)
            brier = self._brier_score(scores, condition_labels)
            reliability_data = self._reliability_diagram_data(scores, condition_labels)
            confidence_hist = self._confidence_histogram(scores)

            metrics[condition] = CalibrationMetrics(
                expected_calibration_error=ece,
                maximum_calibration_error=mce,
                average_calibration_error=ace,
                brier_score=brier,
                reliability_diagram_data=reliability_data,
                confidence_histogram=confidence_hist
            )

            logger.info(f"Calibration metrics for {condition}: ECE={ece:.3f}, MCE={mce:.3f}, Brier={brier:.3f}")

        self.calibration_metrics = metrics
        return metrics

    def _expected_calibration_error(self, scores: npt.NDArray[np.float64],
                                  labels: npt.NDArray[np.int32]) -> float:
        """Calculate Expected Calibration Error (ECE)"""
        bin_boundaries = np.linspace(0, 1, self.config.n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]

        ece = 0.0
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers, strict=False):
            in_bin = (scores > bin_lower) & (scores <= bin_upper)
            prop_in_bin = in_bin.mean()

            if prop_in_bin > 0:
                accuracy_in_bin = labels[in_bin].mean()
                avg_confidence_in_bin = scores[in_bin].mean()
                ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

        return float(ece)

    def _maximum_calibration_error(self, scores: npt.NDArray[np.float64],
                                 labels: npt.NDArray[np.int32]) -> float:
        """Calculate Maximum Calibration Error (MCE)"""
        bin_boundaries = np.linspace(0, 1, self.config.n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]

        mce = 0.0
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers, strict=False):
            in_bin = (scores > bin_lower) & (scores <= bin_upper)

            if in_bin.sum() > 0:
                accuracy_in_bin = labels[in_bin].mean()
                avg_confidence_in_bin = scores[in_bin].mean()
                mce = max(mce, np.abs(avg_confidence_in_bin - accuracy_in_bin))

        return float(mce)

    def _average_calibration_error(self, scores: npt.NDArray[np.float64],
                                 labels: npt.NDArray[np.int32]) -> float:
        """Calculate Average Calibration Error (ACE)"""
        bin_boundaries = np.linspace(0, 1, self.config.n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]

        calibration_errors = []
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers, strict=False):
            in_bin = (scores > bin_lower) & (scores <= bin_upper)

            if in_bin.sum() > 0:
                accuracy_in_bin = labels[in_bin].mean()
                avg_confidence_in_bin = scores[in_bin].mean()
                calibration_errors.append(np.abs(avg_confidence_in_bin - accuracy_in_bin))

        return float(np.mean(calibration_errors)) if calibration_errors else 0.0

    def _brier_score(self, scores: npt.NDArray[np.float64],
                    labels: npt.NDArray[np.int32]) -> float:
        """Calculate Brier Score"""
        return float(np.mean((scores - labels) ** 2))

    def _reliability_diagram_data(self, scores: npt.NDArray[np.float64],
                                labels: npt.NDArray[np.int32]) -> dict[str, list[float]]:
        """Generate data for reliability diagram"""
        bin_boundaries = np.linspace(0, 1, self.config.n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]

        bin_centers = []
        accuracies = []
        confidences = []
        counts = []

        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers, strict=False):
            in_bin = (scores > bin_lower) & (scores <= bin_upper)

            if in_bin.sum() > 0:
                bin_centers.append((bin_lower + bin_upper) / 2)
                accuracies.append(float(labels[in_bin].mean()))
                confidences.append(float(scores[in_bin].mean()))
                counts.append(int(in_bin.sum()))

        return {
            'bin_centers': bin_centers,
            'accuracies': accuracies,
            'confidences': confidences,
            'counts': [float(x) for x in counts]
        }

    def _confidence_histogram(self, scores: npt.NDArray[np.float64]) -> dict[str, list[float]]:
        """Generate confidence histogram data"""
        hist, bin_edges = np.histogram(scores, bins=self.config.n_bins, range=(0, 1))
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

        return {
            'bin_centers': [float(x) for x in bin_centers],
            'counts': [float(x) for x in hist],
            'frequencies': [float(x) for x in (hist / len(scores))]
        }


def create_confidence_calibration_system(method: str = "isotonic", **kwargs: Any) -> ConfidenceCalibrationSystem:
    """
    Factory function to create confidence calibration system

    Args:
        method: Calibration method ("platt", "isotonic", "temperature")
        **kwargs: Additional configuration parameters

    Returns:
        Configured confidence calibration system
    """
    config = CalibrationConfig(method=method, **kwargs)
    return ConfidenceCalibrationSystem(config)


def create_calibrated_prediction(model: Any) -> CalibratedPrediction:
    """
    Factory function to create calibrated prediction system as per guide specification

    Args:
        model: The base ML model to calibrate

    Returns:
        Configured calibrated prediction system
    """
    return CalibratedPrediction(model)


if __name__ == "__main__":
    calibration_system = create_confidence_calibration_system(method="isotonic")

    np.random.seed(42)
    n_samples = 1000

    conditions = ['atrial_fibrillation', 'ventricular_tachycardia', 'normal_sinus_rhythm']

    predictions = {}
    labels = {}

    for condition in conditions:
        true_labels = np.random.binomial(1, 0.3, n_samples)  # 30% positive rate

        raw_scores = np.random.beta(2, 2, n_samples)  # Base scores
        raw_scores[true_labels == 1] += 0.3
        raw_scores = np.clip(raw_scores, 0, 1)

        predictions[condition] = raw_scores
        labels[condition] = true_labels

    calibration_system.fit_calibration(predictions, labels)

    test_predictions = {
        'atrial_fibrillation': 0.85,
        'ventricular_tachycardia': 0.92,
        'normal_sinus_rhythm': 0.15
    }

    calibrated = calibration_system.calibrate_predictions(test_predictions)

    print("Original vs Calibrated Predictions:")
    for condition in test_predictions:
        original = test_predictions[condition]
        calibrated_score = calibrated[condition]
        print(f"{condition}: {original:.3f} -> {calibrated_score:.3f}")

    metrics = calibration_system.evaluate_calibration(predictions, labels)

    print("\nCalibration Quality Metrics:")
    for condition, metric in metrics.items():
        print(f"{condition}:")
        print(f"  Expected Calibration Error: {metric.expected_calibration_error:.3f}")
        print(f"  Maximum Calibration Error: {metric.maximum_calibration_error:.3f}")
        print(f"  Brier Score: {metric.brier_score:.3f}")
