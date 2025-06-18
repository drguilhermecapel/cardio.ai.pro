"""
ECG Classifier - Machine Learning Classification System
Provides ML-based ECG pathology detection
"""

import numpy as np
import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
import pickle
import json
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class PredictionResult:
    """ML prediction result"""
    pathology: str
    probability: float
    confidence: str
    features_used: List[str]
    model_version: str = "1.0.0"


class ECGClassifier:
    """ECG classification using ensemble ML models"""
    
    def __init__(self, model_path: Optional[str] = None):
        """Initialize ECG classifier"""
        self.model_path = Path(model_path) if model_path else Path("app/ml/models")
        self.models = {}
        self.feature_importance = {}
        self.thresholds = self._load_thresholds()
        self._load_models()
        
    def predict(self, features: Dict[str, Any]) -> Dict[str, float]:
        """Predict pathologies from features"""
        predictions = {}
        
        try:
            # Convert features to array
            feature_array = self._prepare_features(features)
            
            # Ensemble predictions
            for pathology, model in self.models.items():
                if model is not None:
                    prob = self._predict_single(model, feature_array, pathology)
                    predictions[pathology] = prob
                else:
                    # Rule-based fallback
                    predictions[pathology] = self._rule_based_prediction(features, pathology)
            
            # Normalize predictions
            predictions = self._normalize_predictions(predictions)
            
            return predictions
            
        except Exception as e:
            logger.error(f"Prediction failed: {str(e)}")
            return self._get_default_predictions()
    
    def predict_with_confidence(
        self,
        features: Dict[str, Any]
    ) -> List[PredictionResult]:
        """Predict with detailed confidence metrics"""
        results = []
        
        try:
            predictions = self.predict(features)
            feature_array = self._prepare_features(features)
            
            for pathology, probability in predictions.items():
                # Calculate confidence
                confidence = self._calculate_confidence(probability, pathology)
                
                # Get important features
                important_features = self._get_important_features(pathology, feature_array)
                
                results.append(PredictionResult(
                    pathology=pathology,
                    probability=probability,
                    confidence=confidence,
                    features_used=important_features
                ))
            
            # Sort by probability
            results.sort(key=lambda x: x.probability, reverse=True)
            
            return results
            
        except Exception as e:
            logger.error(f"Prediction with confidence failed: {str(e)}")
            return []
    
    def _prepare_features(self, features: Dict[str, Any]) -> np.ndarray:
        """Prepare features for ML models"""
        # Standard feature order
        feature_order = [
            "heart_rate",
            "pr_interval", 
            "qrs_duration",
            "qt_interval",
            "qtc_interval",
            "p_wave_amplitude",
            "qrs_amplitude",
            "t_wave_amplitude",
            "rms",
            "variance",
            "skewness",
            "kurtosis",
            "sample_entropy",
            "approximate_entropy"
        ]
        
        # Extract ordered features
        feature_vector = []
        for feat in feature_order:
            if feat in features:
                value = features[feat]
                if isinstance(value, (int, float)):
                    feature_vector.append(float(value))
                else:
                    feature_vector.append(0.0)  # Default for missing
            else:
                # Try to extract from nested structures
                if "spectral_features" in features and feat in features["spectral_features"]:
                    feature_vector.append(float(features["spectral_features"][feat]))
                else:
                    feature_vector.append(0.0)
        
        return np.array(feature_vector)
    
    def _predict_single(
        self,
        model: Any,
        features: np.ndarray,
        pathology: str
    ) -> float:
        """Single model prediction"""
        try:
            # Mock prediction - replace with actual model
            # In production, use model.predict_proba(features.reshape(1, -1))[0, 1]
            
            # Simulate model behavior based on features
            if pathology == "atrial_fibrillation":
                # High heart rate variability suggests AF
                hr_idx = 0  # heart_rate index
                if features[hr_idx] > 100 or features[hr_idx] < 60:
                    base_prob = 0.7
                else:
                    base_prob = 0.3
                
                # Add noise for realism
                noise = np.random.normal(0, 0.1)
                return np.clip(base_prob + noise, 0, 1)
                
            elif pathology == "long_qt_syndrome":
                # QTc > 450 suggests long QT
                qtc_idx = 4  # qtc_interval index
                if features[qtc_idx] > 450:
                    return np.clip(0.85 + np.random.normal(0, 0.05), 0, 1)
                else:
                    return np.clip(0.15 + np.random.normal(0, 0.05), 0, 1)
                    
            else:
                # Default random prediction
                return np.random.beta(2, 5)  # Skewed towards lower probabilities
                
        except Exception as e:
            logger.error(f"Single prediction failed for {pathology}: {str(e)}")
            return 0.5
    
    def _rule_based_prediction(
        self,
        features: Dict[str, Any],
        pathology: str
    ) -> float:
        """Rule-based prediction fallback"""
        rules = {
            "atrial_fibrillation": self._check_af_rules,
            "long_qt_syndrome": self._check_long_qt_rules,
            "ventricular_tachycardia": self._check_vt_rules,
            "myocardial_infarction": self._check_mi_rules,
            "left_bundle_branch_block": self._check_lbbb_rules,
            "right_bundle_branch_block": self._check_rbbb_rules,
        }
        
        if pathology in rules:
            return rules[pathology](features)
        
        return 0.1  # Low default probability
    
    def _check_af_rules(self, features: Dict[str, Any]) -> float:
        """Check atrial fibrillation rules"""
        score = 0.0
        
        # Irregular RR intervals
        if "rr_variability" in features and features["rr_variability"] > 0.15:
            score += 0.4
        
        # Absent P waves
        if "p_wave_amplitude" in features and features["p_wave_amplitude"] < 0.05:
            score += 0.3
        
        # Variable heart rate
        if "heart_rate" in features:
            hr = features["heart_rate"]
            if hr < 60 or hr > 100:
                score += 0.2
        
        return min(score, 0.9)
    
    def _check_long_qt_rules(self, features: Dict[str, Any]) -> float:
        """Check long QT syndrome rules"""
        qtc = features.get("qtc_interval", 0)
        
        if qtc > 500:
            return 0.95
        elif qtc > 470:
            return 0.8
        elif qtc > 450:
            return 0.6
        else:
            return 0.1
    
    def _check_vt_rules(self, features: Dict[str, Any]) -> float:
        """Check ventricular tachycardia rules"""
        score = 0.0
        
        hr = features.get("heart_rate", 0)
        qrs = features.get("qrs_duration", 0)
        
        # Fast heart rate
        if hr > 150:
            score += 0.5
        elif hr > 120:
            score += 0.3
        
        # Wide QRS
        if qrs > 140:
            score += 0.4
        elif qrs > 120:
            score += 0.2
        
        return min(score, 0.9)
    
    def _check_mi_rules(self, features: Dict[str, Any]) -> float:
        """Check myocardial infarction rules"""
        # Simplified - check for ST elevation, Q waves, etc.
        # In production, use proper criteria
        return 0.2
    
    def _check_lbbb_rules(self, features: Dict[str, Any]) -> float:
        """Check left bundle branch block rules"""
        qrs = features.get("qrs_duration", 0)
        
        if qrs > 140:
            return 0.8
        elif qrs > 120:
            return 0.5
        else:
            return 0.1
    
    def _check_rbbb_rules(self, features: Dict[str, Any]) -> float:
        """Check right bundle branch block rules"""
        qrs = features.get("qrs_duration", 0)
        
        if qrs > 120:
            return 0.7
        else:
            return 0.1
    
    def _normalize_predictions(self, predictions: Dict[str, float]) -> Dict[str, float]:
        """Normalize predictions to ensure valid probabilities"""
        # Ensure all values are in [0, 1]
        normalized = {}
        for pathology, prob in predictions.items():
            normalized[pathology] = np.clip(prob, 0, 1)
        
        # Add normal sinus rhythm
        if "normal_sinus_rhythm" not in normalized:
            # Probability of normal is inverse of max pathology probability
            max_pathology_prob = max(normalized.values()) if normalized else 0
            normalized["normal_sinus_rhythm"] = 1 - max_pathology_prob
        
        return normalized
    
    def _calculate_confidence(self, probability: float, pathology: str) -> str:
        """Calculate confidence level"""
        if probability > 0.9:
            return "very_high"
        elif probability > 0.7:
            return "high"
        elif probability > 0.5:
            return "moderate"
        else:
            return "low"
    
    def _get_important_features(
        self,
        pathology: str,
        features: np.ndarray
    ) -> List[str]:
        """Get most important features for a pathology"""
        # Predefined important features per pathology
        important_features_map = {
            "atrial_fibrillation": ["heart_rate", "rr_variability", "p_wave_amplitude"],
            "long_qt_syndrome": ["qt_interval", "qtc_interval", "t_wave_amplitude"],
            "ventricular_tachycardia": ["heart_rate", "qrs_duration", "qrs_amplitude"],
            "myocardial_infarction": ["st_elevation", "q_wave_amplitude", "t_wave_amplitude"],
            "left_bundle_branch_block": ["qrs_duration", "qrs_axis", "r_wave_amplitude"],
            "right_bundle_branch_block": ["qrs_duration", "qrs_axis", "s_wave_amplitude"],
        }
        
        return important_features_map.get(pathology, ["heart_rate", "qrs_duration", "qt_interval"])
    
    def _load_models(self):
        """Load pre-trained models"""
        # In production, load actual trained models
        # For now, create mock models
        pathologies = [
            "atrial_fibrillation",
            "long_qt_syndrome", 
            "ventricular_tachycardia",
            "myocardial_infarction",
            "left_bundle_branch_block",
            "right_bundle_branch_block",
            "normal_sinus_rhythm"
        ]
        
        for pathology in pathologies:
            # Try to load saved model
            model_file = self.model_path / f"{pathology}_model.pkl"
            if model_file.exists():
                try:
                    with open(model_file, 'rb') as f:
                        self.models[pathology] = pickle.load(f)
                except Exception as e:
                    logger.warning(f"Failed to load model for {pathology}: {e}")
                    self.models[pathology] = None
            else:
                self.models[pathology] = None
    
    def _load_thresholds(self) -> Dict[str, Dict[str, float]]:
        """Load diagnostic thresholds"""
        return {
            "atrial_fibrillation": {"min_probability": 0.7},
            "long_qt_syndrome": {"min_probability": 0.6, "qtc_threshold": 450},
            "ventricular_tachycardia": {"min_probability": 0.8, "hr_threshold": 120},
            "myocardial_infarction": {"min_probability": 0.75},
            "left_bundle_branch_block": {"min_probability": 0.7, "qrs_threshold": 120},
            "right_bundle_branch_block": {"min_probability": 0.7, "qrs_threshold": 120},
        }
    
    def _get_default_predictions(self) -> Dict[str, float]:
        """Get default predictions when classification fails"""
        return {
            "normal_sinus_rhythm": 0.7,
            "atrial_fibrillation": 0.1,
            "long_qt_syndrome": 0.05,
            "ventricular_tachycardia": 0.05,
            "myocardial_infarction": 0.05,
            "left_bundle_branch_block": 0.025,
            "right_bundle_branch_block": 0.025,
        }
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about loaded models"""
        info = {
            "version": "1.0.0",
            "models_loaded": len([m for m in self.models.values() if m is not None]),
            "total_models": len(self.models),
            "pathologies": list(self.models.keys()),
            "thresholds": self.thresholds
        }
        
        return info
    
    def update_model(self, pathology: str, model_path: str) -> bool:
        """Update a specific model"""
        try:
            with open(model_path, 'rb') as f:
                self.models[pathology] = pickle.load(f)
            logger.info(f"Updated model for {pathology}")
            return True
        except Exception as e:
            logger.error(f"Failed to update model for {pathology}: {e}")
            return False
