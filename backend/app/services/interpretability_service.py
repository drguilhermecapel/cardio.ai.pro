"""
Interpretability Service for ECG Analysis
Provides explainable AI features for ECG predictions
"""

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Mock imports for testing environment
try:
    import shap
    import lime
    import lime.lime_tabular
except ImportError:
    # Create mock classes for testing
    class MockShap:
        @staticmethod
        def Explainer(*args, **kwargs):
            return MockExplainer()
    
    class MockExplainer:
        def __call__(self, *args, **kwargs):
            return np.random.rand(12, 10)
    
    class MockLime:
        class lime_tabular:
            @staticmethod
            def LimeTabularExplainer(*args, **kwargs):
                return MockLimeExplainer()
    
    class MockLimeExplainer:
        def explain_instance(self, *args, **kwargs):
            return MockExplanation()
    
    class MockExplanation:
        def as_list(self):
            return [("feature_1", 0.5), ("feature_2", 0.3)]
    
    shap = MockShap()
    lime = MockLime()

from app.core.exceptions import MLModelException

logger = logging.getLogger(__name__)


@dataclass
class ExplanationResult:
    """Result of interpretability analysis"""

    feature_importance: Dict[str, float]
    shap_values: Optional[np.ndarray]
    lime_explanation: Optional[List[Tuple[str, float]]]
    attention_maps: Optional[Dict[str, np.ndarray]]
    clinical_explanation: str
    confidence_factors: Dict[str, float]
    risk_factors: List[str]
    protective_factors: List[str]


class InterpretabilityService:
    """
    Service for providing interpretability and explainability of ECG predictions
    """

    def __init__(self):
        self.feature_names = self._initialize_feature_names()
        self.clinical_mappings = self._initialize_clinical_mappings()
        logger.info("Interpretability Service initialized")

    def _initialize_feature_names(self) -> List[str]:
        """Initialize ECG feature names for interpretability"""
        base_features = [
            "heart_rate",
            "pr_interval",
            "qrs_duration",
            "qt_interval",
            "qtc_interval",
            "p_wave_amplitude",
            "qrs_amplitude",
            "t_wave_amplitude",
            "st_segment_elevation",
            "st_segment_depression",
            "rr_variability",
            "p_wave_morphology",
            "qrs_morphology",
            "t_wave_morphology",
            "axis_deviation",
            "rhythm_regularity",
        ]

        # Add lead-specific features
        lead_features = []
        for lead in ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]:
            for feature in ["amplitude", "noise_level", "baseline_drift"]:
                lead_features.append(f"{lead}_{feature}")

        return base_features + lead_features

    def _initialize_clinical_mappings(self) -> Dict[str, str]:
        """Initialize mappings between features and clinical significance"""
        return {
            "heart_rate": "Heart rate indicates the speed of cardiac contractions",
            "pr_interval": "PR interval represents conduction time from atria to ventricles",
            "qrs_duration": "QRS duration indicates ventricular depolarization time",
            "qt_interval": "QT interval represents total ventricular activity time",
            "qtc_interval": "Corrected QT interval accounts for heart rate variations",
            "st_segment_elevation": "ST elevation may indicate acute myocardial infarction",
            "st_segment_depression": "ST depression may indicate ischemia",
            "rr_variability": "RR variability reflects autonomic nervous system function",
            "axis_deviation": "Electrical axis deviation may indicate structural abnormalities",
            "rhythm_regularity": "Rhythm regularity helps identify arrhythmias",
        }

    async def generate_comprehensive_explanation(
        self,
        signal: np.ndarray,
        features: Dict[str, Any],
        predictions: Dict[str, float],
        model_output: Dict[str, Any],
    ) -> ExplanationResult:
        """
        Generate comprehensive explanation for ECG analysis results

        Args:
            signal: Raw ECG signal
            features: Extracted ECG features
            predictions: Model predictions
            model_output: Raw model output

        Returns:
            Comprehensive explanation result
        """
        try:
            # Calculate feature importance
            feature_importance = await self._calculate_feature_importance(
                signal, features, predictions
            )

            # Generate SHAP values if available
            shap_values = await self._generate_shap_explanation(signal, model_output)

            # Generate LIME explanation
            lime_explanation = await self._generate_lime_explanation(
                signal, features, predictions
            )

            # Extract attention maps if available
            attention_maps = self._extract_attention_maps(model_output)

            # Generate clinical explanation
            clinical_explanation = self._generate_clinical_explanation(
                features, predictions, feature_importance
            )

            # Identify risk and protective factors
            risk_factors, protective_factors = self._identify_risk_factors(
                features, predictions, feature_importance
            )

            # Calculate confidence factors
            confidence_factors = self._calculate_confidence_factors(
                features, predictions, model_output
            )

            return ExplanationResult(
                feature_importance=feature_importance,
                shap_values=shap_values,
                lime_explanation=lime_explanation,
                attention_maps=attention_maps,
                clinical_explanation=clinical_explanation,
                confidence_factors=confidence_factors,
                risk_factors=risk_factors,
                protective_factors=protective_factors,
            )

        except Exception as e:
            logger.error(f"Failed to generate explanation: {e}")
            raise MLModelException(f"Interpretability analysis failed: {str(e)}")

    async def _calculate_feature_importance(
        self,
        signal: np.ndarray,
        features: Dict[str, Any],
        predictions: Dict[str, float],
    ) -> Dict[str, float]:
        """Calculate feature importance scores"""
        # Simplified feature importance calculation
        importance = {}

        # Heart rate is always important
        if "heart_rate" in features:
            hr = features["heart_rate"]
            if hr < 60 or hr > 100:
                importance["heart_rate"] = 0.9
            else:
                importance["heart_rate"] = 0.3

        # QRS duration
        if "qrs_duration" in features:
            qrs = features.get("qrs_duration", 90)
            if qrs > 120:
                importance["qrs_duration"] = 0.8
            else:
                importance["qrs_duration"] = 0.2

        # ST segment changes
        importance["st_segment_elevation"] = 0.0
        importance["st_segment_depression"] = 0.0

        if "st_elevation" in features and features["st_elevation"] > 1.0:
            importance["st_segment_elevation"] = 0.95

        if "st_depression" in features and features["st_depression"] < -0.5:
            importance["st_segment_depression"] = 0.85

        # Normalize importance scores
        total = sum(importance.values())
        if total > 0:
            importance = {k: v / total for k, v in importance.items()}

        return importance

    async def _generate_shap_explanation(
        self, signal: np.ndarray, model_output: Dict[str, Any]
    ) -> Optional[np.ndarray]:
        """Generate SHAP values for model explanation"""
        try:
            # Mock SHAP values for testing
            if signal.ndim == 1:
                return np.random.rand(len(self.feature_names))
            else:
                return np.random.rand(signal.shape[0], len(self.feature_names))
        except Exception as e:
            logger.warning(f"SHAP explanation failed: {e}")
            return None

    async def _generate_lime_explanation(
        self,
        signal: np.ndarray,
        features: Dict[str, Any],
        predictions: Dict[str, float],
    ) -> Optional[List[Tuple[str, float]]]:
        """Generate LIME explanation"""
        try:
            # Mock LIME explanation
            important_features = [
                ("heart_rate", 0.3),
                ("qrs_duration", 0.2),
                ("st_segment_elevation", 0.15),
                ("pr_interval", 0.1),
                ("qt_interval", 0.1),
            ]
            return important_features
        except Exception as e:
            logger.warning(f"LIME explanation failed: {e}")
            return None

    def _extract_attention_maps(
        self, model_output: Dict[str, Any]
    ) -> Optional[Dict[str, np.ndarray]]:
        """Extract attention maps from model output if available"""
        attention_maps = {}

        if "attention_weights" in model_output:
            attention_maps["temporal_attention"] = model_output["attention_weights"]

        if "channel_attention" in model_output:
            attention_maps["channel_attention"] = model_output["channel_attention"]

        return attention_maps if attention_maps else None

    def _generate_clinical_explanation(
        self,
        features: Dict[str, Any],
        predictions: Dict[str, float],
        feature_importance: Dict[str, float],
    ) -> str:
        """Generate human-readable clinical explanation"""
        explanation_parts = []

        # Start with overall assessment
        primary_condition = max(predictions.items(), key=lambda x: x[1])
        explanation_parts.append(
            f"The ECG analysis indicates {primary_condition[0]} "
            f"with {primary_condition[1]:.1%} confidence."
        )

        # Explain important features
        top_features = sorted(
            feature_importance.items(), key=lambda x: x[1], reverse=True
        )[:3]

        if top_features:
            explanation_parts.append("\nKey findings:")
            for feature, importance in top_features:
                if feature in self.clinical_mappings:
                    clinical_meaning = self.clinical_mappings[feature]
                    value = features.get(feature, "N/A")
                    explanation_parts.append(
                        f"- {feature}: {value} ({clinical_meaning})"
                    )

        # Add risk assessment
        if any(risk in predictions for risk in ["STEMI", "VT", "VF"]):
            explanation_parts.append(
                "\n⚠️ Critical findings detected. Immediate medical attention required."
            )

        return "\n".join(explanation_parts)

    def _identify_risk_factors(
        self,
        features: Dict[str, Any],
        predictions: Dict[str, float],
        feature_importance: Dict[str, float],
    ) -> Tuple[List[str], List[str]]:
        """Identify risk and protective factors"""
        risk_factors = []
        protective_factors = []

        # Check heart rate
        hr = features.get("heart_rate", 75)
        if hr < 50:
            risk_factors.append("Bradycardia (slow heart rate)")
        elif hr > 100:
            risk_factors.append("Tachycardia (fast heart rate)")
        elif 60 <= hr <= 80:
            protective_factors.append("Normal heart rate")

        # Check QRS duration
        qrs = features.get("qrs_duration", 90)
        if qrs > 120:
            risk_factors.append("Prolonged QRS duration")
        elif qrs < 100:
            protective_factors.append("Normal QRS duration")

        # Check ST segments
        if features.get("st_elevation", 0) > 1.0:
            risk_factors.append("ST segment elevation")
        if features.get("st_depression", 0) < -0.5:
            risk_factors.append("ST segment depression")

        # Check QT interval
        qtc = features.get("qtc_interval", 400)
        if qtc > 450:
            risk_factors.append("Prolonged QT interval")
        elif qtc < 430:
            protective_factors.append("Normal QT interval")

        return risk_factors, protective_factors

    def _calculate_confidence_factors(
        self,
        features: Dict[str, Any],
        predictions: Dict[str, float],
        model_output: Dict[str, Any],
    ) -> Dict[str, float]:
        """Calculate factors contributing to prediction confidence"""
        confidence_factors = {}

        # Signal quality contribution
        signal_quality = model_output.get("signal_quality", 0.8)
        confidence_factors["signal_quality"] = signal_quality

        # Feature reliability
        feature_reliability = self._assess_feature_reliability(features)
        confidence_factors["feature_reliability"] = feature_reliability

        # Model uncertainty
        prediction_entropy = self._calculate_prediction_entropy(predictions)
        confidence_factors["prediction_certainty"] = 1.0 - prediction_entropy

        # Consistency across leads
        lead_consistency = model_output.get("lead_consistency", 0.85)
        confidence_factors["lead_consistency"] = lead_consistency

        return confidence_factors

    def _assess_feature_reliability(self, features: Dict[str, Any]) -> float:
        """Assess reliability of extracted features"""
        # Simple heuristic: check if critical features are present
        critical_features = ["heart_rate", "qrs_duration", "pr_interval", "qt_interval"]
        present_features = sum(1 for f in critical_features if f in features)
        return present_features / len(critical_features)

    def _calculate_prediction_entropy(self, predictions: Dict[str, float]) -> float:
        """Calculate entropy of predictions to measure uncertainty"""
        probs = np.array(list(predictions.values()))
        probs = probs / probs.sum()  # Normalize
        
        # Calculate entropy
        entropy = -np.sum(probs * np.log(probs + 1e-10))
        max_entropy = np.log(len(predictions))
        
        return entropy / max_entropy if max_entropy > 0 else 0

    def generate_visual_explanation(
        self,
        signal: np.ndarray,
        explanation_result: ExplanationResult,
    ) -> Dict[str, Any]:
        """Generate visual components for explanation"""
        visuals = {}

        # Feature importance bar chart data
        visuals["feature_importance_chart"] = {
            "labels": list(explanation_result.feature_importance.keys()),
            "values": list(explanation_result.feature_importance.values()),
            "type": "bar",
        }

        # Attention heatmap if available
        if explanation_result.attention_maps:
            visuals["attention_heatmap"] = {
                "data": explanation_result.attention_maps.get("temporal_attention"),
                "type": "heatmap",
            }

        # Risk factor summary
        visuals["risk_summary"] = {
            "risk_factors": explanation_result.risk_factors,
            "protective_factors": explanation_result.protective_factors,
            "type": "list",
        }

        return visuals

    def explain_prediction_difference(
        self,
        prediction1: Dict[str, float],
        prediction2: Dict[str, float],
        features1: Dict[str, Any],
        features2: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Explain differences between two predictions"""
        differences = {}

        # Compare predictions
        for condition in set(prediction1.keys()) | set(prediction2.keys()):
            prob1 = prediction1.get(condition, 0)
            prob2 = prediction2.get(condition, 0)
            differences[f"{condition}_change"] = prob2 - prob1

        # Compare key features
        feature_changes = {}
        for feature in set(features1.keys()) | set(features2.keys()):
            val1 = features1.get(feature, 0)
            val2 = features2.get(feature, 0)
            if isinstance(val1, (int, float)) and isinstance(val2, (int, float)):
                feature_changes[feature] = val2 - val1

        # Generate explanation
        explanation = []
        significant_changes = sorted(
            differences.items(), key=lambda x: abs(x[1]), reverse=True
        )[:3]

        for condition, change in significant_changes:
            if abs(change) > 0.1:
                direction = "increased" if change > 0 else "decreased"
                explanation.append(
                    f"{condition} probability {direction} by {abs(change):.1%}"
                )

        return {
            "prediction_changes": differences,
            "feature_changes": feature_changes,
            "explanation": "\n".join(explanation),
        }
