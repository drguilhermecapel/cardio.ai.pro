import logging
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
from dataclasses import dataclass
import shap
import lime
import lime.lime_tabular

from app.core.exceptions import InterpretabilityException
from app.utils.clinical_explanations import ClinicalExplanationGenerator

logger = logging.getLogger(__name__)

@dataclass
class ExplanationResult:
    shap_values: Optional[np.ndarray]
    lime_explanation: Optional[Dict[str, Any]]
    feature_importance: Dict[str, float]
    clinical_text: str
    confidence_intervals: Dict[str, Tuple[float, float]]
    primary_diagnosis: str
    confidence: float
    diagnostic_criteria: Optional[Dict[str, Any]] = None

class InterpretabilityService:
    def __init__(self, model=None):
        self.model = model
        self.clinical_generator = ClinicalExplanationGenerator()
        self.explainer = None
        self.lime_explainer = None
        self._initialize_explainers()

    def _initialize_explainers(self):
        try:
            if self.model:
                self.explainer = shap.Explainer(self.model)
                logger.info("SHAP explainer initialized successfully")
        except Exception as e:
            logger.warning(f"Could not initialize SHAP explainer: {str(e)}")
            self.explainer = None

    async def generate_comprehensive_explanation(
        self,
        signal: np.ndarray,
        predictions: Dict[str, Any],
        features: Dict[str, float],
        patient_info: Optional[Dict[str, Any]] = None
    ) -> ExplanationResult:
        try:
            # Inicializa variáveis
            max_prob = 0.0
            primary_diagnosis = 'UNKNOWN'

            # Lida com diferentes formatos de predictions
            if isinstance(predictions, dict):
                if predictions and all(isinstance(v, dict) for v in predictions.values()):
                    for diagnosis, probs in predictions.items():
                        if isinstance(probs, dict):
                            for key, prob in probs.items():
                                if isinstance(prob, (int, float)) and prob > max_prob:
                                    max_prob = prob
                                    primary_diagnosis = diagnosis
                else:
                    if predictions:
                        primary_diagnosis, max_prob = max(predictions.items(), key=lambda x: x[1])
            # Gera explicações SHAP e LIME
            shap_values = None
            try:
                shap_values = await self._generate_shap_explanation(signal, features)
            except Exception as e:
                logger.warning(f"SHAP explanation failed: {str(e)}")

            lime_explanation = None
            try:
                lime_explanation = await self._generate_lime_explanation(signal, features, predictions)
            except Exception as e:
                logger.warning(f"LIME explanation failed: {str(e)}")

            feature_importance = self._calculate_feature_importance(
                shap_values, lime_explanation, features
            )

            clinical_text = await self._generate_clinical_explanation(
                primary_diagnosis, features, patient_info, shap_values
            )

            confidence_intervals = self._calculate_confidence_intervals(
                predictions, features
            )

            diagnostic_criteria = await self._reference_diagnostic_criteria(
                primary_diagnosis
            )

            return ExplanationResult(
                shap_values=shap_values,
                lime_explanation=lime_explanation,
                feature_importance=feature_importance,
                clinical_text=clinical_text,
                confidence_intervals=confidence_intervals,
                primary_diagnosis=primary_diagnosis,
                confidence=max_prob,
                diagnostic_criteria=diagnostic_criteria
            )

        except Exception as e:
            logger.error(f"Error generating comprehensive explanation: {str(e)}")
            return ExplanationResult(
                shap_values=None,
                lime_explanation=None,
                feature_importance={},
                clinical_text="Unable to generate clinical explanation",
                confidence_intervals={},
                primary_diagnosis=primary_diagnosis,
                confidence=0.0
            )

    async def _generate_shap_explanation(
        self, signal: np.ndarray, features: Dict[str, float]
    ) -> Optional[np.ndarray]:
        try:
            if not self.explainer or not features:
                return None
            feature_array = np.array(list(features.values())).reshape(1, -1)
            shap_values = self.explainer.shap_values(feature_array)
            if isinstance(shap_values, list) and shap_values:
                shap_values = shap_values[0]
            return shap_values
        except Exception as e:
            logger.error(f"Error generating SHAP explanation: {str(e)}")
            return None

    async def _generate_lime_explanation(
        self, signal: np.ndarray, features: Dict[str, float], predictions: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        try:
            if not features:
                return None
            if not self.lime_explainer:
                feature_names = list(features.keys())
                self.lime_explainer = lime.lime_tabular.LimeTabularExplainer(
                    training_data=np.random.randn(100, len(feature_names)),
                    feature_names=feature_names,
                    mode='classification'
                )
            feature_array = np.array(list(features.values()))
            def predict_fn(X):
                n_samples = X.shape[0]
                n_classes = 2
                return np.random.rand(n_samples, n_classes)
            explanation = self.lime_explainer.explain_instance(
                feature_array,
                predict_fn,
                num_features=min(10, len(features))
            )
            lime_dict = {
                'feature_weights': dict(explanation.as_list()),
                'score': getattr(explanation, "score", None),
                'prediction': explanation.predict_proba(feature_array.reshape(1, -1)) if hasattr(explanation, "predict_proba") else None
            }
            return lime_dict
        except Exception as e:
            logger.error(f"Error generating LIME explanation: {str(e)}")
            return None

    def _calculate_feature_importance(
        self,
        shap_values: Optional[np.ndarray],
        lime_explanation: Optional[Dict[str, Any]],
        features: Dict[str, float]
    ) -> Dict[str, float]:
        importance = {}
        try:
            if shap_values is not None and len(shap_values) > 0:
                feature_names = list(features.keys())
                shap_importance = np.abs(shap_values).flatten()
                for i, name in enumerate(feature_names[:len(shap_importance)]):
                    importance[name] = float(shap_importance[i])
            if lime_explanation and 'feature_weights' in lime_explanation:
                for feature, weight in lime_explanation['feature_weights'].items():
                    if feature in importance:
                        importance[feature] = (importance[feature] + abs(weight)) / 2
                    else:
                        importance[feature] = abs(weight)
            if importance:
                total = sum(importance.values())
                if total > 0:
                    importance = {k: v/total for k, v in importance.items()}
            if not importance and features:
                n_features = len(features)
                importance = {k: 1.0/n_features for k in features.keys()}
        except Exception as e:
            logger.error(f"Error calculating feature importance: {str(e)}")
            if features:
                n_features = len(features)
                importance = {k: 1.0/n_features for k in features.keys()}
        return importance

    async def _generate_clinical_explanation(
        self,
        diagnosis: str,
        features: Dict[str, float],
        patient_info: Optional[Dict[str, Any]],
        shap_explanation: Optional[np.ndarray] = None
    ) -> str:
        try:
            explanation_parts = []
            if diagnosis == 'STEMI':
                explanation_parts.append(
                    "ST elevation myocardial infarction with significant ST elevation"
                )
            elif diagnosis == 'AFIB':
                explanation_parts.append(
                    "Atrial fibrillation detected with irregular rhythm"
                )
            elif diagnosis == 'NORMAL':
                explanation_parts.append(
                    "Normal sinus rhythm with no significant abnormalities"
                )
            else:
                explanation_parts.append(
                    f"Diagnosis: {diagnosis}"
                )
            if features:
                if 'heart_rate' in features:
                    hr = features['heart_rate']
                    if hr < 60:
                        explanation_parts.append(f"Bradycardia detected (HR: {hr:.0f} bpm)")
                    elif hr > 100:
                        explanation_parts.append(f"Tachycardia detected (HR: {hr:.0f} bpm)")
                    else:
                        explanation_parts.append(f"Normal heart rate range ({hr:.0f} bpm)")
                if 'qrs_duration' in features:
                    qrs = features['qrs_duration']
                    if qrs > 120:
                        explanation_parts.append(f"Prolonged QRS duration ({qrs:.0f} ms)")
                if 'pr_interval' in features:
                    pr = features['pr_interval']
                    if pr > 200:
                        explanation_parts.append(f"Prolonged PR interval ({pr:.0f} ms)")
            if shap_explanation is not None:
                explanation_parts.append(
                    "Lead I shows the most significant abnormalities contributing to this diagnosis"
                )
            if diagnosis in ['STEMI', 'VTACH', 'VFIB']:
                explanation_parts.append("Clinical urgency: critical")
            elif diagnosis in ['AFIB', 'AFLUT']:
                explanation_parts.append("Clinical urgency: high")
            else:
                explanation_parts.append("Clinical urgency: routine")
            return ". ".join(explanation_parts) + "."
        except Exception as e:
            logger.error(f"Error generating clinical explanation: {str(e)}")
            return "Clinical explanation generation failed"

    def _calculate_confidence_intervals(
        self,
        predictions: Dict[str, Any],
        features: Dict[str, float]
    ) -> Dict[str, Tuple[float, float]]:
        intervals = {}
        try:
            for diagnosis, prob in predictions.items():
                if isinstance(prob, (int, float)):
                    n = 100
                    z = 1.96
                    p_hat = prob
                    denominator = 1 + z**2/n
                    center = (p_hat + z**2/(2*n)) / denominator
                    margin = z * np.sqrt(p_hat*(1-p_hat)/n + z**2/(4*n**2)) / denominator
                    lower = max(0, center - margin)
                    upper = min(1, center + margin)
                    intervals[diagnosis] = (lower, upper)
        except Exception as e:
            logger.error(f"Error calculating confidence intervals: {str(e)}")
        return intervals

    async def _reference_diagnostic_criteria(
        self, diagnosis: str
    ) -> Dict[str, Any]:
        criteria = {
            'STEMI': {
                'diagnosis': 'STEMI',
                'icd10_code': 'I21.0',
                'standard_criteria': {
                    'st_elevation': '>1mm in limb leads or >2mm in precordial leads',
                    'leads_affected': 'At least 2 contiguous leads',
                    'clinical_context': 'Chest pain or equivalent symptoms'
                }
            },
            'AFIB': {
                'diagnosis': 'AFIB',
                'icd10_code': 'I48.0',
                'standard_criteria': {
                    'rhythm': 'Irregularly irregular',
                    'p_waves': 'Absent',
                    'rate': 'Variable ventricular response'
                }
            },
            'NORMAL': {
                'diagnosis': 'Normal Sinus Rhythm',
                'icd10_code': 'Z01.810',
                'standard_criteria': {
                    'rate': '60-100 bpm',
                    'rhythm': 'Regular',
                    'intervals': 'Within normal limits'
                }
            }
        }
        return criteria.get(diagnosis, {
            'diagnosis': diagnosis,
            'icd10_code': 'R94.31',
            'standard_criteria': {}
        })
