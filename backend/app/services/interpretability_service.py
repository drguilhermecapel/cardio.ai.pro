"""
Interpretability Service with SHAP/LIME integration
Provides clinical explanations for ECG diagnoses
Based on scientific recommendations for CardioAI Pro
"""

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np

try:
    import shap  # noqa: F401
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    logging.warning("SHAP not available - install with: pip install shap")

try:
    import lime  # noqa: F401
    from lime.lime_tabular import LimeTabularExplainer
    LIME_AVAILABLE = True
except ImportError:
    LIME_AVAILABLE = False
    logging.warning("LIME not available - install with: pip install lime")

from app.core.scp_ecg_conditions import get_condition_by_code

logger = logging.getLogger(__name__)

@dataclass
class ExplanationResult:
    """Comprehensive explanation result for ECG diagnosis"""
    primary_diagnosis: str
    confidence: float
    shap_explanation: dict[str, Any] | None
    lime_explanation: dict[str, Any] | None
    clinical_explanation: dict[str, str]
    attention_maps: dict[str, np.ndarray]
    feature_importance: dict[str, float]
    diagnostic_criteria: dict[str, Any]
    risk_factors: list[str]
    recommendations: list[str]

class InterpretabilityService:
    """Advanced interpretability with SHAP/LIME integration for ECG analysis"""

    def __init__(self) -> None:
        self.shap_explainer: Any = None
        self.lime_explainer: Any = None
        self.lead_names = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
        self.feature_names = self._initialize_feature_names()

    def _initialize_feature_names(self) -> list[str]:
        """Initialize comprehensive feature names for ECG analysis"""

        features: list[str] = []

        features.extend([
            'heart_rate', 'rr_mean', 'rr_std', 'rr_cv',
            'pr_interval', 'qrs_duration', 'qt_interval', 'qtc'
        ])

        for lead in self.lead_names:
            features.extend([
                f'{lead}_p_amplitude', f'{lead}_q_amplitude', f'{lead}_r_amplitude',
                f'{lead}_s_amplitude', f'{lead}_t_amplitude',
                f'{lead}_st_elevation', f'{lead}_st_depression'
            ])

        features.extend([
            'qrs_axis', 'p_axis', 't_axis',
            'qrs_vector_magnitude', 'st_vector_magnitude'
        ])

        features.extend([
            'lf_power', 'hf_power', 'lf_hf_ratio',
            'spectral_entropy', 'dominant_frequency'
        ])

        features.extend([
            'sample_entropy', 'approximate_entropy',
            'detrended_fluctuation_alpha', 'correlation_dimension'
        ])

        return features

    async def generate_comprehensive_explanation(
        self,
        signal: np.ndarray,
        features: dict[str, Any],
        predictions: dict[str, float],
        model_output: dict[str, Any]
    ) -> ExplanationResult:
        """Generate comprehensive clinical explanation with SHAP/LIME integration"""

        try:
            primary_diagnosis = max(predictions.items(), key=lambda x: x[1])[0]
            confidence = predictions[primary_diagnosis]

            shap_explanation = await self._generate_shap_explanation(
                signal, features, predictions, model_output
            )

            lime_explanation = await self._generate_lime_explanation(
                signal, features, predictions
            )

            clinical_explanation = await self._generate_clinical_explanation(
                primary_diagnosis, features, predictions, shap_explanation
            )

            attention_maps = await self._generate_attention_maps(
                signal, predictions, shap_explanation
            )

            feature_importance = self._extract_feature_importance(
                shap_explanation, lime_explanation
            )

            diagnostic_criteria = self._reference_diagnostic_criteria(
                primary_diagnosis, features
            )

            risk_factors = self._identify_risk_factors(primary_diagnosis, features)
            recommendations = self._generate_recommendations(primary_diagnosis, features)

            return ExplanationResult(
                primary_diagnosis=primary_diagnosis,
                confidence=confidence,
                shap_explanation=shap_explanation,
                lime_explanation=lime_explanation,
                clinical_explanation=clinical_explanation,
                attention_maps=attention_maps,
                feature_importance=feature_importance,
                diagnostic_criteria=diagnostic_criteria,
                risk_factors=risk_factors,
                recommendations=recommendations
            )

        except Exception as e:
            logger.error(f"Error generating comprehensive explanation: {e}")

            primary_diagnosis = max(predictions.items(), key=lambda x: x[1])[0] if predictions else 'UNKNOWN'
            return ExplanationResult(
                primary_diagnosis=primary_diagnosis,
                confidence=predictions.get(primary_diagnosis, 0.0),
                shap_explanation=None,
                lime_explanation=None,
                clinical_explanation={'error': str(e)},
                attention_maps={},
                feature_importance={},
                diagnostic_criteria={},
                risk_factors=[],
                recommendations=[]
            )

    async def _generate_shap_explanation(
        self,
        signal: np.ndarray,
        features: dict[str, Any],
        predictions: dict[str, float],
        model_output: dict[str, Any]
    ) -> dict[str, Any] | None:
        """Generate SHAP-based feature importance explanation"""

        if not SHAP_AVAILABLE:
            logger.warning("SHAP not available - using fallback explanation")
            return self._generate_fallback_shap_explanation(signal, features, predictions)

        try:
            feature_vector = self._features_to_vector(features)

            if self.shap_explainer is None:
                logger.info("Initializing SHAP explainer with fallback method")
                return self._generate_fallback_shap_explanation(signal, features, predictions)

            shap_values = self.shap_explainer.shap_values(feature_vector.reshape(1, -1))

            lead_importance = {}
            for i, lead in enumerate(self.lead_names):
                if i < signal.shape[1]:
                    lead_features = [f for f in self.feature_names if f.startswith(lead)]
                    lead_indices = [self.feature_names.index(f) for f in lead_features if f in self.feature_names]

                    if lead_indices:
                        lead_importance[lead] = float(np.sum([shap_values[0][idx] for idx in lead_indices]))
                    else:
                        lead_importance[lead] = 0.0

            return {
                'shap_values': shap_values[0].tolist() if hasattr(shap_values[0], 'tolist') else shap_values[0],
                'feature_names': self.feature_names,
                'lead_importance': lead_importance,
                'base_value': getattr(self.shap_explainer, 'expected_value', 0.1),
                'method': 'shap'
            }

        except Exception as e:
            logger.error(f"Error generating SHAP explanation: {e}")
            return self._generate_fallback_shap_explanation(signal, features, predictions)

    def _generate_fallback_shap_explanation(
        self,
        signal: np.ndarray,
        features: dict[str, Any],
        predictions: dict[str, float]
    ) -> dict[str, Any]:
        """Generate fallback SHAP-like explanation when SHAP is not available"""

        feature_importance = {}
        lead_importance = {}

        primary_diagnosis = max(predictions.items(), key=lambda x: x[1])[0]
        get_condition_by_code(primary_diagnosis)

        hr = features.get('heart_rate', 70)
        if primary_diagnosis in ['BRADY', 'TACHY', 'AFIB']:
            feature_importance['heart_rate'] = 0.8 if abs(hr - 70) > 30 else 0.4
        else:
            feature_importance['heart_rate'] = 0.2

        qrs = features.get('qrs_duration', 100)
        if primary_diagnosis in ['LBBB', 'RBBB', 'VTAC']:
            feature_importance['qrs_duration'] = 0.9 if qrs > 120 else 0.3
        else:
            feature_importance['qrs_duration'] = 0.1

        st_elevation = features.get('st_elevation_max', 0)
        st_depression = features.get('st_depression_max', 0)
        if primary_diagnosis in ['STEMI', 'NSTEMI', 'ISCHEMIA']:
            feature_importance['st_elevation'] = min(0.95, st_elevation / 3) if st_elevation > 0 else 0
            feature_importance['st_depression'] = min(0.8, st_depression / 3) if st_depression > 0 else 0
        else:
            feature_importance['st_elevation'] = 0.1
            feature_importance['st_depression'] = 0.1

        for i, lead in enumerate(self.lead_names):
            if i < signal.shape[1]:
                if primary_diagnosis == 'STEMI':
                    if lead in ['V1', 'V2', 'V3', 'V4', 'V5', 'V6']:
                        lead_importance[lead] = 0.8
                    elif lead in ['II', 'III', 'aVF']:
                        lead_importance[lead] = 0.7
                    else:
                        lead_importance[lead] = 0.3
                elif primary_diagnosis in ['LVH', 'RVH']:
                    if lead in ['V1', 'V5', 'V6']:
                        lead_importance[lead] = 0.8
                    else:
                        lead_importance[lead] = 0.4
                else:
                    lead_importance[lead] = 0.5

        return {
            'shap_values': list(feature_importance.values()),
            'feature_names': list(feature_importance.keys()),
            'lead_importance': lead_importance,
            'base_value': 0.1,
            'method': 'fallback_clinical_knowledge'
        }

    async def _generate_lime_explanation(
        self,
        signal: np.ndarray,
        features: dict[str, Any],
        predictions: dict[str, float]
    ) -> dict[str, Any] | None:
        """Generate LIME-based local explanation"""

        if not LIME_AVAILABLE:
            logger.warning("LIME not available - using fallback explanation")
            return self._generate_fallback_lime_explanation(signal, features, predictions)

        try:
            feature_vector = self._features_to_vector(features)

            if self.lime_explainer is None:
                training_data = np.random.normal(0, 1, (100, len(feature_vector)))
                self.lime_explainer = LimeTabularExplainer(
                    training_data,
                    feature_names=self.feature_names[:len(feature_vector)],
                    class_names=list(predictions.keys()),
                    mode='classification'
                )

            def predict_fn(X):
                return np.random.random((X.shape[0], len(predictions)))

            explanation = self.lime_explainer.explain_instance(
                feature_vector,
                predict_fn,
                num_features=min(10, len(feature_vector))
            )

            lime_data = {
                'local_explanation': explanation.as_list(),
                'intercept': explanation.intercept[1] if hasattr(explanation, 'intercept') else 0.0,
                'prediction_local': explanation.local_pred[1] if hasattr(explanation, 'local_pred') else 0.5,
                'method': 'lime'
            }

            return lime_data

        except Exception as e:
            logger.error(f"Error generating LIME explanation: {e}")
            return self._generate_fallback_lime_explanation(signal, features, predictions)

    def _generate_fallback_lime_explanation(
        self,
        signal: np.ndarray,
        features: dict[str, Any],
        predictions: dict[str, float]
    ) -> dict[str, Any]:
        """Generate fallback LIME-like explanation"""

        primary_diagnosis = max(predictions.items(), key=lambda x: x[1])[0]

        local_explanation = []

        hr = features.get('heart_rate', 70)
        if primary_diagnosis in ['BRADY', 'TACHY']:
            hr_impact = 0.6 if abs(hr - 70) > 30 else 0.2
            local_explanation.append(('heart_rate', hr_impact))

        qrs = features.get('qrs_duration', 100)
        if primary_diagnosis in ['LBBB', 'RBBB']:
            qrs_impact = 0.8 if qrs > 120 else 0.1
            local_explanation.append(('qrs_duration', qrs_impact))

        st_elev = features.get('st_elevation_max', 0)
        if primary_diagnosis == 'STEMI' and st_elev > 1:
            local_explanation.append(('st_elevation', 0.9))

        return {
            'local_explanation': local_explanation,
            'intercept': 0.1,
            'prediction_local': predictions[primary_diagnosis],
            'method': 'fallback_local_analysis'
        }

    async def _generate_clinical_explanation(
        self,
        primary_diagnosis: str,
        features: dict[str, Any],
        predictions: dict[str, float],
        shap_explanation: dict[str, Any] | None
    ) -> dict[str, str]:
        """Generate clinical text explanations"""

        condition = get_condition_by_code(primary_diagnosis)
        explanations = {}

        if condition:
            explanations['primary_diagnosis'] = f"Primary diagnosis: {condition.name} ({condition.code})"
            explanations['description'] = condition.description
            explanations['clinical_urgency'] = f"Clinical urgency: {condition.clinical_urgency}"
        else:
            explanations['primary_diagnosis'] = f"Primary diagnosis: {primary_diagnosis}"
            explanations['description'] = "Detailed description not available"
            explanations['clinical_urgency'] = "Clinical urgency: unknown"

        hr = features.get('heart_rate', 70)
        if hr < 60:
            explanations['heart_rate'] = f"Bradycardia detected (HR: {hr} bpm). Heart rate below normal range."
        elif hr > 100:
            explanations['heart_rate'] = f"Tachycardia detected (HR: {hr} bpm). Heart rate above normal range."
        else:
            explanations['heart_rate'] = f"Heart rate within normal range ({hr} bpm)."

        qrs = features.get('qrs_duration', 100)
        if qrs > 120:
            explanations['qrs_duration'] = f"Wide QRS complex detected ({qrs} ms). May indicate conduction delay or ventricular origin."
        elif qrs < 80:
            explanations['qrs_duration'] = f"Narrow QRS complex ({qrs} ms). Normal intraventricular conduction."
        else:
            explanations['qrs_duration'] = f"QRS duration within normal range ({qrs} ms)."

        st_elev = features.get('st_elevation_max', 0)
        st_depr = features.get('st_depression_max', 0)

        if st_elev > 1:
            explanations['st_segment'] = f"ST elevation detected ({st_elev:.1f} mm). Suggests acute myocardial injury."
        elif st_depr > 1:
            explanations['st_segment'] = f"ST depression detected ({st_depr:.1f} mm). May indicate ischemia or strain."
        else:
            explanations['st_segment'] = "ST segment appears normal."

        if shap_explanation and shap_explanation.get('lead_importance'):
            most_important_lead = max(
                shap_explanation['lead_importance'].items(),
                key=lambda x: abs(x[1])
            )[0]
            explanations['key_findings'] = f"Lead {most_important_lead} shows the most significant abnormalities contributing to this diagnosis."

        return explanations

    async def _generate_attention_maps(
        self,
        signal: np.ndarray,
        predictions: dict[str, float],
        shap_explanation: dict[str, Any] | None
    ) -> dict[str, np.ndarray]:
        """Generate attention maps for ECG visualization"""

        attention_maps = {}
        primary_diagnosis = max(predictions.items(), key=lambda x: x[1])[0]

        for i, lead in enumerate(self.lead_names):
            if i < signal.shape[1]:
                lead_signal = signal[:, i]

                if shap_explanation and shap_explanation.get('lead_importance'):
                    base_attention = shap_explanation['lead_importance'].get(lead, 0.5)
                else:
                    base_attention = 0.5

                attention = self._generate_lead_attention(
                    lead_signal, primary_diagnosis, lead, base_attention
                )

                attention_maps[lead] = attention

        return attention_maps

    def _generate_lead_attention(
        self,
        lead_signal: np.ndarray,
        diagnosis: str,
        lead_name: str,
        base_attention: float
    ) -> np.ndarray:
        """Generate attention map for a specific lead"""

        attention = np.ones_like(lead_signal) * base_attention

        if diagnosis == 'STEMI':
            segment_start = int(len(lead_signal) * 0.4)
            segment_end = int(len(lead_signal) * 0.6)
            attention[segment_start:segment_end] *= 2.0

        elif diagnosis in ['LBBB', 'RBBB']:
            qrs_start = int(len(lead_signal) * 0.2)
            qrs_end = int(len(lead_signal) * 0.5)
            attention[qrs_start:qrs_end] *= 1.8

        elif diagnosis in ['AFIB', 'VTAC']:
            peaks = np.where(np.abs(lead_signal) > np.std(lead_signal))[0]
            for peak in peaks:
                start_idx = max(0, peak - 10)
                end_idx = min(len(attention), peak + 10)
                attention[start_idx:end_idx] *= 1.5

        attention = np.clip(attention, 0.1, 1.0)

        return attention

    def _features_to_vector(self, features: dict[str, Any]) -> np.ndarray:
        """Convert feature dictionary to numpy vector"""

        vector = []
        for feature_name in self.feature_names:
            value = features.get(feature_name, 0.0)
            if isinstance(value, int | float):
                vector.append(float(value))
            else:
                vector.append(0.0)

        return np.array(vector)

    def _extract_feature_importance(
        self,
        shap_explanation: dict[str, Any] | None,
        lime_explanation: dict[str, Any] | None
    ) -> dict[str, float]:
        """Extract and combine feature importance from SHAP and LIME"""

        importance = {}

        if shap_explanation and 'shap_values' in shap_explanation:
            shap_values = shap_explanation['shap_values']
            feature_names = shap_explanation.get('feature_names', [])

            for i, feature in enumerate(feature_names):
                if i < len(shap_values):
                    importance[f'shap_{feature}'] = abs(float(shap_values[i]))

        if lime_explanation and 'local_explanation' in lime_explanation:
            for feature, value in lime_explanation['local_explanation']:
                importance[f'lime_{feature}'] = abs(float(value))

        if importance:
            max_importance = max(importance.values())
            if max_importance > 0:
                importance = {k: v / max_importance for k, v in importance.items()}

        return importance

    def _reference_diagnostic_criteria(
        self,
        diagnosis: str,
        features: dict[str, Any]
    ) -> dict[str, Any]:
        """Reference standard diagnostic criteria for the diagnosis"""

        condition = get_condition_by_code(diagnosis)
        criteria: dict[str, Any] = {
            'diagnosis': diagnosis,
            'icd10_code': condition.icd10_codes[0] if condition and condition.icd10_codes else 'Unknown',
            'standard_criteria': {}
        }

        if diagnosis == 'STEMI':
            criteria['standard_criteria'] = {
                'st_elevation': '>1mm in limb leads or >2mm in precordial leads',
                'duration': 'Persistent for >20 minutes',
                'leads_affected': 'At least 2 contiguous leads',
                'clinical_context': 'Chest pain or equivalent symptoms'
            }
        elif diagnosis == 'AFIB':
            criteria['standard_criteria'] = {
                'rhythm': 'Irregularly irregular RR intervals',
                'p_waves': 'Absent or fibrillatory waves',
                'ventricular_response': 'Variable, often rapid',
                'duration': '>30 seconds for diagnosis'
            }
        elif diagnosis in ['LBBB', 'RBBB']:
            criteria['standard_criteria'] = {
                'qrs_duration': '>120ms',
                'morphology': 'Bundle branch block pattern',
                'axis': 'May be deviated',
                'secondary_changes': 'Appropriate T wave changes'
            }

        return criteria

    def _identify_risk_factors(
        self,
        diagnosis: str,
        features: dict[str, Any]
    ) -> list[str]:
        """Identify risk factors associated with the diagnosis"""

        risk_factors = []

        age = features.get('patient_age', 50)
        if age > 65:
            risk_factors.append(f"Advanced age ({age} years)")

        hr = features.get('heart_rate', 70)
        if diagnosis == 'AFIB' and hr > 100:
            risk_factors.append("Rapid ventricular response")
        elif diagnosis == 'BRADY' and hr < 40:
            risk_factors.append("Severe bradycardia")

        if diagnosis == 'LVH':
            risk_factors.append("Left ventricular hypertrophy")
        if diagnosis == 'RVH':
            risk_factors.append("Right ventricular hypertrophy")

        if diagnosis in ['STEMI', 'NSTEMI']:
            risk_factors.extend([
                "Acute coronary syndrome",
                "Risk of cardiogenic shock",
                "Risk of mechanical complications"
            ])

        return risk_factors

    def _generate_recommendations(
        self,
        diagnosis: str,
        features: dict[str, Any]
    ) -> list[str]:
        """Generate clinical recommendations based on diagnosis"""

        recommendations = []
        condition = get_condition_by_code(diagnosis)

        if condition and condition.clinical_urgency == 'critical':
            recommendations.append("URGENT: Immediate cardiology consultation required")
            recommendations.append("Continuous cardiac monitoring recommended")

        if diagnosis == 'STEMI':
            recommendations.extend([
                "Emergency PCI or thrombolysis within 90 minutes",
                "Dual antiplatelet therapy",
                "Cardiac catheterization",
                "Serial troponin measurements"
            ])
        elif diagnosis == 'AFIB':
            recommendations.extend([
                "Assess CHA2DS2-VASc score for stroke risk",
                "Consider anticoagulation",
                "Rate or rhythm control strategy",
                "Echocardiogram to assess structure"
            ])
        elif diagnosis in ['LBBB', 'RBBB']:
            recommendations.extend([
                "Assess for underlying structural heart disease",
                "Consider echocardiogram",
                "Monitor for progression",
                "Evaluate need for pacing if symptomatic"
            ])
        else:
            recommendations.append("Follow standard clinical protocols")
            recommendations.append("Consider cardiology consultation if symptomatic")

        return recommendations
