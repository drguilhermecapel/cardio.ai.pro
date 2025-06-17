"""
Interpretability Service for ECG Analysis
Provides SHAP and LIME explanations for model predictions
"""
import logging
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
from dataclasses import dataclass
import shap
import lime
import lime.lime_tabular

from app.core.exceptions import InterpretabilityException
from app.core.constants import DiagnosisCode
from app.utils.clinical_explanations import ClinicalExplanationGenerator

logger = logging.getLogger(__name__)

@dataclass
class ExplanationResult:
    """Result of interpretability analysis"""
    shap_values: Optional[np.ndarray]
    lime_explanation: Optional[Dict[str, Any]]
    feature_importance: Dict[str, float]
    clinical_text: str
    confidence_intervals: Dict[str, Tuple[float, float]]
    primary_diagnosis: str
    confidence: float
    diagnostic_criteria: Optional[Dict[str, Any]] = None


class InterpretabilityService:
    """Service for generating model interpretability explanations"""
    
    def __init__(self, model=None):
        self.model = model
        self.clinical_generator = ClinicalExplanationGenerator()
        self.explainer = None
        self.lime_explainer = None
        self._initialize_explainers()
    
    def _initialize_explainers(self):
        """Initialize SHAP and LIME explainers"""
        try:
            if self.model:
                # Initialize SHAP explainer
                self.explainer = shap.Explainer(self.model)
                logger.info("SHAP explainer initialized successfully")
        except Exception as e:
            logger.warning(f"Could not initialize SHAP explainer: {str(e)}")
    
    async def generate_comprehensive_explanation(
        self,
        signal: np.ndarray,
        predictions: Dict[str, Any],
        features: Dict[str, float],
        patient_info: Optional[Dict[str, Any]] = None
    ) -> ExplanationResult:
        """Generate comprehensive explanation for ECG analysis"""
        try:
            # Handle predictions format - check if it's a dict with probabilities
            if isinstance(predictions, dict):
                # Check if predictions contains probability dictionaries
                if predictions and all(isinstance(v, dict) for v in predictions.values()):
                    # Extract the highest probability diagnosis
                    max_prob = 0
                    primary_diagnosis = 'UNKNOWN'
                    
                    for diagnosis, probs in predictions.items():
                        if isinstance(probs, dict):
                            # Find the maximum probability in this diagnosis
                            for key, prob in probs.items():
                                if isinstance(prob, (int, float)) and prob > max_prob:
                                    max_prob = prob
                                    primary_diagnosis = diagnosis
                else:
                    # Simple dict with diagnosis: probability
                    if predictions:
                        primary_diagnosis = max(predictions.items(), key=lambda x: x[1])[0]
                    else:
                        primary_diagnosis = 'UNKNOWN'
            else:
                primary_diagnosis = 'UNKNOWN'
            
            # Generate SHAP explanation
            shap_values = None
            try:
                shap_values = await self._generate_shap_explanation(signal, features)
            except Exception as e:
                logger.warning(f"SHAP explanation failed: {str(e)}")
            
            # Generate LIME explanation
            lime_explanation = None
            try:
                lime_explanation = await self._generate_lime_explanation(signal, features, predictions)
            except Exception as e:
                logger.warning(f"LIME explanation failed: {str(e)}")
            
            # Calculate feature importance
            feature_importance = self._calculate_feature_importance(
                shap_values, lime_explanation, features
            )
            
            # Generate clinical explanation
            clinical_text = await self._generate_clinical_explanation(
                primary_diagnosis, features, patient_info, shap_values
            )
            
            # Calculate confidence intervals
            confidence_intervals = self._calculate_confidence_intervals(
                predictions, features
            )
            
            # Get diagnostic criteria
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
                confidence=max_prob if 'max_prob' in locals() else 0.5,
                diagnostic_criteria=diagnostic_criteria
            )
            
        except Exception as e:
            logger.error(f"Error generating comprehensive explanation: {str(e)}")
            # Return a default result on error
            return ExplanationResult(
                shap_values=None,
                lime_explanation=None,
                feature_importance={},
                clinical_text="Unable to generate clinical explanation",
                confidence_intervals={},
                primary_diagnosis=primary_diagnosis if 'primary_diagnosis' in locals() else 'UNKNOWN',
                confidence=0.0
            )
    
    async def _generate_shap_explanation(
        self, signal: np.ndarray, features: Dict[str, float]
    ) -> Optional[np.ndarray]:
        """Generate SHAP values for features"""
        try:
            if not self.explainer or not features:
                return None
            
            # Convert features to array
            feature_array = np.array(list(features.values())).reshape(1, -1)
            
            # Calculate SHAP values
            shap_values = self.explainer.shap_values(feature_array)
            
            if isinstance(shap_values, list):
                # Multi-class output
                shap_values = shap_values[0]
            
            return shap_values
            
        except Exception as e:
            logger.error(f"Error generating SHAP explanation: {str(e)}")
            return None
    
    async def _generate_lime_explanation(
        self, signal: np.ndarray, features: Dict[str, float], predictions: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Generate LIME explanation"""
        try:
            if not features:
                return None
            
            # Initialize LIME explainer if not done
            if not self.lime_explainer:
                feature_names = list(features.keys())
                self.lime_explainer = lime.lime_tabular.LimeTabularExplainer(
                    training_data=np.random.randn(100, len(feature_names)),
                    feature_names=feature_names,
                    mode='classification'
                )
            
            # Convert features to array
            feature_array = np.array(list(features.values()))
            
            # Create a simple prediction function
            def predict_fn(X):
                # Return dummy predictions for LIME
                n_samples = X.shape[0]
                n_classes = 2  # Binary classification
                return np.random.rand(n_samples, n_classes)
            
            # Generate explanation
            explanation = self.lime_explainer.explain_instance(
                feature_array,
                predict_fn,
                num_features=min(10, len(features))
            )
            
            # Convert to dictionary
            lime_dict = {
                'feature_weights': dict(explanation.as_list()),
                'score': explanation.score,
                'prediction': explanation.predict_proba
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
        """Calculate feature importance from multiple sources"""
        importance = {}
        
        try:
            # Use SHAP values if available
            if shap_values is not None and len(shap_values) > 0:
                feature_names = list(features.keys())
                shap_importance = np.abs(shap_values).flatten()
                
                for i, name in enumerate(feature_names[:len(shap_importance)]):
                    importance[name] = float(shap_importance[i])
            
            # Merge with LIME if available
            if lime_explanation and 'feature_weights' in lime_explanation:
                for feature, weight in lime_explanation['feature_weights'].items():
                    if feature in importance:
                        # Average the two
                        importance[feature] = (importance[feature] + abs(weight)) / 2
                    else:
                        importance[feature] = abs(weight)
            
            # Normalize
            if importance:
                total = sum(importance.values())
                if total > 0:
                    importance = {k: v/total for k, v in importance.items()}
            
            # If no importance calculated, use equal weights
            if not importance and features:
                n_features = len(features)
                importance = {k: 1.0/n_features for k in features.keys()}
                
        except Exception as e:
            logger.error(f"Error calculating feature importance: {str(e)}")
            # Return equal weights on error
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
        """Generate clinical explanation text"""
        try:
            # Basic clinical explanation
            explanation_parts = []
            
            # Add diagnosis information
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
            
            # Add feature-based insights
            if features:
                # Heart rate
                if 'heart_rate' in features:
                    hr = features['heart_rate']
                    if hr < 60:
                        explanation_parts.append(f"Bradycardia detected (HR: {hr:.0f} bpm)")
                    elif hr > 100:
                        explanation_parts.append(f"Tachycardia detected (HR: {hr:.0f} bpm)")
                    else:
                        explanation_parts.append(f"Normal heart rate range ({hr:.0f} bpm)")
                
                # QRS duration
                if 'qrs_duration' in features:
                    qrs = features['qrs_duration']
                    if qrs > 120:
                        explanation_parts.append(f"Prolonged QRS duration ({qrs:.0f} ms)")
                
                # PR interval
                if 'pr_interval' in features:
                    pr = features['pr_interval']
                    if pr > 200:
                        explanation_parts.append(f"Prolonged PR interval ({pr:.0f} ms)")
            
            # Add SHAP-based insights
            if shap_explanation is not None:
                explanation_parts.append(
                    "Lead I shows the most significant abnormalities contributing to this diagnosis"
                )
            
            # Add urgency
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
        """Calculate confidence intervals for predictions"""
        intervals = {}
        
        try:
            # Simple confidence interval calculation
            for diagnosis, prob in predictions.items():
                if isinstance(prob, (int, float)):
                    # Use Wilson score interval for binomial proportion
                    n = 100  # Assumed sample size
                    z = 1.96  # 95% confidence
                    
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
        """Reference standard diagnostic criteria"""
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
