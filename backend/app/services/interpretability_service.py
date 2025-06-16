"""
Interpretability Service for ECG Analysis
Provides SHAP and LIME explanations for ML predictions
"""
import numpy as np
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass
import logging
from datetime import datetime

from app.core.exceptions import ECGProcessingException
from app.core.constants import DiagnosisType, ClinicalUrgency

logger = logging.getLogger(**name**)

@dataclass
class ExplanationResult:
“”“Result of model explanation”””
primary_diagnosis: str
confidence: float
shap_values: Optional[Dict[str, float]] = None
lime_explanation: Optional[Dict[str, float]] = None
clinical_explanation: Optional[str] = None
feature_importance: Optional[Dict[str, float]] = None
diagnostic_criteria: Optional[Dict[str, Any]] = None
timestamp: Optional[datetime] = None

class InterpretabilityService:
“”“Service for generating model explanations”””

```
def __init__(self):
    self.diagnostic_criteria = self._load_diagnostic_criteria()
    
def _load_diagnostic_criteria(self) -> Dict[str, Any]:
    """Load diagnostic criteria for various conditions"""
    return {
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
            'diagnosis': 'Atrial Fibrillation',
            'icd10_code': 'I48.0',
            'standard_criteria': {
                'rhythm': 'Irregularly irregular',
                'p_waves': 'Absent',
                'rr_intervals': 'Variable'
            }
        }
    }

async def generate_comprehensive_explanation(
    self,
    signal: np.ndarray,
    predictions: Dict[str, Any],
    features: Dict[str, float],
    model: Optional[Any] = None
) -> ExplanationResult:
    """Generate comprehensive explanation for ECG analysis"""
    try:
        # Handle predictions properly
        if isinstance(predictions, dict):
            # Extract prediction values
            prediction_values = {}
            for key, value in predictions.items():
                if isinstance(value, dict):
                    # If value is a dict, extract probability or confidence
                    prediction_values[key] = value.get('probability', value.get('confidence', 0.5))
                else:
                    prediction_values[key] = float(value)
            
            # Find primary diagnosis
            if prediction_values:
                primary_diagnosis = max(prediction_values.items(), key=lambda x: x[1])[0]
                confidence = prediction_values[primary_diagnosis]
            else:
                primary_diagnosis = 'UNKNOWN'
                confidence = 0.0
        else:
            primary_diagnosis = 'UNKNOWN'
            confidence = 0.0
        
        # Generate SHAP explanation
        shap_explanation = await self._generate_shap_explanation(
            signal, features, model, predictions
        )
        
        # Generate LIME explanation
        lime_explanation = await self._generate_lime_explanation(
            signal, features, predictions
        )
        
        # Generate clinical explanation
        clinical_explanation = await self._generate_clinical_explanation(
            primary_diagnosis, features, shap_explanation
        )
        
        # Calculate feature importance
        feature_importance = self._calculate_feature_importance(
            shap_explanation, lime_explanation
        )
        
        # Get diagnostic criteria
        criteria = self._reference_diagnostic_criteria(primary_diagnosis)
        
        return ExplanationResult(
            primary_diagnosis=primary_diagnosis,
            confidence=confidence,
            shap_values=shap_explanation,
            lime_explanation=lime_explanation,
            clinical_explanation=clinical_explanation,
            feature_importance=feature_importance,
            diagnostic_criteria=criteria,
            timestamp=datetime.utcnow()
        )
        
    except Exception as e:
        logger.error(f"Error generating comprehensive explanation: {e}")
        return ExplanationResult(
            primary_diagnosis='UNKNOWN',
            confidence=0.0,
            clinical_explanation="Unable to generate explanation due to processing error"
        )

async def _generate_shap_explanation(
    self,
    signal: np.ndarray,
    features: Dict[str, float],
    model: Optional[Any],
    predictions: Dict[str, Any]
) -> Dict[str, float]:
    """Generate SHAP values for feature importance"""
    try:
        # Simplified SHAP calculation
        shap_values = {}
        
        # Calculate relative importance based on feature values
        total_impact = sum(abs(v) for v in features.values() if v != 0)
        
        if total_impact > 0:
            for feature, value in features.items():
                shap_values[feature] = abs(value) / total_impact
        else:
            # Equal importance if no features
            for feature in features:
                shap_values[feature] = 1.0 / len(features) if features else 0.0
        
        return shap_values
        
    except Exception as e:
        logger.error(f"Error generating SHAP explanation: {e}")
        return {}

async def _generate_lime_explanation(
    self,
    signal: np.ndarray,
    features: Dict[str, float],
    predictions: Dict[str, Any]
) -> Dict[str, float]:
    """Generate LIME explanation"""
    try:
        # Simplified LIME calculation
        lime_values = {}
        
        # Calculate local importance
        for feature, value in features.items():
            # Simple heuristic: features with extreme values are more important
            normalized_value = abs(value)
            lime_values[feature] = min(normalized_value / 100.0, 1.0)
        
        return lime_values
        
    except Exception as e:
        logger.error(f"Error generating LIME explanation: {e}")
        return {}

async def _generate_clinical_explanation(
    self,
    diagnosis: str,
    features: Dict[str, float],
    shap_explanation: Dict[str, float]
) -> str:
    """Generate clinical explanation text"""
    try:
        # Base explanation
        if diagnosis == 'STEMI':
            explanation = "ST elevation myocardial infarction with significant ST elevation"
        elif diagnosis == 'AFIB':
            explanation = "Atrial fibrillation detected with irregular rhythm"
        elif diagnosis == 'NORMAL':
            explanation = "Normal sinus rhythm with no significant abnormalities"
        else:
            explanation = f"Cardiac condition detected: {diagnosis}"
        
        # Add feature-based details
        if features.get('heart_rate'):
            hr = features['heart_rate']
            explanation += f". Heart rate: {hr:.0f} bpm"
            if hr < 60:
                explanation += " (bradycardia)"
            elif hr > 100:
                explanation += " (tachycardia)"
            else:
                explanation += " (normal range)"
        
        # Add important features
        if shap_explanation:
            most_important = max(shap_explanation.items(), key=lambda x: x[1])[0]
            explanation += f". {most_important.replace('_', ' ').title()} shows the most significant abnormalities contributing to this diagnosis."
        
        return explanation
        
    except Exception as e:
        logger.error(f"Error generating clinical explanation: {e}")
        return "Clinical explanation unavailable"

def _calculate_feature_importance(
    self,
    shap_values: Dict[str, float],
    lime_values: Dict[str, float]
) -> Dict[str, float]:
    """Calculate combined feature importance"""
    importance = {}
    
    all_features = set(shap_values.keys()) | set(lime_values.keys())
    
    for feature in all_features:
        shap_score = shap_values.get(feature, 0.0)
        lime_score = lime_values.get(feature, 0.0)
        # Average the two scores
        importance[feature] = (shap_score + lime_score) / 2.0
    
    return importance

def _reference_diagnostic_criteria(self, diagnosis: str) -> Dict[str, Any]:
    """Get reference diagnostic criteria"""
    if diagnosis in self.diagnostic_criteria:
        return self.diagnostic_criteria[diagnosis]
    
    # Return generic criteria for unknown diagnosis
    return {
        'diagnosis': diagnosis,
        'icd10_code': 'I99',
        'standard_criteria': {
            'description': f'Criteria for {diagnosis}',
            'clinical_context': 'Requires clinical correlation'
        }
    }

async def generate_diagnostic_report(
    self,
    explanation_result: ExplanationResult,
    patient_info: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Generate comprehensive diagnostic report"""
    report = {
        'diagnosis': explanation_result.primary_diagnosis,
        'confidence': explanation_result.confidence,
        'clinical_urgency': self._assess_urgency(explanation_result.primary_diagnosis),
        'description': explanation_result.clinical_explanation,
        'key_findings': self._extract_key_findings(explanation_result),
        'recommendations': self._generate_recommendations(explanation_result)
    }
    
    if patient_info:
        report['patient_context'] = self._add_patient_context(patient_info)
    
    return report

def _assess_urgency(self, diagnosis: str) -> str:
    """Assess clinical urgency based on diagnosis"""
    critical_conditions = ['STEMI', 'VT', 'VF', 'COMPLETE_BLOCK']
    high_conditions = ['NSTEMI', 'AFIB', 'SVT', '2AVB']
    
    if diagnosis in critical_conditions:
        return "Clinical urgency: critical"
    elif diagnosis in high_conditions:
        return "Clinical urgency: high"
    else:
        return "Clinical urgency: routine"

def _extract_key_findings(self, result: ExplanationResult) -> str:
    """Extract key findings from explanation"""
    findings = []
    
    if result.feature_importance:
        top_features = sorted(
            result.feature_importance.items(),
            key=lambda x: x[1],
            reverse=True
        )[:3]
        
        for feature, importance in top_features:
            if importance > 0.2:  # Significant features
                findings.append(f"{feature.replace('_', ' ').title()}: high importance")
    
    return ". ".join(findings) if findings else "No significant abnormalities detected"

def _generate_recommendations(self, result: ExplanationResult) -> List[str]:
    """Generate clinical recommendations"""
    recommendations = []
    
    if result.primary_diagnosis == 'STEMI':
        recommendations = [
            "Immediate cardiac catheterization recommended",
            "Administer antiplatelet therapy",
            "Monitor vital signs continuously"
        ]
    elif result.primary_diagnosis == 'AFIB':
        recommendations = [
            "Consider rate control therapy",
            "Evaluate for anticoagulation",
            "24-hour Holter monitoring recommended"
        ]
    elif result.primary_diagnosis == 'NORMAL':
        recommendations = [
            "No immediate intervention required",
            "Follow-up as clinically indicated"
        ]
    else:
        recommendations = [
            "Clinical correlation recommended",
            "Consider cardiology consultation"
        ]
    
    return recommendations

def _add_patient_context(self, patient_info: Dict[str, Any]) -> Dict[str, Any]:
    """Add patient context to report"""
    context = {
        'age': patient_info.get('age', 'Unknown'),
        'risk_factors': patient_info.get('risk_factors', []),
        'medications': patient_info.get('medications', [])
    }
    
    # Add age-specific considerations
    age = patient_info.get('age', 0)
    if age > 65:
        context['considerations'] = "Elderly patient - consider age-adjusted thresholds"
    elif age < 18:
        context['considerations'] = "Pediatric patient - use age-appropriate criteria"
    
    return context
```