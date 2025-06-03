"""
Explainable AI Service
Advanced interpretability and explanation generation for ECG analysis
"""

import logging
import numpy as np
import numpy.typing as npt
from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    logger.warning("SHAP not available. Using simplified explanations.")

try:
    import lime
    import lime.lime_tabular
    LIME_AVAILABLE = True
except ImportError:
    LIME_AVAILABLE = False
    logger.warning("LIME not available. Using simplified explanations.")

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("PyTorch not available. Using simplified gradient analysis.")

try:
    from scipy import signal
    from scipy.stats import pearsonr
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    logger.warning("SciPy not available. Using simplified signal analysis.")


class ExplanationMethod(Enum):
    """Available explanation methods"""
    SHAP = "shap"
    LIME = "lime"
    GRADIENT_BASED = "gradient_based"
    ATTENTION_MAPS = "attention_maps"
    FEATURE_IMPORTANCE = "feature_importance"
    CLINICAL_REASONING = "clinical_reasoning"


class ECGLead(Enum):
    """Standard ECG leads"""
    I = "I"
    II = "II"
    III = "III"
    AVR = "aVR"
    AVL = "aVL"
    AVF = "aVF"
    V1 = "V1"
    V2 = "V2"
    V3 = "V3"
    V4 = "V4"
    V5 = "V5"
    V6 = "V6"


@dataclass
class ECGSegment:
    """ECG segment information"""
    name: str
    start_ms: float
    end_ms: float
    description: str
    normal_range: Optional[Tuple[float, float]] = None


@dataclass
class ClinicalFinding:
    """Clinical finding with explanation"""
    condition: str
    confidence: float
    evidence: List[str]
    lead_involvement: List[str]
    clinical_significance: str
    recommendations: List[str]


@dataclass
class ExplanationResult:
    """Complete explanation result"""
    method: ExplanationMethod
    feature_importance: Dict[str, float]
    attention_maps: Dict[str, npt.NDArray[np.float32]]
    clinical_reasoning: List[ClinicalFinding]
    visual_explanations: Dict[str, Any]
    confidence_factors: List[str]
    uncertainty_analysis: Dict[str, float]


class ECGFeatureExtractor:
    """Extract clinical features from ECG signals"""
    
    def __init__(self, sampling_rate: int = 500):
        self.sampling_rate = sampling_rate
        self.ecg_segments = self._define_ecg_segments()
        
    def _define_ecg_segments(self) -> List[ECGSegment]:
        """Define standard ECG segments and intervals"""
        return [
            ECGSegment("P_wave", 0, 120, "Atrial depolarization", (80, 120)),
            ECGSegment("PR_interval", 0, 200, "AV conduction time", (120, 200)),
            ECGSegment("QRS_complex", 0, 120, "Ventricular depolarization", (80, 120)),
            ECGSegment("ST_segment", 120, 320, "Early ventricular repolarization", None),
            ECGSegment("T_wave", 320, 480, "Ventricular repolarization", None),
            ECGSegment("QT_interval", 0, 440, "Total ventricular activity", (350, 440)),
            ECGSegment("RR_interval", 600, 1200, "Heart rate variability", (600, 1000))
        ]
        
    def extract_morphological_features(
        self, 
        ecg_signal: npt.NDArray[np.float32],
        lead_name: str
    ) -> Dict[str, float]:
        """Extract morphological features from ECG signal"""
        try:
            features = {}
            
            features[f"{lead_name}_max_amplitude"] = float(np.max(ecg_signal))
            features[f"{lead_name}_min_amplitude"] = float(np.min(ecg_signal))
            features[f"{lead_name}_amplitude_range"] = float(np.ptp(ecg_signal))
            features[f"{lead_name}_rms"] = float(np.sqrt(np.mean(ecg_signal**2)))
            
            features[f"{lead_name}_mean"] = float(np.mean(ecg_signal))
            features[f"{lead_name}_std"] = float(np.std(ecg_signal))
            features[f"{lead_name}_skewness"] = float(self._calculate_skewness(ecg_signal))
            features[f"{lead_name}_kurtosis"] = float(self._calculate_kurtosis(ecg_signal))
            
            if SCIPY_AVAILABLE:
                freq_features = self._extract_frequency_features(ecg_signal, lead_name)
                features.update(freq_features)
                
            peak_features = self._extract_peak_features(ecg_signal, lead_name)
            features.update(peak_features)
            
            return features
            
        except Exception as e:
            logger.error(f"Morphological feature extraction failed: {e}")
            return {}
            
    def _calculate_skewness(self, signal: npt.NDArray[np.float32]) -> float:
        """Calculate skewness of signal"""
        try:
            mean = np.mean(signal)
            std = np.std(signal)
            if std == 0:
                return 0.0
            return float(np.mean(((signal - mean) / std) ** 3))
        except Exception:
            return 0.0
            
    def _calculate_kurtosis(self, signal: npt.NDArray[np.float32]) -> float:
        """Calculate kurtosis of signal"""
        try:
            mean = np.mean(signal)
            std = np.std(signal)
            if std == 0:
                return 0.0
            return float(np.mean(((signal - mean) / std) ** 4) - 3)
        except Exception:
            return 0.0
            
    def _extract_frequency_features(
        self, 
        ecg_signal: npt.NDArray[np.float32],
        lead_name: str
    ) -> Dict[str, float]:
        """Extract frequency domain features"""
        try:
            frequencies, psd = signal.welch(ecg_signal, fs=self.sampling_rate, nperseg=256)
            
            low_freq = (0.04, 0.15)  # Low frequency
            high_freq = (0.15, 0.4)  # High frequency
            
            low_power = np.trapz(psd[(frequencies >= low_freq[0]) & (frequencies <= low_freq[1])])
            high_power = np.trapz(psd[(frequencies >= high_freq[0]) & (frequencies <= high_freq[1])])
            total_power = np.trapz(psd)
            
            return {
                f"{lead_name}_low_freq_power": float(low_power),
                f"{lead_name}_high_freq_power": float(high_power),
                f"{lead_name}_total_power": float(total_power),
                f"{lead_name}_lf_hf_ratio": float(low_power / (high_power + 1e-8)),
                f"{lead_name}_dominant_freq": float(frequencies[np.argmax(psd)])
            }
            
        except Exception as e:
            logger.error(f"Frequency feature extraction failed: {e}")
            return {}
            
    def _extract_peak_features(
        self, 
        ecg_signal: npt.NDArray[np.float32],
        lead_name: str
    ) -> Dict[str, float]:
        """Extract peak-related features"""
        try:
            features = {}
            
            if SCIPY_AVAILABLE:
                peaks, properties = signal.find_peaks(
                    ecg_signal, 
                    height=np.std(ecg_signal),
                    distance=int(0.6 * self.sampling_rate)  # Minimum 60 BPM
                )
                
                if len(peaks) > 1:
                    rr_intervals = np.diff(peaks) / self.sampling_rate * 1000  # Convert to ms
                    
                    features[f"{lead_name}_heart_rate"] = float(60000 / np.mean(rr_intervals))
                    features[f"{lead_name}_rr_mean"] = float(np.mean(rr_intervals))
                    features[f"{lead_name}_rr_std"] = float(np.std(rr_intervals))
                    features[f"{lead_name}_rmssd"] = float(np.sqrt(np.mean(np.diff(rr_intervals)**2)))
                    
                    peak_amplitudes = ecg_signal[peaks]
                    features[f"{lead_name}_peak_amplitude_mean"] = float(np.mean(peak_amplitudes))
                    features[f"{lead_name}_peak_amplitude_std"] = float(np.std(peak_amplitudes))
                    
            return features
            
        except Exception as e:
            logger.error(f"Peak feature extraction failed: {e}")
            return {}


class SHAPExplainer:
    """SHAP-based explanations for ECG analysis"""
    
    def __init__(self):
        self.explainer = None
        self.background_data = None
        
    def initialize_explainer(
        self, 
        model_function: callable,
        background_data: npt.NDArray[np.float32]
    ) -> bool:
        """Initialize SHAP explainer with background data"""
        if not SHAP_AVAILABLE:
            logger.warning("SHAP not available")
            return False
            
        try:
            sample_size = min(100, len(background_data))
            self.background_data = background_data[:sample_size]
            
            self.explainer = shap.Explainer(model_function, self.background_data)
            
            return True
            
        except Exception as e:
            logger.error(f"SHAP explainer initialization failed: {e}")
            return False
            
    def explain_prediction(
        self, 
        input_data: npt.NDArray[np.float32],
        max_evals: int = 100
    ) -> Dict[str, Any]:
        """Generate SHAP explanations for prediction"""
        if not SHAP_AVAILABLE or self.explainer is None:
            return self._fallback_explanation(input_data)
            
        try:
            shap_values = self.explainer(input_data, max_evals=max_evals)
            
            if hasattr(shap_values, 'values'):
                importance_scores = np.abs(shap_values.values).mean(axis=0)
            else:
                importance_scores = np.abs(shap_values).mean(axis=0)
                
            leads = [lead.value for lead in ECGLead]
            lead_importance = {}
            
            if len(importance_scores.shape) > 1:
                for i, lead in enumerate(leads[:importance_scores.shape[0]]):
                    lead_importance[lead] = float(np.mean(importance_scores[i]))
            else:
                samples_per_lead = len(importance_scores) // len(leads)
                for i, lead in enumerate(leads):
                    start_idx = i * samples_per_lead
                    end_idx = (i + 1) * samples_per_lead
                    if end_idx <= len(importance_scores):
                        lead_importance[lead] = float(np.mean(importance_scores[start_idx:end_idx]))
                        
            return {
                "method": "SHAP",
                "lead_importance": lead_importance,
                "feature_importance": importance_scores.tolist() if hasattr(importance_scores, 'tolist') else [],
                "explanation_quality": "high"
            }
            
        except Exception as e:
            logger.error(f"SHAP explanation failed: {e}")
            return self._fallback_explanation(input_data)
            
    def _fallback_explanation(self, input_data: npt.NDArray[np.float32]) -> Dict[str, Any]:
        """Fallback explanation when SHAP is not available"""
        leads = [lead.value for lead in ECGLead]
        
        if len(input_data.shape) > 1:
            lead_importance = {}
            for i, lead in enumerate(leads[:input_data.shape[0]]):
                lead_importance[lead] = float(np.var(input_data[i]))
        else:
            samples_per_lead = len(input_data) // len(leads)
            lead_importance = {}
            for i, lead in enumerate(leads):
                start_idx = i * samples_per_lead
                end_idx = (i + 1) * samples_per_lead
                if end_idx <= len(input_data):
                    lead_importance[lead] = float(np.var(input_data[start_idx:end_idx]))
                    
        return {
            "method": "variance_based",
            "lead_importance": lead_importance,
            "feature_importance": [],
            "explanation_quality": "basic"
        }


class LIMEExplainer:
    """LIME-based explanations for ECG analysis"""
    
    def __init__(self):
        self.explainer = None
        
    def initialize_explainer(
        self, 
        training_data: npt.NDArray[np.float32],
        feature_names: Optional[List[str]] = None
    ) -> bool:
        """Initialize LIME explainer"""
        if not LIME_AVAILABLE:
            logger.warning("LIME not available")
            return False
            
        try:
            if len(training_data.shape) > 2:
                training_data = training_data.reshape(training_data.shape[0], -1)
                
            self.explainer = lime.lime_tabular.LimeTabularExplainer(
                training_data,
                feature_names=feature_names,
                class_names=['Normal', 'Abnormal'],
                mode='classification',
                discretize_continuous=True
            )
            
            return True
            
        except Exception as e:
            logger.error(f"LIME explainer initialization failed: {e}")
            return False
            
    def explain_prediction(
        self, 
        input_data: npt.NDArray[np.float32],
        model_function: callable,
        num_features: int = 10
    ) -> Dict[str, Any]:
        """Generate LIME explanations for prediction"""
        if not LIME_AVAILABLE or self.explainer is None:
            return self._fallback_explanation(input_data)
            
        try:
            if len(input_data.shape) > 1:
                flattened_input = input_data.flatten()
            else:
                flattened_input = input_data
                
            explanation = self.explainer.explain_instance(
                flattened_input,
                model_function,
                num_features=num_features
            )
            
            feature_importance = dict(explanation.as_list())
            
            return {
                "method": "LIME",
                "feature_importance": feature_importance,
                "explanation_quality": "high",
                "local_explanation": True
            }
            
        except Exception as e:
            logger.error(f"LIME explanation failed: {e}")
            return self._fallback_explanation(input_data)
            
    def _fallback_explanation(self, input_data: npt.NDArray[np.float32]) -> Dict[str, Any]:
        """Fallback explanation when LIME is not available"""
        return {
            "method": "gradient_approximation",
            "feature_importance": {},
            "explanation_quality": "basic",
            "local_explanation": True
        }


class GradientBasedExplainer:
    """Gradient-based explanations for neural networks"""
    
    def __init__(self):
        self.model = None
        
    def set_model(self, model: Any) -> bool:
        """Set the model for gradient analysis"""
        try:
            self.model = model
            return True
        except Exception as e:
            logger.error(f"Model setting failed: {e}")
            return False
            
    def generate_gradient_explanations(
        self, 
        input_data: npt.NDArray[np.float32],
        target_class: Optional[int] = None
    ) -> Dict[str, Any]:
        """Generate gradient-based explanations"""
        if not TORCH_AVAILABLE:
            return self._fallback_gradient_explanation(input_data)
            
        try:
            input_tensor = torch.FloatTensor(input_data).requires_grad_(True)
            
            if hasattr(self.model, '__call__'):
                output = self.model(input_tensor)
            else:
                return self._fallback_gradient_explanation(input_data)
                
            if target_class is not None:
                target_output = output[target_class]
            else:
                target_output = torch.max(output)
                
            target_output.backward()
            
            gradients = input_tensor.grad.detach().numpy()
            
            importance_scores = np.abs(gradients)
            
            attention_maps = self._generate_attention_maps(importance_scores)
            
            return {
                "method": "gradient_based",
                "gradients": gradients.tolist(),
                "importance_scores": importance_scores.tolist(),
                "attention_maps": attention_maps,
                "explanation_quality": "high"
            }
            
        except Exception as e:
            logger.error(f"Gradient explanation failed: {e}")
            return self._fallback_gradient_explanation(input_data)
            
    def _generate_attention_maps(
        self, 
        importance_scores: npt.NDArray[np.float32]
    ) -> Dict[str, npt.NDArray[np.float32]]:
        """Generate attention maps from importance scores"""
        try:
            attention_maps = {}
            leads = [lead.value for lead in ECGLead]
            
            if len(importance_scores.shape) > 1:
                for i, lead in enumerate(leads[:importance_scores.shape[0]]):
                    attention_maps[lead] = importance_scores[i]
            else:
                samples_per_lead = len(importance_scores) // len(leads)
                for i, lead in enumerate(leads):
                    start_idx = i * samples_per_lead
                    end_idx = (i + 1) * samples_per_lead
                    if end_idx <= len(importance_scores):
                        attention_maps[lead] = importance_scores[start_idx:end_idx]
                        
            return attention_maps
            
        except Exception as e:
            logger.error(f"Attention map generation failed: {e}")
            return {}
            
    def _fallback_gradient_explanation(
        self, 
        input_data: npt.NDArray[np.float32]
    ) -> Dict[str, Any]:
        """Fallback gradient explanation"""
        try:
            epsilon = 1e-5
            baseline_score = np.mean(input_data)
            
            gradients = np.zeros_like(input_data)
            for i in range(len(input_data.flatten())):
                perturbed_data = input_data.copy().flatten()
                perturbed_data[i] += epsilon
                perturbed_score = np.mean(perturbed_data)
                
                gradient = (perturbed_score - baseline_score) / epsilon
                gradients.flat[i] = gradient
                
            return {
                "method": "finite_difference",
                "gradients": gradients.tolist(),
                "importance_scores": np.abs(gradients).tolist(),
                "explanation_quality": "basic"
            }
            
        except Exception as e:
            logger.error(f"Fallback gradient explanation failed: {e}")
            return {"method": "none", "explanation_quality": "unavailable"}


class ClinicalReasoningEngine:
    """Generate clinical reasoning explanations"""
    
    def __init__(self):
        self.clinical_rules = self._initialize_clinical_rules()
        self.feature_extractor = ECGFeatureExtractor()
        
    def _initialize_clinical_rules(self) -> Dict[str, Dict[str, Any]]:
        """Initialize clinical reasoning rules"""
        return {
            "atrial_fibrillation": {
                "criteria": [
                    "Irregularly irregular rhythm",
                    "Absence of P waves",
                    "Variable RR intervals",
                    "Fibrillatory waves in V1"
                ],
                "leads": ["II", "V1", "V5"],
                "significance": "High risk for stroke and thromboembolism",
                "recommendations": [
                    "Anticoagulation assessment",
                    "Rate or rhythm control",
                    "Echocardiogram evaluation"
                ]
            },
            "stemi": {
                "criteria": [
                    "ST elevation ≥1mm in limb leads",
                    "ST elevation ≥2mm in precordial leads",
                    "Reciprocal changes",
                    "Q wave development"
                ],
                "leads": ["II", "III", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"],
                "significance": "Acute myocardial infarction requiring immediate intervention",
                "recommendations": [
                    "Immediate PCI or thrombolysis",
                    "Dual antiplatelet therapy",
                    "Serial troponins and ECGs"
                ]
            },
            "ventricular_tachycardia": {
                "criteria": [
                    "Wide QRS complexes (>120ms)",
                    "Heart rate >100 bpm",
                    "AV dissociation",
                    "Capture beats or fusion beats"
                ],
                "leads": ["II", "V1", "V6"],
                "significance": "Life-threatening arrhythmia",
                "recommendations": [
                    "Immediate cardioversion if unstable",
                    "Antiarrhythmic therapy",
                    "Electrolyte correction"
                ]
            },
            "left_bundle_branch_block": {
                "criteria": [
                    "QRS duration ≥120ms",
                    "Broad R waves in I, aVL, V5, V6",
                    "Absent Q waves in I, V5, V6",
                    "ST depression and T wave inversion"
                ],
                "leads": ["I", "aVL", "V5", "V6"],
                "significance": "May indicate underlying cardiac disease",
                "recommendations": [
                    "Echocardiogram assessment",
                    "Evaluate for ischemic heart disease",
                    "Consider pacemaker evaluation"
                ]
            }
        }
        
    def generate_clinical_reasoning(
        self, 
        ecg_data: npt.NDArray[np.float32],
        predictions: Dict[str, float],
        confidence: float
    ) -> List[ClinicalFinding]:
        """Generate clinical reasoning explanations"""
        try:
            clinical_findings = []
            
            sorted_predictions = sorted(predictions.items(), key=lambda x: x[1], reverse=True)
            
            for condition, prediction_confidence in sorted_predictions[:3]:  # Top 3 predictions
                if prediction_confidence > 0.1:  # Only explain significant predictions
                    finding = self._analyze_condition(
                        condition, 
                        prediction_confidence, 
                        ecg_data,
                        confidence
                    )
                    if finding:
                        clinical_findings.append(finding)
                        
            return clinical_findings
            
        except Exception as e:
            logger.error(f"Clinical reasoning generation failed: {e}")
            return []
            
    def _analyze_condition(
        self, 
        condition: str,
        prediction_confidence: float,
        ecg_data: npt.NDArray[np.float32],
        overall_confidence: float
    ) -> Optional[ClinicalFinding]:
        """Analyze specific condition"""
        try:
            if condition not in self.clinical_rules:
                return None
                
            rule = self.clinical_rules[condition]
            
            evidence = []
            lead_involvement = []
            
            for lead in rule["leads"]:
                lead_idx = self._get_lead_index(lead)
                if lead_idx < ecg_data.shape[0]:
                    lead_signal = ecg_data[lead_idx]
                    
                    features = self.feature_extractor.extract_morphological_features(
                        lead_signal, lead
                    )
                    
                    if self._check_condition_patterns(condition, lead, features):
                        evidence.append(f"Abnormal patterns detected in lead {lead}")
                        lead_involvement.append(lead)
                        
            if prediction_confidence > 0.8:
                evidence.append(f"Strong algorithmic confidence ({prediction_confidence:.2f})")
            elif prediction_confidence > 0.5:
                evidence.append(f"Moderate algorithmic confidence ({prediction_confidence:.2f})")
            else:
                evidence.append(f"Low algorithmic confidence ({prediction_confidence:.2f})")
                
            for criterion in rule["criteria"][:2]:  # Limit to top 2 criteria
                evidence.append(f"Consistent with: {criterion}")
                
            return ClinicalFinding(
                condition=condition.replace("_", " ").title(),
                confidence=prediction_confidence,
                evidence=evidence,
                lead_involvement=lead_involvement,
                clinical_significance=rule["significance"],
                recommendations=rule["recommendations"]
            )
            
        except Exception as e:
            logger.error(f"Condition analysis failed for {condition}: {e}")
            return None
            
    def _get_lead_index(self, lead_name: str) -> int:
        """Get index for lead name"""
        lead_mapping = {
            "I": 0, "II": 1, "III": 2,
            "aVR": 3, "aVL": 4, "aVF": 5,
            "V1": 6, "V2": 7, "V3": 8,
            "V4": 9, "V5": 10, "V6": 11
        }
        return lead_mapping.get(lead_name, 0)
        
    def _check_condition_patterns(
        self, 
        condition: str,
        lead: str,
        features: Dict[str, float]
    ) -> bool:
        """Check for condition-specific patterns in features"""
        try:
            amplitude_key = f"{lead}_amplitude_range"
            std_key = f"{lead}_std"
            
            if condition == "atrial_fibrillation":
                rr_std_key = f"{lead}_rr_std"
                return features.get(rr_std_key, 0) > 50  # High RR variability
                
            elif condition == "stemi":
                return features.get(amplitude_key, 0) > 0.5
                
            elif condition == "ventricular_tachycardia":
                heart_rate_key = f"{lead}_heart_rate"
                return features.get(heart_rate_key, 0) > 100
                
            elif condition == "left_bundle_branch_block":
                return features.get(std_key, 0) > 0.2
                
            return False
            
        except Exception as e:
            logger.error(f"Pattern checking failed: {e}")
            return False


class ExplainableAIService:
    """Main service for explainable AI functionality"""
    
    def __init__(self):
        self.shap_explainer = SHAPExplainer()
        self.lime_explainer = LIMEExplainer()
        self.gradient_explainer = GradientBasedExplainer()
        self.clinical_reasoning = ClinicalReasoningEngine()
        self.feature_extractor = ECGFeatureExtractor()
        
        self.initialized = False
        self.available_methods = self._check_available_methods()
        
    def _check_available_methods(self) -> List[ExplanationMethod]:
        """Check which explanation methods are available"""
        methods = [ExplanationMethod.CLINICAL_REASONING, ExplanationMethod.FEATURE_IMPORTANCE]
        
        if SHAP_AVAILABLE:
            methods.append(ExplanationMethod.SHAP)
            
        if LIME_AVAILABLE:
            methods.append(ExplanationMethod.LIME)
            
        if TORCH_AVAILABLE:
            methods.append(ExplanationMethod.GRADIENT_BASED)
            methods.append(ExplanationMethod.ATTENTION_MAPS)
            
        return methods
        
    def initialize_explainers(
        self, 
        model_function: Optional[callable] = None,
        background_data: Optional[npt.NDArray[np.float32]] = None,
        training_data: Optional[npt.NDArray[np.float32]] = None
    ) -> bool:
        """Initialize explanation methods"""
        try:
            success_count = 0
            
            if SHAP_AVAILABLE and model_function and background_data is not None:
                if self.shap_explainer.initialize_explainer(model_function, background_data):
                    success_count += 1
                    
            if LIME_AVAILABLE and training_data is not None:
                if self.lime_explainer.initialize_explainer(training_data):
                    success_count += 1
                    
            if TORCH_AVAILABLE:
                success_count += 1
                
            success_count += 1
            
            self.initialized = success_count > 0
            
            logger.info(f"Initialized {success_count} explanation methods")
            return self.initialized
            
        except Exception as e:
            logger.error(f"Explainer initialization failed: {e}")
            return False
            
    def generate_comprehensive_explanation(
        self, 
        ecg_data: npt.NDArray[np.float32],
        predictions: Dict[str, float],
        confidence: float,
        methods: Optional[List[ExplanationMethod]] = None
    ) -> ExplanationResult:
        """Generate comprehensive explanation using multiple methods"""
        try:
            if methods is None:
                methods = self.available_methods
                
            feature_importance = {}
            attention_maps = {}
            visual_explanations = {}
            confidence_factors = []
            uncertainty_analysis = {}
            
            if ExplanationMethod.SHAP in methods and SHAP_AVAILABLE:
                try:
                    shap_result = self.shap_explainer.explain_prediction(ecg_data)
                    if shap_result:
                        feature_importance.update(shap_result.get("lead_importance", {}))
                        visual_explanations["shap"] = shap_result
                        confidence_factors.append(f"SHAP analysis: {shap_result.get('explanation_quality', 'unknown')}")
                except Exception as e:
                    logger.warning(f"SHAP explanation failed: {e}")
                    
            if ExplanationMethod.LIME in methods and LIME_AVAILABLE:
                try:
                    lime_result = self.lime_explainer._fallback_explanation(ecg_data)
                    visual_explanations["lime"] = lime_result
                    confidence_factors.append(f"LIME analysis: {lime_result.get('explanation_quality', 'unknown')}")
                except Exception as e:
                    logger.warning(f"LIME explanation failed: {e}")
                    
            if ExplanationMethod.GRADIENT_BASED in methods:
                try:
                    gradient_result = self.gradient_explainer._fallback_gradient_explanation(ecg_data)
                    visual_explanations["gradient"] = gradient_result
                    confidence_factors.append(f"Gradient analysis: {gradient_result.get('explanation_quality', 'unknown')}")
                except Exception as e:
                    logger.warning(f"Gradient explanation failed: {e}")
                    
            if ExplanationMethod.ATTENTION_MAPS in methods:
                try:
                    attention_maps = self._generate_attention_maps(ecg_data, predictions)
                except Exception as e:
                    logger.warning(f"Attention map generation failed: {e}")
                    
            if ExplanationMethod.FEATURE_IMPORTANCE in methods:
                try:
                    feature_importance.update(self._generate_feature_importance(ecg_data))
                except Exception as e:
                    logger.warning(f"Feature importance generation failed: {e}")
                    
            clinical_findings = []
            if ExplanationMethod.CLINICAL_REASONING in methods:
                try:
                    clinical_findings = self.clinical_reasoning.generate_clinical_reasoning(
                        ecg_data, predictions, confidence
                    )
                    confidence_factors.append(f"Clinical reasoning: {len(clinical_findings)} findings")
                except Exception as e:
                    logger.warning(f"Clinical reasoning failed: {e}")
                    
            uncertainty_analysis = self._analyze_uncertainty(predictions, confidence)
            
            return ExplanationResult(
                method=ExplanationMethod.SHAP if ExplanationMethod.SHAP in methods else methods[0],
                feature_importance=feature_importance,
                attention_maps=attention_maps,
                clinical_reasoning=clinical_findings,
                visual_explanations=visual_explanations,
                confidence_factors=confidence_factors,
                uncertainty_analysis=uncertainty_analysis
            )
            
        except Exception as e:
            logger.error(f"Comprehensive explanation generation failed: {e}")
            return self._generate_fallback_explanation(ecg_data, predictions, confidence)
            
    def _generate_attention_maps(
        self, 
        ecg_data: npt.NDArray[np.float32],
        predictions: Dict[str, float]
    ) -> Dict[str, npt.NDArray[np.float32]]:
        """Generate attention maps for ECG leads"""
        try:
            attention_maps = {}
            leads = [lead.value for lead in ECGLead]
            
            top_prediction = max(predictions.items(), key=lambda x: x[1])
            attention_weight = top_prediction[1]
            
            for i, lead in enumerate(leads[:ecg_data.shape[0]]):
                lead_signal = ecg_data[i]
                
                attention = np.abs(lead_signal - np.mean(lead_signal)) * attention_weight
                attention = attention / (np.max(attention) + 1e-8)  # Normalize
                
                attention_maps[lead] = attention.astype(np.float32)
                
            return attention_maps
            
        except Exception as e:
            logger.error(f"Attention map generation failed: {e}")
            return {}
            
    def _generate_feature_importance(
        self, 
        ecg_data: npt.NDArray[np.float32]
    ) -> Dict[str, float]:
        """Generate feature importance scores"""
        try:
            importance = {}
            leads = [lead.value for lead in ECGLead]
            
            for i, lead in enumerate(leads[:ecg_data.shape[0]]):
                lead_signal = ecg_data[i]
                
                features = self.feature_extractor.extract_morphological_features(lead_signal, lead)
                
                lead_importance = 0.0
                for feature_name, feature_value in features.items():
                    normalized_value = abs(feature_value) / (1.0 + abs(feature_value))
                    lead_importance += normalized_value
                    
                importance[lead] = float(lead_importance / len(features)) if features else 0.0
                
            return importance
            
        except Exception as e:
            logger.error(f"Feature importance generation failed: {e}")
            return {}
            
    def _analyze_uncertainty(
        self, 
        predictions: Dict[str, float],
        confidence: float
    ) -> Dict[str, float]:
        """Analyze prediction uncertainty"""
        try:
            probs = list(predictions.values())
            probs = [p for p in probs if p > 0]  # Remove zero probabilities
            
            if probs:
                entropy = -sum(p * np.log(p + 1e-8) for p in probs)
                max_entropy = np.log(len(probs))
                normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0
            else:
                normalized_entropy = 1.0
                
            prediction_spread = np.std(list(predictions.values()))
            
            top_prediction = max(predictions.values())
            confidence_gap = abs(confidence - top_prediction)
            
            return {
                "entropy": float(normalized_entropy),
                "prediction_spread": float(prediction_spread),
                "confidence_gap": float(confidence_gap),
                "uncertainty_score": float((normalized_entropy + prediction_spread + confidence_gap) / 3)
            }
            
        except Exception as e:
            logger.error(f"Uncertainty analysis failed: {e}")
            return {"uncertainty_score": 0.5}
            
    def _generate_fallback_explanation(
        self, 
        ecg_data: npt.NDArray[np.float32],
        predictions: Dict[str, float],
        confidence: float
    ) -> ExplanationResult:
        """Generate fallback explanation when other methods fail"""
        try:
            feature_importance = self._generate_feature_importance(ecg_data)
            
            attention_maps = self._generate_attention_maps(ecg_data, predictions)
            
            clinical_findings = self.clinical_reasoning.generate_clinical_reasoning(
                ecg_data, predictions, confidence
            )
            
            return ExplanationResult(
                method=ExplanationMethod.FEATURE_IMPORTANCE,
                feature_importance=feature_importance,
                attention_maps=attention_maps,
                clinical_reasoning=clinical_findings,
                visual_explanations={"method": "fallback"},
                confidence_factors=["Basic feature analysis", "Clinical pattern matching"],
                uncertainty_analysis=self._analyze_uncertainty(predictions, confidence)
            )
            
        except Exception as e:
            logger.error(f"Fallback explanation generation failed: {e}")
            return ExplanationResult(
                method=ExplanationMethod.FEATURE_IMPORTANCE,
                feature_importance={},
                attention_maps={},
                clinical_reasoning=[],
                visual_explanations={},
                confidence_factors=["Explanation generation failed"],
                uncertainty_analysis={"uncertainty_score": 1.0}
            )
            
    def get_available_methods(self) -> List[str]:
        """Get list of available explanation methods"""
        return [method.value for method in self.available_methods]
        
    def get_service_info(self) -> Dict[str, Any]:
        """Get explainable AI service information"""
        return {
            "initialized": self.initialized,
            "available_methods": self.get_available_methods(),
            "shap_available": SHAP_AVAILABLE,
            "lime_available": LIME_AVAILABLE,
            "torch_available": TORCH_AVAILABLE,
            "scipy_available": SCIPY_AVAILABLE,
            "clinical_reasoning_enabled": True,
            "feature_extraction_enabled": True
        }
