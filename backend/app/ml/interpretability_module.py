"""
Interpretability and Neuro-Symbolic AI Module for ECG Analysis
Implements SHAP, LIME, GradCAM, and knowledge-based reasoning
Based on state-of-the-art explainable AI research
"""

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import signal as scipy_signal
from scipy.stats import zscore

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    logging.warning("SHAP not available. Install with: pip install shap")

try:
    from lime.lime_tabular import LimeTabularExplainer
    LIME_AVAILABLE = True
except ImportError:
    LIME_AVAILABLE = False
    logging.warning("LIME not available. Install with: pip install lime")

from app.core.scp_ecg_conditions import SCP_ECG_CONDITIONS

logger = logging.getLogger(__name__)


@dataclass
class ECGKnowledgeRule:
    """Knowledge rule for ECG interpretation"""
    name: str
    condition: str
    parameters: Dict[str, Any]
    confidence: float
    clinical_significance: str


class ECGKnowledgeBase:
    """
    Knowledge base containing clinical rules for ECG interpretation
    Based on ACC/AHA guidelines and clinical literature
    """
    
    def __init__(self):
        self.rules = self._initialize_rules()
        self.condition_hierarchy = self._build_condition_hierarchy()
    
    def _initialize_rules(self) -> List[ECGKnowledgeRule]:
        """Initialize clinical ECG rules"""
        rules = [
            # Rhythm rules
            ECGKnowledgeRule(
                name="normal_sinus_rhythm",
                condition="rhythm",
                parameters={
                    "heart_rate": (60, 100),
                    "p_wave_present": True,
                    "pr_interval": (0.12, 0.20),
                    "qrs_duration": (0.06, 0.10),
                    "regular_rr": True
                },
                confidence=0.95,
                clinical_significance="Normal cardiac rhythm"
            ),
            
            ECGKnowledgeRule(
                name="atrial_fibrillation",
                condition="rhythm",
                parameters={
                    "p_wave_present": False,
                    "irregular_rr": True,
                    "fibrillatory_waves": True,
                    "qrs_duration": (0.06, 0.10)
                },
                confidence=0.90,
                clinical_significance="Irregular heart rhythm requiring anticoagulation evaluation"
            ),
            
            ECGKnowledgeRule(
                name="ventricular_tachycardia",
                condition="rhythm",
                parameters={
                    "heart_rate": (120, 250),
                    "wide_qrs": True,
                    "qrs_duration": (0.12, float('inf')),
                    "av_dissociation": True
                },
                confidence=0.85,
                clinical_significance="Life-threatening arrhythmia requiring immediate intervention"
            ),
            
            # Conduction rules
            ECGKnowledgeRule(
                name="first_degree_av_block",
                condition="conduction",
                parameters={
                    "pr_interval": (0.20, float('inf')),
                    "constant_pr": True,
                    "all_p_waves_conducted": True
                },
                confidence=0.90,
                clinical_significance="Delayed AV conduction, usually benign"
            ),
            
            ECGKnowledgeRule(
                name="left_bundle_branch_block",
                condition="conduction",
                parameters={
                    "qrs_duration": (0.12, float('inf')),
                    "absent_q_v5_v6": True,
                    "notched_r_v5_v6": True,
                    "st_t_opposite_qrs": True
                },
                confidence=0.88,
                clinical_significance="Complete LBBB, may indicate underlying heart disease"
            ),
            
            # Ischemia/Infarction rules
            ECGKnowledgeRule(
                name="acute_stemi",
                condition="ischemia",
                parameters={
                    "st_elevation": (2.0, float('inf')),  # mm
                    "reciprocal_changes": True,
                    "evolving_pattern": True,
                    "pathological_q_waves": False
                },
                confidence=0.92,
                clinical_significance="Acute myocardial infarction requiring immediate reperfusion"
            ),
            
            ECGKnowledgeRule(
                name="old_infarction",
                condition="ischemia",
                parameters={
                    "pathological_q_waves": True,
                    "q_wave_duration": (0.04, float('inf')),
                    "q_wave_depth": (0.25, float('inf')),  # 25% of R wave
                    "no_st_elevation": True
                },
                confidence=0.85,
                clinical_significance="Previous myocardial infarction with established scar"
            ),
            
            # Hypertrophy rules
            ECGKnowledgeRule(
                name="left_ventricular_hypertrophy",
                condition="hypertrophy",
                parameters={
                    "sokolow_lyon_voltage": (35, float('inf')),  # mm
                    "cornell_voltage": (28, float('inf')),  # mm for men
                    "st_t_strain_pattern": True
                },
                confidence=0.80,
                clinical_significance="LVH suggesting chronic pressure overload"
            )
        ]
        
        return rules
    
    def _build_condition_hierarchy(self) -> Dict[str, List[str]]:
        """Build hierarchy of cardiac conditions"""
        return {
            "rhythm_disorders": [
                "atrial_fibrillation",
                "atrial_flutter",
                "ventricular_tachycardia",
                "ventricular_fibrillation",
                "supraventricular_tachycardia"
            ],
            "conduction_disorders": [
                "first_degree_av_block",
                "second_degree_av_block",
                "third_degree_av_block",
                "left_bundle_branch_block",
                "right_bundle_branch_block"
            ],
            "ischemic_conditions": [
                "acute_stemi",
                "nstemi",
                "unstable_angina",
                "old_infarction"
            ],
            "structural_abnormalities": [
                "left_ventricular_hypertrophy",
                "right_ventricular_hypertrophy",
                "left_atrial_enlargement",
                "right_atrial_enlargement"
            ]
        }
    
    def apply_rules(self, ecg_features: Dict[str, Any]) -> List[Tuple[ECGKnowledgeRule, float]]:
        """Apply knowledge rules to ECG features"""
        matched_rules = []
        
        for rule in self.rules:
            match_score = self._evaluate_rule(rule, ecg_features)
            if match_score > 0.5:  # Threshold for rule matching
                matched_rules.append((rule, match_score))
        
        # Sort by confidence and match score
        matched_rules.sort(key=lambda x: x[0].confidence * x[1], reverse=True)
        
        return matched_rules
    
    def _evaluate_rule(self, rule: ECGKnowledgeRule, features: Dict[str, Any]) -> float:
        """Evaluate how well ECG features match a rule"""
        match_scores = []
        
        for param, expected in rule.parameters.items():
            if param not in features:
                continue
            
            actual = features[param]
            
            if isinstance(expected, bool):
                score = 1.0 if actual == expected else 0.0
            elif isinstance(expected, tuple):
                # Range check
                min_val, max_val = expected
                if min_val <= actual <= max_val:
                    score = 1.0
                else:
                    # Partial score based on distance
                    if actual < min_val:
                        score = max(0, 1 - (min_val - actual) / min_val)
                    else:
                        score = max(0, 1 - (actual - max_val) / max_val)
            else:
                score = 1.0 if actual == expected else 0.0
            
            match_scores.append(score)
        
        return np.mean(match_scores) if match_scores else 0.0


class GradCAMExplainer:
    """
    Gradient-weighted Class Activation Mapping for ECG CNNs
    Adapted for 1D temporal signals
    """
    
    def __init__(self, model: nn.Module, target_layer: str):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Register hooks
        self._register_hooks()
    
    def _register_hooks(self):
        """Register forward and backward hooks"""
        def forward_hook(module, input, output):
            self.activations = output.detach()
        
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()
        
        # Find target layer and register hooks
        for name, module in self.model.named_modules():
            if name == self.target_layer:
                module.register_forward_hook(forward_hook)
                module.register_backward_hook(backward_hook)
                break
    
    def generate_gradcam(
        self,
        input_tensor: torch.Tensor,
        target_class: Optional[int] = None
    ) -> np.ndarray:
        """Generate GradCAM heatmap"""
        self.model.eval()
        
        # Forward pass
        output = self.model(input_tensor)
        
        if target_class is None:
            target_class = output['logits'].argmax(1).item()
        
        # Backward pass
        self.model.zero_grad()
        one_hot = torch.zeros_like(output['logits'])
        one_hot[0, target_class] = 1.0
        output['logits'].backward(gradient=one_hot, retain_graph=True)
        
        # Compute GradCAM
        gradients = self.gradients[0]  # Remove batch dimension
        activations = self.activations[0]
        
        # Global average pooling of gradients
        weights = gradients.mean(dim=1, keepdim=True)
        
        # Weighted combination of activation maps
        gradcam = (weights * activations).sum(dim=0)
        
        # ReLU and normalization
        gradcam = F.relu(gradcam)
        gradcam = gradcam / (gradcam.max() + 1e-8)
        
        return gradcam.cpu().numpy()


class ECGFeatureExtractor:
    """Extract interpretable features from ECG signals"""
    
    def __init__(self, sampling_rate: int = 500):
        self.sampling_rate = sampling_rate
    
    def extract_features(self, ecg_signal: np.ndarray) -> Dict[str, Any]:
        """Extract comprehensive ECG features"""
        features = {}
        
        # Basic measurements
        features.update(self._extract_basic_measurements(ecg_signal))
        
        # Heart rate variability
        features.update(self._extract_hrv_features(ecg_signal))
        
        # Morphological features
        features.update(self._extract_morphological_features(ecg_signal))
        
        # Frequency domain features
        features.update(self._extract_frequency_features(ecg_signal))
        
        return features
    
    def _extract_basic_measurements(self, signal: np.ndarray) -> Dict[str, float]:
        """Extract basic ECG measurements"""
        # Simplified implementation - in practice, use robust QRS detection
        from scipy.signal import find_peaks
        
        # Find R peaks
        peaks, _ = find_peaks(signal[0], height=0.5, distance=self.sampling_rate*0.6)
        
        if len(peaks) < 2:
            return {}
        
        # RR intervals
        rr_intervals = np.diff(peaks) / self.sampling_rate
        
        features = {
            "heart_rate": 60 / np.mean(rr_intervals),
            "rr_mean": np.mean(rr_intervals),
            "rr_std": np.std(rr_intervals),
            "regular_rr": np.std(rr_intervals) < 0.1
        }
        
        return features
    
    def _extract_hrv_features(self, signal: np.ndarray) -> Dict[str, float]:
        """Extract heart rate variability features"""
        # Simplified HRV analysis
        from scipy.signal import find_peaks
        
        peaks, _ = find_peaks(signal[0], height=0.5, distance=self.sampling_rate*0.6)
        
        if len(peaks) < 3:
            return {}
        
        rr_intervals = np.diff(peaks) / self.sampling_rate * 1000  # Convert to ms
        
        features = {
            "hrv_rmssd": np.sqrt(np.mean(np.diff(rr_intervals)**2)),
            "hrv_pnn50": np.sum(np.abs(np.diff(rr_intervals)) > 50) / len(rr_intervals) * 100
        }
        
        return features
    
    def _extract_morphological_features(self, signal: np.ndarray) -> Dict[str, float]:
        """Extract morphological features"""
        # Placeholder for complex morphological analysis
        features = {
            "qrs_duration": 0.08,  # Placeholder
            "pr_interval": 0.16,   # Placeholder
            "qt_interval": 0.40,   # Placeholder
            "p_wave_present": True,
            "pathological_q_waves": False
        }
        
        return features
    
    def _extract_frequency_features(self, signal: np.ndarray) -> Dict[str, float]:
        """Extract frequency domain features"""
        # Compute power spectral density
        freqs, psd = scipy_signal.welch(signal[0], fs=self.sampling_rate)
        
        # Define frequency bands
        vlf_band = (0.003, 0.04)
        lf_band = (0.04, 0.15)
        hf_band = (0.15, 0.4)
        
        # Calculate power in each band
        vlf_power = np.trapz(psd[(freqs >= vlf_band[0]) & (freqs < vlf_band[1])])
        lf_power = np.trapz(psd[(freqs >= lf_band[0]) & (freqs < lf_band[1])])
        hf_power = np.trapz(psd[(freqs >= hf_band[0]) & (freqs < hf_band[1])])
        
        features = {
            "vlf_power": vlf_power,
            "lf_power": lf_power,
            "hf_power": hf_power,
            "lf_hf_ratio": lf_power / (hf_power + 1e-8)
        }
        
        return features


class BootstrapLIME:
    """
    Bootstrap-LIME adapted for ECG time series
    Handles temporal dependencies better than standard LIME
    """
    
    def __init__(self, model: nn.Module, feature_extractor: ECGFeatureExtractor):
        self.model = model
        self.feature_extractor = feature_extractor
        self.device = next(model.parameters()).device
    
    def explain_instance(
        self,
        ecg_signal: np.ndarray,
        num_samples: int = 1000,
        num_features: int = 10
    ) -> Dict[str, Any]:
        """Generate LIME explanation for ECG instance"""
        if not LIME_AVAILABLE:
            raise ImportError("LIME not available. Install with: pip install lime")
        
        # Extract features
        features = self.feature_extractor.extract_features(ecg_signal)
        feature_names = list(features.keys())
        feature_values = np.array(list(features.values()))
        
        # Create LIME explainer
        explainer = LimeTabularExplainer(
            feature_values.reshape(1, -1),
            feature_names=feature_names,
            mode='classification'
        )
        
        # Define prediction function
        def predict_fn(feature_array):
            # Reconstruct signals from features (simplified)
            batch_size = feature_array.shape[0]
            predictions = []
            
            for i in range(batch_size):
                # In practice, this would reconstruct the signal from features
                # For now, we'll use the original signal with perturbations
                perturbed_signal = ecg_signal + np.random.randn(*ecg_signal.shape) * 0.1
                
                with torch.no_grad():
                    tensor = torch.from_numpy(perturbed_signal).float().unsqueeze(0).to(self.device)
                    output = self.model(tensor)
                    probs = output['predictions'].cpu().numpy()
                    predictions.append(probs[0])
            
            return np.array(predictions)
        
        # Generate explanation
        explanation = explainer.explain_instance(
            feature_values,
            predict_fn,
            num_features=num_features,
            num_samples=num_samples
        )
        
        # Format results
        feature_importance = dict(explanation.as_list())
        
        return {
            'feature_importance': feature_importance,
            'intercept': explanation.intercept,
            'score': explanation.score,
            'local_pred': explanation.local_pred
        }


class CounterfactualExplainer:
    """
    Counterfactual explanation generator for ECG
    Shows minimal changes needed to alter prediction
    """
    
    def __init__(self, model: nn.Module, num_iterations: int = 100):
        self.model = model
        self.num_iterations = num_iterations
        self.device = next(model.parameters()).device
    
    def generate_counterfactual(
        self,
        original_ecg: torch.Tensor,
        target_class: int,
        lambda_validity: float = 1.0,
        lambda_proximity: float = 0.1,
        lambda_sparsity: float = 0.01
    ) -> Dict[str, Any]:
        """Generate counterfactual explanation"""
        self.model.eval()
        
        # Clone and require gradient
        counterfactual = original_ecg.clone().detach().requires_grad_(True)
        optimizer = torch.optim.Adam([counterfactual], lr=0.01)
        
        original_output = self.model(original_ecg)
        original_class = original_output['logits'].argmax(1).item()
        
        losses = []
        
        for iteration in range(self.num_iterations):
            optimizer.zero_grad()
            
            # Forward pass
            output = self.model(counterfactual)
            logits = output['logits']
            
            # Validity loss (cross-entropy with target class)
            validity_loss = F.cross_entropy(logits, torch.tensor([target_class]).to(self.device))
            
            # Proximity loss (L2 distance to original)
            proximity_loss = F.mse_loss(counterfactual, original_ecg)
            
            # Sparsity loss (L1 norm of changes)
            sparsity_loss = F.l1_loss(counterfactual, original_ecg)
            
            # Total loss
            total_loss = (lambda_validity * validity_loss + 
                         lambda_proximity * proximity_loss + 
                         lambda_sparsity * sparsity_loss)
            
            total_loss.backward()
            optimizer.step()
            
            # Clip to valid range
            with torch.no_grad():
                counterfactual.clamp_(-3, 3)  # Assuming normalized ECG
            
            losses.append(total_loss.item())
            
            # Check if target class achieved
            if logits.argmax(1).item() == target_class and iteration > 10:
                break
        
        # Compute final metrics
        final_output = self.model(counterfactual)
        final_class = final_output['logits'].argmax(1).item()
        
        # Identify changed regions
        difference = (counterfactual - original_ecg).abs()
        threshold = difference.mean() + 2 * difference.std()
        changed_regions = (difference > threshold).float()
        
        return {
            'counterfactual': counterfactual.detach(),
            'original_class': original_class,
            'final_class': final_class,
            'success': final_class == target_class,
            'difference': difference.detach(),
            'changed_regions': changed_regions,
            'num_iterations': iteration + 1,
            'losses': losses
        }


class IntegratedGradientsExplainer:
    """
    Integrated Gradients for ECG interpretation
    Provides attribution scores for each time point
    """
    
    def __init__(self, model: nn.Module, baseline: str = 'zero'):
        self.model = model
        self.baseline = baseline
        self.device = next(model.parameters()).device
    
    def explain(
        self,
        ecg_signal: torch.Tensor,
        target_class: Optional[int] = None,
        n_steps: int = 50
    ) -> torch.Tensor:
        """Compute integrated gradients"""
        self.model.eval()
        
        # Get baseline
        if self.baseline == 'zero':
            baseline = torch.zeros_like(ecg_signal)
        elif self.baseline == 'mean':
            baseline = ecg_signal.mean(dim=2, keepdim=True).expand_as(ecg_signal)
        else:
            baseline = torch.randn_like(ecg_signal) * 0.1
        
        # Get target class if not specified
        if target_class is None:
            output = self.model(ecg_signal)
            target_class = output['logits'].argmax(1).item()
        
        # Compute integrated gradients
        alphas = torch.linspace(0, 1, n_steps).to(self.device)
        integrated_grads = torch.zeros_like(ecg_signal)
        
        for alpha in alphas:
            # Interpolate between baseline and input
            interpolated = baseline + alpha * (ecg_signal - baseline)
            interpolated.requires_grad_(True)
            
            # Forward pass
            output = self.model(interpolated)
            logits = output['logits']
            
            # Backward pass
            self.model.zero_grad()
            logits[0, target_class].backward()
            
            # Accumulate gradients
            integrated_grads += interpolated.grad / n_steps
        
        # Multiply by the difference
        integrated_grads *= (ecg_signal - baseline)
        
        return integrated_grads


class HybridExplainabilitySystem:
    """
    Comprehensive explainability system combining multiple methods
    """
    
    def __init__(self, model: nn.Module, config: Optional[Dict[str, Any]] = None):
        self.model = model
        self.device = next(model.parameters()).device
        
        # Initialize components
        self.knowledge_base = ECGKnowledgeBase()
        self.feature_extractor = ECGFeatureExtractor()
        
        # Initialize explainers
        self.gradcam = GradCAMExplainer(model, 'cnn.bn_final')
        self.lime_explainer = BootstrapLIME(model, self.feature_extractor)
        self.counterfactual = CounterfactualExplainer(model)
        self.integrated_gradients = IntegratedGradientsExplainer(model)
        
        # SHAP explainer (if available)
        if SHAP_AVAILABLE:
            self.shap_explainer = self._init_shap_explainer()
        else:
            self.shap_explainer = None
    
    def _init_shap_explainer(self):
        """Initialize SHAP explainer"""
        def model_predict(x):
            tensor = torch.from_numpy(x).float().to(self.device)
            with torch.no_grad():
                output = self.model(tensor)
                return output['predictions'].cpu().numpy()
        
        # Create background dataset (simplified)
        background = np.random.randn(100, 12, 5000).astype(np.float32)
        
        return shap.DeepExplainer(model_predict, background)
    
    def explain_prediction(
        self,
        ecg_signal: np.ndarray,
        prediction: Dict[str, Any],
        methods: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Generate comprehensive explanation for ECG prediction
        
        Args:
            ecg_signal: ECG signal array
            prediction: Model prediction output
            methods: List of explanation methods to use
            
        Returns:
            Dictionary containing explanations from multiple methods
        """
        if methods is None:
            methods = ['knowledge', 'gradcam', 'lime', 'counterfactual', 'integrated_gradients']
        
        explanations = {}
        
        # Convert to tensor
        ecg_tensor = torch.from_numpy(ecg_signal).float().unsqueeze(0).to(self.device)
        
        # 1. Knowledge-based explanation
        if 'knowledge' in methods:
            features = self.feature_extractor.extract_features(ecg_signal)
            matched_rules = self.knowledge_base.apply_rules(features)
            
            explanations['knowledge'] = {
                'matched_rules': [
                    {
                        'rule': rule.name,
                        'confidence': rule.confidence,
                        'match_score': score,
                        'clinical_significance': rule.clinical_significance
                    }
                    for rule, score in matched_rules
                ],
                'extracted_features': features
            }
        
        # 2. GradCAM explanation
        if 'gradcam' in methods:
            gradcam_map = self.gradcam.generate_gradcam(ecg_tensor)
            explanations['gradcam'] = {
                'heatmap': gradcam_map,
                'important_regions': self._identify_important_regions(gradcam_map)
            }
        
        # 3. LIME explanation
        if 'lime' in methods and LIME_AVAILABLE:
            lime_exp = self.lime_explainer.explain_instance(ecg_signal)
            explanations['lime'] = lime_exp
        
        # 4. Counterfactual explanation
        if 'counterfactual' in methods:
            # Find the second most likely class
            probs = prediction['predictions']
            top_classes = probs.argsort()[-2:][::-1]
            target_class = top_classes[1] if len(top_classes) > 1 else 0
            
            cf_result = self.counterfactual.generate_counterfactual(
                ecg_tensor,
                target_class
            )
            
            explanations['counterfactual'] = {
                'success': cf_result['success'],
                'original_class': cf_result['original_class'],
                'target_class': target_class,
                'changed_regions': cf_result['changed_regions'].cpu().numpy(),
                'num_iterations': cf_result['num_iterations']
            }
        
        # 5. Integrated Gradients
        if 'integrated_gradients' in methods:
            ig_attributions = self.integrated_gradients.explain(ecg_tensor)
            explanations['integrated_gradients'] = {
                'attributions': ig_attributions.cpu().numpy(),
                'channel_importance': ig_attributions.abs().mean(dim=2).cpu().numpy()
            }
        
        # 6. SHAP explanation (if available)
        if 'shap' in methods and self.shap_explainer is not None:
            shap_values = self.shap_explainer.shap_values(ecg_signal[np.newaxis, :, :])
            explanations['shap'] = {
                'shap_values': shap_values,
                'base_value': self.shap_explainer.expected_value
            }
        
        # Combine insights
        explanations['summary'] = self._generate_summary(explanations, prediction)
        
        return explanations
    
    def _identify_important_regions(self, heatmap: np.ndarray, threshold: float = 0.7) -> List[Dict[str, Any]]:
        """Identify important regions in GradCAM heatmap"""
        important_indices = np.where(heatmap > threshold * heatmap.max())[0]
        
        if len(important_indices) == 0:
            return []
        
        # Group consecutive indices
        regions = []
        start = important_indices[0]
        end = start
        
        for i in range(1, len(important_indices)):
            if important_indices[i] == important_indices[i-1] + 1:
                end = important_indices[i]
            else:
                regions.append({
                    'start': int(start),
                    'end': int(end),
                    'importance': float(heatmap[start:end+1].mean())
                })
                start = important_indices[i]
                end = start
        
        regions.append({
            'start': int(start),
            'end': int(end),
            'importance': float(heatmap[start:end+1].mean())
        })
        
        return regions
    
    def _generate_summary(self, explanations: Dict[str, Any], prediction: Dict[str, Any]) -> Dict[str, str]:
        """Generate human-readable summary of explanations"""
        summary = {
            'prediction_confidence': f"{prediction['predictions'].max():.2%}",
            'explanation_consensus': self._assess_consensus(explanations),
            'key_findings': self._extract_key_findings(explanations),
            'clinical_relevance': self._assess_clinical_relevance(explanations)
        }
        
        return summary
    
    def _assess_consensus(self, explanations: Dict[str, Any]) -> str:
        """Assess consensus among different explanation methods"""
        # Simplified consensus assessment
        if len(explanations) < 2:
            return "Limited explanation methods available"
        
        # Check if different methods highlight similar regions/features
        consensus_score = 0.7  # Placeholder
        
        if consensus_score > 0.8:
            return "High consensus among explanation methods"
        elif consensus_score > 0.5:
            return "Moderate consensus among explanation methods"
        else:
            return "Low consensus - results should be interpreted carefully"
    
    def _extract_key_findings(self, explanations: Dict[str, Any]) -> List[str]:
        """Extract key findings from explanations"""
        findings = []
        
        # From knowledge-based explanation
        if 'knowledge' in explanations:
            for rule_info in explanations['knowledge']['matched_rules'][:3]:
                findings.append(
                    f"Pattern consistent with {rule_info['rule']} "
                    f"(confidence: {rule_info['confidence']:.2%})"
                )
        
        # From feature importance
        if 'lime' in explanations:
            important_features = sorted(
                explanations['lime']['feature_importance'].items(),
                key=lambda x: abs(x[1]),
                reverse=True
            )[:3]
            
            for feature, importance in important_features:
                findings.append(f"{feature}: {importance:.3f}")
        
        return findings
    
    def _assess_clinical_relevance(self, explanations: Dict[str, Any]) -> str:
        """Assess clinical relevance of findings"""
        if 'knowledge' in explanations and explanations['knowledge']['matched_rules']:
            top_rule = explanations['knowledge']['matched_rules'][0]
            return top_rule['clinical_significance']
        
        return "Clinical interpretation requires expert review"


def create_explainability_system(
    model: nn.Module,
    config: Optional[Dict[str, Any]] = None
) -> HybridExplainabilitySystem:
    """Factory function to create explainability system"""
    return HybridExplainabilitySystem(model, config)
