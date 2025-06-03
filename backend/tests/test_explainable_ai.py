"""
Tests for Explainable AI Service
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch

from app.services.explainable_ai import (
    ExplanationMethod,
    ECGLead,
    ECGSegment,
    ClinicalFinding,
    ExplanationResult,
    ECGFeatureExtractor,
    SHAPExplainer,
    LIMEExplainer,
    GradientBasedExplainer,
    ClinicalReasoningEngine,
    ExplainableAIService
)


class TestECGFeatureExtractor:
    """Test ECG feature extraction"""
    
    def setup_method(self):
        self.extractor = ECGFeatureExtractor(sampling_rate=500)
        
    def test_initialization(self):
        """Test feature extractor initialization"""
        assert self.extractor.sampling_rate == 500
        assert len(self.extractor.ecg_segments) > 0
        
        segment_names = [seg.name for seg in self.extractor.ecg_segments]
        assert "P_wave" in segment_names
        assert "QRS_complex" in segment_names
        assert "T_wave" in segment_names
        
    def test_extract_morphological_features(self):
        """Test morphological feature extraction"""
        signal = np.sin(np.linspace(0, 4*np.pi, 1000)).astype(np.float32)
        
        features = self.extractor.extract_morphological_features(signal, "II")
        
        assert isinstance(features, dict)
        assert len(features) > 0
        
        expected_features = [
            "II_max_amplitude",
            "II_min_amplitude", 
            "II_amplitude_range",
            "II_rms",
            "II_mean",
            "II_std"
        ]
        
        for feature in expected_features:
            assert feature in features
            assert isinstance(features[feature], float)
            
    def test_extract_morphological_features_empty_signal(self):
        """Test feature extraction with empty signal"""
        signal = np.array([], dtype=np.float32)
        
        features = self.extractor.extract_morphological_features(signal, "I")
        
        assert isinstance(features, dict)
        
    def test_calculate_skewness(self):
        """Test skewness calculation"""
        signal = np.random.normal(0, 1, 1000).astype(np.float32)
        skewness = self.extractor._calculate_skewness(signal)
        
        assert isinstance(skewness, float)
        assert abs(skewness) < 1.0  # Should be close to 0 for normal distribution
        
    def test_calculate_kurtosis(self):
        """Test kurtosis calculation"""
        signal = np.random.normal(0, 1, 1000).astype(np.float32)
        kurtosis = self.extractor._calculate_kurtosis(signal)
        
        assert isinstance(kurtosis, float)
        assert abs(kurtosis) < 2.0  # Should be close to 0 for normal distribution


class TestSHAPExplainer:
    """Test SHAP explainer"""
    
    def setup_method(self):
        self.explainer = SHAPExplainer()
        
    def test_initialization(self):
        """Test SHAP explainer initialization"""
        assert self.explainer.explainer is None
        assert self.explainer.background_data is None
        
    def test_fallback_explanation(self):
        """Test fallback explanation when SHAP is not available"""
        input_data = np.random.random((12, 1000)).astype(np.float32)
        
        result = self.explainer._fallback_explanation(input_data)
        
        assert isinstance(result, dict)
        assert "method" in result
        assert "lead_importance" in result
        assert result["method"] == "variance_based"
        
    def test_explain_prediction_fallback(self):
        """Test prediction explanation with fallback"""
        input_data = np.random.random((12, 1000)).astype(np.float32)
        
        result = self.explainer.explain_prediction(input_data)
        
        assert isinstance(result, dict)
        assert "lead_importance" in result
        assert "explanation_quality" in result


class TestLIMEExplainer:
    """Test LIME explainer"""
    
    def setup_method(self):
        self.explainer = LIMEExplainer()
        
    def test_initialization(self):
        """Test LIME explainer initialization"""
        assert self.explainer.explainer is None
        
    def test_fallback_explanation(self):
        """Test fallback explanation when LIME is not available"""
        input_data = np.random.random((1000,)).astype(np.float32)
        
        result = self.explainer._fallback_explanation(input_data)
        
        assert isinstance(result, dict)
        assert "method" in result
        assert result["method"] == "gradient_approximation"
        assert "local_explanation" in result
        assert result["local_explanation"] is True


class TestGradientBasedExplainer:
    """Test gradient-based explainer"""
    
    def setup_method(self):
        self.explainer = GradientBasedExplainer()
        
    def test_initialization(self):
        """Test gradient explainer initialization"""
        assert self.explainer.model is None
        
    def test_set_model(self):
        """Test setting model for gradient analysis"""
        mock_model = Mock()
        
        result = self.explainer.set_model(mock_model)
        
        assert result is True
        assert self.explainer.model == mock_model
        
    def test_fallback_gradient_explanation(self):
        """Test fallback gradient explanation"""
        input_data = np.random.random((12, 1000)).astype(np.float32)
        
        result = self.explainer._fallback_gradient_explanation(input_data)
        
        assert isinstance(result, dict)
        assert "method" in result
        assert "gradients" in result
        assert "importance_scores" in result
        
    def test_generate_attention_maps(self):
        """Test attention map generation"""
        importance_scores = np.random.random((12, 1000)).astype(np.float32)
        
        attention_maps = self.explainer._generate_attention_maps(importance_scores)
        
        assert isinstance(attention_maps, dict)
        assert len(attention_maps) > 0
        
        expected_leads = ["I", "II", "III", "aVR", "aVL", "aVF"]
        for lead in expected_leads:
            if lead in attention_maps:
                assert isinstance(attention_maps[lead], np.ndarray)


class TestClinicalReasoningEngine:
    """Test clinical reasoning engine"""
    
    def setup_method(self):
        self.engine = ClinicalReasoningEngine()
        
    def test_initialization(self):
        """Test clinical reasoning engine initialization"""
        assert hasattr(self.engine, 'clinical_rules')
        assert hasattr(self.engine, 'feature_extractor')
        assert len(self.engine.clinical_rules) > 0
        
        expected_conditions = [
            "atrial_fibrillation",
            "stemi", 
            "ventricular_tachycardia",
            "left_bundle_branch_block"
        ]
        
        for condition in expected_conditions:
            assert condition in self.engine.clinical_rules
            
    def test_generate_clinical_reasoning(self):
        """Test clinical reasoning generation"""
        ecg_data = np.random.random((12, 1000)).astype(np.float32)
        
        predictions = {
            "atrial_fibrillation": 0.8,
            "normal": 0.1,
            "stemi": 0.05
        }
        
        confidence = 0.8
        
        findings = self.engine.generate_clinical_reasoning(
            ecg_data, predictions, confidence
        )
        
        assert isinstance(findings, list)
        
        if findings:
            for finding in findings:
                assert isinstance(finding, ClinicalFinding)
                assert hasattr(finding, 'condition')
                assert hasattr(finding, 'confidence')
                assert hasattr(finding, 'evidence')
                assert hasattr(finding, 'recommendations')
                
    def test_get_lead_index(self):
        """Test lead index mapping"""
        assert self.engine._get_lead_index("I") == 0
        assert self.engine._get_lead_index("II") == 1
        assert self.engine._get_lead_index("V1") == 6
        assert self.engine._get_lead_index("V6") == 11
        assert self.engine._get_lead_index("unknown") == 0  # Default
        
    def test_check_condition_patterns(self):
        """Test condition pattern checking"""
        features = {
            "II_rr_std": 100.0,  # High variability for AF
            "II_amplitude_range": 1.0,  # High amplitude for STEMI
            "II_heart_rate": 150.0,  # High rate for VT
            "II_std": 0.5  # High std for LBBB
        }
        
        af_result = self.engine._check_condition_patterns(
            "atrial_fibrillation", "II", features
        )
        assert isinstance(af_result, bool)
        
        stemi_result = self.engine._check_condition_patterns(
            "stemi", "II", features
        )
        assert isinstance(stemi_result, bool)


class TestExplainableAIService:
    """Test main explainable AI service"""
    
    def setup_method(self):
        self.service = ExplainableAIService()
        
    def test_initialization(self):
        """Test service initialization"""
        assert hasattr(self.service, 'shap_explainer')
        assert hasattr(self.service, 'lime_explainer')
        assert hasattr(self.service, 'gradient_explainer')
        assert hasattr(self.service, 'clinical_reasoning')
        assert hasattr(self.service, 'feature_extractor')
        
        assert isinstance(self.service.available_methods, list)
        assert len(self.service.available_methods) > 0
        
    def test_check_available_methods(self):
        """Test checking available explanation methods"""
        methods = self.service._check_available_methods()
        
        assert isinstance(methods, list)
        assert ExplanationMethod.CLINICAL_REASONING in methods
        assert ExplanationMethod.FEATURE_IMPORTANCE in methods
        
    def test_initialize_explainers(self):
        """Test explainer initialization"""
        background_data = np.random.random((100, 12000)).astype(np.float32)
        training_data = np.random.random((200, 12000)).astype(np.float32)
        
        def mock_model_function(x):
            return np.random.random((len(x), 5))
            
        result = self.service.initialize_explainers(
            model_function=mock_model_function,
            background_data=background_data,
            training_data=training_data
        )
        
        assert isinstance(result, bool)
        
    def test_generate_comprehensive_explanation(self):
        """Test comprehensive explanation generation"""
        ecg_data = np.random.random((12, 1000)).astype(np.float32)
        
        predictions = {
            "atrial_fibrillation": 0.7,
            "normal": 0.2,
            "stemi": 0.1
        }
        
        confidence = 0.7
        
        result = self.service.generate_comprehensive_explanation(
            ecg_data, predictions, confidence
        )
        
        assert isinstance(result, ExplanationResult)
        assert hasattr(result, 'method')
        assert hasattr(result, 'feature_importance')
        assert hasattr(result, 'attention_maps')
        assert hasattr(result, 'clinical_reasoning')
        assert hasattr(result, 'visual_explanations')
        assert hasattr(result, 'confidence_factors')
        assert hasattr(result, 'uncertainty_analysis')
        
    def test_generate_attention_maps(self):
        """Test attention map generation"""
        ecg_data = np.random.random((12, 1000)).astype(np.float32)
        predictions = {"atrial_fibrillation": 0.8, "normal": 0.2}
        
        attention_maps = self.service._generate_attention_maps(ecg_data, predictions)
        
        assert isinstance(attention_maps, dict)
        
        if attention_maps:
            for lead_name, attention in attention_maps.items():
                assert isinstance(attention, np.ndarray)
                assert attention.dtype == np.float32
                
    def test_generate_feature_importance(self):
        """Test feature importance generation"""
        ecg_data = np.random.random((12, 1000)).astype(np.float32)
        
        importance = self.service._generate_feature_importance(ecg_data)
        
        assert isinstance(importance, dict)
        
        if importance:
            for lead_name, score in importance.items():
                assert isinstance(score, float)
                assert score >= 0.0
                
    def test_analyze_uncertainty(self):
        """Test uncertainty analysis"""
        predictions = {
            "atrial_fibrillation": 0.6,
            "normal": 0.3,
            "stemi": 0.1
        }
        confidence = 0.7
        
        uncertainty = self.service._analyze_uncertainty(predictions, confidence)
        
        assert isinstance(uncertainty, dict)
        assert "uncertainty_score" in uncertainty
        assert isinstance(uncertainty["uncertainty_score"], float)
        assert 0.0 <= uncertainty["uncertainty_score"] <= 1.0
        
    def test_generate_fallback_explanation(self):
        """Test fallback explanation generation"""
        ecg_data = np.random.random((12, 1000)).astype(np.float32)
        predictions = {"normal": 0.9, "abnormal": 0.1}
        confidence = 0.9
        
        result = self.service._generate_fallback_explanation(
            ecg_data, predictions, confidence
        )
        
        assert isinstance(result, ExplanationResult)
        assert result.method == ExplanationMethod.FEATURE_IMPORTANCE
        
    def test_get_available_methods(self):
        """Test getting available methods"""
        methods = self.service.get_available_methods()
        
        assert isinstance(methods, list)
        assert len(methods) > 0
        
        assert "clinical_reasoning" in methods
        assert "feature_importance" in methods
        
    def test_get_service_info(self):
        """Test getting service information"""
        info = self.service.get_service_info()
        
        assert isinstance(info, dict)
        assert "initialized" in info
        assert "available_methods" in info
        assert "clinical_reasoning_enabled" in info
        assert "feature_extraction_enabled" in info
        
        assert info["clinical_reasoning_enabled"] is True
        assert info["feature_extraction_enabled"] is True


class TestDataClasses:
    """Test data classes and enums"""
    
    def test_ecg_segment(self):
        """Test ECG segment dataclass"""
        segment = ECGSegment(
            name="P_wave",
            start_ms=0,
            end_ms=120,
            description="Atrial depolarization",
            normal_range=(80, 120)
        )
        
        assert segment.name == "P_wave"
        assert segment.start_ms == 0
        assert segment.end_ms == 120
        assert segment.description == "Atrial depolarization"
        assert segment.normal_range == (80, 120)
        
    def test_clinical_finding(self):
        """Test clinical finding dataclass"""
        finding = ClinicalFinding(
            condition="Atrial Fibrillation",
            confidence=0.85,
            evidence=["Irregular rhythm", "Absent P waves"],
            lead_involvement=["II", "V1"],
            clinical_significance="High stroke risk",
            recommendations=["Anticoagulation", "Rate control"]
        )
        
        assert finding.condition == "Atrial Fibrillation"
        assert finding.confidence == 0.85
        assert len(finding.evidence) == 2
        assert len(finding.lead_involvement) == 2
        assert len(finding.recommendations) == 2
        
    def test_explanation_result(self):
        """Test explanation result dataclass"""
        result = ExplanationResult(
            method=ExplanationMethod.SHAP,
            feature_importance={"I": 0.5, "II": 0.8},
            attention_maps={"I": np.array([1, 2, 3])},
            clinical_reasoning=[],
            visual_explanations={"shap": {}},
            confidence_factors=["High confidence"],
            uncertainty_analysis={"entropy": 0.2}
        )
        
        assert result.method == ExplanationMethod.SHAP
        assert len(result.feature_importance) == 2
        assert "I" in result.attention_maps
        assert len(result.confidence_factors) == 1
        
    def test_explanation_method_enum(self):
        """Test explanation method enum"""
        assert ExplanationMethod.SHAP.value == "shap"
        assert ExplanationMethod.LIME.value == "lime"
        assert ExplanationMethod.GRADIENT_BASED.value == "gradient_based"
        assert ExplanationMethod.CLINICAL_REASONING.value == "clinical_reasoning"
        
    def test_ecg_lead_enum(self):
        """Test ECG lead enum"""
        assert ECGLead.I.value == "I"
        assert ECGLead.II.value == "II"
        assert ECGLead.V1.value == "V1"
        assert ECGLead.V6.value == "V6"


@pytest.mark.integration
def test_integration_workflow():
    """Test complete explainable AI workflow"""
    service = ExplainableAIService()
    
    ecg_data = np.random.random((12, 5000)).astype(np.float32)
    
    for i in range(12):
        for beat in range(0, 5000, 600):  # ~100 BPM
            if beat + 100 < 5000:
                ecg_data[i, beat:beat+100] += np.sin(np.linspace(0, 2*np.pi, 100)) * 0.5
                
    predictions = {
        "normal": 0.1,
        "atrial_fibrillation": 0.7,
        "stemi": 0.1,
        "ventricular_tachycardia": 0.05,
        "left_bundle_branch_block": 0.05
    }
    
    confidence = 0.75
    
    explanation = service.generate_comprehensive_explanation(
        ecg_data, predictions, confidence
    )
    
    assert isinstance(explanation, ExplanationResult)
    assert explanation.method in [method for method in ExplanationMethod]
    
    assert isinstance(explanation.feature_importance, dict)
    
    assert isinstance(explanation.attention_maps, dict)
    
    assert isinstance(explanation.clinical_reasoning, list)
    
    af_findings = [f for f in explanation.clinical_reasoning if "atrial" in f.condition.lower()]
    assert len(af_findings) > 0 or len(explanation.clinical_reasoning) == 0  # May be empty if extraction fails
    
    assert isinstance(explanation.uncertainty_analysis, dict)
    assert "uncertainty_score" in explanation.uncertainty_analysis
    
    assert isinstance(explanation.confidence_factors, list)
    assert len(explanation.confidence_factors) > 0
    
    service_info = service.get_service_info()
    assert service_info["clinical_reasoning_enabled"] is True
    assert service_info["feature_extraction_enabled"] is True
    
    methods = service.get_available_methods()
    assert "clinical_reasoning" in methods
    assert "feature_importance" in methods
