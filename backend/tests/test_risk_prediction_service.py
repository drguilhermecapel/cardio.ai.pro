"""
Tests for Risk Prediction Service
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock

from app.services.risk_prediction_service import (
    RiskPredictionService,
    CardiovascularRiskModel,
    SuddenCardiacDeathModel
)


class TestRiskPredictionService:
    """Test risk prediction service functionality"""

    def test_init_without_torch(self):
        """Test initialization when PyTorch is not available"""
        with patch('app.services.risk_prediction_service.TORCH_AVAILABLE', False):
            service = RiskPredictionService()
            
            assert service.cv_risk_model is None
            assert service.scd_model is None
            assert service.device == "cpu"

    @patch('app.services.risk_prediction_service.TORCH_AVAILABLE', True)
    def test_init_with_torch(self):
        """Test initialization when PyTorch is available"""
        with patch('app.services.risk_prediction_service.torch') as mock_torch:
            mock_torch.cuda.is_available.return_value = True
            mock_torch.device.return_value = "cuda"
            
            service = RiskPredictionService()
            
            assert len(service.event_types) == 8
            assert "myocardial_infarction" in service.event_types
            assert service.risk_thresholds["low"] == 0.075

    @patch('app.services.risk_prediction_service.TORCH_AVAILABLE', False)
    async def test_load_models_without_torch(self):
        """Test model loading when PyTorch is not available"""
        service = RiskPredictionService()
        
        with patch.object(service, '_load_traditional_models', return_value=True):
            result = await service.load_models()
            
            assert result is True

    @patch('app.services.risk_prediction_service.TORCH_AVAILABLE', True)
    async def test_load_models_with_torch(self):
        """Test model loading when PyTorch is available"""
        with patch('app.services.risk_prediction_service.torch') as mock_torch, \
             patch('app.services.risk_prediction_service.CardiovascularRiskModel') as mock_cv_model, \
             patch('app.services.risk_prediction_service.SuddenCardiacDeathModel') as mock_scd_model:
            
            mock_cv_instance = Mock()
            mock_scd_instance = Mock()
            mock_cv_model.return_value = mock_cv_instance
            mock_scd_model.return_value = mock_scd_instance
            mock_torch.device.return_value = "cpu"
            
            service = RiskPredictionService()
            
            with patch.object(service, '_load_traditional_models', return_value=True):
                result = await service.load_models()
                
                assert result is True
                assert service.models_loaded is True

    async def test_predict_cardiovascular_risk_basic(self):
        """Test basic cardiovascular risk prediction"""
        service = RiskPredictionService()
        
        patient_data = {
            "age": 65,
            "sex": "male",
            "systolic_bp": 140,
            "total_cholesterol": 220,
            "hdl_cholesterol": 40,
            "smoking": True,
            "diabetes": False
        }
        
        with patch.object(service, '_calculate_traditional_risk_scores') as mock_traditional:
            mock_traditional.return_value = {
                "ten_year_risk": 0.15,
                "framingham_score": 0.12,
                "pooled_cohort_score": 0.18,
                "score2_score": 0.15
            }
            
            result = await service.predict_cardiovascular_risk(
                patient_data=patient_data,
                use_neural_model=False
            )
            
            assert "ten_year_risk" in result
            assert "risk_category" in result
            assert "risk_factors" in result
            assert "recommendations" in result
            assert result["ten_year_risk"] == 0.15

    async def test_predict_scd_risk_without_model(self):
        """Test SCD risk prediction without neural model"""
        service = RiskPredictionService()
        
        patient_data = {
            "ejection_fraction": 30,
            "previous_cardiac_arrest": False,
            "sustained_vt": True,
            "heart_failure": True
        }
        
        ecg_features = np.random.randn(512).astype(np.float32)
        
        with patch.object(service, '_traditional_scd_assessment') as mock_traditional:
            mock_traditional.return_value = {
                "scd_risk_score": 0.08,
                "model_confidence": 0.7
            }
            
            result = await service.predict_sudden_cardiac_death_risk(
                patient_data=patient_data,
                ecg_features=ecg_features
            )
            
            assert "scd_risk_score" in result
            assert "risk_level" in result
            assert "risk_factors" in result
            assert result["scd_risk_score"] == 0.08

    def test_framingham_risk_calculation(self):
        """Test Framingham risk score calculation"""
        service = RiskPredictionService()
        
        patient_data = {
            "age": 60,
            "sex": "male",
            "total_cholesterol": 200,
            "hdl_cholesterol": 45,
            "systolic_bp": 130,
            "bp_treated": False,
            "smoking": False,
            "diabetes": False
        }
        
        risk = service._calculate_framingham_risk(patient_data)
        
        assert isinstance(risk, float)
        assert 0.0 <= risk <= 1.0

    def test_pooled_cohort_risk_calculation(self):
        """Test Pooled Cohort Equations calculation"""
        service = RiskPredictionService()
        
        patient_data = {
            "age": 55,
            "sex": "female",
            "race": "white",
            "total_cholesterol": 180,
            "hdl_cholesterol": 60,
            "systolic_bp": 120,
            "bp_treated": False,
            "smoking": False,
            "diabetes": False
        }
        
        risk = service._calculate_pooled_cohort_risk(patient_data)
        
        assert isinstance(risk, float)
        assert 0.0 <= risk <= 1.0

    def test_score2_risk_calculation(self):
        """Test SCORE2 risk calculation"""
        service = RiskPredictionService()
        
        patient_data = {
            "age": 50,
            "smoking": True,
            "systolic_bp": 140,
            "total_cholesterol": 220,
            "hdl_cholesterol": 40,
            "diabetes": False,
            "region_risk": "moderate"
        }
        
        risk = service._calculate_score2_risk(patient_data)
        
        assert isinstance(risk, float)
        assert 0.0 <= risk <= 1.0

    def test_risk_categorization(self):
        """Test risk level categorization"""
        service = RiskPredictionService()
        
        assert service._categorize_risk(0.05) == "low"
        assert service._categorize_risk(0.10) == "intermediate"
        assert service._categorize_risk(0.25) == "high"

    def test_risk_factors_identification(self):
        """Test risk factors identification"""
        service = RiskPredictionService()
        
        patient_data = {
            "age": 70,
            "sex": "male",
            "smoking": True,
            "diabetes": True,
            "hypertension": True,
            "total_cholesterol": 250,
            "hdl_cholesterol": 35,
            "bmi": 32,
            "family_history_cad": True
        }
        
        risk_factors = service._identify_risk_factors(patient_data)
        
        assert "Advanced age" in risk_factors
        assert "Male sex" in risk_factors
        assert "Current smoking" in risk_factors
        assert "Diabetes mellitus" in risk_factors
        assert "Hypertension" in risk_factors
        assert "High cholesterol" in risk_factors
        assert "Low HDL cholesterol" in risk_factors
        assert "Obesity" in risk_factors
        assert "Family history of CAD" in risk_factors

    def test_scd_risk_factors_identification(self):
        """Test SCD risk factors identification"""
        service = RiskPredictionService()
        
        patient_data = {
            "ejection_fraction": 25,
            "previous_cardiac_arrest": True,
            "sustained_vt": True,
            "ischemic_cardiomyopathy": True,
            "heart_failure": True,
            "syncope": True,
            "family_history_scd": True
        }
        
        risk_factors = service._identify_scd_risk_factors(patient_data)
        
        assert "Reduced ejection fraction" in risk_factors
        assert "Previous cardiac arrest" in risk_factors
        assert "Sustained ventricular tachycardia" in risk_factors
        assert "Ischemic cardiomyopathy" in risk_factors
        assert "Heart failure" in risk_factors
        assert "Unexplained syncope" in risk_factors
        assert "Family history of SCD" in risk_factors

    def test_protective_factors_identification(self):
        """Test protective factors identification"""
        service = RiskPredictionService()
        
        patient_data = {
            "icd_implanted": True,
            "beta_blocker": True,
            "ace_inhibitor": True,
            "statin_use": True,
            "exercise_regular": True
        }
        
        protective_factors = service._identify_protective_factors(patient_data)
        
        assert "ICD implanted" in protective_factors
        assert "Beta-blocker therapy" in protective_factors
        assert "ACE inhibitor therapy" in protective_factors
        assert "Statin therapy" in protective_factors
        assert "Regular exercise" in protective_factors

    def test_recommendations_generation(self):
        """Test treatment recommendations generation"""
        service = RiskPredictionService()
        
        high_risk_factors = ["Current smoking", "Diabetes mellitus", "High cholesterol"]
        high_recommendations = service._generate_recommendations("high", high_risk_factors)
        
        assert "Consider high-intensity statin therapy" in high_recommendations
        assert "Smoking cessation counseling and support" in high_recommendations
        assert "Optimal diabetes management (HbA1c <7%)" in high_recommendations
        
        low_recommendations = service._generate_recommendations("low", [])
        
        assert "Lifestyle modifications" in low_recommendations
        assert "Regular cardiovascular risk assessment" in low_recommendations

    def test_scd_recommendations_generation(self):
        """Test SCD recommendations generation"""
        service = RiskPredictionService()
        
        very_high_risk_factors = ["Reduced ejection fraction", "Heart failure"]
        very_high_recommendations = service._generate_scd_recommendations("very_high", very_high_risk_factors)
        
        assert "Urgent cardiology/electrophysiology consultation" in very_high_recommendations
        assert "Consider ICD implantation" in very_high_recommendations
        assert "ACE inhibitor/ARB and beta-blocker optimization" in very_high_recommendations
        
        low_recommendations = service._generate_scd_recommendations("low", [])
        
        assert "Routine cardiology follow-up" in low_recommendations

    def test_clinical_features_extraction(self):
        """Test clinical features extraction"""
        service = RiskPredictionService()
        
        patient_data = {
            "age": 65,
            "sex": "male",
            "systolic_bp": 140,
            "diastolic_bp": 90,
            "total_cholesterol": 220,
            "hdl_cholesterol": 40,
            "ldl_cholesterol": 150,
            "triglycerides": 200,
            "bmi": 28,
            "smoking": True,
            "diabetes": True,
            "hypertension": True,
            "family_history_cad": True,
            "previous_mi": False,
            "previous_stroke": False,
            "statin_use": True,
            "aspirin_use": True,
            "ace_inhibitor": False,
            "beta_blocker": True
        }
        
        features = service._extract_clinical_features(patient_data)
        
        assert len(features) == 20
        assert features[0] == 0.65  # age normalized
        assert features[1] == 1.0   # male sex
        assert features[9] == 1.0   # smoking
        assert features[10] == 1.0  # diabetes

    def test_scd_clinical_features_extraction(self):
        """Test SCD-specific clinical features extraction"""
        service = RiskPredictionService()
        
        patient_data = {
            "age": 60,
            "sex": "female",
            "ejection_fraction": 30,
            "lv_mass_index": 120,
            "ischemic_cardiomyopathy": True,
            "sustained_vt": True,
            "atrial_fibrillation": False,
            "previous_cardiac_arrest": False,
            "syncope": True,
            "heart_failure": True,
            "family_history_scd": True,
            "diabetes": False,
            "ckd": False,
            "icd_implanted": False,
            "crt_device": False,
            "antiarrhythmic_drugs": True
        }
        
        features = service._extract_clinical_features_scd(patient_data)
        
        assert len(features) == 20
        assert features[0] == 0.60  # age normalized
        assert features[1] == 0.0   # female sex
        assert features[2] == 0.30  # ejection fraction normalized
        assert features[4] == 1.0   # ischemic cardiomyopathy

    def test_get_service_info(self):
        """Test service information retrieval"""
        service = RiskPredictionService()
        
        info = service.get_service_info()
        
        assert "models_loaded" in info
        assert "torch_available" in info
        assert "sklearn_available" in info
        assert "supported_scores" in info
        assert "event_types" in info
        assert "risk_thresholds" in info
        
        assert "framingham" in info["supported_scores"]
        assert "pooled_cohort_equations" in info["supported_scores"]
        assert "score2" in info["supported_scores"]
        assert "neural_network" in info["supported_scores"]
        assert "sudden_cardiac_death" in info["supported_scores"]


@patch('app.services.risk_prediction_service.TORCH_AVAILABLE', True)
class TestCardiovascularRiskModel:
    """Test cardiovascular risk model"""

    def test_model_initialization(self):
        """Test model initialization"""
        with patch('app.services.risk_prediction_service.torch') as mock_torch:
            mock_torch.nn = Mock()
            
            model = CardiovascularRiskModel(
                input_dim=50,
                hidden_dims=[256, 128, 64],
                dropout=0.3
            )
            
            assert hasattr(model, 'risk_head')
            assert hasattr(model, 'event_head')
            assert hasattr(model, 'time_head')

    def test_model_forward_pass(self):
        """Test model forward pass"""
        with patch('app.services.risk_prediction_service.torch') as mock_torch:
            mock_tensor = Mock()
            mock_tensor.shape = (2, 50)
            
            model = CardiovascularRiskModel()
            
            model.risk_head = Mock(return_value=mock_tensor)
            model.event_head = Mock(return_value=mock_tensor)
            model.time_head = Mock(return_value=mock_tensor)
            
            result = model.forward(mock_tensor)
            
            assert "risk_probability" in result
            assert "event_logits" in result
            assert "time_to_event" in result


@patch('app.services.risk_prediction_service.TORCH_AVAILABLE', True)
class TestSuddenCardiacDeathModel:
    """Test sudden cardiac death model"""

    def test_model_initialization(self):
        """Test SCD model initialization"""
        with patch('app.services.risk_prediction_service.torch') as mock_torch:
            mock_torch.nn = Mock()
            
            model = SuddenCardiacDeathModel(
                ecg_features_dim=512,
                clinical_features_dim=20,
                hidden_dim=256
            )
            
            assert hasattr(model, 'ecg_processor')
            assert hasattr(model, 'clinical_processor')
            assert hasattr(model, 'fusion')

    def test_model_forward_pass(self):
        """Test SCD model forward pass"""
        with patch('app.services.risk_prediction_service.torch') as mock_torch:
            mock_ecg_tensor = Mock()
            mock_clinical_tensor = Mock()
            mock_output = Mock()
            
            mock_torch.cat.return_value = mock_output
            
            model = SuddenCardiacDeathModel()
            
            model.ecg_processor = Mock(return_value=mock_output)
            model.clinical_processor = Mock(return_value=mock_output)
            model.fusion = Mock(return_value=mock_output)
            
            result = model.forward(mock_ecg_tensor, mock_clinical_tensor)
            
            assert result == mock_output
