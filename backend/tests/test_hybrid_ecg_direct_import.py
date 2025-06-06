"""
Direct import test for hybrid_ecg_service to achieve real coverage
"""
import pytest
import numpy as np
from unittest.mock import AsyncMock, MagicMock, patch
from app.services.hybrid_ecg_service import HybridECGAnalysisService, UniversalECGReader, AdvancedPreprocessor, FeatureExtractor


class TestHybridECGDirectImport:
    """Test hybrid ECG service with direct imports for real coverage"""

    @pytest.fixture
    def mock_db_session(self):
        """Mock database session"""
        session = AsyncMock()
        return session

    @pytest.fixture
    def mock_ml_service(self):
        """Mock ML service"""
        service = MagicMock()
        service.predict_arrhythmia = AsyncMock(return_value={
            "predictions": {"normal": 0.8, "abnormal": 0.2},
            "confidence": 0.85
        })
        return service

    @pytest.fixture
    def hybrid_service(self, mock_db_session):
        """Create hybrid ECG service instance"""
        return HybridECGAnalysisService(
            db=mock_db_session,
            validation_service=None,
            sampling_rate=500
        )

    @pytest.fixture
    def sample_ecg_signal(self):
        """Sample ECG signal data"""
        return np.random.randn(5000, 12).astype(np.float64)

    def test_universal_ecg_reader_init(self):
        """Test UniversalECGReader initialization"""
        reader = UniversalECGReader()
        assert reader is not None

    def test_advanced_preprocessor_init(self):
        """Test AdvancedPreprocessor initialization"""
        preprocessor = AdvancedPreprocessor()
        assert preprocessor is not None

    def test_feature_extractor_init(self):
        """Test FeatureExtractor initialization"""
        extractor = FeatureExtractor()
        assert extractor is not None

    def test_hybrid_service_init(self, hybrid_service):
        """Test HybridECGAnalysisService initialization"""
        assert hybrid_service is not None
        assert hasattr(hybrid_service, 'ecg_reader')
        assert hasattr(hybrid_service, 'preprocessor')
        assert hasattr(hybrid_service, 'feature_extractor')

    def test_get_supported_pathologies(self, hybrid_service):
        """Test get_supported_pathologies method"""
        pathologies = hybrid_service.get_supported_pathologies()
        assert isinstance(pathologies, list)

    def test_validate_signal(valid_signal):
        """Test validate_signal method"""
        result = hybrid_service.validate_signal(valid_signal)
        assert "is_valid" in result

    def test_simulate_predictions(self, hybrid_service):
        """Test _simulate_predictions method"""
        features = {"heart_rate": 75, "qt_interval": 400}
        result = hybrid_service._simulate_predictions(features)
        assert isinstance(result, dict)
        assert "class_probabilities" in result
        assert "confidence" in result

    def test_detect_atrial_fibrillation(self, hybrid_service):
        """Test _detect_atrial_fibrillation method"""
        features = {"rr_std": 250, "hrv_rmssd": 60, "spectral_entropy": 0.9}
        result = hybrid_service._detect_atrial_fibrillation(features)
        assert isinstance(result, dict)
        assert "detected" in result
        assert "probability" in result

    def test_detect_long_qt(self, hybrid_service):
        """Test _detect_long_qt method"""
        features = {"qtc_bazett": 480}
        result = hybrid_service._detect_long_qt(features)
        assert isinstance(result, dict)
        assert "detected" in result
        assert "probability" in result

    @pytest.mark.asyncio
    async def test_generate_clinical_assessment(self, hybrid_service):
        """Test _generate_clinical_assessment method"""
        ai_predictions = {"normal": 0.8}
        pathology_results = {"atrial_fibrillation": {"detected": False}}
        features = {"heart_rate": 75}
        
        result = await hybrid_service._generate_clinical_assessment(
            ai_predictions, pathology_results, features
        )
        assert isinstance(result, dict)
        assert "clinical_urgency" in result
        assert "assessment" in result

    def test_analyze_emergency_patterns(self, hybrid_service, sample_ecg_signal):
        """Test _analyze_emergency_patterns method"""
        signal_1d = sample_ecg_signal[:, 0]
        result = hybrid_service._analyze_emergency_patterns(signal_1d)
        assert isinstance(result, dict)
        assert "stemi_detected" in result
        assert "vfib_detected" in result

    def test_generate_audit_trail(self, hybrid_service):
        """Test _generate_audit_trail method"""
        analysis_result = {"test": "data"}
        result = hybrid_service._generate_audit_trail(analysis_result)
        assert isinstance(result, dict)
        assert "timestamp" in result
        assert "analysis_id" in result

    def test_preprocess_signal(self, hybrid_service, sample_ecg_signal):
        """Test _preprocess_signal method"""
        signal_1d = sample_ecg_signal[:, 0]
        result = hybrid_service._preprocess_signal(signal_1d)
        assert isinstance(result, np.ndarray)

    def test_analyze_with_ai(self, hybrid_service, sample_ecg_signal):
        """Test _analyze_with_ai method"""
        signal_1d = sample_ecg_signal[:, 0]
        result = hybrid_service._analyze_with_ai(signal_1d)
        assert isinstance(result, dict)
        assert "classification" in result
        assert "confidence" in result

    def test_validate_ecg_signal(self, hybrid_service):
        """Test _validate_ecg_signal method"""
        signal_data = {
            "leads": {
                "I": [0.1, 0.2, 0.3, 0.4, 0.5],
                "II": [0.2, 0.3, 0.4, 0.5, 0.6]
            }
        }
        result = hybrid_service._validate_ecg_signal(signal_data)
        assert result is True

    def test_validate_signal_quality(self, hybrid_service, sample_ecg_signal):
        """Test _validate_signal_quality method"""
        signal_1d = sample_ecg_signal[:, 0]
        result = hybrid_service._validate_signal_quality(signal_1d)
        assert isinstance(result, dict)
        assert "is_valid" in result
        assert "quality" in result

    @pytest.mark.asyncio
    @pytest.mark.asyncio
    async def test_assess_signal_quality(self, hybrid_service, sample_ecg_signal):
        """Test _assess_signal_quality method"""
        signal_1d = sample_ecg_signal[:, 0]
        result = await hybrid_service._assess_signal_quality(signal_1d)
        assert isinstance(result, dict)
        assert "quality" in result
        assert "score" in result

    def test_get_system_status(self, hybrid_service):
        """Test get_system_status method"""
        result = hybrid_service.get_model_info()
        assert isinstance(result, dict)
        assert "status" in result
        assert "version" in result

    def test_apply_advanced_preprocessing(self, hybrid_service, sample_ecg_signal):
        """Test _apply_advanced_preprocessing method"""
        signal_1d = sample_ecg_signal[:, 0]
        result = hybrid_service._apply_advanced_preprocessing(signal_1d, 500)
        assert isinstance(result, np.ndarray)

    def test_extract_comprehensive_features(self, hybrid_service, sample_ecg_signal):
        """Test _extract_comprehensive_features method"""
        signal_1d = sample_ecg_signal[:, 0]
        result = hybrid_service._extract_comprehensive_features(signal_1d, 500)
        assert isinstance(result, dict)

    def test_analyze_leads(self, hybrid_service, sample_ecg_signal):
        """Test _analyze_leads method"""
        leads = ["I", "II", "III"]
        result = hybrid_service._analyze_leads(sample_ecg_signal, leads)
        assert isinstance(result, dict)

    def test_analyze_rhythm_patterns(self, hybrid_service, sample_ecg_signal):
        """Test _analyze_rhythm_patterns method"""
        signal_1d = sample_ecg_signal[:, 0]
        result = hybrid_service._analyze_rhythm_patterns(signal_1d, 500)
        assert isinstance(result, dict)
        assert "rhythm" in result

    def test_read_ecg_file_fallback(self, hybrid_service):
        """Test _read_ecg_file_fallback method"""
        result = hybrid_service._read_ecg_file_fallback("test.wfdb")
        assert isinstance(result, np.ndarray)

    def test_get_supported_formats(self, hybrid_service):
        """Test get_supported_formats method"""
        result = hybrid_service.supported_formats
        assert isinstance(result, list)
        assert len(result) > 0

    def test_analyze_ecg_signal(self, hybrid_service, sample_ecg_signal):
        """Test analyze_ecg_signal method"""
        result = hybrid_service.analyze_ecg_signal(sample_ecg_signal, sampling_rate=500)
        assert isinstance(result, dict)

    @pytest.mark.asyncio
    @pytest.mark.asyncio
    async def test_analyze_ecg_comprehensive_async(self, hybrid_service):
        """Test analyze_ecg_comprehensive_async method"""
        with patch.object(hybrid_service.reader, 'read_ecg') as mock_read:
            mock_read.return_value = np.random.randn(1000).astype(np.float64)
            
            result = await hybrid_service.analyze_ecg_comprehensive_async(
                file_path="test.csv",
                patient_id=1,
                analysis_id=1
            )
            assert isinstance(result, dict)

    @pytest.mark.asyncio
    async def test_analyze_ecg_comprehensive(self, hybrid_service, sample_ecg_signal):
        """Test analyze_ecg_comprehensive method"""
        signal_data = {
            "leads": {
                "I": sample_ecg_signal[:, 0].tolist(),
                "II": sample_ecg_signal[:, 1].tolist()
            }
        }
        
        result = await hybrid_service.analyze_ecg_comprehensive(signal_data)
        assert isinstance(result, dict)

    @pytest.mark.asyncio
    @pytest.mark.asyncio
    async def test_run_simplified_analysis(self, hybrid_service):
        """Test _run_simplified_analysis method"""
        signal_data = {"test": "data"}
        features = {"heart_rate": 75}
        
        result = await hybrid_service._run_simplified_analysis(signal_data, features)
        assert isinstance(result, dict)
        assert "simplified" in result

    @pytest.mark.asyncio
    @pytest.mark.asyncio
    async def test_detect_pathologies(self, hybrid_service):
        """Test _detect_pathologies method"""
        signal_data = {"test": "data"}
        features = {"heart_rate": 105, "qt_interval": 460}
        
        result = await hybrid_service._detect_pathologies(signal_data, features)
        assert isinstance(result, dict)
