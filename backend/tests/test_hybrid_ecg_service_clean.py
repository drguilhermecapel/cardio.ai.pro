"""
Comprehensive tests for Hybrid ECG Service - targeting 829 uncovered lines
Phase 1: Zero-coverage critical modules for regulatory compliance
"""

import pytest
from unittest.mock import Mock, patch, AsyncMock, MagicMock
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, Any

from app.services.hybrid_ecg_service import (
    HybridECGAnalysisService,
    UniversalECGReader,
    AdvancedPreprocessor,
    FeatureExtractor,
    ClinicalUrgency
)


class TestHybridECGAnalysisServiceComprehensive:
    """Comprehensive tests for Hybrid ECG Analysis Service - 829 uncovered lines"""
    
    @pytest.fixture
    def service(self):
        """Create service instance with mocked dependencies"""
        return HybridECGAnalysisService()
    
    @pytest.fixture
    def sample_ecg_data(self):
        """Generate realistic ECG test data"""
        duration = 10
        fs = 500
        t = np.linspace(0, duration, duration * fs)
        ecg = 0.5 * np.sin(2 * np.pi * 1.2 * t) + 0.1 * np.random.randn(len(t))
        return ecg.astype(np.float64), fs
    
    def test_service_initialization(self, service):
        """Test service initialization - covers lines 740-766"""
        assert service is not None
        assert hasattr(service, 'reader')
        assert hasattr(service, 'preprocessor')
        assert hasattr(service, 'feature_extractor')
        assert isinstance(service.reader, UniversalECGReader)
        assert isinstance(service.preprocessor, AdvancedPreprocessor)
        assert isinstance(service.feature_extractor, FeatureExtractor)
    
    @patch('app.services.hybrid_ecg_service.UniversalECGReader.read_ecg')
    def test_analyze_ecg_file_success(self, mock_read_ecg, service, sample_ecg_data):
        """Test ECG file analysis - covers lines 768-801"""
        ecg_data, fs = sample_ecg_data
        mock_read_ecg.return_value = {
            'signal': ecg_data.reshape(-1, 1),
            'sampling_rate': fs,
            'metadata': {'format': 'csv'},
            'labels': ['Lead I']
        }
        
        result = service.analyze_ecg_file("test_file.csv")
        
        assert isinstance(result, dict)
        assert 'features' in result
        assert 'predictions' in result
        assert 'metadata' in result
        assert 'processing_info' in result
        mock_read_ecg.assert_called_once()
    
    def test_analyze_ecg_signal_comprehensive(self, service, sample_ecg_data):
        """Test ECG signal analysis - covers lines 803-838"""
        ecg_data, fs = sample_ecg_data
        
        result = service.analyze_ecg_signal(ecg_data, fs)
        
        assert isinstance(result, dict)
        assert 'predictions' in result
        assert 'confidence' in result
        assert 'pathologies' in result
        assert 'clinical_assessment' in result
    
    def test_simulate_predictions(self, service):
        """Test prediction simulation - covers lines 840-865"""
        features = {'heart_rate': 75, 'qrs_duration': 100}
        
        predictions = service._simulate_predictions(features)
        
        assert isinstance(predictions, dict)
        assert 'normal' in predictions
        assert 'atrial_fibrillation' in predictions
        assert 'ventricular_tachycardia' in predictions
        assert all(0 <= v <= 1 for v in predictions.values())
    
    def test_get_supported_pathologies(self, service):
        """Test supported pathologies - covers lines 867-869"""
        pathologies = service.get_supported_pathologies()
        
        assert isinstance(pathologies, list)
        assert len(pathologies) > 0
        assert 'atrial_fibrillation' in pathologies
    
    def test_validate_signal_comprehensive(self, service, sample_ecg_data):
        """Test signal validation - covers lines 871-918"""
        ecg_data, fs = sample_ecg_data
        
        is_valid, issues = service.validate_signal(sample_signal)
        assert isinstance(is_valid, bool)
        assert isinstance(issues, list)
        
        short_signal = np.array([1, 2, 3])
        is_valid, issues = service.validate_signal(sample_signal)
        assert is_valid == False
        assert len(issues) > 0
        
        is_valid, issues = service.validate_signal(sample_signal)
        assert is_valid == False
        assert 'Invalid sampling rate' in str(issues)
    
    def test_analyze_ecg_comprehensive_sync(self, service, sample_ecg_data):
        """Test comprehensive ECG analysis - covers lines 920-952"""
        ecg_data, fs = sample_ecg_data
        
        result = await service.analyze_ecg_comprehensive(ecg_data, fs)
        
        assert isinstance(result, dict)
        assert 'analysis_id' in result
        assert 'timestamp' in result
        assert 'signal_quality' in result
        assert 'pathology_predictions' in result
        assert 'clinical_assessment' in result
    
    def test_detect_atrial_fibrillation(self, service):
        """Test AF detection - covers lines 1165-1187"""
        features = {
            'rr_intervals': np.array([0.8, 1.2, 0.9, 1.1, 0.7]),
            'heart_rate_variability': 0.15
        }
        
        af_probability = service._detect_atrial_fibrillation(features)
        
        assert isinstance(af_probability, float)
        assert 0 <= af_probability <= 1
    
    def test_detect_long_qt(self, service):
        """Test Long QT detection - covers lines 1189-1210"""
        features = {
            'qt_interval': 450,
            'heart_rate': 60,
            'qtc_interval': 460
        }
        
        long_qt_probability = service._detect_long_qt(features)
        
        assert isinstance(long_qt_probability, float)
        assert 0 <= long_qt_probability <= 1
    
    def test_generate_clinical_assessment(self, service):
        """Test clinical assessment generation - covers lines 1212-1251"""
        predictions = {
            'normal': 0.3,
            'atrial_fibrillation': 0.6,
            'ventricular_tachycardia': 0.1
        }
        
        assessment = await service._generate_clinical_assessment(predictions)
        
        assert isinstance(assessment, dict)
        assert 'primary_diagnosis' in assessment
        assert 'urgency_level' in assessment
        assert 'recommendations' in assessment
        assert assessment['urgency_level'] in [
            ClinicalUrgency.LOW, ClinicalUrgency.MEDIUM, 
            ClinicalUrgency.HIGH, ClinicalUrgency.CRITICAL
        ]
    
    def test_analyze_emergency_patterns(self, service):
        """Test emergency pattern analysis - covers lines 1252-1284"""
        predictions = {
            'ventricular_fibrillation': 0.8,
            'ventricular_tachycardia': 0.1,
            'normal': 0.1
        }
        
        emergency_assessment = service._analyze_emergency_patterns(predictions)
        
        assert isinstance(emergency_assessment, dict)
        assert 'is_emergency' in emergency_assessment
        assert 'emergency_type' in emergency_assessment
        assert 'immediate_actions' in emergency_assessment
    
    def test_generate_audit_trail(self, service):
        """Test audit trail generation - covers lines 1286-1310"""
        analysis_data = {
            'analysis_id': 'test_123',
            'timestamp': datetime.now(),
            'predictions': {'normal': 0.8}
        }
        
        audit_trail = service._generate_audit_trail(analysis_data)
        
        assert isinstance(audit_trail, dict)
        assert 'analysis_id' in audit_trail
        assert 'timestamp' in audit_trail
        assert 'processing_steps' in audit_trail
        assert 'system_version' in audit_trail
    
    def test_get_system_status(self, service):
        """Test system status - covers lines 1567-1575"""
        status = service.get_model_info()
        
        assert isinstance(status, dict)
        assert 'status' in status
        assert 'version' in status
        assert 'supported_formats' in status
    
    def test_get_supported_formats(self, service):
        """Test supported formats - covers lines 1635-1637"""
        formats = service.supported_formats)
        
        assert isinstance(formats, list)
        assert len(formats) > 0
        assert '.csv' in formats


class TestUniversalECGReader:
    """Test Universal ECG Reader - covers lines 34-275"""
    
    @pytest.fixture
    def reader(self):
        return UniversalECGReader()
    
    def test_reader_initialization(self, reader):
        """Test reader initialization - covers lines 39-51"""
        assert reader is not None
        assert hasattr(reader, 'supported_formats')
        assert '.csv' in reader.supported_formats
        assert '.txt' in reader.supported_formats
    
    @patch('numpy.loadtxt')
    def test_read_csv_format(self, mock_loadtxt, reader):
        """Test CSV reading - covers lines 136-156"""
        mock_loadtxt.return_value = np.random.randn(1000)
        
        signal, fs, metadata = reader._read_csv("test.csv")
        
        assert isinstance(signal, np.ndarray)
        assert isinstance(fs, int)
        assert isinstance(metadata, dict)
        mock_loadtxt.assert_called_once()
    
    @patch('numpy.loadtxt')
    def test_read_text_format(self, mock_loadtxt, reader):
        """Test text reading - covers lines 183-204"""
        mock_loadtxt.return_value = np.random.randn(1000)
        
        signal, fs, metadata = reader._read_text("test.txt")
        
        assert isinstance(signal, np.ndarray)
        assert isinstance(fs, int)
        assert isinstance(metadata, dict)


class TestAdvancedPreprocessor:
    """Test Advanced Preprocessor - covers lines 278-413"""
    
    @pytest.fixture
    def preprocessor(self):
        return AdvancedPreprocessor()
    
    @pytest.fixture
    def sample_signal(self):
        return np.random.randn(1000).astype(np.float64)
    
    def test_preprocessor_initialization(self, preprocessor):
        """Test preprocessor initialization - covers lines 283-285"""
        assert preprocessor is not None
    
    def test_preprocess_signal(self, preprocessor, sample_signal):
        """Test signal preprocessing - covers lines 287-342"""
        fs = 500
        
        processed = await preprocessor.preprocess_signal(sample_signal, fs)
        
        assert isinstance(processed, np.ndarray)
        assert len(processed) == len(sample_signal)
    
    def test_filter_signal(self, preprocessor, sample_signal):
        """Test signal filtering - covers lines 361-373"""
        fs = 500
        
        filtered = preprocessor.filter_signal(sample_signal, fs)
        
        assert isinstance(filtered, np.ndarray)
        assert len(filtered) == len(sample_signal)


class TestFeatureExtractor:
    """Test Feature Extractor - covers lines 416-732"""
    
    @pytest.fixture
    def extractor(self):
        return FeatureExtractor()
    
    @pytest.fixture
    def sample_signal(self):
        return np.random.randn(1000).astype(np.float64)
    
    def test_extractor_initialization(self, extractor):
        """Test extractor initialization - covers lines 421-423"""
        assert extractor is not None
    
    def test_extract_all_features(self, extractor, sample_signal):
        """Test feature extraction - covers lines 425-448"""
        fs = 500
        
        features = extractor.extract_all_features(sample_signal, fs)
        
        assert isinstance(features, dict)
        assert len(features) > 0
    
    def test_extract_time_domain_features(self, extractor, sample_signal):
        """Test time domain features - covers lines 485-498"""
        fs = 500
        
        features = extractor.extract_time_domain_features(sample_signal, fs)
        
        assert isinstance(features, dict)
        assert 'heart_rate' in features or 'rr_intervals' in features
    
    def test_extract_frequency_domain_features(self, extractor, sample_signal):
        """Test frequency domain features - covers lines 500-517"""
        fs = 500
        
        features = extractor.extract_frequency_domain_features(sample_signal, fs)
        
        assert isinstance(features, dict)
    
    def test_extract_morphological_features(self, extractor, sample_signal):
        """Test morphological features - covers lines 519-546"""
        fs = 500
        
        features = extractor.extract_morphological_features(sample_signal, fs)
        
        assert isinstance(features, dict)
