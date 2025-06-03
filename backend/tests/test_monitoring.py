"""
Tests for monitoring components
"""

import pytest
from unittest.mock import Mock, patch
from prometheus_client import CollectorRegistry
from app.monitoring.ecg_metrics import ECGMetricsCollector
from app.monitoring.structured_logging import get_ecg_logger, setup_structured_logging
from app.monitoring.health_checks import check_ml_models, test_ecg_processing


class TestECGMetricsCollector:
    """Test ECG metrics collector"""
    
    def test_metrics_collector_initialization(self):
        """Test metrics collector initializes correctly"""
        registry = CollectorRegistry()
        collector = ECGMetricsCollector(registry=registry)
        
        assert collector.ecg_analysis_total is not None
        assert collector.ecg_processing_duration is not None
        assert collector.ecg_quality_score is not None
        assert collector.pathology_detections is not None
        assert collector.model_inference_time is not None
        assert collector.model_memory_usage is not None
        assert collector.regulatory_compliance is not None
        assert collector.prediction_confidence is not None
        assert collector.processing_errors is not None
    
    def test_record_analysis(self):
        """Test recording analysis metrics"""
        registry = CollectorRegistry()
        collector = ECGMetricsCollector(registry=registry)
        
        collector.record_analysis("csv", "success", "FDA")
        collector.record_analysis("edf", "error", "EU_MDR")
    
    def test_record_pathology_detection(self):
        """Test recording pathology detection metrics"""
        registry = CollectorRegistry()
        collector = ECGMetricsCollector(registry=registry)
        
        collector.record_pathology_detection("atrial_fibrillation", 0.85, "hybrid_v1.0")
        collector.record_pathology_detection("long_qt", 0.65, "hybrid_v1.0")
    
    def test_confidence_level_categorization(self):
        """Test confidence level categorization"""
        registry = CollectorRegistry()
        collector = ECGMetricsCollector(registry=registry)
        
        assert collector._get_confidence_level(0.95) == "high"
        assert collector._get_confidence_level(0.75) == "medium"
        assert collector._get_confidence_level(0.55) == "low"
        assert collector._get_confidence_level(0.35) == "very_low"
    
    def test_time_operation_context_manager(self):
        """Test timing context manager"""
        registry = CollectorRegistry()
        collector = ECGMetricsCollector(registry=registry)
        
        with collector.time_operation("test_step", "test_model"):
            pass


class TestStructuredLogging:
    """Test structured logging"""
    
    def test_setup_structured_logging(self):
        """Test structured logging setup"""
        setup_structured_logging("DEBUG")
        setup_structured_logging("INFO")
    
    def test_ecg_logger_creation(self):
        """Test ECG logger creation"""
        logger = get_ecg_logger("test_module")
        assert logger is not None
        assert hasattr(logger, 'log_analysis_start')
        assert hasattr(logger, 'log_analysis_complete')
        assert hasattr(logger, 'log_analysis_error')
        assert hasattr(logger, 'log_pathology_detection')
    
    def test_ecg_logger_methods(self):
        """Test ECG logger methods don't raise exceptions"""
        logger = get_ecg_logger("test_module")
        
        logger.log_analysis_start("patient_123", "csv", 12, 500, "analysis_456")
        logger.log_analysis_complete("patient_123", "analysis_456", 2, 0.85, 2.5, True)
        logger.log_analysis_error("patient_123", "analysis_456", "TestError", "Test message", "test_step")
        logger.log_pathology_detection("patient_123", "analysis_456", "atrial_fibrillation", 0.85, "hybrid_v1.0", "high")


@pytest.mark.asyncio
class TestHealthChecks:
    """Test health check functions"""
    
    async def test_check_ml_models(self):
        """Test ML models health check"""
        result = await check_ml_models()
        
        assert "count" in result
        assert "memory_mb" in result
        assert "models" in result
        assert isinstance(result["count"], int)
        assert isinstance(result["memory_mb"], (int, float))
        assert isinstance(result["models"], dict)
    
    async def test_ecg_processing_test(self):
        """Test ECG processing health check"""
        result = await test_ecg_processing()
        
        assert "duration_ms" in result
        assert "result" in result
        assert isinstance(result["duration_ms"], (int, float))
        assert isinstance(result["result"], dict)
        
        assert "pathologies_detected" in result["result"]
        assert "confidence" in result["result"]
        assert "signal_quality" in result["result"]
