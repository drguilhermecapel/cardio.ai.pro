"""
Tests for monitoring components
"""

import pytest
from unittest.mock import Mock, patch, AsyncMock
from prometheus_client import CollectorRegistry
from fastapi import HTTPException
from starlette.requests import Request
from starlette.responses import Response
from app.monitoring.ecg_metrics import ECGMetricsCollector
from app.monitoring.structured_logging import get_ecg_logger, setup_structured_logging
from app.monitoring.health_checks import (
    check_ml_models, test_ecg_processing, check_regulatory_services,
    check_filesystem, check_system_resources, check_network_connectivity,
    detailed_health_check, health_metrics, readiness_check, liveness_check
)
from app.monitoring.middleware import ECGMetricsMiddleware


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
    
    def test_record_quality_score(self):
        """Test recording quality score"""
        registry = CollectorRegistry()
        collector = ECGMetricsCollector(registry=registry)
        
        collector.record_quality_score("lead_I", "patient_123", 0.95)
        collector.record_quality_score("lead_II", "patient_456", 0.87)
    
    def test_record_model_inference(self):
        """Test recording model inference time"""
        registry = CollectorRegistry()
        collector = ECGMetricsCollector(registry=registry)
        
        collector.record_model_inference("cnn_model", "v1.0", 0.25)
        collector.record_model_inference("transformer", "v2.1", 0.45)
    
    def test_record_model_memory(self):
        """Test recording model memory usage"""
        registry = CollectorRegistry()
        collector = ECGMetricsCollector(registry=registry)
        
        collector.record_model_memory("cnn_model", "tensorflow", 512000000)
        collector.record_model_memory("transformer", "pytorch", 768000000)
    
    def test_record_regulatory_compliance(self):
        """Test recording regulatory compliance"""
        registry = CollectorRegistry()
        collector = ECGMetricsCollector(registry=registry)
        
        collector.record_regulatory_compliance("FDA", True, "full_validation")
        collector.record_regulatory_compliance("EU_MDR", False, "partial_validation")
    
    def test_record_processing_error(self):
        """Test recording processing errors"""
        registry = CollectorRegistry()
        collector = ECGMetricsCollector(registry=registry)
        
        collector.record_processing_error("ValueError", "preprocessing", "csv")
        collector.record_processing_error("TimeoutError", "inference", "edf")


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
    
    def test_ecg_logger_additional_methods(self):
        """Test additional ECG logger methods"""
        logger = get_ecg_logger("test_module")
        
        logger.log_regulatory_validation("analysis_123", "FDA", True, {"score": 0.95})
        logger.log_model_performance("cnn_model", "v1.0", 0.25, 512.0, 32)
        logger.log_signal_quality("patient_123", "analysis_456", {"I": 0.95, "II": 0.87}, 0.91, ["noise"])
        logger.log_preprocessing_step("analysis_123", "filter", (1000, 12), (1000, 12), {"cutoff": 40}, 0.1)
        logger.log_feature_extraction("analysis_123", ["rr_intervals", "qrs_width"], 150, 0.05)


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
    
    async def test_check_regulatory_services(self):
        """Test regulatory services health check"""
        result = await check_regulatory_services()
        
        assert "standards" in result
        assert "services" in result
        assert isinstance(result["standards"], list)
        assert isinstance(result["services"], dict)
        assert "FDA" in result["standards"]
        assert "validation_engine" in result["services"]
    
    async def test_check_filesystem(self):
        """Test filesystem health check"""
        result = await check_filesystem()
        
        assert "disk_usage" in result
        assert "available_gb" in result
        assert "total_gb" in result
        assert isinstance(result["disk_usage"], (int, float))
        assert isinstance(result["available_gb"], (int, float))
        assert isinstance(result["total_gb"], (int, float))
    
    async def test_check_system_resources(self):
        """Test system resources health check"""
        result = await check_system_resources()
        
        assert "memory_percent" in result
        assert "cpu_percent" in result
        assert "available_memory_gb" in result
        assert "total_memory_gb" in result
        assert isinstance(result["memory_percent"], (int, float))
        assert isinstance(result["cpu_percent"], (int, float))
    
    async def test_check_network_connectivity(self):
        """Test network connectivity health check"""
        result = await check_network_connectivity()
        
        assert "external" in result
        assert "dns" in result
        assert isinstance(result["external"], bool)
        assert isinstance(result["dns"], bool)
    
    async def test_detailed_health_check(self):
        """Test detailed health check endpoint"""
        result = await detailed_health_check()
        
        assert "status" in result
        assert "timestamp" in result
        assert "check_duration_seconds" in result
        assert "checks" in result
        assert "summary" in result
        
        assert "ml_models" in result["checks"]
        assert "ecg_processing" in result["checks"]
        assert "regulatory" in result["checks"]
        assert "filesystem" in result["checks"]
        assert "system_resources" in result["checks"]
        assert "network" in result["checks"]
        
        assert "total_checks" in result["summary"]
        assert "healthy" in result["summary"]
        assert "warning" in result["summary"]
        assert "unhealthy" in result["summary"]
    
    async def test_health_metrics(self):
        """Test health metrics endpoint"""
        result = await health_metrics()
        
        assert "timestamp" in result
        assert "system" in result
        assert "ecg_system" in result
        
        assert "memory_usage_percent" in result["system"]
        assert "disk_usage_percent" in result["system"]
        assert "cpu_usage_percent" in result["system"]
        assert "available_memory_gb" in result["system"]
        
        assert "models_loaded" in result["ecg_system"]
        assert "analyses_today" in result["ecg_system"]
        assert "avg_processing_time" in result["ecg_system"]
    
    async def test_readiness_check(self):
        """Test readiness check endpoint"""
        result = await readiness_check()
        
        assert "status" in result
        assert result["status"] == "ready"
    
    async def test_liveness_check(self):
        """Test liveness check endpoint"""
        result = await liveness_check()
        
        assert "status" in result
        assert "timestamp" in result
        assert result["status"] == "alive"
    
    @patch('app.monitoring.health_checks.check_ml_models')
    async def test_detailed_health_check_with_ml_error(self, mock_check_ml):
        """Test detailed health check with ML models error"""
        mock_check_ml.side_effect = Exception("ML models unavailable")
        
        result = await detailed_health_check()
        
        assert result["checks"]["ml_models"]["status"] == "unhealthy"
        assert "ML models unavailable" in result["checks"]["ml_models"]["error"]
    
    @patch('app.monitoring.health_checks.test_ecg_processing')
    async def test_detailed_health_check_with_ecg_error(self, mock_ecg_test):
        """Test detailed health check with ECG processing error"""
        mock_ecg_test.side_effect = Exception("ECG processing failed")
        
        result = await detailed_health_check()
        
        assert result["checks"]["ecg_processing"]["status"] == "unhealthy"
        assert "ECG processing failed" in result["checks"]["ecg_processing"]["error"]
    
    async def test_readiness_check_exception(self):
        """Test readiness check with exception"""
        with patch('asyncio.sleep', side_effect=Exception("Service error")):
            with pytest.raises(HTTPException) as exc_info:
                await readiness_check()
            assert exc_info.value.status_code == 503


@pytest.mark.asyncio
class TestECGMetricsMiddleware:
    """Test ECG metrics middleware"""
    
    async def test_middleware_successful_request(self):
        """Test middleware with successful request"""
        middleware = ECGMetricsMiddleware(app=None)
        
        mock_request = Mock(spec=Request)
        mock_request.url.path = "/api/v1/ecg/analyze"
        mock_request.method = "POST"
        
        mock_response = Mock(spec=Response)
        mock_response.status_code = 200
        
        async def mock_call_next(request):
            return mock_response
        
        result = await middleware.dispatch(mock_request, mock_call_next)
        
        assert result == mock_response
    
    async def test_middleware_failed_request(self):
        """Test middleware with failed request"""
        middleware = ECGMetricsMiddleware(app=None)
        
        mock_request = Mock(spec=Request)
        mock_request.url.path = "/api/v1/ecg/analyze"
        mock_request.method = "POST"
        
        async def mock_call_next(request):
            raise ValueError("Processing error")
        
        with pytest.raises(ValueError):
            await middleware.dispatch(mock_request, mock_call_next)
    
    async def test_middleware_timing(self):
        """Test middleware timing functionality"""
        middleware = ECGMetricsMiddleware(app=None)
        
        mock_request = Mock(spec=Request)
        mock_request.url.path = "/api/v1/test"
        mock_request.method = "GET"
        
        mock_response = Mock(spec=Response)
        mock_response.status_code = 200
        
        async def mock_call_next(request):
            import asyncio
            await asyncio.sleep(0.01)  # Simulate processing time
            return mock_response
        
        result = await middleware.dispatch(mock_request, mock_call_next)
        
        assert result == mock_response
