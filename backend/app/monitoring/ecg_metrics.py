"""
ECG-specific metrics collection for Prometheus monitoring
"""

from prometheus_client import Counter, Histogram, Gauge, Summary, CollectorRegistry, REGISTRY
from typing import Dict, Any, Optional
import time
from contextlib import contextmanager


class ECGMetricsCollector:
    """Collector for ECG analysis specific metrics"""
    
    def __init__(self, registry: Optional[CollectorRegistry] = None):
        if registry is None:
            registry = REGISTRY
        
        self.ecg_analysis_total = Counter(
            'ecg_analysis_total',
            'Total de análises ECG realizadas',
            ['format', 'status', 'regulatory_standard'],
            registry=registry
        )
        
        self.ecg_processing_duration = Histogram(
            'ecg_processing_duration_seconds',
            'Tempo de processamento ECG por etapa',
            ['step', 'model_type'],
            buckets=(0.1, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0),
            registry=registry
        )
        
        self.ecg_quality_score = Gauge(
            'ecg_quality_score',
            'Score de qualidade do sinal ECG',
            ['lead', 'patient_id'],
            registry=registry
        )
        
        self.pathology_detections = Counter(
            'ecg_pathology_detections_total',
            'Detecções por tipo de patologia',
            ['pathology_type', 'confidence_level', 'model_version'],
            registry=registry
        )
        
        self.model_inference_time = Summary(
            'ecg_model_inference_seconds',
            'Tempo de inferência por modelo ML',
            ['model_name', 'model_version'],
            registry=registry
        )
        
        self.model_memory_usage = Gauge(
            'ecg_model_memory_usage_bytes',
            'Uso de memória por modelo ML',
            ['model_name', 'model_type'],
            registry=registry
        )
        
        self.regulatory_compliance = Counter(
            'ecg_regulatory_compliance_total',
            'Resultados de conformidade regulatória',
            ['standard', 'compliant', 'validation_type'],
            registry=registry
        )
        
        self.prediction_confidence = Histogram(
            'ecg_prediction_confidence',
            'Distribuição de confiança das predições',
            ['pathology_type', 'model_name'],
            buckets=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 1.0),
            registry=registry
        )
        
        self.processing_errors = Counter(
            'ecg_processing_errors_total',
            'Erros durante processamento ECG',
            ['error_type', 'step', 'format'],
            registry=registry
        )

    @contextmanager
    def time_operation(self, step: str, model_type: str = "hybrid"):
        """Context manager para medir tempo de operações"""
        start_time = time.time()
        try:
            yield
        finally:
            duration = time.time() - start_time
            self.ecg_processing_duration.labels(
                step=step,
                model_type=model_type
            ).observe(duration)

    def record_analysis(self, format: str, status: str, regulatory_standard: str):
        """Registra uma análise ECG completa"""
        self.ecg_analysis_total.labels(
            format=format,
            status=status,
            regulatory_standard=regulatory_standard
        ).inc()

    def record_pathology_detection(
        self, 
        pathology_type: str, 
        confidence: float, 
        model_version: str
    ):
        """Registra detecção de patologia"""
        confidence_level = self._get_confidence_level(confidence)
        self.pathology_detections.labels(
            pathology_type=pathology_type,
            confidence_level=confidence_level,
            model_version=model_version
        ).inc()
        
        self.prediction_confidence.labels(
            pathology_type=pathology_type,
            model_name=model_version
        ).observe(confidence)

    def record_quality_score(self, lead: str, patient_id: str, score: float):
        """Registra score de qualidade do sinal"""
        self.ecg_quality_score.labels(
            lead=lead,
            patient_id=patient_id
        ).set(score)

    def record_model_inference(self, model_name: str, model_version: str, duration: float):
        """Registra tempo de inferência do modelo"""
        self.model_inference_time.labels(
            model_name=model_name,
            model_version=model_version
        ).observe(duration)

    def record_model_memory(self, model_name: str, model_type: str, memory_bytes: int):
        """Registra uso de memória do modelo"""
        self.model_memory_usage.labels(
            model_name=model_name,
            model_type=model_type
        ).set(memory_bytes)

    def record_regulatory_compliance(
        self, 
        standard: str, 
        compliant: bool, 
        validation_type: str
    ):
        """Registra resultado de conformidade regulatória"""
        self.regulatory_compliance.labels(
            standard=standard,
            compliant=str(compliant).lower(),
            validation_type=validation_type
        ).inc()

    def record_processing_error(self, error_type: str, step: str, format: str):
        """Registra erro de processamento"""
        self.processing_errors.labels(
            error_type=error_type,
            step=step,
            format=format
        ).inc()

    def _get_confidence_level(self, confidence: float) -> str:
        """Converte confiança numérica em categoria"""
        if confidence >= 0.9:
            return "high"
        elif confidence >= 0.7:
            return "medium"
        elif confidence >= 0.5:
            return "low"
        else:
            return "very_low"


ecg_metrics = ECGMetricsCollector()
