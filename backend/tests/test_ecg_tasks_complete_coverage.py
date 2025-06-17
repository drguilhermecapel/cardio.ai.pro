"""
Testes completos para as tasks assíncronas de análise de ECG.
Garante cobertura de 100% dos cenários críticos de processamento assíncrono.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock, AsyncMock
from datetime import datetime, timedelta
import json
import numpy as np
from io import BytesIO
import asyncio
from typing import Dict, Any, List

# Import correto do Celery
import celery
from celery import exceptions
Retry = exceptions.Retry

# Imports do projeto
from app.tasks.ecg_tasks import (
    process_ecg_async,
    process_batch_ecgs,
    generate_report_async,
    cleanup_old_analyses,
    monitor_processing_queue,
    retry_failed_analyses,
    schedule_periodic_cleanup,
    send_analysis_notification,
    validate_ecg_data_async,
    compress_ecg_storage
)
from app.models.ecg import ECGAnalysis, ECGFile, ProcessingStatus
from app.core.exceptions import (
    ECGProcessingError,
    InvalidECGDataError,
    ModelNotFoundError
)


class TestECGTasksCompleteCoverage:
    """Suite completa de testes para tasks assíncronas."""
    
    @pytest.fixture
    def mock_ecg_data(self):
        """Dados de ECG mock para testes."""
        return {
            "patient_id": "TEST001",
            "data": np.random.randn(12, 5000).tolist(),
            "sampling_rate": 500,
            "leads": ["I", "II", "III", "aVR", "aVL", "aVF", 
                     "V1", "V2", "V3", "V4", "V5", "V6"],
            "metadata": {
                "device": "TEST_DEVICE",
                "timestamp": datetime.utcnow().isoformat()
            }
        }
    
    @pytest.fixture
    def mock_celery_task(self):
        """Mock para task do Celery."""
        task = Mock()
        task.request = Mock()
        task.request.id = "test-task-id"
        task.request.retries = 0
        task.max_retries = 3
        task.retry = Mock(side_effect=Retry("Retry requested"))
        return task
    
    @pytest.mark.asyncio
    async def test_process_ecg_async_success(self, mock_ecg_data, mock_celery_task):
        """Testa processamento assíncrono bem-sucedido de ECG."""
        with patch('app.tasks.ecg_tasks.ECGAnalysisService') as mock_service:
            # Configurar mock do serviço
            mock_analysis = Mock()
            mock_analysis.id = "analysis-123"
            mock_analysis.status = ProcessingStatus.COMPLETED
            mock_analysis.results = {
                "heart_rate": 72,
                "rhythm": "normal_sinus",
                "abnormalities": []
            }
            
            mock_service_instance = mock_service.return_value
            mock_service_instance.process_ecg = AsyncMock(return_value=mock_analysis)
            
            # Executar task
            result = await process_ecg_async(
                ecg_data=mock_ecg_data,
                priority="high",
                task=mock_celery_task
            )
            
            # Verificações
            assert result["analysis_id"] == "analysis-123"
            assert result["status"] == "completed"
            assert "results" in result
            mock_service_instance.process_ecg.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_process_ecg_async_with_retry(self, mock_ecg_data, mock_celery_task):
        """Testa retry em caso de erro temporário."""
        mock_celery_task.request.retries = 1
        
        with patch('app.tasks.ecg_tasks.ECGAnalysisService') as mock_service:
            # Simular erro temporário
            mock_service_instance = mock_service.return_value
            mock_service_instance.process_ecg = AsyncMock(
                side_effect=ECGProcessingError("Temporary error")
            )
            
            # Deve lançar Retry
            with pytest.raises(Retry):
                await process_ecg_async(
                    ecg_data=mock_ecg_data,
                    priority="normal",
                    task=mock_celery_task
                )
            
            mock_celery_task.retry.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_process_ecg_async_max_retries_exceeded(self, mock_ecg_data, mock_celery_task):
        """Testa falha após exceder máximo de retries."""
        mock_celery_task.request.retries = 3  # Máximo atingido
        
        with patch('app.tasks.ecg_tasks.ECGAnalysisService') as mock_service:
            mock_service_instance = mock_service.return_value
            mock_service_instance.process_ecg = AsyncMock(
                side_effect=ECGProcessingError("Persistent error")
            )
            
            with patch('app.tasks.ecg_tasks.save_failed_analysis') as mock_save:
                result = await process_ecg_async(
                    ecg_data=mock_ecg_data,
                    priority="normal",
                    task=mock_celery_task
                )
                
                assert result["status"] == "failed"
                assert "error" in result
                mock_save.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_process_batch_ecgs(self, mock_ecg_data, mock_celery_task):
        """Testa processamento em lote de ECGs."""
        batch_data = [mock_ecg_data.copy() for _ in range(5)]
        
        with patch('app.tasks.ecg_tasks.process_ecg_async.delay') as mock_delay:
            mock_delay.return_value = Mock(id="task-id")
            
            results = await process_batch_ecgs(
                ecg_batch=batch_data,
                priority="normal",
                task=mock_celery_task
            )
            
            assert len(results) == 5
            assert all("task_id" in r for r in results)
            assert mock_delay.call_count == 5
    
    @pytest.mark.asyncio
    async def test_generate_report_async(self, mock_celery_task):
        """Testa geração assíncrona de relatório."""
        analysis_id = "analysis-123"
        
        with patch('app.tasks.ecg_tasks.ReportService') as mock_report_service:
            mock_report = {
                "id": "report-123",
                "analysis_id": analysis_id,
                "format": "pdf",
                "url": "https://example.com/report.pdf"
            }
            
            mock_service_instance = mock_report_service.return_value
            mock_service_instance.generate_report = AsyncMock(return_value=mock_report)
            
            result = await generate_report_async(
                analysis_id=analysis_id,
                format="pdf",
                include_raw_data=True,
                task=mock_celery_task
            )
            
            assert result["report_id"] == "report-123"
            assert result["format"] == "pdf"
            assert "url" in result
    
    @pytest.mark.asyncio
    async def test_cleanup_old_analyses(self, mock_celery_task):
        """Testa limpeza de análises antigas."""
        with patch('app.tasks.ecg_tasks.db_session') as mock_db:
            # Mock de análises antigas
            old_analyses = [
                Mock(id=f"old-{i}", created_at=datetime.utcnow() - timedelta(days=40))
                for i in range(10)
            ]
            
            mock_query = Mock()
            mock_query.filter.return_value.filter.return_value.all.return_value = old_analyses
            mock_db.query.return_value = mock_query
            
            result = await cleanup_old_analyses(
                days_to_keep=30,
                batch_size=5,
                task=mock_celery_task
            )
            
            assert result["deleted_count"] == 10
            assert result["status"] == "completed"
    
    @pytest.mark.asyncio
    async def test_monitor_processing_queue(self, mock_celery_task):
        """Testa monitoramento da fila de processamento."""
        with patch('app.tasks.ecg_tasks.get_queue_stats') as mock_stats:
            mock_stats.return_value = {
                "pending": 15,
                "processing": 5,
                "completed": 100,
                "failed": 2
            }
            
            with patch('app.tasks.ecg_tasks.send_alert') as mock_alert:
                result = await monitor_processing_queue(
                    alert_threshold=10,
                    task=mock_celery_task
                )
                
                assert result["queue_stats"]["pending"] == 15
                assert result["alert_sent"] is True
                mock_alert.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_retry_failed_analyses(self, mock_celery_task):
        """Testa reprocessamento de análises falhas."""
        with patch('app.tasks.ecg_tasks.db_session') as mock_db:
            # Mock de análises falhas
            failed_analyses = [
                Mock(
                    id=f"failed-{i}",
                    status=ProcessingStatus.FAILED,
                    retry_count=1,
                    ecg_data={"test": "data"}
                )
                for i in range(3)
            ]
            
            mock_query = Mock()
            mock_query.filter.return_value.filter.return_value.limit.return_value.all.return_value = failed_analyses
            mock_db.query.return_value = mock_query
            
            with patch('app.tasks.ecg_tasks.process_ecg_async.delay') as mock_delay:
                result = await retry_failed_analyses(
                    max_retries=3,
                    limit=10,
                    task=mock_celery_task
                )
                
                assert result["retried_count"] == 3
                assert mock_delay.call_count == 3
    
    @pytest.mark.asyncio
    async def test_send_analysis_notification(self, mock_celery_task):
        """Testa envio de notificações de análise."""
        with patch('app.tasks.ecg_tasks.NotificationService') as mock_notif:
            mock_service = mock_notif.return_value
            mock_service.send_notification = AsyncMock(return_value={"sent": True})
            
            result = await send_analysis_notification(
                analysis_id="analysis-123",
                user_id="user-456",
                notification_type="completed",
                task=mock_celery_task
            )
            
            assert result["sent"] is True
            mock_service.send_notification.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_validate_ecg_data_async(self, mock_ecg_data, mock_celery_task):
        """Testa validação assíncrona de dados ECG."""
        with patch('app.tasks.ecg_tasks.ECGValidator') as mock_validator:
            mock_validator_instance = mock_validator.return_value
            mock_validator_instance.validate = Mock(return_value={
                "valid": True,
                "errors": [],
                "warnings": ["Low signal quality in lead V6"]
            })
            
            result = await validate_ecg_data_async(
                ecg_data=mock_ecg_data,
                strict_mode=True,
                task=mock_celery_task
            )
            
            assert result["valid"] is True
            assert len(result["warnings"]) == 1
    
    @pytest.mark.asyncio
    async def test_compress_ecg_storage(self, mock_celery_task):
        """Testa compressão de armazenamento de ECG."""
        with patch('app.tasks.ecg_tasks.StorageService') as mock_storage:
            mock_service = mock_storage.return_value
            mock_service.compress_old_files = AsyncMock(return_value={
                "compressed_count": 50,
                "space_saved_mb": 1024
            })
            
            result = await compress_ecg_storage(
                days_old=7,
                compression_level=6,
                task=mock_celery_task
            )
            
            assert result["compressed_count"] == 50
            assert result["space_saved_mb"] == 1024
    
    @pytest.mark.asyncio
    async def test_task_with_database_error(self, mock_ecg_data, mock_celery_task):
        """Testa tratamento de erro de banco de dados."""
        with patch('app.tasks.ecg_tasks.db_session') as mock_db:
            mock_db.commit.side_effect = Exception("Database error")
            
            with pytest.raises(Retry):
                await process_ecg_async(
                    ecg_data=mock_ecg_data,
                    priority="normal",
                    task=mock_celery_task
                )
    
    @pytest.mark.asyncio
    async def test_task_with_invalid_data(self, mock_celery_task):
        """Testa tratamento de dados inválidos."""
        invalid_data = {"invalid": "data"}
        
        result = await process_ecg_async(
            ecg_data=invalid_data,
            priority="normal",
            task=mock_celery_task
        )
        
        assert result["status"] == "failed"
        assert "validation" in result["error"].lower()
    
    @pytest.mark.asyncio
    async def test_schedule_periodic_cleanup(self, mock_celery_task):
        """Testa agendamento de limpeza periódica."""
        with patch('app.tasks.ecg_tasks.schedule_task') as mock_schedule:
            mock_schedule.return_value = {"scheduled": True}
            
            result = await schedule_periodic_cleanup(
                interval_hours=24,
                days_to_keep=30,
                task=mock_celery_task
            )
            
            assert result["scheduled"] is True
            mock_schedule.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_task_timeout_handling(self, mock_ecg_data, mock_celery_task):
        """Testa tratamento de timeout em tasks."""
        with patch('app.tasks.ecg_tasks.ECGAnalysisService') as mock_service:
            mock_service_instance = mock_service.return_value
            mock_service_instance.process_ecg = AsyncMock(
                side_effect=asyncio.TimeoutError()
            )
            
            with pytest.raises(Retry):
                await process_ecg_async(
                    ecg_data=mock_ecg_data,
                    priority="high",
                    task=mock_celery_task
                )
    
    @pytest.mark.asyncio
    async def test_batch_processing_partial_failure(self, mock_ecg_data, mock_celery_task):
        """Testa processamento em lote com falhas parciais."""
        batch_data = [mock_ecg_data.copy() for _ in range(5)]
        
        with patch('app.tasks.ecg_tasks.process_ecg_async.delay') as mock_delay:
            # Simular sucesso e falha alternados
            mock_delay.side_effect = [
                Mock(id="task-1"),
                Exception("Processing error"),
                Mock(id="task-3"),
                Exception("Processing error"),
                Mock(id="task-5")
            ]
            
            results = await process_batch_ecgs(
                ecg_batch=batch_data,
                priority="normal",
                continue_on_error=True,
                task=mock_celery_task
            )
            
            assert len(results) == 5
            assert sum(1 for r in results if "error" in r) == 2
            assert sum(1 for r in results if "task_id" in r) == 3
    
    @pytest.mark.asyncio
    async def test_priority_queue_handling(self, mock_ecg_data, mock_celery_task):
        """Testa tratamento de filas por prioridade."""
        priorities = ["low", "normal", "high", "urgent"]
        
        with patch('app.tasks.ecg_tasks.route_to_queue') as mock_route:
            for priority in priorities:
                await process_ecg_async(
                    ecg_data=mock_ecg_data,
                    priority=priority,
                    task=mock_celery_task
                )
                
                mock_route.assert_called_with(priority)
    
    @pytest.mark.asyncio
    async def test_memory_optimization_during_processing(self, mock_ecg_data, mock_celery_task):
        """Testa otimização de memória durante processamento."""
        with patch('app.tasks.ecg_tasks.MemoryMonitor') as mock_monitor:
            monitor_instance = mock_monitor.return_value
            monitor_instance.get_memory_usage.return_value = {"rss_mb": 100}
            monitor_instance.optimize_memory.return_value = {"freed_mb": 20}
            
            with patch('app.tasks.ecg_tasks.ECGAnalysisService'):
                await process_ecg_async(
                    ecg_data=mock_ecg_data,
                    priority="normal",
                    optimize_memory=True,
                    task=mock_celery_task
                )
                
                monitor_instance.optimize_memory.assert_called()
    
    @pytest.mark.asyncio
    async def test_distributed_processing(self, mock_ecg_data, mock_celery_task):
        """Testa processamento distribuído em múltiplos workers."""
        with patch('app.tasks.ecg_tasks.get_worker_info') as mock_worker:
            mock_worker.return_value = {
                "worker_id": "worker-001",
                "hostname": "processing-node-1"
            }
            
            with patch('app.tasks.ecg_tasks.ECGAnalysisService'):
                result = await process_ecg_async(
                    ecg_data=mock_ecg_data,
                    priority="normal",
                    task=mock_celery_task
                )
                
                assert "worker_info" in result
                assert result["worker_info"]["worker_id"] == "worker-001"
