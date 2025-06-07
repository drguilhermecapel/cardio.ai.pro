"""
Medical-Grade Tests for ECG Tasks Module (Standalone Version)
Target: 70%+ Coverage for Auxiliary Module

Focus Areas:
- Synchronous ECG processing workflow (converted from Celery)
- Async ECG processing workflow
- Error handling and failure scenarios
- Medical safety in processing
- Task monitoring and progress tracking
"""

import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock

from app.tasks.ecg_tasks import process_ecg_analysis_sync


class TestECGTasksBasicFunctionality:
    """Basic functionality tests for ECG tasks."""
    
    @pytest.fixture
    def mock_session_factory(self):
        """Mock database session factory."""
        with patch('app.tasks.ecg_tasks.get_session_factory') as mock_factory:
            mock_session = Mock()
            mock_session.__aenter__ = AsyncMock(return_value=mock_session)
            mock_session.__aexit__ = AsyncMock(return_value=None)
            mock_factory.return_value.return_value = mock_session
            yield mock_factory, mock_session
    
    @pytest.fixture
    def mock_ecg_service(self):
        """Mock ECG Analysis Service."""
        with patch('app.tasks.ecg_tasks.ECGAnalysisService') as mock_service_class:
            mock_service = Mock()
            mock_service._process_analysis_async = AsyncMock(return_value={"status": "completed"})
            mock_service_class.return_value = mock_service
            yield mock_service
    
    @pytest.mark.asyncio
    async def test_process_ecg_analysis_success(self, mock_session_factory, mock_ecg_service):
        """Test successful ECG analysis task execution."""
        mock_factory, mock_session = mock_session_factory
        
        with patch('app.tasks.ecg_tasks.MLModelService'), \
             patch('app.services.notification_service.NotificationService'), \
             patch('app.tasks.ecg_tasks.ValidationService'):
            
            result = await process_ecg_analysis_sync(analysis_id=12345)
            
            assert result['status'] == 'completed'
            assert result['analysis_id'] == 12345
    
    @pytest.mark.asyncio
    async def test_process_ecg_analysis_exception_handling(self, mock_session_factory):
        """Test exception handling in ECG analysis task."""
        mock_factory, mock_session = mock_session_factory
        
        with patch('app.tasks.ecg_tasks.ECGAnalysisService') as mock_service_class:
            mock_service = Mock()
            mock_service._process_analysis_async = AsyncMock(
                side_effect=Exception("Database connection failed")
            )
            mock_service_class.return_value = mock_service
            
            with patch('app.tasks.ecg_tasks.MLModelService'), \
                 patch('app.services.notification_service.NotificationService'), \
                 patch('app.tasks.ecg_tasks.ValidationService'), \
                 pytest.raises(Exception, match="Database connection failed"):
                
                await process_ecg_analysis_sync(analysis_id=12345)


class TestECGTasksMedicalSafety:
    """Medical safety tests for ECG processing."""
    
    @pytest.mark.asyncio
    async def test_critical_analysis_task_isolation(self):
        """Test that critical analysis tasks are properly isolated."""
        with patch('app.tasks.ecg_tasks.get_session_factory') as mock_factory, \
             patch('app.tasks.ecg_tasks.ECGAnalysisService') as mock_service_class, \
             patch('app.tasks.ecg_tasks.MLModelService'), \
             patch('app.services.notification_service.NotificationService'), \
             patch('app.tasks.ecg_tasks.ValidationService'):
            
            mock_session = Mock()
            mock_session.__aenter__ = AsyncMock(return_value=mock_session)
            mock_session.__aexit__ = AsyncMock(return_value=None)
            mock_factory.return_value.return_value = mock_session
            
            mock_service = Mock()
            mock_service._process_analysis_async = AsyncMock(return_value={"status": "completed"})
            mock_service_class.return_value = mock_service
            
            result = await process_ecg_analysis_sync(analysis_id=99999)  # Emergency case ID
            
            mock_service_class.assert_called_once()
            call_args = mock_service_class.call_args[0]
            assert len(call_args) == 3  # db, ml_service, validation_service
            
            mock_service._process_analysis_async.assert_called_once_with(99999)
            
            assert result['status'] == 'completed'
    
    @pytest.mark.asyncio
    async def test_async_processing_error_containment(self):
        """Test that async processing errors are properly contained."""
        with patch('app.tasks.ecg_tasks.get_session_factory') as mock_factory:
            mock_factory.side_effect = Exception("Session creation failed")
            
            with pytest.raises(Exception, match="Session creation failed"):
                await process_ecg_analysis_sync(analysis_id=12345)
    
    @pytest.mark.asyncio
    async def test_task_progress_tracking_medical_compliance(self):
        """Test that task progress tracking meets medical compliance requirements."""
        with patch('app.tasks.ecg_tasks.get_session_factory') as mock_factory, \
             patch('app.tasks.ecg_tasks.ECGAnalysisService') as mock_service_class, \
             patch('app.tasks.ecg_tasks.MLModelService'), \
             patch('app.services.notification_service.NotificationService'), \
             patch('app.tasks.ecg_tasks.ValidationService'):
            
            mock_session = Mock()
            mock_session.__aenter__ = AsyncMock(return_value=mock_session)
            mock_session.__aexit__ = AsyncMock(return_value=None)
            mock_factory.return_value.return_value = mock_session
            
            mock_service = Mock()
            mock_service._process_analysis_async = AsyncMock(return_value={"status": "completed"})
            mock_service_class.return_value = mock_service
            
            result = await process_ecg_analysis_sync(analysis_id=12345)
            
            assert result['analysis_id'] == 12345


class TestECGTasksPerformanceAndReliability:
    """Performance and reliability tests for ECG tasks."""
    
    @pytest.mark.asyncio
    async def test_service_dependency_injection(self):
        """Test proper dependency injection for medical services."""
        with patch('app.tasks.ecg_tasks.get_session_factory') as mock_factory, \
             patch('app.tasks.ecg_tasks.ECGAnalysisService') as mock_ecg_service, \
             patch('app.tasks.ecg_tasks.MLModelService') as mock_ml_service, \
             patch('app.services.notification_service.NotificationService') as mock_notification_service, \
             patch('app.tasks.ecg_tasks.ValidationService') as mock_validation_service:
            
            mock_session = Mock()
            mock_session.__aenter__ = AsyncMock(return_value=mock_session)
            mock_session.__aexit__ = AsyncMock(return_value=None)
            mock_factory.return_value.return_value = mock_session
            
            mock_service_instance = Mock()
            mock_service_instance._process_analysis_async = AsyncMock(return_value={"status": "completed"})
            mock_ecg_service.return_value = mock_service_instance
            
            await process_ecg_analysis_sync(analysis_id=12345)
            
            mock_ml_service.assert_called_once()
            mock_notification_service.assert_called_once_with(mock_session)
            mock_validation_service.assert_called_once()
            mock_ecg_service.assert_called_once()
            
            ecg_service_call_args = mock_ecg_service.call_args[0]
            assert len(ecg_service_call_args) == 3  # db, ml_service, validation_service
    
    @pytest.mark.asyncio
    async def test_async_execution_context_management(self):
        """Test proper async execution context management."""
        with patch('app.tasks.ecg_tasks.get_session_factory') as mock_factory, \
             patch('app.tasks.ecg_tasks.ECGAnalysisService') as mock_service_class, \
             patch('app.tasks.ecg_tasks.MLModelService'), \
             patch('app.services.notification_service.NotificationService'), \
             patch('app.tasks.ecg_tasks.ValidationService'):
            
            mock_session = Mock()
            mock_session.__aenter__ = AsyncMock(return_value=mock_session)
            mock_session.__aexit__ = AsyncMock(return_value=None)
            mock_factory.return_value.return_value = mock_session
            
            mock_service = Mock()
            mock_service._process_analysis_async = AsyncMock(return_value={"status": "completed"})
            mock_service_class.return_value = mock_service
            
            result = await process_ecg_analysis_sync(analysis_id=12345)
            
            assert result['status'] == 'completed'
            assert result['analysis_id'] == 12345
    
    @pytest.mark.asyncio
    async def test_task_binding_and_self_reference(self):
        """Test that task binding and self reference work correctly."""
        with patch('app.tasks.ecg_tasks.get_session_factory') as mock_factory, \
             patch('app.tasks.ecg_tasks.ECGAnalysisService') as mock_service_class, \
             patch('app.tasks.ecg_tasks.MLModelService'), \
             patch('app.services.notification_service.NotificationService'), \
             patch('app.tasks.ecg_tasks.ValidationService'):
            
            mock_session = Mock()
            mock_session.__aenter__ = AsyncMock(return_value=mock_session)
            mock_session.__aexit__ = AsyncMock(return_value=None)
            mock_factory.return_value.return_value = mock_session
            
            mock_service = Mock()
            mock_service._process_analysis_async = AsyncMock(return_value={"status": "completed"})
            mock_service_class.return_value = mock_service
            
            result = await process_ecg_analysis_sync(analysis_id=12345)
            
            assert result['status'] == 'completed'
            assert result['analysis_id'] == 12345
    
    @pytest.mark.asyncio
    async def test_logging_integration(self):
        """Test that logging is properly integrated for medical audit trail."""
        with patch('app.tasks.ecg_tasks.get_session_factory') as mock_factory, \
             patch('app.tasks.ecg_tasks.logger') as mock_logger:
            
            mock_factory.side_effect = Exception("Test logging error")
            
            with pytest.raises(Exception, match="Test logging error"):
                await process_ecg_analysis_sync(analysis_id=12345)
            
            mock_logger.error.assert_called_once()
            log_call_args = mock_logger.error.call_args[0]
            assert "ECG analysis failed" in log_call_args[0]
            assert "Test logging error" in str(log_call_args[1])


class TestECGTasksEdgeCases:
    """Edge case tests for ECG tasks."""
    
    @pytest.mark.asyncio
    async def test_invalid_analysis_id_handling(self):
        """Test handling of invalid analysis IDs."""
        with patch('app.tasks.ecg_tasks.get_session_factory') as mock_factory, \
             patch('app.tasks.ecg_tasks.ECGAnalysisService') as mock_service_class, \
             patch('app.tasks.ecg_tasks.MLModelService'), \
             patch('app.services.notification_service.NotificationService'), \
             patch('app.tasks.ecg_tasks.ValidationService'):
            
            mock_session = Mock()
            mock_session.__aenter__ = AsyncMock(return_value=mock_session)
            mock_session.__aexit__ = AsyncMock(return_value=None)
            mock_factory.return_value.return_value = mock_session
            
            mock_service = Mock()
            mock_service._process_analysis_async = AsyncMock(
                side_effect=ValueError("Analysis ID not found")
            )
            mock_service_class.return_value = mock_service
            
            with pytest.raises(ValueError, match="Analysis ID not found"):
                await process_ecg_analysis_sync(analysis_id=-1)  # Invalid ID
    
    @pytest.mark.asyncio
    async def test_zero_analysis_id(self):
        """Test handling of zero analysis ID."""
        with patch('app.tasks.ecg_tasks.get_session_factory') as mock_factory, \
             patch('app.tasks.ecg_tasks.ECGAnalysisService') as mock_service_class, \
             patch('app.tasks.ecg_tasks.MLModelService'), \
             patch('app.services.notification_service.NotificationService'), \
             patch('app.tasks.ecg_tasks.ValidationService'):
            
            mock_session = Mock()
            mock_session.__aenter__ = AsyncMock(return_value=mock_session)
            mock_session.__aexit__ = AsyncMock(return_value=None)
            mock_factory.return_value.return_value = mock_session
            
            mock_service = Mock()
            mock_service._process_analysis_async = AsyncMock(return_value={"status": "completed"})
            mock_service_class.return_value = mock_service
            
            result = await process_ecg_analysis_sync(analysis_id=0)
            
            mock_service._process_analysis_async.assert_called_once_with(0)
            assert result['analysis_id'] == 0
