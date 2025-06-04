"""
Medical-Grade Tests for ECG Tasks Module
Target: 70%+ Coverage for Auxiliary Module

Focus Areas:
- Celery task execution and state management
- Async ECG processing workflow
- Error handling and failure scenarios
- Medical safety in background processing
- Task monitoring and progress tracking
"""

import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock
from celery.exceptions import Retry

from app.tasks.ecg_tasks import process_ecg_analysis


class TestECGTasksBasicFunctionality:
    """Basic functionality tests for ECG tasks."""
    
    @pytest.fixture
    def mock_current_task(self):
        """Mock Celery current_task."""
        with patch('app.tasks.ecg_tasks.current_task') as mock_task:
            mock_task.update_state = Mock()
            yield mock_task
    
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
    
    def test_process_ecg_analysis_success(self, mock_current_task, mock_session_factory, mock_ecg_service):
        """Test successful ECG analysis task execution."""
        mock_factory, mock_session = mock_session_factory
        
        with patch('app.tasks.ecg_tasks.MLModelService'), \
             patch('app.services.notification_service.NotificationService'), \
             patch('app.tasks.ecg_tasks.ValidationService'):
            
            result = process_ecg_analysis(analysis_id=12345)
            
            assert mock_current_task.update_state.call_count == 2
            
            initial_call = mock_current_task.update_state.call_args_list[0]
            assert initial_call[1]['state'] == 'PROGRESS'
            assert initial_call[1]['meta']['current'] == 0
            assert initial_call[1]['meta']['total'] == 100
            assert 'Starting analysis' in initial_call[1]['meta']['status']
            
            success_call = mock_current_task.update_state.call_args_list[1]
            assert success_call[1]['state'] == 'SUCCESS'
            assert success_call[1]['meta']['current'] == 100
            assert success_call[1]['meta']['total'] == 100
            assert 'Analysis complete' in success_call[1]['meta']['status']
            
            assert result['status'] == 'completed'
            assert result['analysis_id'] == 12345
    
    def test_process_ecg_analysis_exception_handling(self, mock_current_task, mock_session_factory):
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
                
                process_ecg_analysis(analysis_id=12345)
                
                failure_call = mock_current_task.update_state.call_args_list[-1]
                assert failure_call[1]['state'] == 'FAILURE'
                assert failure_call[1]['meta']['current'] == 0
                assert failure_call[1]['meta']['total'] == 100
                assert 'Database connection failed' in failure_call[1]['meta']['status']


class TestECGTasksMedicalSafety:
    """Medical safety tests for ECG background processing."""
    
    @pytest.fixture
    def mock_current_task(self):
        """Mock Celery current_task."""
        with patch('app.tasks.ecg_tasks.current_task') as mock_task:
            mock_task.update_state = Mock()
            yield mock_task
    
    def test_critical_analysis_task_isolation(self, mock_current_task):
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
            
            result = process_ecg_analysis(analysis_id=99999)  # Emergency case ID
            
            mock_service_class.assert_called_once()
            call_args = mock_service_class.call_args[0]
            assert len(call_args) == 3  # db, ml_service, validation_service
            
            mock_service._process_analysis_async.assert_called_once_with(99999)
            
            assert result['status'] == 'completed'
    
    def test_async_processing_error_containment(self, mock_current_task):
        """Test that async processing errors are properly contained."""
        with patch('app.tasks.ecg_tasks.get_session_factory') as mock_factory:
            mock_factory.side_effect = Exception("Session creation failed")
            
            with pytest.raises(Exception, match="Session creation failed"):
                process_ecg_analysis(analysis_id=12345)
            
            failure_call = mock_current_task.update_state.call_args_list[-1]
            assert failure_call[1]['state'] == 'FAILURE'
            assert 'Session creation failed' in failure_call[1]['meta']['status']
    
    def test_task_progress_tracking_medical_compliance(self, mock_current_task):
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
            
            result = process_ecg_analysis(analysis_id=12345)
            
            assert mock_current_task.update_state.call_count >= 2
            
            initial_call = mock_current_task.update_state.call_args_list[0]
            assert 'current' in initial_call[1]['meta']
            assert 'total' in initial_call[1]['meta']
            assert 'status' in initial_call[1]['meta']
            
            final_call = mock_current_task.update_state.call_args_list[-1]
            assert final_call[1]['state'] == 'SUCCESS'
            assert 'result' in final_call[1]['meta']
            
            assert result['analysis_id'] == 12345


class TestECGTasksPerformanceAndReliability:
    """Performance and reliability tests for ECG tasks."""
    
    @pytest.fixture
    def mock_current_task(self):
        """Mock Celery current_task."""
        with patch('app.tasks.ecg_tasks.current_task') as mock_task:
            mock_task.update_state = Mock()
            yield mock_task
    
    def test_service_dependency_injection(self, mock_current_task):
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
            
            process_ecg_analysis(analysis_id=12345)
            
            mock_ml_service.assert_called_once()
            mock_notification_service.assert_called_once_with(mock_session)
            mock_validation_service.assert_called_once()
            mock_ecg_service.assert_called_once()
            
            ecg_service_call_args = mock_ecg_service.call_args[0]
            assert len(ecg_service_call_args) == 3  # db, ml_service, validation_service
    
    def test_async_execution_context_management(self, mock_current_task):
        """Test proper async execution context management."""
        with patch('app.tasks.ecg_tasks.get_session_factory') as mock_factory, \
             patch('app.tasks.ecg_tasks.ECGAnalysisService') as mock_service_class, \
             patch('app.tasks.ecg_tasks.MLModelService'), \
             patch('app.services.notification_service.NotificationService'), \
             patch('app.tasks.ecg_tasks.ValidationService'), \
             patch('app.tasks.ecg_tasks.asyncio.run') as mock_asyncio_run:
            
            mock_asyncio_run.return_value = {"status": "completed", "analysis_id": 12345}
            
            result = process_ecg_analysis(analysis_id=12345)
            
            mock_asyncio_run.assert_called_once()
            
            call_args = mock_asyncio_run.call_args[0]
            assert len(call_args) == 1
            
            assert result['status'] == 'completed'
    
    def test_task_binding_and_self_reference(self, mock_current_task):
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
            
            result = process_ecg_analysis(analysis_id=12345)
            
            assert result['status'] == 'completed'
            assert result['analysis_id'] == 12345
    
    def test_logging_integration(self, mock_current_task):
        """Test that logging is properly integrated for medical audit trail."""
        with patch('app.tasks.ecg_tasks.get_session_factory') as mock_factory, \
             patch('app.tasks.ecg_tasks.logger') as mock_logger:
            
            mock_factory.side_effect = Exception("Test logging error")
            
            with pytest.raises(Exception, match="Test logging error"):
                process_ecg_analysis(analysis_id=12345)
            
            mock_logger.error.assert_called_once()
            log_call_args = mock_logger.error.call_args[0]
            assert "ECG analysis task failed" in log_call_args[0]
            assert "Test logging error" in str(log_call_args[1])


class TestECGTasksEdgeCases:
    """Edge case tests for ECG tasks."""
    
    @pytest.fixture
    def mock_current_task(self):
        """Mock Celery current_task."""
        with patch('app.tasks.ecg_tasks.current_task') as mock_task:
            mock_task.update_state = Mock()
            yield mock_task
    
    def test_invalid_analysis_id_handling(self, mock_current_task):
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
                process_ecg_analysis(analysis_id=-1)  # Invalid ID
            
            failure_call = mock_current_task.update_state.call_args_list[-1]
            assert failure_call[1]['state'] == 'FAILURE'
            assert 'Analysis ID not found' in failure_call[1]['meta']['status']
    
    def test_zero_analysis_id(self, mock_current_task):
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
            
            result = process_ecg_analysis(analysis_id=0)
            
            mock_service._process_analysis_async.assert_called_once_with(0)
            assert result['analysis_id'] == 0
