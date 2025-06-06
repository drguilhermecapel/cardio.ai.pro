"""
Phase 2: Validation Service Comprehensive Tests
Target: 70%+ coverage for critical medical services
"""

import pytest
from unittest.mock import Mock, patch, AsyncMock
from app.services.validation_service import ValidationService
from app.core.constants import UserRoles

class TestValidationServiceComprehensive:
    """Comprehensive tests for Validation Service - targeting 70%+ coverage"""
    
    @pytest.fixture
    def validation_service(self):
        """Create validation service with mocked repositories"""
        mock_db = Mock()
        mock_notification = Mock()
        mock_notification.send_validation_assignment = AsyncMock(return_value=None)
        mock_notification.send_validation_completion = AsyncMock(return_value=None)
        return ValidationService(
            db=mock_db,
            notification_service=mock_notification
        )
    
    @pytest.mark.asyncio
    async def test_create_validation_all_types(self, validation_service):
        """Test validation creation - covers lines 39-87"""
        
        validation_service.repository.get_validation_by_analysis = AsyncMock(return_value=None)
        validation_service.repository.get_analysis_by_id = AsyncMock(return_value=Mock(clinical_urgency='LOW'))
        validation_service.repository.create_validation = AsyncMock(return_value=Mock(id=1))
        
        validation = await validation_service.create_validation(
            analysis_id=123,
            validator_id=456,
            validator_role=UserRoles.ADMIN,
            validator_experience_years=5
        )
        assert validation is not None
    
    @pytest.mark.asyncio
    async def test_automated_validation_rules(self, validation_service):
        """Test automated validation - covers lines 96-149"""
        analysis_data = {
            'ecg_features': {
                'hr': 150,  # Tachycardia
                'pr_interval': 250,  # Prolonged
                'qt_interval': 500,  # Prolonged
                'st_elevation': 3.0  # Significant
            },
            'ml_predictions': {
                'arrhythmia': 'atrial_fibrillation',
                'confidence': 0.95
            }
        }
        
        result = validation_service.run_automated_validation_rules(
            analysis_data
        )
        
        assert 'rules_passed' in result
        assert 'overall_score' in result
        assert result['overall_score'] >= 0
        
    @pytest.mark.asyncio
    async def test_validation_workflow_complete(self, validation_service):
        """Test complete validation workflow - covers lines 153-189"""
        from app.core.constants import ValidationStatus
        
        validation_service.repository.get_validation_by_analysis = AsyncMock(return_value=None)
        validation_service.repository.get_analysis_by_id = AsyncMock(return_value=Mock(clinical_urgency='LOW'))
        validation_service.repository.create_validation = AsyncMock(return_value=Mock(id=1))
        
        validation = await validation_service.create_validation(
            analysis_id=126,
            validator_id=789,
            validator_role=UserRoles.ADMIN
        )
        
        validation_service.repository.get_validation_by_id = AsyncMock(return_value=Mock(
            id=validation.id,
            validator_id=789,
            status=ValidationStatus.PENDING,
            analysis_id=126
        ))
        validation_service.repository.update_validation = AsyncMock(return_value=Mock(
            id=validation.id,
            status=ValidationStatus.APPROVED, 
            completed_at='2024-01-01',
            analysis_id=126
        ))
        validation_service.repository.update_analysis_validation_status = AsyncMock(return_value=None)
        
        submitted = await validation_service.submit_validation(
            validation_id=validation.id,
            validator_id=789,
            validation_data={
                'approved': True,
                'clinical_notes': 'Normal sinus rhythm',
                'recommendations': ['Continue monitoring']
            }
        )
        
        assert submitted.status == ValidationStatus.APPROVED
        assert submitted.completed_at is not None
        
    @pytest.mark.asyncio
    async def test_quality_metrics_calculation(self, validation_service):
        """Test quality metrics - covers lines 195-224"""
        validation_data = {
            'agreement_score': 0.85,
            'confidence_scores': [0.9, 0.8, 0.85],
            'processing_time': 45.2,
            'validator_experience': 'senior'
        }
        
        mock_validations = [
            Mock(status='approved', confidence_score=0.9),
            Mock(status='approved', confidence_score=0.8)
        ]
        
        metrics = validation_service._calculate_quality_metrics(
            mock_validations, validation_data
        )
        
        assert 'total_validations' in metrics
        assert 'approval_rate' in metrics
        assert 'quality_metrics' in metrics
    
    @pytest.mark.asyncio
    async def test_escalation_workflows(self, validation_service):
        """Test escalation logic - covers lines 233-250"""
        critical_validation = {
            'findings': ['STEMI', 'Ventricular tachycardia'],
            'urgency': 'immediate'
        }
        
        validation_service.repository.get_validation_by_analysis = AsyncMock(return_value=None)
        validation_service.repository.get_analysis_by_id = AsyncMock(return_value=Mock(clinical_urgency='CRITICAL'))
        validation_service.repository.create_validation = AsyncMock(return_value=Mock(id=1))
        validation_service.notification_service.send_validation_assignment = AsyncMock(return_value=None)
        
        validation_service.repository.get_available_validators = AsyncMock(return_value=[Mock(id=456, role='CARDIOLOGIST', experience_years=10)])
        validation_service.notification_service.send_urgent_validation_alert = AsyncMock(return_value=None)
        
        escalated = await validation_service.create_urgent_validation(
            analysis_id=999
        )
        
        assert escalated is None
    
    @pytest.mark.asyncio
    async def test_multi_validator_consensus(self, validation_service):
        """Test consensus mechanisms - covers lines 258-287"""
        validations = [
            {'diagnosis': 'AFib', 'confidence': 0.9},
            {'diagnosis': 'AFib', 'confidence': 0.85},
            {'diagnosis': 'Normal', 'confidence': 0.6}
        ]
        
        consensus = validation_service._calculate_consensus(
            validations
        )
        
        assert 'final_status' in consensus
        assert 'confidence' in consensus
        assert 'total_validations' in consensus
