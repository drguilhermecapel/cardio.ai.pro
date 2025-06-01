"""Test Validation Service."""

import pytest
from unittest.mock import Mock, AsyncMock
from datetime import datetime, timedelta
from app.services.validation_service import ValidationService
from app.models.validation import Validation
from app.models.ecg_analysis import ECGAnalysis
from app.schemas.validation import ValidationCreate, ValidationUpdate


@pytest.fixture
def validation_service(test_db):
    """Create validation service instance."""
    from unittest.mock import Mock
    mock_notification_service = Mock()
    return ValidationService(db=test_db, notification_service=mock_notification_service)


@pytest.fixture
def sample_ecg_analysis(test_db):
    """Create sample ECG analysis."""
    analysis = ECGAnalysis(
        patient_id=1,
        file_path="/tmp/test.txt",
        classification="abnormal",
        confidence=0.85,
        rhythm="atrial_fibrillation",
        heart_rate=95,
        signal_quality=0.88,
        findings=["Irregular rhythm", "Fast heart rate"]
    )
    test_db.add(analysis)
    return analysis


@pytest.fixture
def sample_validation_data():
    """Sample validation data."""
    return ValidationCreate(
        ecg_analysis_id=1,
        validator_id=1,
        validation_type="medical_review",
        priority="high",
        notes="Requires immediate attention"
    )


@pytest.mark.asyncio
async def test_create_validation_success(validation_service, sample_validation_data, test_db):
    """Test successful validation creation."""
    result = await validation_service.create_validation(sample_validation_data)
    
    assert result is not None
    assert result.ecg_analysis_id == sample_validation_data.ecg_analysis_id
    assert result.validator_id == sample_validation_data.validator_id
    assert result.validation_type == sample_validation_data.validation_type
    assert result.status == "pending"


@pytest.mark.asyncio
async def test_get_validation_by_id(validation_service, test_db):
    """Test retrieving validation by ID."""
    validation = Validation(
        ecg_analysis_id=1,
        validator_id=1,
        validation_type="medical_review",
        status="pending",
        priority="high"
    )
    test_db.add(validation)
    await test_db.commit()
    await test_db.refresh(validation)
    
    result = await validation_service.get_validation_by_id(validation.id)
    
    assert result is not None
    assert result.id == validation.id
    assert result.validation_type == "medical_review"


@pytest.mark.asyncio
async def test_get_validation_by_id_not_found(validation_service):
    """Test retrieving non-existent validation."""
    result = await validation_service.get_validation_by_id(99999)
    assert result is None


@pytest.mark.asyncio
async def test_get_validations_by_ecg_analysis(validation_service, test_db):
    """Test retrieving validations by ECG analysis ID."""
    ecg_analysis_id = 1
    
    for i in range(3):
        validation = Validation(
            ecg_analysis_id=ecg_analysis_id,
            validator_id=i + 1,
            validation_type="medical_review",
            status="pending",
            priority="medium"
        )
        test_db.add(validation)
    
    await test_db.commit()
    
    results = await validation_service.get_validations_by_ecg_analysis(ecg_analysis_id)
    
    assert len(results) == 3
    assert all(v.ecg_analysis_id == ecg_analysis_id for v in results)


@pytest.mark.asyncio
async def test_get_pending_validations(validation_service, test_db):
    """Test retrieving pending validations."""
    for i in range(5):
        status = "pending" if i < 3 else "completed"
        validation = Validation(
            ecg_analysis_id=i + 1,
            validator_id=1,
            validation_type="medical_review",
            status=status,
            priority="medium"
        )
        test_db.add(validation)
    
    await test_db.commit()
    
    results = await validation_service.get_pending_validations()
    
    assert len(results) == 3
    assert all(v.status == "pending" for v in results)


@pytest.mark.asyncio
async def test_update_validation_status(validation_service, test_db):
    """Test updating validation status."""
    validation = Validation(
        ecg_analysis_id=1,
        validator_id=1,
        validation_type="medical_review",
        status="pending",
        priority="high"
    )
    test_db.add(validation)
    await test_db.commit()
    await test_db.refresh(validation)
    
    update_data = ValidationUpdate(
        status="approved",
        notes="Analysis confirmed correct",
        validated_at=datetime.utcnow()
    )
    
    result = await validation_service.update_validation(validation.id, update_data)
    
    assert result is not None
    assert result.status == "approved"
    assert result.notes == "Analysis confirmed correct"
    assert result.validated_at is not None


@pytest.mark.asyncio
async def test_approve_validation(validation_service, test_db):
    """Test approving validation."""
    validation = Validation(
        ecg_analysis_id=1,
        validator_id=1,
        validation_type="medical_review",
        status="pending",
        priority="high"
    )
    test_db.add(validation)
    await test_db.commit()
    await test_db.refresh(validation)
    
    result = await validation_service.approve_validation(
        validation.id, 
        validator_id=2,
        notes="Approved by senior cardiologist"
    )
    
    assert result is not None
    assert result.status == "approved"
    assert result.approved_by == 2
    assert result.notes == "Approved by senior cardiologist"


@pytest.mark.asyncio
async def test_reject_validation(validation_service, test_db):
    """Test rejecting validation."""
    validation = Validation(
        ecg_analysis_id=1,
        validator_id=1,
        validation_type="medical_review",
        status="pending",
        priority="high"
    )
    test_db.add(validation)
    await test_db.commit()
    await test_db.refresh(validation)
    
    result = await validation_service.reject_validation(
        validation.id,
        validator_id=2,
        reason="Insufficient signal quality"
    )
    
    assert result is not None
    assert result.status == "rejected"
    assert result.rejected_by == 2
    assert result.rejection_reason == "Insufficient signal quality"


@pytest.mark.asyncio
async def test_validate_ecg_analysis_quality_check(validation_service, sample_ecg_analysis):
    """Test ECG analysis quality validation."""
    await sample_ecg_analysis.db.commit()
    await sample_ecg_analysis.db.refresh(sample_ecg_analysis)
    
    is_valid = await validation_service.validate_ecg_analysis(sample_ecg_analysis.id)
    
    assert isinstance(is_valid, bool)


@pytest.mark.asyncio
async def test_validate_ecg_analysis_high_quality(validation_service, test_db):
    """Test validation of high-quality ECG analysis."""
    analysis = ECGAnalysis(
        patient_id=1,
        file_path="/tmp/test.txt",
        classification="normal",
        confidence=0.95,
        rhythm="sinus",
        heart_rate=72,
        signal_quality=0.92
    )
    test_db.add(analysis)
    await test_db.commit()
    await test_db.refresh(analysis)
    
    is_valid = await validation_service.validate_ecg_analysis(analysis.id)
    
    assert is_valid is True


@pytest.mark.asyncio
async def test_validate_ecg_analysis_low_quality(validation_service, test_db):
    """Test validation of low-quality ECG analysis."""
    analysis = ECGAnalysis(
        patient_id=1,
        file_path="/tmp/test.txt",
        classification="uncertain",
        confidence=0.45,
        rhythm="unknown",
        heart_rate=0,
        signal_quality=0.35
    )
    test_db.add(analysis)
    await test_db.commit()
    await test_db.refresh(analysis)
    
    is_valid = await validation_service.validate_ecg_analysis(analysis.id)
    
    assert is_valid is False


@pytest.mark.asyncio
async def test_get_validation_statistics(validation_service, test_db):
    """Test retrieving validation statistics."""
    statuses = ["pending", "approved", "rejected", "pending", "approved"]
    
    for i, status in enumerate(statuses):
        validation = Validation(
            ecg_analysis_id=i + 1,
            validator_id=1,
            validation_type="medical_review",
            status=status,
            priority="medium"
        )
        test_db.add(validation)
    
    await test_db.commit()
    
    stats = await validation_service.get_validation_statistics()
    
    assert "total" in stats
    assert "pending" in stats
    assert "approved" in stats
    assert "rejected" in stats
    assert stats["total"] == 5
    assert stats["pending"] == 2
    assert stats["approved"] == 2
    assert stats["rejected"] == 1


@pytest.mark.asyncio
async def test_get_validator_performance(validation_service, test_db):
    """Test retrieving validator performance metrics."""
    validator_id = 1
    
    for i in range(10):
        status = "approved" if i < 8 else "rejected"
        validation = Validation(
            ecg_analysis_id=i + 1,
            validator_id=validator_id,
            validation_type="medical_review",
            status=status,
            priority="medium",
            validated_at=datetime.utcnow() - timedelta(days=i)
        )
        test_db.add(validation)
    
    await test_db.commit()
    
    performance = await validation_service.get_validator_performance(validator_id)
    
    assert "total_validations" in performance
    assert "approval_rate" in performance
    assert "average_time" in performance
    assert performance["total_validations"] == 10
    assert performance["approval_rate"] == 0.8


@pytest.mark.asyncio
async def test_get_overdue_validations(validation_service, test_db):
    """Test retrieving overdue validations."""
    old_date = datetime.utcnow() - timedelta(days=5)
    recent_date = datetime.utcnow() - timedelta(hours=1)
    
    old_validation = Validation(
        ecg_analysis_id=1,
        validator_id=1,
        validation_type="medical_review",
        status="pending",
        priority="high",
        created_at=old_date
    )
    
    recent_validation = Validation(
        ecg_analysis_id=2,
        validator_id=1,
        validation_type="medical_review",
        status="pending",
        priority="medium",
        created_at=recent_date
    )
    
    test_db.add(old_validation)
    test_db.add(recent_validation)
    await test_db.commit()
    
    overdue = await validation_service.get_overdue_validations(hours=24)
    
    assert len(overdue) == 1
    assert overdue[0].ecg_analysis_id == 1


@pytest.mark.asyncio
async def test_assign_validator(validation_service, test_db):
    """Test assigning validator to validation."""
    validation = Validation(
        ecg_analysis_id=1,
        validator_id=1,
        validation_type="medical_review",
        status="pending",
        priority="high"
    )
    test_db.add(validation)
    await test_db.commit()
    await test_db.refresh(validation)
    
    result = await validation_service.assign_validator(validation.id, new_validator_id=2)
    
    assert result is not None
    assert result.validator_id == 2
    assert result.status == "assigned"


@pytest.mark.asyncio
async def test_escalate_validation(validation_service, test_db):
    """Test escalating validation priority."""
    validation = Validation(
        ecg_analysis_id=1,
        validator_id=1,
        validation_type="medical_review",
        status="pending",
        priority="medium"
    )
    test_db.add(validation)
    await test_db.commit()
    await test_db.refresh(validation)
    
    result = await validation_service.escalate_validation(
        validation.id,
        new_priority="critical",
        reason="Patient condition deteriorated"
    )
    
    assert result is not None
    assert result.priority == "critical"
    assert result.escalation_reason == "Patient condition deteriorated"


@pytest.mark.asyncio
async def test_bulk_approve_validations(validation_service, test_db):
    """Test bulk approval of validations."""
    validation_ids = []
    
    for i in range(3):
        validation = Validation(
            ecg_analysis_id=i + 1,
            validator_id=1,
            validation_type="medical_review",
            status="pending",
            priority="medium"
        )
        test_db.add(validation)
        await test_db.flush()
        validation_ids.append(validation.id)
    
    await test_db.commit()
    
    results = await validation_service.bulk_approve_validations(
        validation_ids,
        approver_id=2,
        notes="Bulk approval by supervisor"
    )
    
    assert len(results) == 3
    assert all(v.status == "approved" for v in results)
    assert all(v.approved_by == 2 for v in results)


@pytest.mark.asyncio
async def test_compliance_audit_trail(validation_service, test_db):
    """Test compliance audit trail generation."""
    validation = Validation(
        ecg_analysis_id=1,
        validator_id=1,
        validation_type="medical_review",
        status="approved",
        priority="high",
        validated_at=datetime.utcnow(),
        approved_by=2
    )
    test_db.add(validation)
    await test_db.commit()
    await test_db.refresh(validation)
    
    audit_trail = await validation_service.generate_audit_trail(validation.id)
    
    assert audit_trail is not None
    assert "validation_id" in audit_trail
    assert "timeline" in audit_trail
    assert "compliance_status" in audit_trail
    assert audit_trail["compliance_status"] == "compliant"


@pytest.mark.asyncio
async def test_regulatory_compliance_check(validation_service):
    """Test regulatory compliance validation."""
    validation_data = {
        "validator_credentials": "MD, Board Certified Cardiologist",
        "validation_time": datetime.utcnow(),
        "digital_signature": "valid_signature_hash",
        "audit_trail": ["created", "assigned", "reviewed", "approved"]
    }
    
    is_compliant = await validation_service.check_regulatory_compliance(validation_data)
    
    assert isinstance(is_compliant, bool)
    assert is_compliant is True


@pytest.mark.asyncio
async def test_delete_validation(validation_service, test_db):
    """Test deleting validation."""
    validation = Validation(
        ecg_analysis_id=1,
        validator_id=1,
        validation_type="medical_review",
        status="pending",
        priority="medium"
    )
    test_db.add(validation)
    await test_db.commit()
    await test_db.refresh(validation)
    
    success = await validation_service.delete_validation(validation.id)
    assert success is True
    
    deleted = await validation_service.get_validation_by_id(validation.id)
    assert deleted is None
