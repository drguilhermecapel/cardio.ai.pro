"""Test Patient Service."""

import pytest
from datetime import date, datetime
from app.services.patient_service import PatientService
from app.models.patient import Patient
from app.schemas.patient import PatientCreate, PatientUpdate


@pytest.fixture
def patient_service(test_db):
    """Create patient service instance."""
    return PatientService(db=test_db)


@pytest.fixture
def sample_patient_data():
    """Sample patient data."""
    return PatientCreate(
        patient_id="PAT123456",
        first_name="John",
        last_name="Doe",
        date_of_birth=date(1990, 1, 15),
        gender="male",
        phone="+1234567890",
        email="john.doe@example.com",
        address="123 Main St, City, State 12345",
        emergency_contact_name="Jane Doe",
        emergency_contact_phone="+0987654321"
    )


@pytest.mark.asyncio
async def test_create_patient_success(patient_service, sample_patient_data):
    """Test successful patient creation."""
    result = await patient_service.create_patient(sample_patient_data)
    
    assert result is not None
    assert result.first_name == sample_patient_data.first_name
    assert result.last_name == sample_patient_data.last_name
    assert result.date_of_birth == sample_patient_data.date_of_birth
    assert result.gender == sample_patient_data.gender
    assert result.patient_id == sample_patient_data.patient_id


@pytest.mark.asyncio
async def test_create_patient_duplicate_mrn(patient_service, sample_patient_data, test_db):
    """Test creating patient with duplicate medical record number."""
    existing_patient = Patient(
        first_name="Existing",
        last_name="Patient",
        date_of_birth=date(1985, 5, 20),
        gender="female",
        patient_id=sample_patient_data.patient_id
    )
    test_db.add(existing_patient)
    await test_db.commit()
    
    with pytest.raises(ValueError, match="Medical record number already exists"):
        await patient_service.create_patient(sample_patient_data)


@pytest.mark.asyncio
async def test_get_patient_by_id(patient_service, test_db):
    """Test retrieving patient by ID."""
    patient = Patient(
        first_name="Test",
        last_name="Patient",
        date_of_birth=date(1990, 1, 1),
        gender="male",
        patient_id="TEST123"
    )
    test_db.add(patient)
    await test_db.commit()
    await test_db.refresh(patient)
    
    result = await patient_service.get_patient_by_id(patient.id)
    
    assert result is not None
    assert result.id == patient.id
    assert result.first_name == "Test"
    assert result.last_name == "Patient"


@pytest.mark.asyncio
async def test_get_patient_by_id_not_found(patient_service):
    """Test retrieving non-existent patient."""
    result = await patient_service.get_patient_by_id(99999)
    assert result is None


@pytest.mark.asyncio
async def test_get_patient_by_mrn(patient_service, test_db):
    """Test retrieving patient by medical record number."""
    mrn = "UNIQUE123"
    patient = Patient(
        first_name="Test",
        last_name="Patient",
        date_of_birth=date(1990, 1, 1),
        gender="male",
        patient_id=mrn
    )
    test_db.add(patient)
    await test_db.commit()
    
    result = await patient_service.get_patient_by_mrn(mrn)
    
    assert result is not None
    assert result.patient_id == mrn


@pytest.mark.asyncio
async def test_get_patient_by_mrn_not_found(patient_service):
    """Test retrieving patient by non-existent MRN."""
    result = await patient_service.get_patient_by_mrn("NONEXISTENT")
    assert result is None


@pytest.mark.asyncio
async def test_update_patient(patient_service, test_db):
    """Test updating patient information."""
    patient = Patient(
        first_name="Original",
        last_name="Name",
        date_of_birth=date(1990, 1, 1),
        gender="male",
        patient_id="UPDATE123"
    )
    test_db.add(patient)
    await test_db.commit()
    await test_db.refresh(patient)
    
    update_data = PatientUpdate(
        first_name="Updated",
        last_name="Name",
        phone="+1111111111",
        email="updated@example.com"
    )
    
    result = await patient_service.update_patient(patient.id, update_data)
    
    assert result is not None
    assert result.first_name == "Updated"
    assert result.last_name == "Name"
    assert result.phone == "+1111111111"
    assert result.email == "updated@example.com"
    assert result.date_of_birth == date(1990, 1, 1)


@pytest.mark.asyncio
async def test_update_patient_not_found(patient_service):
    """Test updating non-existent patient."""
    update_data = PatientUpdate(first_name="New", last_name="Name")
    
    result = await patient_service.update_patient(99999, update_data)
    assert result is None


@pytest.mark.asyncio
async def test_delete_patient(patient_service, test_db):
    """Test deleting patient."""
    patient = Patient(
        name="To Delete",
        birth_date=date(1990, 1, 1),
        gender="M",
        medical_record_number="DELETE123"
    )
    test_db.add(patient)
    await test_db.commit()
    await test_db.refresh(patient)
    
    success = await patient_service.delete_patient(patient.id)
    assert success is True
    
    deleted = await patient_service.get_patient_by_id(patient.id)
    assert deleted is None


@pytest.mark.asyncio
async def test_delete_patient_not_found(patient_service):
    """Test deleting non-existent patient."""
    success = await patient_service.delete_patient(99999)
    assert success is False


@pytest.mark.asyncio
async def test_search_patients_by_name(patient_service, test_db):
    """Test searching patients by name."""
    patients_data = [
        ("John Smith", "JOHN001"),
        ("Jane Smith", "JANE001"),
        ("John Doe", "JOHN002"),
        ("Bob Johnson", "BOB001")
    ]
    
    for name, mrn in patients_data:
        patient = Patient(
            name=name,
            birth_date=date(1990, 1, 1),
            gender="M",
            medical_record_number=mrn
        )
        test_db.add(patient)
    
    await test_db.commit()
    
    results = await patient_service.search_patients(name="John")
    
    assert len(results) == 2
    assert all("John" in p.name for p in results)


@pytest.mark.asyncio
async def test_search_patients_by_mrn(patient_service, test_db):
    """Test searching patients by medical record number."""
    patient = Patient(
        name="Search Test",
        birth_date=date(1990, 1, 1),
        gender="M",
        medical_record_number="SEARCH123"
    )
    test_db.add(patient)
    await test_db.commit()
    
    results = await patient_service.search_patients(mrn="SEARCH")
    
    assert len(results) == 1
    assert results[0].medical_record_number == "SEARCH123"


@pytest.mark.asyncio
async def test_get_patients_paginated(patient_service, test_db):
    """Test retrieving patients with pagination."""
    for i in range(15):
        patient = Patient(
            name=f"Patient {i}",
            birth_date=date(1990, 1, 1),
            gender="M",
            medical_record_number=f"PAGE{i:03d}"
        )
        test_db.add(patient)
    
    await test_db.commit()
    
    page1 = await patient_service.get_patients(skip=0, limit=10)
    page2 = await patient_service.get_patients(skip=10, limit=10)
    
    assert len(page1) == 10
    assert len(page2) == 5
    assert page1[0].id != page2[0].id


@pytest.mark.asyncio
async def test_get_patient_statistics(patient_service, test_db):
    """Test retrieving patient statistics."""
    genders = ["M", "F", "M", "F", "M"]
    
    for i, gender in enumerate(genders):
        patient = Patient(
            name=f"Patient {i}",
            birth_date=date(1990, 1, 1),
            gender=gender,
            medical_record_number=f"STAT{i:03d}"
        )
        test_db.add(patient)
    
    await test_db.commit()
    
    stats = await patient_service.get_patient_statistics()
    
    assert "total_patients" in stats
    assert "gender_distribution" in stats
    assert stats["total_patients"] == 5
    assert stats["gender_distribution"]["M"] == 3
    assert stats["gender_distribution"]["F"] == 2


@pytest.mark.asyncio
async def test_validate_patient_data(patient_service):
    """Test patient data validation."""
    valid_data = PatientCreate(
        name="Valid Patient",
        birth_date=date(1990, 1, 1),
        gender="M",
        medical_record_number="VALID123",
        email="valid@example.com",
        phone="+1234567890"
    )
    
    is_valid, errors = await patient_service.validate_patient_data(valid_data)
    
    assert is_valid is True
    assert len(errors) == 0


@pytest.mark.asyncio
async def test_validate_patient_data_invalid_email(patient_service):
    """Test patient data validation with invalid email."""
    invalid_data = PatientCreate(
        patient_id="INVALID123",
        first_name="Invalid",
        last_name="Patient",
        date_of_birth=date(1990, 1, 1),
        gender="male",
        email="invalid-email"
    )
    
    is_valid, errors = await patient_service.validate_patient_data(invalid_data)
    
    assert is_valid is False
    assert "email" in str(errors)


@pytest.mark.asyncio
async def test_validate_patient_data_future_birth_date(patient_service):
    """Test patient data validation with future birth date."""
    future_date = date(2030, 1, 1)
    invalid_data = PatientCreate(
        name="Future Patient",
        birth_date=future_date,
        gender="M",
        medical_record_number="FUTURE123"
    )
    
    is_valid, errors = await patient_service.validate_patient_data(invalid_data)
    
    assert is_valid is False
    assert "birth_date" in str(errors)


@pytest.mark.asyncio
async def test_calculate_age(patient_service):
    """Test age calculation."""
    birth_date = date(1990, 6, 15)
    
    age = patient_service.calculate_age(birth_date)
    
    assert isinstance(age, int)
    assert age >= 30


@pytest.mark.asyncio
async def test_get_patients_by_age_range(patient_service, test_db):
    """Test retrieving patients by age range."""
    birth_dates = [
        date(1980, 1, 1),  # ~45 years old
        date(1990, 1, 1),  # ~35 years old
        date(2000, 1, 1),  # ~25 years old
        date(2010, 1, 1)   # ~15 years old
    ]
    
    for i, birth_date in enumerate(birth_dates):
        patient = Patient(
            name=f"Patient {i}",
            birth_date=birth_date,
            gender="M",
            medical_record_number=f"AGE{i:03d}"
        )
        test_db.add(patient)
    
    await test_db.commit()
    
    adults = await patient_service.get_patients_by_age_range(min_age=18, max_age=65)
    
    assert len(adults) == 3


@pytest.mark.asyncio
async def test_anonymize_patient_data(patient_service, test_db):
    """Test patient data anonymization."""
    patient = Patient(
        name="Sensitive Patient",
        birth_date=date(1990, 1, 1),
        gender="M",
        medical_record_number="SENSITIVE123",
        phone="+1234567890",
        email="sensitive@example.com",
        address="123 Secret St"
    )
    test_db.add(patient)
    await test_db.commit()
    await test_db.refresh(patient)
    
    anonymized = await patient_service.anonymize_patient_data(patient.id)
    
    assert anonymized is not None
    assert anonymized.name != "Sensitive Patient"
    assert anonymized.phone is None
    assert anonymized.email is None
    assert anonymized.address is None


@pytest.mark.asyncio
async def test_export_patient_data(patient_service, test_db):
    """Test patient data export."""
    patient = Patient(
        name="Export Patient",
        birth_date=date(1990, 1, 1),
        gender="M",
        medical_record_number="EXPORT123"
    )
    test_db.add(patient)
    await test_db.commit()
    await test_db.refresh(patient)
    
    exported_data = await patient_service.export_patient_data(patient.id)
    
    assert exported_data is not None
    assert "patient_info" in exported_data
    assert "ecg_analyses" in exported_data
    assert "validations" in exported_data


@pytest.mark.asyncio
async def test_merge_patients(patient_service, test_db):
    """Test merging duplicate patients."""
    patient1 = Patient(
        name="John Doe",
        birth_date=date(1990, 1, 1),
        gender="M",
        medical_record_number="MERGE001"
    )
    
    patient2 = Patient(
        name="John Doe",
        birth_date=date(1990, 1, 1),
        gender="M",
        medical_record_number="MERGE002"
    )
    
    test_db.add(patient1)
    test_db.add(patient2)
    await test_db.commit()
    await test_db.refresh(patient1)
    await test_db.refresh(patient2)
    
    merged = await patient_service.merge_patients(
        primary_id=patient1.id,
        secondary_id=patient2.id
    )
    
    assert merged is not None
    assert merged.id == patient1.id
    
    deleted = await patient_service.get_patient_by_id(patient2.id)
    assert deleted is None


@pytest.mark.asyncio
async def test_get_recent_patients(patient_service, test_db):
    """Test retrieving recently created patients."""
    for i in range(5):
        patient = Patient(
            name=f"Recent Patient {i}",
            birth_date=date(1990, 1, 1),
            gender="M",
            medical_record_number=f"RECENT{i:03d}"
        )
        test_db.add(patient)
    
    await test_db.commit()
    
    recent = await patient_service.get_recent_patients(limit=3)
    
    assert len(recent) == 3
    assert all(p.created_at is not None for p in recent)


@pytest.mark.asyncio
async def test_patient_consent_management(patient_service, test_db):
    """Test patient consent management."""
    patient = Patient(
        name="Consent Patient",
        birth_date=date(1990, 1, 1),
        gender="M",
        medical_record_number="CONSENT123"
    )
    test_db.add(patient)
    await test_db.commit()
    await test_db.refresh(patient)
    
    consent_data = {
        "data_processing": True,
        "research_participation": False,
        "marketing_communications": False,
        "consent_date": datetime.utcnow()
    }
    
    result = await patient_service.update_consent(patient.id, consent_data)
    
    assert result is not None
    assert result.consent_data["data_processing"] is True
    assert result.consent_data["research_participation"] is False
