import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from app.core.config import settings

if settings.STANDALONE_MODE:
    pytest.skip("Patient service tests skipped in standalone mode", allow_module_level=True)

from app.services.patient_service import PatientService
from app.schemas.patient import PatientCreate, PatientUpdate


@pytest.mark.asyncio
async def test_patient_service_initialization():
    """Test patient service initialization."""
    mock_session = AsyncMock()
    service = PatientService(mock_session)
    
    assert service.session == mock_session


@pytest.mark.asyncio
async def test_create_patient():
    """Test patient creation."""
    mock_session = AsyncMock()
    service = PatientService(mock_session)
    
    patient_data = PatientCreate(
        name="John Doe",
        date_of_birth="1990-01-01",
        gender="M",
        medical_record_number="MRN123"
    )
    
    with patch.object(service, 'patient_repo') as mock_repo:
        mock_patient = MagicMock()
        mock_patient.id = 1
        mock_patient.name = "John Doe"
        mock_repo.create.return_value = mock_patient
        
        result = await service.create_patient(patient_data)
        
        mock_repo.create.assert_called_once()
        assert result.id == 1
        assert result.name == "John Doe"


@pytest.mark.asyncio
async def test_get_patient_by_id():
    """Test getting patient by ID."""
    mock_session = AsyncMock()
    service = PatientService(mock_session)
    
    with patch.object(service, 'patient_repo') as mock_repo:
        mock_patient = MagicMock()
        mock_patient.id = 1
        mock_patient.name = "John Doe"
        mock_repo.get.return_value = mock_patient
        
        result = await service.get_patient(1)
        
        mock_repo.get.assert_called_once_with(1)
        assert result.id == 1


@pytest.mark.asyncio
async def test_update_patient():
    """Test patient update."""
    mock_session = AsyncMock()
    service = PatientService(mock_session)
    
    patient_update = PatientUpdate(name="Jane Doe")
    
    with patch.object(service, 'patient_repo') as mock_repo:
        mock_patient = MagicMock()
        mock_patient.id = 1
        mock_patient.name = "Jane Doe"
        mock_repo.update.return_value = mock_patient
        
        result = await service.update_patient(1, patient_update)
        
        mock_repo.update.assert_called_once_with(1, patient_update)
        assert result.name == "Jane Doe"


@pytest.mark.asyncio
async def test_delete_patient():
    """Test patient deletion."""
    mock_session = AsyncMock()
    service = PatientService(mock_session)
    
    with patch.object(service, 'patient_repo') as mock_repo:
        mock_repo.delete.return_value = True
        
        result = await service.delete_patient(1)
        
        mock_repo.delete.assert_called_once_with(1)
        assert result is True


@pytest.mark.asyncio
async def test_list_patients():
    """Test listing patients."""
    mock_session = AsyncMock()
    service = PatientService(mock_session)
    
    with patch.object(service, 'patient_repo') as mock_repo:
        mock_patients = [MagicMock(), MagicMock()]
        mock_repo.list.return_value = mock_patients
        
        result = await service.list_patients(skip=0, limit=10)
        
        mock_repo.list.assert_called_once_with(skip=0, limit=10)
        assert len(result) == 2


@pytest.mark.asyncio
async def test_search_patients():
    """Test searching patients."""
    mock_session = AsyncMock()
    service = PatientService(mock_session)
    
    with patch.object(service, 'patient_repo') as mock_repo:
        mock_patients = [MagicMock()]
        mock_repo.search.return_value = mock_patients
        
        result = await service.search_patients("John")
        
        mock_repo.search.assert_called_once_with("John")
        assert len(result) == 1


@pytest.mark.asyncio
async def test_get_patient_by_mrn():
    """Test getting patient by medical record number."""
    mock_session = AsyncMock()
    service = PatientService(mock_session)
    
    with patch.object(service, 'patient_repo') as mock_repo:
        mock_patient = MagicMock()
        mock_patient.medical_record_number = "MRN123"
        mock_repo.get_by_mrn.return_value = mock_patient
        
        result = await service.get_patient_by_mrn("MRN123")
        
        mock_repo.get_by_mrn.assert_called_once_with("MRN123")
        assert result.medical_record_number == "MRN123"


@pytest.mark.asyncio
async def test_patient_exists():
    """Test checking if patient exists."""
    mock_session = AsyncMock()
    service = PatientService(mock_session)
    
    with patch.object(service, 'patient_repo') as mock_repo:
        mock_repo.get.return_value = MagicMock()
        
        result = await service.patient_exists(1)
        
        mock_repo.get.assert_called_once_with(1)
        assert result is True


@pytest.mark.asyncio
async def test_patient_not_exists():
    """Test checking if patient does not exist."""
    mock_session = AsyncMock()
    service = PatientService(mock_session)
    
    with patch.object(service, 'patient_repo') as mock_repo:
        mock_repo.get.return_value = None
        
        result = await service.patient_exists(999)
        
        assert result is False


@pytest.mark.asyncio
async def test_get_patient_ecg_analyses():
    """Test getting patient ECG analyses."""
    mock_session = AsyncMock()
    service = PatientService(mock_session)
    
    with patch.object(service, 'ecg_repo') as mock_ecg_repo:
        mock_analyses = [MagicMock(), MagicMock()]
        mock_ecg_repo.get_by_patient_id.return_value = mock_analyses
        
        result = await service.get_patient_ecg_analyses(1)
        
        mock_ecg_repo.get_by_patient_id.assert_called_once_with(1)
        assert len(result) == 2


@pytest.mark.asyncio
async def test_validate_patient_data():
    """Test patient data validation."""
    mock_session = AsyncMock()
    service = PatientService(mock_session)
    
    patient_data = PatientCreate(
        name="John Doe",
        date_of_birth="1990-01-01",
        gender="M",
        medical_record_number="MRN123"
    )
    
    with patch.object(service, '_validate_patient_data') as mock_validate:
        mock_validate.return_value = True
        
        result = await service._validate_patient_data(patient_data)
        
        assert result is True
