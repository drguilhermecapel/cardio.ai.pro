"""
Script to fix all API endpoint issues
Run this to update all endpoint files with correct imports and dependencies
"""
import os
from pathlib import Path

# Base directory for endpoints
ENDPOINTS_DIR = Path(__file__).parent

# Common imports for all endpoints
COMMON_IMPORTS = """from typing import List, Optional, Dict, Any
from datetime import datetime
from fastapi import APIRouter, Depends, HTTPException, Query, status, File, UploadFile
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.session import get_db
from app.core.security import get_current_user
from app.models.user import User
from app.core.exceptions import (
    NotFoundException, 
    ValidationException, 
    UnauthorizedException
)
"""

# Fix auth.py
AUTH_CONTENT = COMMON_IMPORTS + """
from app.schemas.user import UserCreate, UserResponse, Token
from app.services.user_service import UserService
from app.repositories.user_repository import UserRepository

router = APIRouter(prefix="/auth", tags=["authentication"])

@router.post("/register", response_model=UserResponse)
async def register(
    user_data: UserCreate,
    db: AsyncSession = Depends(get_db)
):
    \"\"\"Register a new user\"\"\"
    try:
        user_repo = UserRepository(db)
        user_service = UserService(db, user_repo)
        user = await user_service.create_user(user_data)
        return user
    except ValidationException as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.post("/login", response_model=Token)
async def login(
    username: str,
    password: str,
    db: AsyncSession = Depends(get_db)
):
    \"\"\"Login and get access token\"\"\"
    try:
        user_repo = UserRepository(db)
        user_service = UserService(db, user_repo)
        user = await user_service.authenticate_user(username, password)
        if not user:
            raise HTTPException(
                status_code=401,
                detail="Invalid credentials"
            )
        # Generate token (simplified for now)
        return {
            "access_token": f"token_for_{user.id}",
            "token_type": "bearer"
        }
    except Exception as e:
        raise HTTPException(status_code=401, detail="Authentication failed")

@router.get("/me", response_model=UserResponse)
async def get_current_user_info(
    current_user: User = Depends(get_current_user)
):
    \"\"\"Get current user information\"\"\"
    return current_user

@router.post("/logout")
async def logout(
    current_user: User = Depends(get_current_user)
):
    \"\"\"Logout current user\"\"\"
    # In a real implementation, you would invalidate the token
    return {"message": "Logged out successfully"}

@router.post("/refresh", response_model=Token)
async def refresh_token(
    current_user: User = Depends(get_current_user)
):
    \"\"\"Refresh access token\"\"\"
    return {
        "access_token": f"refreshed_token_for_{current_user.id}",
        "token_type": "bearer"
    }
"""

# Fix users.py
USERS_CONTENT = COMMON_IMPORTS + """
from app.schemas.user import UserResponse, UserUpdate
from app.services.user_service import UserService
from app.repositories.user_repository import UserRepository
from app.models.user import UserRoles

router = APIRouter(prefix="/users", tags=["users"])

@router.get("/", response_model=List[UserResponse])
async def get_users(
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=100),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    \"\"\"Get list of users (admin only)\"\"\"
    if current_user.role != UserRoles.ADMIN:
        raise HTTPException(status_code=403, detail="Not authorized")
    
    user_repo = UserRepository(db)
    users = await user_repo.get_users(skip=skip, limit=limit)
    return users

@router.get("/{user_id}", response_model=UserResponse)
async def get_user(
    user_id: int,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    \"\"\"Get user by ID\"\"\"
    user_repo = UserRepository(db)
    user = await user_repo.get_user_by_id(user_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    return user

@router.put("/{user_id}", response_model=UserResponse)
async def update_user(
    user_id: int,
    user_update: UserUpdate,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    \"\"\"Update user information\"\"\"
    if current_user.id != user_id and current_user.role != UserRoles.ADMIN:
        raise HTTPException(status_code=403, detail="Not authorized")
    
    user_repo = UserRepository(db)
    user_service = UserService(db, user_repo)
    user = await user_service.update_user(user_id, user_update)
    return user

@router.delete("/{user_id}")
async def delete_user(
    user_id: int,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    \"\"\"Delete user (admin only)\"\"\"
    if current_user.role != UserRoles.ADMIN:
        raise HTTPException(status_code=403, detail="Not authorized")
    
    user_repo = UserRepository(db)
    await user_repo.delete_user(user_id)
    return {"message": "User deleted successfully"}
"""

# Fix patients.py
PATIENTS_CONTENT = COMMON_IMPORTS + """
from app.schemas.patient import PatientCreate, PatientResponse, PatientUpdate
from app.services.patient_service import PatientService
from app.repositories.patient_repository import PatientRepository

router = APIRouter(prefix="/patients", tags=["patients"])

@router.post("/", response_model=PatientResponse)
async def create_patient(
    patient_data: PatientCreate,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    \"\"\"Create a new patient\"\"\"
    patient_repo = PatientRepository(db)
    patient_service = PatientService(db, patient_repo)
    patient = await patient_service.create_patient(patient_data)
    return patient

@router.get("/", response_model=List[PatientResponse])
async def get_patients(
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=100),
    search: Optional[str] = None,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    \"\"\"Get list of patients\"\"\"
    patient_repo = PatientRepository(db)
    patient_service = PatientService(db, patient_repo)
    
    if search:
        patients = await patient_service.search_patients(search, skip, limit)
    else:
        patients = await patient_service.get_patients(skip, limit)
    
    return patients

@router.get("/{patient_id}", response_model=PatientResponse)
async def get_patient(
    patient_id: str,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    \"\"\"Get patient by ID\"\"\"
    patient_repo = PatientRepository(db)
    patient_service = PatientService(db, patient_repo)
    patient = await patient_service.get_patient_by_patient_id(patient_id)
    
    if not patient:
        raise HTTPException(status_code=404, detail="Patient not found")
    
    return patient

@router.put("/{patient_id}", response_model=PatientResponse)
async def update_patient(
    patient_id: str,
    patient_update: PatientUpdate,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    \"\"\"Update patient information\"\"\"
    patient_repo = PatientRepository(db)
    patient_service = PatientService(db, patient_repo)
    patient = await patient_service.update_patient(patient_id, patient_update)
    return patient

@router.delete("/{patient_id}")
async def delete_patient(
    patient_id: str,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    \"\"\"Delete patient\"\"\"
    patient_repo = PatientRepository(db)
    await patient_repo.delete_patient(patient_id)
    return {"message": "Patient deleted successfully"}
"""

# Fix ecg_analysis.py
ECG_ANALYSIS_CONTENT = COMMON_IMPORTS + """
from app.schemas.ecg_analysis import (
    ECGAnalysisCreate, 
    ECGAnalysisResponse, 
    ECGAnalysisUpdate,
    ECGSearchParams
)
from app.services.ecg_service import ECGAnalysisService
from app.repositories.ecg_repository import ECGRepository
from app.services.patient_service import PatientService
from app.repositories.patient_repository import PatientRepository
from app.services.ml_model_service import MLModelService
from app.services.notification_service import NotificationService
from app.repositories.notification_repository import NotificationRepository
from app.services.interpretability_service import InterpretabilityService
from app.services.multi_pathology_service import MultiPathologyService
from app.core.constants import FileType

router = APIRouter(prefix="/ecg", tags=["ecg_analysis"])

@router.post("/analyses", response_model=ECGAnalysisResponse)
async def create_analysis(
    analysis_data: ECGAnalysisCreate,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    \"\"\"Create a new ECG analysis\"\"\"
    # Initialize all services
    ecg_repo = ECGRepository(db)
    patient_repo = PatientRepository(db)
    patient_service = PatientService(db, patient_repo)
    notification_repo = NotificationRepository(db)
    notification_service = NotificationService(db, notification_repo)
    ml_service = MLModelService()
    interpretability_service = InterpretabilityService()
    multi_pathology_service = MultiPathologyService(ml_service)
    
    ecg_service = ECGAnalysisService(
        db=db,
        ecg_repository=ecg_repo,
        patient_service=patient_service,
        ml_service=ml_service,
        notification_service=notification_service,
        interpretability_service=interpretability_service,
        multi_pathology_service=multi_pathology_service
    )
    
    analysis = await ecg_service.create_analysis(analysis_data, current_user.id)
    return analysis

@router.post("/upload", response_model=ECGAnalysisResponse)
async def upload_ecg_file(
    patient_id: str,
    file: UploadFile = File(...),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    \"\"\"Upload and analyze ECG file\"\"\"
    # Save uploaded file
    file_path = f"/tmp/{file.filename}"
    content = await file.read()
    
    with open(file_path, "wb") as f:
        f.write(content)
    
    # Determine file type
    file_extension = file.filename.split('.')[-1].lower()
    file_type_map = {
        'csv': FileType.CSV,
        'edf': FileType.EDF,
        'dat': FileType.MIT,
        'dcm': FileType.DICOM,
        'json': FileType.JSON
    }
    file_type = file_type_map.get(file_extension, FileType.OTHER)
    
    # Create analysis
    analysis_data = ECGAnalysisCreate(
        patient_id=patient_id,
        recording_date=datetime.utcnow(),
        file_path=file_path,
        file_type=file_type
    )
    
    return await create_analysis(analysis_data, current_user, db)

@router.get("/analyses", response_model=List[ECGAnalysisResponse])
async def search_analyses(
    search_params: ECGSearchParams = Depends(),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    \"\"\"Search ECG analyses\"\"\"
    ecg_repo = ECGRepository(db)
    analyses = await ecg_repo.search_analyses(
        patient_id=search_params.patient_id,
        start_date=search_params.start_date,
        end_date=search_params.end_date,
        status=search_params.status,
        diagnosis=search_params.diagnosis,
        skip=search_params.skip,
        limit=search_params.limit
    )
    return analyses

@router.get("/analyses/{analysis_id}", response_model=ECGAnalysisResponse)
async def get_analysis(
    analysis_id: str,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    \"\"\"Get ECG analysis by ID\"\"\"
    ecg_repo = ECGRepository(db)
    analysis = await ecg_repo.get_analysis_by_analysis_id(analysis_id)
    
    if not analysis:
        raise HTTPException(status_code=404, detail="Analysis not found")
    
    return analysis

@router.get("/analyses/{analysis_id}/report")
async def get_analysis_report(
    analysis_id: str,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    \"\"\"Generate analysis report\"\"\"
    # Initialize services
    ecg_repo = ECGRepository(db)
    patient_repo = PatientRepository(db)
    patient_service = PatientService(db, patient_repo)
    notification_repo = NotificationRepository(db)
    notification_service = NotificationService(db, notification_repo)
    ml_service = MLModelService()
    interpretability_service = InterpretabilityService()
    multi_pathology_service = MultiPathologyService(ml_service)
    
    ecg_service = ECGAnalysisService(
        db=db,
        ecg_repository=ecg_repo,
        patient_service=patient_service,
        ml_service=ml_service,
        notification_service=notification_service,
        interpretability_service=interpretability_service,
        multi_pathology_service=multi_pathology_service
    )
    
    report = await ecg_service.generate_report(analysis_id)
    return report

@router.delete("/analyses/{analysis_id}")
async def delete_analysis(
    analysis_id: str,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    \"\"\"Delete ECG analysis\"\"\"
    ecg_repo = ECGRepository(db)
    await ecg_repo.delete_analysis(analysis_id)
    return {"message": "Analysis deleted successfully"}
"""

# Fix validations.py
VALIDATIONS_CONTENT = COMMON_IMPORTS + """
from app.schemas.validation import (
    ValidationCreate,
    ValidationResponse,
    ValidationSubmit
)
from app.services.validation_service import ValidationService
from app.repositories.validation_repository import ValidationRepository
from app.repositories.ecg_repository import ECGRepository
from app.repositories.notification_repository import NotificationRepository
from app.services.notification_service import NotificationService
from app.models.user import UserRoles

router = APIRouter(prefix="/validations", tags=["validations"])

@router.post("/", response_model=ValidationResponse)
async def create_validation(
    validation_data: ValidationCreate,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    \"\"\"Create a new validation request\"\"\"
    validation_repo = ValidationRepository(db)
    ecg_repo = ECGRepository(db)
    notification_repo = NotificationRepository(db)
    notification_service = NotificationService(db, notification_repo)
    
    validation_service = ValidationService(
        db=db,
        validation_repository=validation_repo,
        ecg_repository=ecg_repo,
        notification_service=notification_service
    )
    
    validation = await validation_service.create_validation(
        validation_data,
        current_user.id
    )
    return validation

@router.get("/pending", response_model=List[ValidationResponse])
async def get_pending_validations(
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=100),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    \"\"\"Get pending validations for current user\"\"\"
    if current_user.role not in [UserRoles.PHYSICIAN, UserRoles.CARDIOLOGIST, UserRoles.ADMIN]:
        raise HTTPException(status_code=403, detail="Not authorized")
    
    validation_repo = ValidationRepository(db)
    validations = await validation_repo.get_pending_validations_for_user(
        current_user.id,
        skip=skip,
        limit=limit
    )
    return validations

@router.get("/{validation_id}", response_model=ValidationResponse)
async def get_validation(
    validation_id: int,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    \"\"\"Get validation by ID\"\"\"
    validation_repo = ValidationRepository(db)
    validation = await validation_repo.get_validation_by_id(validation_id)
    
    if not validation:
        raise HTTPException(status_code=404, detail="Validation not found")
    
    return validation

@router.post("/{validation_id}/submit", response_model=ValidationResponse)
async def submit_validation(
    validation_id: int,
    submission: ValidationSubmit,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    \"\"\"Submit validation results\"\"\"
    validation_repo = ValidationRepository(db)
    ecg_repo = ECGRepository(db)
    notification_repo = NotificationRepository(db)
    notification_service = NotificationService(db, notification_repo)
    
    validation_service = ValidationService(
        db=db,
        validation_repository=validation_repo,
        ecg_repository=ecg_repo,
        notification_service=notification_service
    )
    
    validation = await validation_service.submit_validation(
        validation_id,
        submission,
        current_user.id
    )
    return validation
"""

# Fix notifications.py
NOTIFICATIONS_CONTENT = COMMON_IMPORTS + """
from app.schemas.notification import NotificationResponse
from app.services.notification_service import NotificationService
from app.repositories.notification_repository import NotificationRepository

router = APIRouter(prefix="/notifications", tags=["notifications"])

@router.get("/", response_model=List[NotificationResponse])
async def get_notifications(
    unread_only: bool = Query(False),
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=100),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    \"\"\"Get user notifications\"\"\"
    notification_repo = NotificationRepository(db)
    notification_service = NotificationService(db, notification_repo)
    
    notifications = await notification_service.get_user_notifications(
        current_user.id,
        unread_only=unread_only,
        skip=skip,
        limit=limit
    )
    return notifications

@router.post("/{notification_id}/read")
async def mark_notification_read(
    notification_id: int,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    \"\"\"Mark notification as read\"\"\"
    notification_repo = NotificationRepository(db)
    notification_service = NotificationService(db, notification_repo)
    
    await notification_service.mark_notification_read(notification_id, current_user.id)
    return {"message": "Notification marked as read"}

@router.post("/read-all")
async def mark_all_read(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    \"\"\"Mark all notifications as read\"\"\"
    notification_repo = NotificationRepository(db)
    notification_service = NotificationService(db, notification_repo)
    
    await notification_service.mark_all_read(current_user.id)
    return {"message": "All notifications marked as read"}

@router.get("/unread-count")
async def get_unread_count(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    \"\"\"Get count of unread notifications\"\"\"
    notification_repo = NotificationRepository(db)
    notification_service = NotificationService(db, notification_repo)
    
    count = await notification_service.get_unread_count(current_user.id)
    return {"count": count}
"""

def fix_all_endpoints():
    """Fix all endpoint files"""
    files_to_fix = {
        'auth.py': AUTH_CONTENT,
        'users.py': USERS_CONTENT,
        'patients.py': PATIENTS_CONTENT,
        'ecg_analysis.py': ECG_ANALYSIS_CONTENT,
        'validations.py': VALIDATIONS_CONTENT,
        'notifications.py': NOTIFICATIONS_CONTENT
    }
    
    for filename, content in files_to_fix.items():
        file_path = ENDPOINTS_DIR / filename
        print(f"Fixing {filename}...")
        
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        print(f"✓ {filename} fixed successfully")
    
    print("\nAll endpoints fixed!")

if __name__ == "__main__":
    fix_all_endpoints()
