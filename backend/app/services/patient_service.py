"""
Patient Service - Patient management functionality.
"""

import logging
from datetime import date, datetime

from sqlalchemy.ext.asyncio import AsyncSession

from app.models.patient import Patient
from app.repositories.patient_repository import PatientRepository
from app.schemas.patient import PatientCreate

logger = logging.getLogger(__name__)


class PatientService:
    """Service for patient management."""

    def __init__(self, db: AsyncSession) -> None:
        self.db = db
        self.repository = PatientRepository(db)

    async def create_patient(self, patient_data: PatientCreate, created_by: int) -> Patient:
        """Create a new patient."""
        today = date.today()
        age = today.year - patient_data.date_of_birth.year
        if today.month < patient_data.date_of_birth.month or \
           (today.month == patient_data.date_of_birth.month and today.day < patient_data.date_of_birth.day):
            age -= 1

        bmi = None
        if patient_data.height_cm and patient_data.weight_kg:
            height_m = patient_data.height_cm / 100
            bmi = patient_data.weight_kg / (height_m ** 2)

        patient = Patient(
            patient_id=patient_data.patient_id,
            mrn=patient_data.mrn,
            first_name=patient_data.first_name,
            last_name=patient_data.last_name,
            date_of_birth=patient_data.date_of_birth,
            age=age,
            gender=patient_data.gender,
            phone=patient_data.phone,
            email=patient_data.email,
            address=patient_data.address,
            height_cm=patient_data.height_cm,
            weight_kg=patient_data.weight_kg,
            bmi=bmi,
            blood_type=patient_data.blood_type,
            emergency_contact_name=patient_data.emergency_contact_name,
            emergency_contact_phone=patient_data.emergency_contact_phone,
            emergency_contact_relationship=patient_data.emergency_contact_relationship,
            allergies=patient_data.allergies,
            medications=patient_data.medications,
            medical_history=patient_data.medical_history,
            family_history=patient_data.family_history,
            insurance_provider=patient_data.insurance_provider,
            insurance_number=patient_data.insurance_number,
            consent_for_research=patient_data.consent_for_research,
            consent_date=datetime.utcnow() if patient_data.consent_for_research else None,
            created_by=created_by,
        )

        return await self.repository.create_patient(patient)

    async def get_patient_by_patient_id(self, patient_id: str) -> Patient | None:
        """Get patient by patient ID."""
        return await self.repository.get_patient_by_patient_id(patient_id)

    async def update_patient(self, patient_id: int, update_data: dict) -> Patient | None:
        """Update patient information."""
        if 'height_cm' in update_data or 'weight_kg' in update_data:
            patient = await self.repository.get_patient_by_id(patient_id)
            if patient:
                height_cm = update_data.get('height_cm', patient.height_cm)
                weight_kg = update_data.get('weight_kg', patient.weight_kg)

                if height_cm and weight_kg:
                    height_m = height_cm / 100
                    update_data['bmi'] = weight_kg / (height_m ** 2)

        return await self.repository.update_patient(patient_id, update_data)

    async def get_patients(self, limit: int = 50, offset: int = 0) -> tuple[list[Patient], int]:
        """Get patients with pagination."""
        return await self.repository.get_patients(limit, offset)

    async def search_patients(
        self, query: str, search_fields: list[str], limit: int = 50, offset: int = 0
    ) -> tuple[list[Patient], int]:
        """Search patients."""
        return await self.repository.search_patients(query, search_fields, limit, offset)
