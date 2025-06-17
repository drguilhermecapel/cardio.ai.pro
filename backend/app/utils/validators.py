"""Validation utilities for the application."""

import numpy as np
from typing import Dict, Any, Optional
from datetime import datetime
import re


def validate_ecg_signal(signal: np.ndarray) -> bool:
    """Validate ECG signal data.

    Args:
        signal: ECG signal array

    Returns:
        True if signal is valid
    """
    if signal is None:
        return False

    if not isinstance(signal, np.ndarray):
        return False

    # Check dimensions
    if signal.ndim not in [1, 2]:
        return False

    # Check signal length
    if signal.shape[0] < 1000:  # Minimum 2 seconds at 500Hz
        return False

    # Check for NaN or Inf values
    if np.any(np.isnan(signal)) or np.any(np.isinf(signal)):
        return False

    return True


def validate_patient_data(data: Dict[str, Any]) -> bool:
    """Validate patient data.

    Args:
        data: Patient data dictionary

    Returns:
        True if data is valid
    """
    if not data:
        return False

    # Required fields
    required_fields = ["name", "birth_date", "gender"]
    for field in required_fields:
        if field not in data or not data[field]:
            return False

    # Validate name
    if not isinstance(data["name"], str) or len(data["name"]) < 2:
        return False

    # Validate birth date
    try:
        birth_date = datetime.fromisoformat(data["birth_date"].replace("Z", "+00:00"))
        if birth_date > datetime.now():
            return False
    except:
        return False

    # Validate gender
    if data["gender"] not in ["M", "F", "O"]:
        return False

    return True


def validate_email(email: str) -> bool:
    """Validate email address.

    Args:
        email: Email address to validate

    Returns:
        True if email is valid
    """
    if not email:
        return False

    pattern = r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"
    return bool(re.match(pattern, email))


def validate_phone_number(phone: str) -> bool:
    """Validate phone number.

    Args:
        phone: Phone number to validate

    Returns:
        True if phone number is valid
    """
    if not phone:
        return False

    # Remove common separators
    cleaned = re.sub(r"[\s\-\.\(\)]", "", phone)

    # Check if it's a valid phone number (10-15 digits)
    if not re.match(r"^\+?\d{10,15}$", cleaned):
        return False

    return True


def validate_medical_record_number(mrn: str) -> bool:
    """Validate medical record number.

    Args:
        mrn: Medical record number to validate

    Returns:
        True if MRN is valid
    """
    if not mrn:
        return False

    # MRN should be alphanumeric, 6-20 characters
    if not re.match(r"^[A-Z0-9]{6,20}$", mrn.upper()):
        return False

    return True


def validate_heart_rate(hr: float) -> bool:
    """Validate heart rate value.

    Args:
        hr: Heart rate in bpm

    Returns:
        True if heart rate is physiologically valid
    """
    # Normal range: 30-250 bpm
    return 30 <= hr <= 250


def validate_blood_pressure(systolic: float, diastolic: float) -> bool:
    """Validate blood pressure values.

    Args:
        systolic: Systolic blood pressure
        diastolic: Diastolic blood pressure

    Returns:
        True if blood pressure values are valid
    """
    # Validate ranges
    if not (50 <= systolic <= 300):
        return False

    if not (30 <= diastolic <= 200):
        return False

    # Systolic should be higher than diastolic
    if systolic <= diastolic:
        return False

    return True
